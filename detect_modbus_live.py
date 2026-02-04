#!/usr/bin/env python3
"""Live detector for abnormal Modbus usage."""
from __future__ import annotations

import glob
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import detect_modbus_core as core
import detect_modbus_stream as stream


def resolve_latest_file(pattern: str) -> Path | None:
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return Path(files[-1])


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def detect_outliers_live(
    input_pattern: str,
    output_path: Path,
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
    end_gap_s: float,
    update_interval_s: float,
    poll_interval_s: int,
    lookback_s: float,
    follow_latest: bool,
    state_path: Path,
    *,
    alert_to_console: bool,
    alert_to_log: bool,
    alert_log_path: Path,
    alert_include_stop: bool,
    stop_event: threading.Event | None = None,
    stop_after_idle_s: float | None = None,
) -> None:
    state = load_state(state_path)
    last_reported_end = None
    last_reported_file = state.get("last_reported_file")
    last_end_ts = state.get("last_reported_end_ts")
    next_event_id = state.get("next_event_id", 1)
    if last_end_ts is not None:
        try:
            last_reported_end = datetime.fromtimestamp(float(last_end_ts))
        except (TypeError, ValueError):
            last_reported_end = None

    file_path: Path | None = None
    file_pos = 0
    last_seen_ts: datetime | None = None
    last_new_data_time = time.monotonic()
    detector = stream.StreamingDetector(
        source_file="",
        min_duration_s=min_duration_s,
        tolerance=tolerance,
        min_rate=min_rate,
        min_flow_fraction=min_flow_fraction,
        constant_fraction=constant_fraction,
        end_gap_s=end_gap_s,
        update_interval_s=update_interval_s,
        lookback_s=lookback_s,
        event_id_start=next_event_id,
        last_reported_end=last_reported_end,
    )

    while True:
        if follow_latest or file_path is None:
            latest = resolve_latest_file(input_pattern)
            if latest is None:
                if stop_event and stop_event.is_set():
                    if stop_after_idle_s is None:
                        break
                    if (time.monotonic() - last_new_data_time) >= stop_after_idle_s:
                        break
                time.sleep(poll_interval_s)
                continue
            if file_path is None or latest != file_path:
                # Finalize any active event before switching to a new live file.
                if detector.active_event:
                    rows, ended_event = detector.finalize_active()
                    if rows:
                        core.write_report_rows(output_path, rows)
                    if ended_event and alert_include_stop:
                        core.emit_alert(
                            "stop",
                            ended_event["source_file"],
                            ended_event["start_time"],
                            end_time=ended_event["end_time"],
                            duration_s=ended_event["duration_s"],
                            alert_to_console=alert_to_console,
                            alert_to_log=alert_to_log,
                            alert_log_path=alert_log_path,
                        )
                    if ended_event:
                        save_state(
                            state_path,
                            {
                                "last_reported_end_ts": detector.last_reported_end.timestamp(),
                                "last_reported_file": str(file_path),
                                "next_event_id": detector.event_id,
                            },
                        )
                file_path = latest
                file_pos = 0
                last_seen_ts = None
                # Reset window state, but keep event IDs monotonic across files.
                detector.reset(keep_event_id=True)
                detector.set_source(file_path.name)
                if last_reported_file and str(file_path) != last_reported_file:
                    last_reported_end = None
                    detector.last_reported_end = None

        if file_path is None:
            time.sleep(poll_interval_s)
            continue

        new_lines = 0
        with file_path.open(newline="") as input_file:
            input_file.seek(file_pos)
            for line in input_file:
                parsed = core.parse_csv_line(line)
                if parsed is None:
                    continue
                ts, flow, vol = parsed
                if last_seen_ts is not None and ts < last_seen_ts:
                    last_reported_end = None
                    detector.reset(keep_event_id=True)
                detector.set_source(file_path.name)
                rows, started_event, ended_event = detector.process_sample(ts, flow, vol)
                if rows:
                    core.write_report_rows(output_path, rows)
                if started_event:
                    core.emit_alert(
                        "start",
                        started_event["source_file"],
                        started_event["start_time"],
                        avg_flow=started_event["avg_flow"],
                        alert_to_console=alert_to_console,
                        alert_to_log=alert_to_log,
                        alert_log_path=alert_log_path,
                    )
                if ended_event and alert_include_stop:
                    core.emit_alert(
                        "stop",
                        ended_event["source_file"],
                        ended_event["start_time"],
                        end_time=ended_event["end_time"],
                        duration_s=ended_event["duration_s"],
                        alert_to_console=alert_to_console,
                        alert_to_log=alert_to_log,
                        alert_log_path=alert_log_path,
                    )
                if ended_event:
                    detector.last_reported_end = ended_event["end_time"]
                    save_state(
                        state_path,
                        {
                            "last_reported_end_ts": detector.last_reported_end.timestamp(),
                            "last_reported_file": str(file_path),
                            "next_event_id": detector.event_id,
                        },
                    )
                last_seen_ts = ts
                new_lines += 1
            file_pos = input_file.tell()

        if new_lines:
            last_new_data_time = time.monotonic()

        if stop_event and stop_event.is_set():
            if stop_after_idle_s is None:
                if detector.active_event:
                    rows, ended_event = detector.finalize_active()
                    if rows:
                        core.write_report_rows(output_path, rows)
                    if ended_event and alert_include_stop:
                        core.emit_alert(
                            "stop",
                            ended_event["source_file"],
                            ended_event["start_time"],
                            end_time=ended_event["end_time"],
                            duration_s=ended_event["duration_s"],
                            alert_to_console=alert_to_console,
                            alert_to_log=alert_to_log,
                            alert_log_path=alert_log_path,
                        )
                    if ended_event and file_path is not None:
                        save_state(
                            state_path,
                            {
                                "last_reported_end_ts": detector.last_reported_end.timestamp(),
                                "last_reported_file": str(file_path),
                                "next_event_id": detector.event_id,
                            },
                        )
                break
            if (time.monotonic() - last_new_data_time) >= stop_after_idle_s:
                if detector.active_event:
                    rows, ended_event = detector.finalize_active()
                    if rows:
                        core.write_report_rows(output_path, rows)
                    if ended_event and alert_include_stop:
                        core.emit_alert(
                            "stop",
                            ended_event["source_file"],
                            ended_event["start_time"],
                            end_time=ended_event["end_time"],
                            duration_s=ended_event["duration_s"],
                            alert_to_console=alert_to_console,
                            alert_to_log=alert_to_log,
                            alert_log_path=alert_log_path,
                        )
                    if ended_event and file_path is not None:
                        save_state(
                            state_path,
                            {
                                "last_reported_end_ts": detector.last_reported_end.timestamp(),
                                "last_reported_file": str(file_path),
                                "next_event_id": detector.event_id,
                            },
                        )
                break
        time.sleep(poll_interval_s)


if __name__ == "__main__":
    raise SystemExit("This module is not runnable directly. Use launch.py.")
