#!/usr/bin/env python3
"""Batch detector for abnormal Modbus usage."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List

import detect_modbus_core as core
import detect_modbus_stream as stream
from modbus_io import iter_input_files, load_series


def detect_outliers_batch(
    input_pattern: str,
    output_path: Path,
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
    end_gap_s: float,
    *,
    batch_start_day: str | None,
    batch_end_day: str | None,
    alert_to_console: bool,
    alert_to_log: bool,
    alert_log_path: Path,
    alert_include_stop: bool,
    show_progress: bool,
) -> None:
    def filter_paths(paths: List[Path]) -> List[Path]:
        if not batch_start_day and not batch_end_day:
            return paths
        selected = []
        for path in paths:
            stem = path.stem
            if not stem.startswith("Modbus_readings_"):
                continue
            day = stem.split("_", 2)[-1]
            if batch_start_day and day < batch_start_day:
                continue
            if batch_end_day and day > batch_end_day:
                continue
            selected.append(path)
        return selected

    paths = filter_paths(list(iter_input_files(input_pattern)))
    total = len(paths)
    last_progress_len = 0
    use_single_line = show_progress and sys.stdout.isatty()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(core.REPORT_HEADER)
        for idx, path in enumerate(paths):
            if show_progress and total:
                pct = int((idx / total) * 100)
                line = f"Batch {idx + 1}/{total} {path.name} | {pct}%"
                padding = " " * max(0, last_progress_len - len(line))
                if use_single_line:
                    core._safe_print(line + padding, end="\r", flush=True)
                else:
                    core._safe_print(line + padding, flush=True)
                last_progress_len = len(line)

            timestamps, flow_rates, volumes = load_series(path)
            detector = stream.StreamingDetector(
                source_file=path.name,
                min_duration_s=min_duration_s,
                tolerance=tolerance,
                min_rate=min_rate,
                min_flow_fraction=min_flow_fraction,
                constant_fraction=constant_fraction,
                end_gap_s=end_gap_s,
                update_interval_s=0.0,
                lookback_s=max(min_duration_s, end_gap_s) * 2,
            )
            for ts, flow, vol in zip(timestamps, flow_rates, volumes):
                rows, started_event, ended_event = detector.process_sample(ts, flow, vol)
                if rows:
                    writer.writerows(rows)
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
            final_rows, ended_event = detector.finalize_active()
            if final_rows:
                writer.writerows(final_rows)
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

            if show_progress and total:
                pct = int(((idx + 1) / total) * 100)
                line = f"Batch {idx + 1}/{total} {path.name} | {pct}%"
                padding = " " * max(0, last_progress_len - len(line))
                if use_single_line:
                    core._safe_print(line + padding, flush=True)
                else:
                    core._safe_print(line + padding, flush=True)
                last_progress_len = len(line)

    if show_progress and total and use_single_line:
        core._safe_print("", flush=True)


if __name__ == "__main__":
    raise SystemExit("This module is not runnable directly. Use launch.py.")
