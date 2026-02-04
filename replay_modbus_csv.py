#!/usr/bin/env python3
"""Replay static Modbus CSV data as a live stream.

Reads from static_data/ and appends rows to live_data/ while respecting
the original timestamp spacing (scaled by SPEEDUP).
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from modbus_io import iter_input_files


@dataclass
class ReplayConfig:
    input_pattern: str
    output_dir: Path
    speedup: float
    min_speedup: float
    max_speedup: float
    start_day: str | None
    end_day: str | None
    loop: bool
    show_progress: bool
    progress_interval_sec: float
    output_time_mode: str
    output_time_format: str
    flush_each_row: bool


def _scan_first_last_ts(input_file) -> tuple[float | None, float | None]:
    first_ts = None
    last_ts = None
    reader = csv.reader(input_file)
    for row in reader:
        if not row:
            continue
        try:
            ts = float(row[0])
        except ValueError:
            continue
        if first_ts is None:
            first_ts = ts
        last_ts = ts
    return first_ts, last_ts


def _format_hhmmss_compact(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}"


def _safe_print(text: str, *, end: str = "\n", flush: bool = False) -> None:
    try:
        print(text, end=end, flush=flush)
    except OSError:
        # Ignore broken/invalid stdout (e.g., when output is closed).
        return


def replay_file(
    path: Path,
    config: ReplayConfig,
    speedup: float,
    prev_ts: float | None,
    progress_prefix: str,
) -> float | None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / path.name
    with path.open(newline="") as input_file, output_path.open("a", newline="") as output_file:
        first_ts = None
        last_ts_file = None
        if config.show_progress:
            first_ts, last_ts_file = _scan_first_last_ts(input_file)
            input_file.seek(0)
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        writerow = writer.writerow
        sleep = time.sleep
        localtime = time.localtime
        strftime = time.strftime
        elapsed_mode = config.output_time_mode == "elapsed"
        clock_mode = config.output_time_mode == "clock"
        format_elapsed = _format_hhmmss_compact
        last_ts = prev_ts
        last_progress = time.monotonic()
        last_progress_len = 0
        progress_enabled = config.show_progress and last_ts_file is not None
        progress_span = (
            max(0.0, last_ts_file - first_ts) if progress_enabled and first_ts is not None else 0.0
        )
        use_single_line = config.show_progress and sys.stdout.isatty()
        if config.show_progress:
            line = f"{progress_prefix} | 0%"
            if use_single_line:
                _safe_print(line, end="\r", flush=True)
            else:
                _safe_print(line, flush=True)
            last_progress_len = len(line)
        for row in reader:
            if not row:
                continue
            try:
                ts = float(row[0])
            except ValueError:
                continue
            if first_ts is None:
                first_ts = ts
                if progress_enabled and last_ts_file is not None:
                    progress_span = max(0.0, last_ts_file - first_ts)
            if last_ts is not None:
                delta = ts - last_ts
                if delta > 0:
                    sleep(delta / speedup)
            if elapsed_mode and first_ts is not None:
                elapsed = max(0.0, ts - first_ts)
                row[0] = format_elapsed(elapsed)
            elif clock_mode:
                row[0] = strftime(config.output_time_format, localtime(ts))
            writerow(row)
            if config.flush_each_row:
                output_file.flush()
            last_ts = ts
            if progress_enabled:
                now = time.monotonic()
                if now - last_progress >= config.progress_interval_sec:
                    if progress_span > 0 and first_ts is not None:
                        pct = int((ts - first_ts) / progress_span * 100)
                        pct = max(0, min(100, pct))
                    else:
                        pct = 0
                    line = f"{progress_prefix} | {pct}%"
                    padding = " " * max(0, last_progress_len - len(line))
                    if use_single_line:
                        _safe_print(line + padding, end="\r", flush=True)
                    else:
                        _safe_print(line + padding, flush=True)
                    last_progress_len = len(line)
                    last_progress = now
        if config.show_progress:
            pct = 100 if progress_span > 0 else 0
            line = f"{progress_prefix} | {pct}%"
            padding = " " * max(0, last_progress_len - len(line))
            if use_single_line:
                _safe_print(line + padding)
            else:
                _safe_print(line + padding, flush=True)
    return last_ts


def iter_files(pattern: str, start_day: str | None, end_day: str | None) -> Iterable[Path]:
    files = list(iter_input_files(pattern))
    if start_day or end_day:
        selected = []
        for path in files:
            stem = path.stem
            if not stem.startswith("Modbus_readings_"):
                continue
            day = stem.split("_", 2)[-1]
            if start_day and day < start_day:
                continue
            if end_day and day > end_day:
                continue
            selected.append(path)
        return selected
    return files


def run_replay(config: ReplayConfig) -> None:
    speedup = max(config.min_speedup, min(config.speedup, config.max_speedup))
    if speedup != config.speedup and config.show_progress:
        print(
            f"SPEEDUP {config.speedup} out of bounds; clamped to {speedup} "
            f"(range {config.min_speedup}..{config.max_speedup})."
        )
    loop_count = 0
    while True:
        loop_count += 1
        last_ts = None
        for path in iter_files(config.input_pattern, config.start_day, config.end_day):
            day = path.stem.replace("Modbus_readings_", "")
            progress_prefix = f"Replaying {path.name} (day {day}) | loop {loop_count}"
            last_ts = replay_file(path, config, speedup, last_ts, progress_prefix)
        if not config.loop:
            break
        if config.show_progress:
            print("Replay loop completed, restarting...")


if __name__ == "__main__":
    raise SystemExit("This module is not runnable directly. Use launch.py.")
