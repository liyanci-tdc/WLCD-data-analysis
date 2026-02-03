#!/usr/bin/env python3
"""Replay static Modbus CSV data as a live stream.

Reads from static_data/ and appends rows to live_data/ while respecting
the original timestamp spacing (scaled by SPEEDUP).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Iterable

from modbus_io import iter_input_files

# =========================
# Replay Settings
# =========================
INPUT_PATTERN = "static_data/Modbus_readings_*.csv"
OUTPUT_DIR = Path("live_data")

# Replay speed:
# SPEEDUP = 1.0 is real-time, 10.0 is 10x faster, etc.
# Cap/clamp: MIN_SPEEDUP <= SPEEDUP <= MAX_SPEEDUP (hardware safety limits).
# Example: SPEEDUP = 25.0
SPEEDUP = 6000.0
# Lower/upper bounds for SPEEDUP (hardware safety limits).
MIN_SPEEDUP = 0.1
MAX_SPEEDUP = 9999.0

# Output time for the first column in live_data.
# OUTPUT_TIME_MODE:
# - "raw":     keep the original epoch timestamp from the source CSV.
# - "clock":   HHMMSS clock time from the timestamp.
# - "elapsed": HHMMSS since the first row in each file (starts at 000000).
OUTPUT_TIME_MODE = "raw"  # "raw", "clock", or "elapsed"
OUTPUT_TIME_FORMAT = "%H%M%S"

# Progress line during replay (single-line update).
# Example: SHOW_PROGRESS = True
SHOW_PROGRESS = True
PROGRESS_INTERVAL_SEC = 10.0

# Restrict replay range by date (YYYYMMDD). Use None for full range.
# Example: START_DAY = "20251015"
START_DAY = "20260120"
# Example: END_DAY = "20251020"
END_DAY = None

# Whether to loop the dataset forever.
# Example: LOOP = False
LOOP = False

# Remove old live_data files at startup.
# Example: CLEAR_OUTPUT_ON_START = False
CLEAR_OUTPUT_ON_START = True


def clear_live_data(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("Modbus_readings_*.csv"):
        path.unlink()


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


def replay_file(
    path: Path,
    output_dir: Path,
    speedup: float,
    prev_ts: float | None,
    progress_prefix: str,
) -> float | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / path.name
    with path.open(newline="") as input_file, output_path.open("a", newline="") as output_file:
        first_ts = None
        last_ts_file = None
        if SHOW_PROGRESS:
            first_ts, last_ts_file = _scan_first_last_ts(input_file)
            input_file.seek(0)
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        writerow = writer.writerow
        sleep = time.sleep
        localtime = time.localtime
        strftime = time.strftime
        elapsed_mode = OUTPUT_TIME_MODE == "elapsed"
        clock_mode = OUTPUT_TIME_MODE == "clock"
        format_elapsed = _format_hhmmss_compact
        last_ts = prev_ts
        last_progress = time.monotonic()
        last_progress_len = 0
        progress_enabled = SHOW_PROGRESS and last_ts_file is not None
        progress_span = (
            max(0.0, last_ts_file - first_ts) if progress_enabled and first_ts is not None else 0.0
        )
        use_single_line = SHOW_PROGRESS and sys.stdout.isatty()
        if SHOW_PROGRESS:
            line = f"{progress_prefix} | 0%"
            if use_single_line:
                print(line, end="\r", flush=True)
            else:
                print(line, flush=True)
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
                row[0] = strftime(OUTPUT_TIME_FORMAT, localtime(ts))
            writerow(row)
            last_ts = ts
            if progress_enabled:
                now = time.monotonic()
                if now - last_progress >= PROGRESS_INTERVAL_SEC:
                    if progress_span > 0 and first_ts is not None:
                        pct = int((ts - first_ts) / progress_span * 100)
                        pct = max(0, min(100, pct))
                    else:
                        pct = 0
                    line = f"{progress_prefix} | {pct}%"
                    padding = " " * max(0, last_progress_len - len(line))
                    if use_single_line:
                        print(line + padding, end="\r", flush=True)
                    else:
                        print(line + padding, flush=True)
                    last_progress_len = len(line)
                    last_progress = now
        if SHOW_PROGRESS:
            pct = 100 if progress_span > 0 else 0
            line = f"{progress_prefix} | {pct}%"
            padding = " " * max(0, last_progress_len - len(line))
            if use_single_line:
                print(line + padding)
            else:
                print(line + padding, flush=True)
    return last_ts


def iter_files(pattern: str) -> Iterable[Path]:
    files = list(iter_input_files(pattern))
    if START_DAY or END_DAY:
        selected = []
        for path in files:
            stem = path.stem
            if not stem.startswith("Modbus_readings_"):
                continue
            day = stem.split("_", 2)[-1]
            if START_DAY and day < START_DAY:
                continue
            if END_DAY and day > END_DAY:
                continue
            selected.append(path)
        return selected
    return files


def main() -> None:
    speedup = max(MIN_SPEEDUP, min(SPEEDUP, MAX_SPEEDUP))
    if speedup != SPEEDUP and SHOW_PROGRESS:
        print(
            f"SPEEDUP {SPEEDUP} out of bounds; clamped to {speedup} "
            f"(range {MIN_SPEEDUP}..{MAX_SPEEDUP})."
        )
    if CLEAR_OUTPUT_ON_START:
        clear_live_data(OUTPUT_DIR)

    loop_count = 0
    while True:
        loop_count += 1
        last_ts = None
        for path in iter_files(INPUT_PATTERN):
            day = path.stem.replace("Modbus_readings_", "")
            progress_prefix = f"Replaying {path.name} (day {day}) | loop {loop_count}"
            last_ts = replay_file(path, OUTPUT_DIR, speedup, last_ts, progress_prefix)
        if not LOOP:
            break
        if SHOW_PROGRESS:
            print("Replay loop completed, restarting...")


if __name__ == "__main__":
    main()
