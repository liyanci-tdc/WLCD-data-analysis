#!/usr/bin/env python3
"""Detect suspicious long-duration water usage.

Flags intervals where usage is continuous for >= 2 hours and either:
1) Flow rate is relatively constant (+/- tolerance).
2) Cumulative volume increases nearly linearly (+/- tolerance).

Writes an interval-based CSV report that marks suspicious long-duration usage.
"""
from __future__ import annotations

import csv
import glob
import json
import statistics
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from modbus_io import iter_input_files, load_series
# =========================
# Detection Parameters
# =========================
# Run mode:
# - "batch": process files once and exit.
# - "live": keep watching the latest file and append alerts as data arrives.
# Example: RUN_MODE = "live"
RUN_MODE = "batch"

# Detection mode:
# - "minute_refined": detect on 1-minute bins, then refine start/end to seconds.
# - "second": detect directly on second-level samples (slower, more precise).
# Example: DETECTION_MODE = "second"
DETECTION_MODE = "minute_refined"

# Minimum length of suspicious usage in seconds.
# Increase to only flag very long events; decrease to catch shorter events.
# Example: MIN_DURATION_S = 60 * 60  # 1 hour
MIN_DURATION_S = 60 * 60

# Allowed relative variation for "constant" flow and "near-linear" volume.
# 0.10 means +/-10%. Lower = stricter, higher = more tolerant.
# Example: TOLERANCE = 0.05  # +/-5%
TOLERANCE = 0.10

# Minimum per-minute volume rate (L/min) to consider the water "flowing".
# Raise to ignore tiny drips; lower to catch very small leaks.
# Example: MIN_RATE_L_MIN = 0.2
MIN_RATE_L_MIN = 1

# Fraction of minutes in a window that must be flowing (>= MIN_RATE_L_MIN).
# 0.95 means 95% of minutes must show flow.
# Example: MIN_FLOW_FRACTION = 0.90
MIN_FLOW_FRACTION = 0.9

# Fraction of flowing minutes that must stay within TOLERANCE of the median rate
# to count as "constant" flow.
# Example: CONSTANT_FRACTION = 0.70
CONSTANT_FRACTION = 0.9

# Second-level detection: step size for the sliding window in seconds.
# Lower = more precise start times, higher = faster runtime.
# Example: SECOND_WINDOW_STEP_S = 1  # check every second
SECOND_WINDOW_STEP_S = 10

# Hysteresis time (seconds): short dropouts <= this time are treated as continuous flow.
# Example: HYSTERESIS_TIME_S = 2  # allow 2-second gaps
HYSTERESIS_TIME_S = 60

# Live mode settings.
# Example: LIVE_INPUT_PATTERN = "data/Modbus_readings_20260127.csv"
LIVE_INPUT_PATTERN = "data/Modbus_readings_*.csv"
# Example: LIVE_POLL_INTERVAL_S = 1
LIVE_POLL_INTERVAL_S = 2
# Example: LIVE_LOOKBACK_S = MIN_DURATION_S * 3
LIVE_LOOKBACK_S = MIN_DURATION_S * 2
# Example: LIVE_FOLLOW_LATEST = False  # stick to first file found
LIVE_FOLLOW_LATEST = True
# Example: LIVE_STATE_PATH = Path("output") / "my_state.json"
LIVE_STATE_PATH = Path("output") / "abnormal_state.json"

# Alert output.
# Example: ALERT_LOG_PATH = Path("output") / "alerts.log"
ALERT_LOG_PATH = Path("output") / "abnormal_alerts.log"
# Example: ALERT_TO_CONSOLE = False
ALERT_TO_CONSOLE = True
# Example: ALERT_TO_LOG = False
ALERT_TO_LOG = True

# Progress output for batch processing.
# Example: SHOW_PROGRESS = False
SHOW_PROGRESS = True

REPORT_HEADER = [
    "file",
    "start_date",
    "start_time",
    "end_date",
    "end_time",
    "duration_hours",
    "note",
    "avg_flow_rate_L_min",
    "avg_volume_rate_L_min",
]


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


def parse_csv_line(line: str) -> Tuple[datetime, float, float] | None:
    try:
        row = next(csv.reader([line]))
    except (csv.Error, ValueError):
        return None
    if len(row) < 3:
        return None
    try:
        timestamp = float(row[0])
        flow_rate = float(row[1])
        volume = float(row[2])
    except ValueError:
        return None
    return datetime.fromtimestamp(timestamp), flow_rate, volume


def resample_minutes(
    timestamps: List[datetime],
    flow_rates: List[float],
    volumes: List[float],
) -> Tuple[List[datetime], List[float], List[float], List[float]]:
    minute_volumes: dict[datetime, float] = {}
    minute_flow_sum: dict[datetime, float] = {}
    minute_flow_count: dict[datetime, int] = {}

    for ts, flow, vol in zip(timestamps, flow_rates, volumes):
        minute = ts.replace(second=0, microsecond=0)
        minute_volumes[minute] = vol
        minute_flow_sum[minute] = minute_flow_sum.get(minute, 0.0) + flow
        minute_flow_count[minute] = minute_flow_count.get(minute, 0) + 1

    minutes = sorted(minute_volumes)
    if len(minutes) < 2:
        return [], [], [], []

    volume_series = [minute_volumes[m] for m in minutes]
    flow_series = [minute_flow_sum[m] / minute_flow_count[m] for m in minutes]

    rates: List[float] = []
    for idx in range(1, len(minutes)):
        delta_min = (minutes[idx] - minutes[idx - 1]).total_seconds() / 60.0
        if delta_min <= 0:
            rate = 0.0
        else:
            rate = (volume_series[idx] - volume_series[idx - 1]) / delta_min
        rates.append(rate)

    # Align to the rate timestamps (end of each minute interval).
    return minutes[1:], flow_series[1:], volume_series[1:], rates


def build_rate_series(
    timestamps: List[datetime],
    volumes: List[float],
) -> Tuple[List[datetime], List[float], List[float]]:
    rate_times: List[datetime] = []
    rates: List[float] = []
    rate_volumes: List[float] = []
    for idx in range(1, len(timestamps)):
        delta_s = (timestamps[idx] - timestamps[idx - 1]).total_seconds()
        if delta_s <= 0:
            rate = 0.0
        else:
            rate = (volumes[idx] - volumes[idx - 1]) / (delta_s / 60.0)
        rate_times.append(timestamps[idx])
        rates.append(rate)
        rate_volumes.append(volumes[idx])
    return rate_times, rates, rate_volumes


def linear_fit_relative_error(times: List[datetime], volumes: List[float]) -> float:
    if len(times) < 2:
        return float("inf")
    xs = [t.timestamp() for t in times]
    ys = volumes
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return float("inf")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]

    total_increase = ys[-1] - ys[0]
    if total_increase <= 0:
        return float("inf")
    max_residual = max(abs(r) for r in residuals)
    return max_residual / total_increase


def is_constant_flow(rates: List[float], tolerance: float, min_fraction: float) -> bool:
    if not rates:
        return False
    median_rate = statistics.median(rates)
    if median_rate == 0:
        return False
    within = sum(1 for rate in rates if abs(rate - median_rate) / median_rate <= tolerance)
    return (within / len(rates)) >= min_fraction


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def average_in_time_range(
    timestamps: List[datetime],
    values: List[float],
    start_time: datetime,
    end_time: datetime,
) -> float | None:
    total = 0.0
    count = 0
    for ts, value in zip(timestamps, values):
        if start_time <= ts <= end_time:
            total += value
            count += 1
    if count == 0:
        return None
    return total / count


def write_report_rows(output_path: Path, rows: List[List[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", newline="") as output_file:
        writer = csv.writer(output_file)
        if not file_exists or output_path.stat().st_size == 0:
            writer.writerow(REPORT_HEADER)
        writer.writerows(rows)


def emit_alert(source_file: str, start_time: datetime, avg_flow: float) -> None:
    message = (
        "ALERT: abnormal usage detected | "
        f"file={source_file} | start={start_time:%Y-%m-%d %H:%M:%S} | "
        f"avg_flow={avg_flow:.3f} L/min"
    )
    if ALERT_TO_CONSOLE:
        print(message)
    if ALERT_TO_LOG:
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ALERT_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")


def refine_interval_to_seconds(
    rate_times: List[datetime],
    rates: List[float],
    start_time: datetime,
    end_time: datetime,
    min_rate: float,
) -> Tuple[datetime, datetime]:
    first_time = None
    last_time = None
    for ts, rate in zip(rate_times, rates):
        if ts < start_time:
            continue
        if ts > end_time:
            break
        if rate >= min_rate:
            if first_time is None:
                first_time = ts
            last_time = ts
    if first_time is None or last_time is None:
        return start_time, end_time
    return first_time, last_time


def find_suspicious_intervals(
    minute_times: List[datetime],
    minute_volumes: List[float],
    minute_rates: List[float],
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
) -> List[Tuple[int, int]]:
    if not minute_times:
        return []
    window = int(min_duration_s / 60)
    if window < 2 or len(minute_rates) < window:
        return []

    flagged: List[Tuple[int, int]] = []
    for start in range(0, len(minute_rates) - window + 1):
        window_rates = minute_rates[start : start + window]
        active_rates = [rate for rate in window_rates if rate >= min_rate]
        flow_fraction = len(active_rates) / window
        if flow_fraction < min_flow_fraction:
            continue

        constant = is_constant_flow(active_rates, tolerance, constant_fraction)
        linear_error = linear_fit_relative_error(
            minute_times[start : start + window],
            minute_volumes[start : start + window],
        )
        linear = linear_error <= tolerance

        if constant or linear:
            flagged.append((start, start + window - 1))

    return _merge_intervals(flagged)


def find_suspicious_intervals_seconds(
    rate_times: List[datetime],
    rate_volumes: List[float],
    rates: List[float],
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
    hysteresis_time_s: float,
    step_s: int,
) -> List[Tuple[int, int]]:
    if not rate_times:
        return []

    flagged: List[Tuple[int, int]] = []
    start = 0
    while start < len(rate_times):
        start_time = rate_times[start]
        end = start
        while end < len(rate_times) and (
            rate_times[end] - start_time
        ).total_seconds() < min_duration_s:
            end += 1
        if end >= len(rate_times):
            break

        gap_ok = True
        for idx in range(start + 1, end + 1):
            if (rate_times[idx] - rate_times[idx - 1]).total_seconds() > hysteresis_time_s:
                gap_ok = False
                break

        if gap_ok:
            window_rates = rates[start : end + 1]
            active_rates = [rate for rate in window_rates if rate >= min_rate]
            flow_fraction = len(active_rates) / len(window_rates)
            if flow_fraction >= min_flow_fraction and active_rates:
                constant = is_constant_flow(active_rates, tolerance, constant_fraction)
                linear_error = linear_fit_relative_error(
                    rate_times[start : end + 1],
                    rate_volumes[start : end + 1],
                )
                linear = linear_error <= tolerance
                if constant or linear:
                    flagged.append((start, end))

        next_time = start_time + timedelta(seconds=step_s)
        while start < len(rate_times) and rate_times[start] < next_time:
            start += 1

    return _merge_intervals(flagged)


def build_intervals_minute_refined(
    timestamps: List[datetime],
    flow_rates: List[float],
    volumes: List[float],
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
) -> Tuple[List[Tuple[datetime, datetime]], List[datetime], List[float], int]:
    if not timestamps:
        return [], [], [], 0
    rate_times, rates, _ = build_rate_series(timestamps, volumes)
    minute_times, _, minute_volumes, minute_rates = resample_minutes(
        timestamps, flow_rates, volumes
    )
    intervals = find_suspicious_intervals(
        minute_times,
        minute_volumes,
        minute_rates,
        min_duration_s,
        tolerance,
        min_rate,
        min_flow_fraction,
        constant_fraction,
    )
    refined: List[Tuple[datetime, datetime]] = []
    for start_idx, end_idx in intervals:
        minute_start = minute_times[start_idx]
        minute_end = minute_times[end_idx]
        refined.append(
            refine_interval_to_seconds(
                rate_times,
                rates,
                minute_start,
                minute_end,
                min_rate,
            )
        )
    return refined, rate_times, rates, len(intervals)


def build_intervals_second(
    timestamps: List[datetime],
    volumes: List[float],
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
    hysteresis_time_s: float,
    step_s: int,
) -> Tuple[List[Tuple[datetime, datetime]], List[datetime], List[float], int]:
    if len(timestamps) < 2:
        return [], [], [], 0
    rate_times, rates, rate_volumes = build_rate_series(timestamps, volumes)
    intervals = find_suspicious_intervals_seconds(
        rate_times,
        rate_volumes,
        rates,
        min_duration_s,
        tolerance,
        min_rate,
        min_flow_fraction,
        constant_fraction,
        hysteresis_time_s,
        step_s,
    )
    resolved = [(rate_times[start], rate_times[end]) for start, end in intervals]
    return resolved, rate_times, rates, len(intervals)


def build_intervals_for_mode(
    detection_mode: str,
    timestamps: List[datetime],
    flow_rates: List[float],
    volumes: List[float],
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
    hysteresis_time_s: float,
    step_s: int,
) -> Tuple[List[Tuple[datetime, datetime]], List[datetime], List[float], int]:
    if detection_mode == "minute_refined":
        return build_intervals_minute_refined(
            timestamps,
            flow_rates,
            volumes,
            min_duration_s,
            tolerance,
            min_rate,
            min_flow_fraction,
            constant_fraction,
        )
    if detection_mode == "second":
        return build_intervals_second(
            timestamps,
            volumes,
            min_duration_s,
            tolerance,
            min_rate,
            min_flow_fraction,
            constant_fraction,
            hysteresis_time_s,
            step_s,
        )
    raise ValueError(
        f"Unknown DETECTION_MODE={detection_mode!r}. "
        "Use 'minute_refined' or 'second'."
    )


def collect_report_rows(
    source_name: str,
    timestamps: List[datetime],
    flow_rates: List[float],
    rate_times: List[datetime],
    rates: List[float],
    intervals: List[Tuple[datetime, datetime]],
    min_rate: float,
    last_reported_end: datetime | None = None,
) -> Tuple[List[List[str]], datetime | None]:
    rows: List[List[str]] = []
    last_end = last_reported_end
    for start_time, end_time in intervals:
        if last_end and start_time <= last_end:
            continue
        avg_flow = average_in_time_range(timestamps, flow_rates, start_time, end_time)
        avg_rate = average_in_time_range(rate_times, rates, start_time, end_time)
        if avg_flow is None or avg_rate is None or avg_rate < min_rate:
            continue
        duration_s = (end_time - start_time).total_seconds()
        rows.append(
            [
                source_name,
                start_time.date().isoformat(),
                start_time.time().replace(microsecond=0).isoformat(),
                end_time.date().isoformat(),
                end_time.time().replace(microsecond=0).isoformat(),
                f"{duration_s / 3600:.2f}",
                "suspicious_long_duration_usage",
                f"{avg_flow:.3f}",
                f"{avg_rate:.3f}",
            ]
        )
        emit_alert(source_name, start_time, avg_flow)
        last_end = end_time
    return rows, last_end


def detect_outliers_batch(
    input_pattern: str,
    output_path: Path,
    detection_mode: str,
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
    hysteresis_time_s: float,
    step_s: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(REPORT_HEADER)
        for path in iter_input_files(input_pattern):
            timestamps, flow_rates, volumes = load_series(path)
            intervals, rate_times, rates, raw_count = build_intervals_for_mode(
                detection_mode,
                timestamps,
                flow_rates,
                volumes,
                min_duration_s,
                tolerance,
                min_rate,
                min_flow_fraction,
                constant_fraction,
                hysteresis_time_s,
                step_s,
            )
            rows, _ = collect_report_rows(
                path.name,
                timestamps,
                flow_rates,
                rate_times,
                rates,
                intervals,
                min_rate,
            )
            if rows:
                writer.writerows(rows)
            if SHOW_PROGRESS:
                print(
                    f"Finished {path.name} | intervals={raw_count} | "
                    f"mode={detection_mode}"
                )


def detect_outliers_live(
    input_pattern: str,
    output_path: Path,
    min_duration_s: float,
    tolerance: float,
    min_rate: float,
    min_flow_fraction: float,
    constant_fraction: float,
    detection_mode: str,
    hysteresis_time_s: float,
    step_s: int,
    poll_interval_s: int,
    lookback_s: float,
    follow_latest: bool,
    state_path: Path,
) -> None:
    state = load_state(state_path)
    last_reported_end = None
    last_reported_file = state.get("last_reported_file")
    last_end_ts = state.get("last_reported_end_ts")
    if last_end_ts is not None:
        try:
            last_reported_end = datetime.fromtimestamp(float(last_end_ts))
        except (TypeError, ValueError):
            last_reported_end = None

    buffer: deque[Tuple[datetime, float, float]] = deque()
    file_path: Path | None = None
    file_pos = 0

    while True:
        if follow_latest or file_path is None:
            latest = resolve_latest_file(input_pattern)
            if latest is None:
                time.sleep(poll_interval_s)
                continue
            if file_path is None or latest != file_path:
                file_path = latest
                file_pos = 0
                buffer.clear()
                if last_reported_file and str(file_path) != last_reported_file:
                    last_reported_end = None

        if file_path is None:
            time.sleep(poll_interval_s)
            continue

        with file_path.open(newline="") as input_file:
            input_file.seek(file_pos)
            for line in input_file:
                parsed = parse_csv_line(line)
                if parsed is None:
                    continue
                buffer.append(parsed)
            file_pos = input_file.tell()

        if buffer:
            latest_ts = buffer[-1][0]
            while buffer and (latest_ts - buffer[0][0]).total_seconds() > lookback_s:
                buffer.popleft()

            timestamps = [item[0] for item in buffer]
            flow_rates = [item[1] for item in buffer]
            volumes = [item[2] for item in buffer]

            intervals, rate_times, rates, _ = build_intervals_for_mode(
                detection_mode,
                timestamps,
                flow_rates,
                volumes,
                min_duration_s,
                tolerance,
                min_rate,
                min_flow_fraction,
                constant_fraction,
                hysteresis_time_s,
                step_s,
            )
            rows, last_reported_end = collect_report_rows(
                file_path.name,
                timestamps,
                flow_rates,
                rate_times,
                rates,
                intervals,
                min_rate,
                last_reported_end,
            )

            if rows:
                write_report_rows(output_path, rows)
                if last_reported_end:
                    save_state(
                        state_path,
                        {
                            "last_reported_end_ts": last_reported_end.timestamp(),
                            "last_reported_file": str(file_path),
                        },
                    )

        time.sleep(poll_interval_s)


def main() -> None:
    input_pattern = "data/Modbus_readings_*.csv"
    output_path = Path("output") / "abnormal_report.csv"
    if RUN_MODE == "batch":
        detect_outliers_batch(
            input_pattern,
            output_path,
            DETECTION_MODE,
            MIN_DURATION_S,
            TOLERANCE,
            MIN_RATE_L_MIN,
            MIN_FLOW_FRACTION,
            CONSTANT_FRACTION,
            HYSTERESIS_TIME_S,
            SECOND_WINDOW_STEP_S,
        )
        print(f"Saved outlier report to {output_path}")
    elif RUN_MODE == "live":
        detect_outliers_live(
            input_pattern,
            output_path,
            MIN_DURATION_S,
            TOLERANCE,
            MIN_RATE_L_MIN,
            MIN_FLOW_FRACTION,
            CONSTANT_FRACTION,
            DETECTION_MODE,
            HYSTERESIS_TIME_S,
            SECOND_WINDOW_STEP_S,
            LIVE_POLL_INTERVAL_S,
            LIVE_LOOKBACK_S,
            LIVE_FOLLOW_LATEST,
            LIVE_STATE_PATH,
        )
    else:
        raise ValueError("RUN_MODE must be 'batch' or 'live'.")


if __name__ == "__main__":
    main()
