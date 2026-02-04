#!/usr/bin/env python3
"""Shared utilities for detecting suspicious Modbus usage."""
from __future__ import annotations

import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

REPORT_HEADER = [
    "file",
    "event_id",
    "status",
    "update_time",
    "start_day_num",
    "start_date",
    "start_time",
    "end_day_num",
    "end_date",
    "end_time",
    "duration_hours",
    "note",
    "avg_flow_rate_L_min",
    "total_volume_L",
]


def _safe_print(text: str, *, end: str = "\n", flush: bool = False) -> None:
    try:
        print(text, end=end, flush=flush)
    except OSError:
        return


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
    min_residual = min(residuals)
    if min_residual >= 0:
        return 0.0
    return abs(min_residual) / total_increase


def is_constant_flow(rates: List[float], tolerance: float, min_fraction: float) -> bool:
    if not rates:
        return False
    median_rate = statistics.median(rates)
    if median_rate == 0:
        return False
    # Lower-sided check only: spikes above median are allowed.
    lower_bound = median_rate * (1 - tolerance)
    within = sum(1 for rate in rates if rate >= lower_bound)
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


def volume_delta_in_time_range(
    timestamps: List[datetime],
    volumes: List[float],
    start_time: datetime,
    end_time: datetime,
) -> float | None:
    first_volume = None
    last_volume = None
    for ts, volume in zip(timestamps, volumes):
        if ts < start_time:
            continue
        if ts > end_time:
            break
        if first_volume is None:
            first_volume = volume
        last_volume = volume
    if first_volume is None or last_volume is None:
        return None
    return last_volume - first_volume


def write_report_rows(output_path: Path, rows: List[List[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            with output_path.open(newline="") as existing_file:
                reader = csv.reader(existing_file)
                existing_header = next(reader, None)
                existing_rows = list(reader)
            if existing_header != REPORT_HEADER:
                with output_path.open("w", newline="") as output_file:
                    writer = csv.writer(output_file)
                    writer.writerow(REPORT_HEADER)
                    writer.writerows(existing_rows)
        except OSError:
            pass

    file_exists = output_path.exists()
    with output_path.open("a", newline="") as output_file:
        writer = csv.writer(output_file)
        if not file_exists or output_path.stat().st_size == 0:
            writer.writerow(REPORT_HEADER)
        writer.writerows(rows)


def emit_alert(
    event: str,
    source_file: str,
    start_time: datetime,
    avg_flow: float | None = None,
    end_time: datetime | None = None,
    duration_s: float | None = None,
    pushed_time: datetime | None = None,
    *,
    alert_to_console: bool,
    alert_to_log: bool,
    alert_log_path: Path,
) -> None:
    if pushed_time is None:
        pushed_time = datetime.now()
    if event == "start":
        label = "abnormal usage started"
    elif event == "stop":
        label = "abnormal usage stopped"
    else:
        label = event
    parts = [
        f"ALERT: {label}",
        f"file={source_file}",
        f"start={start_time:%Y-%m-%d %H:%M:%S}",
    ]
    if end_time is not None:
        parts.append(f"end={end_time:%Y-%m-%d %H:%M:%S}")
    if duration_s is not None:
        parts.append(f"duration_hours={duration_s / 3600:.2f}")
    if avg_flow is not None:
        parts.append(f"avg_flow={avg_flow:.3f} L/min")
    parts.append(f"pushed={pushed_time:%Y-%m-%d %H:%M:%S}")
    message = " | ".join(parts)
    if alert_to_console:
        print(message)
    if alert_to_log:
        alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        with alert_log_path.open("a", encoding="utf-8") as log_file:
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


def collect_report_rows(
    source_name: str,
    timestamps: List[datetime],
    flow_rates: List[float],
    volumes: List[float],
    rate_times: List[datetime],
    rates: List[float],
    intervals: List[Tuple[datetime, datetime]],
    min_rate: float,
    last_reported_end: datetime | None = None,
) -> Tuple[List[List[str]], List[dict], datetime | None]:
    rows: List[List[str]] = []
    alerts: List[dict] = []
    last_end = last_reported_end
    for start_time, end_time in intervals:
        if last_end and start_time <= last_end:
            continue
        avg_flow = average_in_time_range(timestamps, flow_rates, start_time, end_time)
        avg_rate = average_in_time_range(rate_times, rates, start_time, end_time)
        volume_delta = volume_delta_in_time_range(timestamps, volumes, start_time, end_time)
        if avg_flow is None or avg_rate is None or avg_rate < min_rate:
            continue
        if volume_delta is None:
            continue
        duration_s = (end_time - start_time).total_seconds()
        rows.append(
            build_report_row(
                source_name=source_name,
                event_id="0",
                status="final",
                update_time=end_time,
                start_time=start_time,
                end_time=end_time,
                duration_s=duration_s,
                avg_flow=avg_flow,
                total_volume=volume_delta,
            )
        )
        alerts.append(
            {
                "source_file": source_name,
                "start_time": start_time,
                "end_time": end_time,
                "avg_flow": avg_flow,
                "duration_s": duration_s,
            }
        )
        last_end = end_time
    return rows, alerts, last_end


def build_report_row(
    *,
    source_name: str,
    event_id: str,
    status: str,
    update_time: datetime,
    start_time: datetime,
    end_time: datetime,
    duration_s: float,
    avg_flow: float,
    total_volume: float,
) -> List[str]:
    start_day_num = str(start_time.weekday() + 1)
    end_day_num = str(end_time.weekday() + 1)
    return [
        source_name,
        event_id,
        status,
        update_time.replace(microsecond=0).isoformat(),
        start_day_num,
        start_time.date().isoformat(),
        start_time.time().replace(microsecond=0).isoformat(),
        end_day_num,
        end_time.date().isoformat(),
        end_time.time().replace(microsecond=0).isoformat(),
        f"{duration_s / 3600:.2f}",
        "suspicious_long_duration_usage",
        f"{avg_flow:.4f}",
        f"{total_volume:.3f}",
    ]


if __name__ == "__main__":
    raise SystemExit("This module is not runnable directly. Use launch.py.")
