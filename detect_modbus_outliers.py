#!/usr/bin/env python3
"""Detect continuous, near-constant usage anomalies.

Flags intervals where flow stays relatively constant (±10%) for >= 2 hours,
or volume increases at a relatively constant rate (±10%) for >= 2 hours.

Writes an interval-based CSV report that marks suspicious long-duration usage.
"""
from __future__ import annotations

import csv
import glob
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


def iter_input_files(pattern: str) -> Iterable[Path]:
    for filename in sorted(glob.glob(pattern)):
        path = Path(filename)
        if path.is_file():
            yield path


def load_series(path: Path) -> Tuple[List[datetime], List[float], List[float]]:
    timestamps: List[datetime] = []
    flow_rates: List[float] = []
    volumes: List[float] = []
    with path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                timestamp = float(row[0])
                flow_rate = float(row[1])
                volume = float(row[2])
            except ValueError:
                continue
            timestamps.append(datetime.fromtimestamp(timestamp))
            flow_rates.append(flow_rate)
            volumes.append(volume)
    return timestamps, flow_rates, volumes


def _within_tolerance(value: float, baseline: float, tolerance: float) -> bool:
    if baseline == 0:
        return value == 0
    return abs(value - baseline) / abs(baseline) <= tolerance


def find_constant_flow_intervals(
    timestamps: List[datetime],
    flow_rates: List[float],
    min_duration_s: float,
    tolerance: float,
    max_gap_s: float,
) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start = 0
    while start < len(timestamps):
        baseline = flow_rates[start]
        end = start + 1
        last_good = start
        gap_start = None
        while end < len(timestamps):
            if _within_tolerance(flow_rates[end], baseline, tolerance):
                last_good = end
                gap_start = None
            else:
                if gap_start is None:
                    gap_start = end
                gap_duration = (timestamps[end] - timestamps[gap_start]).total_seconds()
                if gap_duration > max_gap_s:
                    break
            end += 1
        if last_good > start:
            duration_s = (timestamps[last_good] - timestamps[start]).total_seconds()
            if duration_s >= min_duration_s:
                intervals.append((start, last_good))
        start = max(end, start + 1)
    return intervals


def find_constant_volume_rate_intervals(
    timestamps: List[datetime],
    volumes: List[float],
    min_duration_s: float,
    tolerance: float,
    max_gap_s: float,
) -> List[Tuple[int, int]]:
    if len(timestamps) < 2:
        return []
    rates: List[float] = []
    for idx in range(1, len(timestamps)):
        delta_v = volumes[idx] - volumes[idx - 1]
        delta_t = (timestamps[idx] - timestamps[idx - 1]).total_seconds()
        if delta_t <= 0:
            rates.append(0.0)
        else:
            rates.append(delta_v / (delta_t / 60.0))
    intervals: List[Tuple[int, int]] = []
    start = 0
    while start < len(rates):
        baseline = rates[start]
        end = start + 1
        last_good = start
        gap_start = None
        while end < len(rates):
            if _within_tolerance(rates[end], baseline, tolerance):
                last_good = end
                gap_start = None
            else:
                if gap_start is None:
                    gap_start = end
                gap_duration = (timestamps[end + 1] - timestamps[gap_start]).total_seconds()
                if gap_duration > max_gap_s:
                    break
            end += 1
        if last_good > start:
            duration_s = (timestamps[last_good + 1] - timestamps[start]).total_seconds()
            if duration_s >= min_duration_s:
                intervals.append((start, last_good + 1))
        start = max(end, start + 1)
    return intervals


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_outliers(
    input_pattern: str,
    output_path: Path,
    min_duration_s: float,
    tolerance: float,
    max_gap_s: float,
) -> None:
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
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
        )
        for path in iter_input_files(input_pattern):
            timestamps, flow_rates, volumes = load_series(path)
            if not timestamps:
                continue
            flow_intervals = find_constant_flow_intervals(
                timestamps, flow_rates, min_duration_s, tolerance, max_gap_s
            )
            rate_intervals = find_constant_volume_rate_intervals(
                timestamps, volumes, min_duration_s, tolerance, max_gap_s
            )
            merged_intervals = _merge_intervals(flow_intervals + rate_intervals)
            for start, end in merged_intervals:
                duration_s = (timestamps[end] - timestamps[start]).total_seconds()
                avg_flow = sum(flow_rates[start : end + 1]) / (end - start + 1)
                if avg_flow < 0.5:
                    continue
                rates = []
                for idx in range(start + 1, end + 1):
                    delta_v = volumes[idx] - volumes[idx - 1]
                    delta_t = (timestamps[idx] - timestamps[idx - 1]).total_seconds()
                    if delta_t <= 0:
                        continue
                    rates.append(delta_v / (delta_t / 60.0))
                avg_rate = sum(rates) / len(rates) if rates else 0.0
                writer.writerow(
                    [
                        path.name,
                        timestamps[start].date().isoformat(),
                        timestamps[start].time().replace(microsecond=0).isoformat(),
                        timestamps[end].date().isoformat(),
                        timestamps[end].time().replace(microsecond=0).isoformat(),
                        f"{duration_s / 3600:.2f}",
                        "suspicious_long_duration_usage",
                        f"{avg_flow:.3f}",
                        f"{avg_rate:.3f}",
                    ]
                )


def main() -> None:
    input_pattern = "Modbus_readings_*.csv"
    output_path = Path("outlier_report.csv")
    min_duration_s = 2 * 60 * 60
    tolerance = 0.10
    max_gap_s = 5 * 60
    detect_outliers(input_pattern, output_path, min_duration_s, tolerance, max_gap_s)
    print(f"Saved outlier report to {output_path}")


if __name__ == "__main__":
    main()
