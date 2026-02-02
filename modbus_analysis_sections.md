# Modbus daily CSV merge + plots (JupyterLab-style sections)

Use the following sections as separate JupyterLab cells (Markdown headings + code cells).

## 1) Imports

```python
import csv
import glob
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
```

## 2) Helpers (file iteration + merge)

```python
def iter_input_files(pattern: str):
    for filename in sorted(glob.glob(pattern)):
        path = Path(filename)
        if path.is_file():
            yield path


def merge_files(input_pattern: str, output_path: Path) -> None:
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["date", "time", "flow_rate_L_min", "volume"])

        for path in iter_input_files(input_pattern):
            with path.open(newline="") as input_file:
                reader = csv.reader(input_file)
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        timestamp = float(row[0])
                    except ValueError:
                        continue
                    dt = datetime.fromtimestamp(timestamp)
                    writer.writerow([
                        dt.date().isoformat(),
                        dt.time().replace(microsecond=0).isoformat(),
                        row[1],
                        row[2],
                    ])
```

## 3) Helpers (load series + plotting)

```python
def load_series(path: Path):
    timestamps = []
    flow_rates = []
    volumes = []
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


def save_plot(x, y, title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## 4) Merge all daily CSVs into one

```python
merge_files("Modbus_readings_*.csv", Path("merged_modbus_readings.csv"))
print("Merged files into merged_modbus_readings.csv")
```

## 5) Plot each day (time vs flow rate, time vs volume)

```python
output_dir = Path("plots")
for path in iter_input_files("Modbus_readings_*.csv"):
    timestamps, flow_rates, volumes = load_series(path)
    if not timestamps:
        continue
    stem = path.stem
    save_plot(
        timestamps,
        flow_rates,
        title=f"{stem} - Time vs Flow Rate",
        ylabel="Flow Rate (L/min)",
        output_path=output_dir / f"{stem}_flow_rate.png",
    )
    save_plot(
        timestamps,
        volumes,
        title=f"{stem} - Time vs Volume",
        ylabel="Volume",
        output_path=output_dir / f"{stem}_volume.png",
    )
print(f"Saved plots to {output_dir}")
```

## 6) Detect anomalies (long-duration continuous usage)

```python
def within_tolerance(value: float, baseline: float, tolerance: float) -> bool:
    if baseline == 0:
        return value == 0
    return abs(value - baseline) / abs(baseline) <= tolerance


def find_constant_flow_intervals(
    timestamps, flow_rates, min_duration_s, tolerance, max_gap_s
):
    intervals = []
    start = 0
    while start < len(timestamps):
        baseline = flow_rates[start]
        end = start + 1
        last_good = start
        gap_start = None
        while end < len(timestamps):
            if within_tolerance(flow_rates[end], baseline, tolerance):
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
    timestamps, volumes, min_duration_s, tolerance, max_gap_s
):
    if len(timestamps) < 2:
        return []
    rates = []
    for idx in range(1, len(timestamps)):
        delta_v = volumes[idx] - volumes[idx - 1]
        delta_t = (timestamps[idx] - timestamps[idx - 1]).total_seconds()
        if delta_t <= 0:
            rates.append(0.0)
        else:
            rates.append(delta_v / (delta_t / 60.0))
    intervals = []
    start = 0
    while start < len(rates):
        baseline = rates[start]
        end = start + 1
        last_good = start
        gap_start = None
        while end < len(rates):
            if within_tolerance(rates[end], baseline, tolerance):
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


def merge_intervals(intervals):
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


min_duration_s = 2 * 60 * 60
tolerance = 0.10
max_gap_s = 5 * 60
anomaly_rows = []
for path in iter_input_files("Modbus_readings_*.csv"):
    timestamps, flow_rates, volumes = load_series(path)
    if not timestamps:
        continue
    flow_intervals = find_constant_flow_intervals(
        timestamps, flow_rates, min_duration_s, tolerance, max_gap_s
    )
    rate_intervals = find_constant_volume_rate_intervals(
        timestamps, volumes, min_duration_s, tolerance, max_gap_s
    )
    merged_intervals = merge_intervals(flow_intervals + rate_intervals)
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
        anomaly_rows.append(
            {
                "file": path.name,
                "start_date": timestamps[start].date().isoformat(),
                "start_time": timestamps[start].time().replace(microsecond=0).isoformat(),
                "end_date": timestamps[end].date().isoformat(),
                "end_time": timestamps[end].time().replace(microsecond=0).isoformat(),
                "duration_hours": round(duration_s / 3600, 2),
                "note": "suspicious_long_duration_usage",
                "avg_flow_rate_L_min": round(avg_flow, 3),
                "avg_volume_rate_L_min": round(avg_rate, 3),
            }
        )

anomaly_rows[:5]
```

## 7) Save anomaly report (CSV)

```python
with open("outlier_report.csv", "w", newline="") as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=[
            "file",
            "start_date",
            "start_time",
            "end_date",
            "end_time",
            "duration_hours",
            "note",
            "avg_flow_rate_L_min",
            "avg_volume_rate_L_min",
        ],
    )
    writer.writeheader()
    writer.writerows(anomaly_rows)

print("Saved outlier_report.csv")
```
