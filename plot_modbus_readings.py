#!/usr/bin/env python3
"""Plot Modbus reading CSV files as time vs flow rate and time vs volume.

Each input file is expected to match Modbus_readings_YYYYMMDD.csv and contain:
  1) epoch timestamp (seconds, float)
  2) flow rate (L/min)
  3) accumulated volume

Outputs two PNG files per input file in the `plots/` directory:
  - <stem>_flow_rate.png
  - <stem>_volume.png
"""
from __future__ import annotations

import csv
import glob
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt


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


def save_plot(x: List[datetime], y: List[float], title: str, ylabel: str, output_path: Path) -> None:
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


def plot_file(path: Path, output_dir: Path) -> None:
    timestamps, flow_rates, volumes = load_series(path)
    if not timestamps:
        return
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


def main() -> None:
    input_pattern = "Modbus_readings_*.csv"
    output_dir = Path("plots")
    for path in iter_input_files(input_pattern):
        plot_file(path, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
