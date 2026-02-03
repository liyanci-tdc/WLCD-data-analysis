#!/usr/bin/env python3
"""Merge Modbus reading CSV files into a single normalized dataset.

Input files are expected to match Modbus_readings_YYYYMMDD.csv and contain
at least three columns per row:
  1) epoch timestamp (seconds, float)
  2) flow rate (L/min)
  3) accumulated volume

Output columns:
  1) date (YYYY-MM-DD)
  2) time (HH:MM:SS)
  3) flow rate
  4) volume
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from modbus_io import iter_input_files


def merge_files(input_pattern: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
                    writer.writerow(
                        [
                            dt.date().isoformat(),
                            dt.time().replace(microsecond=0).isoformat(),
                            row[1],
                            row[2],
                        ]
                    )


def main() -> None:
    input_pattern = "data/Modbus_readings_*.csv"
    output_path = Path("output") / "merged_modbus_readings.csv"
    merge_files(input_pattern, output_path)
    print(f"Merged files into {output_path}")


if __name__ == "__main__":
    main()
