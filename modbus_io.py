#!/usr/bin/env python3
"""Shared IO helpers for Modbus CSV files."""
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
