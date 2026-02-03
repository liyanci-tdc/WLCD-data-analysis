#!/usr/bin/env python3
"""Build a SQLite database from Modbus CSV files."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

# =========================
# Build Parameters
# =========================
INPUT_PATTERN = "data/Modbus_readings_*.csv"
OUTPUT_DB_PATH = Path("output") / "modbus_readings.sqlite"
OVERWRITE_DB = True
BATCH_SIZE = 10000


def iter_input_files(pattern: str) -> Iterable[Path]:
    return (path for path in sorted(Path().glob(pattern)) if path.is_file())


def ensure_db_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            flow_rate REAL,
            volume REAL,
            col4 REAL,
            col5 REAL,
            col6 REAL,
            source_file TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(timestamp)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_readings_file_ts ON readings(source_file, timestamp)"
    )


def parse_row(row: List[str], source_file: str) -> Tuple:
    values = [None] * 6
    for idx in range(min(6, len(row))):
        try:
            values[idx] = float(row[idx])
        except ValueError:
            values[idx] = None
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        source_file,
    )


def load_file(conn: sqlite3.Connection, path: Path) -> int:
    rows: List[Tuple] = []
    inserted = 0
    with path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            if len(row) < 3:
                continue
            parsed = parse_row(row, path.name)
            if parsed[0] is None:
                continue
            rows.append(parsed)
            if len(rows) >= BATCH_SIZE:
                conn.executemany(
                    """
                    INSERT INTO readings (
                        timestamp, flow_rate, volume, col4, col5, col6, source_file
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                inserted += len(rows)
                rows.clear()
        if rows:
            conn.executemany(
                """
                INSERT INTO readings (
                    timestamp, flow_rate, volume, col4, col5, col6, source_file
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted += len(rows)
    return inserted


def main() -> None:
    ensure_db_path(OUTPUT_DB_PATH, OVERWRITE_DB)
    total = 0
    with sqlite3.connect(OUTPUT_DB_PATH) as conn:
        init_db(conn)
        for path in iter_input_files(INPUT_PATTERN):
            total += load_file(conn, path)
            conn.commit()
    print(f"Saved {total} rows to {OUTPUT_DB_PATH}")


if __name__ == "__main__":
    main()
