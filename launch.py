#!/usr/bin/env python3
"""Launch replay + detector features together."""
from __future__ import annotations

import atexit
import csv
import os
import threading
import time
from collections import Counter
from pathlib import Path

import clear_previous
import detect_modbus_batch as detect_batch
import detect_modbus_live as detect_live
import replay_modbus_csv as replay

# =========================
# Configuration
# =========================
# Mode switches.
DEBUG_MODE = 0  # 0 = off, 1 = on (runs batch then live + compare).
REPLAY_MODE = 0  # 0 = off, 1 = on.
DETECT_MODE = 1  # 0 = off, 1 = batch, 2 = live.

# Housekeeping.
CLEAR_PREVIOUS_LIVE_DATA = 1  # 0 = keep live_data, 1 = clear live_data.
CLEAR_PREVIOUS_OUTPUT = 1  # 0 = keep output, 1 = clear output.

# Shared date range (YYYYMMDD) for replay + batch detector.
RANGE_START_DAY = "20260120"  # Inclusive (None = no lower bound).
RANGE_END_DAY = "20260130"  # Inclusive (None = no upper bound).

# Replay settings (simulation).
REPLAY_COMMON_INPUT_PATTERN = "static_data/Modbus_readings_*.csv"  # Source CSV glob.
REPLAY_COMMON_OUTPUT_DIR = Path("live_data")  # Output folder for live files.
REPLAY_COMMON_LOOP = False  # Loop over the date range forever.
REPLAY_SPEED_VALUE = 1000.0  # 1.0 = real-time, 10.0 = 10x.
REPLAY_SPEED_MIN = 0.1  # Lower bound (hardware safety).
REPLAY_SPEED_MAX = 9999.0  # Upper bound (hardware safety).
REPLAY_PROGRESS_ENABLED = True  # Print single-line progress while replaying.
REPLAY_PROGRESS_INTERVAL_SEC = 5.0  # Progress update interval in seconds.
REPLAY_TIME_MODE = "raw"  # "raw" (epoch), "clock" (HHMMSS), "elapsed" (HHMMSS).
REPLAY_TIME_FORMAT = "%H%M%S"  # Only used when TIME_MODE="clock".
REPLAY_OUTPUT_FLUSH_EACH_ROW = True  # True = immediate live visibility (slower).

# Detector settings (common).
DETECT_BATCH_OUTPUT_PATH = Path("output") / "abnormal_report_batch.csv"  # Batch CSV report.
DETECT_LIVE_OUTPUT_PATH = Path("output") / "abnormal_report_live.csv"  # Live CSV report.
DETECT_COMMON_MIN_DURATION_S = 10 * 60  # Minimum suspicious duration in seconds.
DETECT_COMMON_TOLERANCE = 0.20  # Lower-side tolerance; high spikes are allowed.
DETECT_COMMON_MIN_RATE_L_MIN = 1  # Minimum flow rate to count as flowing (L/min).
DETECT_COMMON_MIN_FLOW_FRACTION = 0.8  # Fraction of window minutes that must show flow.
DETECT_COMMON_CONSTANT_FRACTION = 0.8  # Fraction of flowing minutes that must be "constant" (0 = skip).
DETECT_COMMON_ALERT_LOG_PATH = Path("output") / "abnormal_alerts.log"  # Alert log file.
DETECT_COMMON_ALERT_TO_CONSOLE = True  # Print alerts to console.
DETECT_COMMON_ALERT_TO_LOG = False  # Write alerts to log file.
DETECT_COMMON_ALERT_INCLUDE_STOP = False  # Emit stop alerts as well as start alerts.

# Detector settings (live).
DETECT_LIVE_INPUT_PATTERN = "live_data/Modbus_readings_*.csv"  # Live file glob.
DETECT_LIVE_END_GAP_S = 120  # End event if no suspicious/continuous flow for this long (seconds).
DETECT_LIVE_UPDATE_INTERVAL_S = 10 * 60  # Write ongoing rows this often (seconds).
DETECT_LIVE_POLL_INTERVAL_S = 5  # How often to poll the live file(s).
DETECT_LIVE_LOOKBACK_S = DETECT_COMMON_MIN_DURATION_S * 2  # Base window for live analysis.
DETECT_LIVE_FOLLOW_LATEST = True  # Follow newest live file if multiple exist.
DETECT_LIVE_STATE_PATH = Path("output") / "abnormal_state.json"  # Resume state file.

# Detector settings (batch).
DETECT_BATCH_INPUT_PATTERN = "static_data/Modbus_readings_*.csv"  # Batch file glob.
DETECT_BATCH_SHOW_PROGRESS = True  # Batch progress output.

# Debug settings.
DETECT_DEBUG_IDLE_S = 10  # Seconds to wait for live data to go idle after replay ends.
DETECT_DEBUG_COMPARE_MAX_DIFF = 5  # Max mismatched rows to include in report.
DETECT_DEBUG_COMPARE_REPORT_PATH = Path("output") / "abnormal_report_compare.csv"


def _normalize_day(value: str | int | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _clamp_speed(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _compute_live_lookback_s() -> float:
    """Ensure live lookback is large enough when replay speedup skips data."""
    base = float(DETECT_LIVE_LOOKBACK_S)
    if REPLAY_MODE != 1 and DEBUG_MODE != 1:
        return base
    effective_speed = _clamp_speed(REPLAY_SPEED_VALUE, REPLAY_SPEED_MIN, REPLAY_SPEED_MAX)
    # If data jumps too far between polls, we can miss intervals; expand lookback to cover it.
    data_jump_s = DETECT_LIVE_POLL_INTERVAL_S * effective_speed
    required = DETECT_COMMON_MIN_DURATION_S + data_jump_s
    return max(base, required)


_LOCK_HANDLE = None


def _release_lock() -> None:
    global _LOCK_HANDLE
    if _LOCK_HANDLE is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(_LOCK_HANDLE.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(_LOCK_HANDLE, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        _LOCK_HANDLE.close()
    except OSError:
        pass
    _LOCK_HANDLE = None


def _acquire_lock(lock_path: Path) -> None:
    """Prevent multiple launch.py instances from running simultaneously."""
    global _LOCK_HANDLE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            handle.seek(0)
            existing_pid = handle.read().strip()
        except OSError:
            existing_pid = ""
        handle.close()
        suffix = f" (pid {existing_pid})" if existing_pid else ""
        raise RuntimeError(f"Another launch.py instance is already running{suffix}.")

    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _LOCK_HANDLE = handle
    atexit.register(_release_lock)


def build_replay_config() -> replay.ReplayConfig:
    return replay.ReplayConfig(
        input_pattern=REPLAY_COMMON_INPUT_PATTERN,
        output_dir=REPLAY_COMMON_OUTPUT_DIR,
        speedup=REPLAY_SPEED_VALUE,
        min_speedup=REPLAY_SPEED_MIN,
        max_speedup=REPLAY_SPEED_MAX,
        start_day=_normalize_day(RANGE_START_DAY),
        end_day=_normalize_day(RANGE_END_DAY),
        loop=REPLAY_COMMON_LOOP,
        show_progress=REPLAY_PROGRESS_ENABLED,
        progress_interval_sec=REPLAY_PROGRESS_INTERVAL_SEC,
        output_time_mode=REPLAY_TIME_MODE,
        output_time_format=REPLAY_TIME_FORMAT,
        flush_each_row=REPLAY_OUTPUT_FLUSH_EACH_ROW,
    )


def run_replay(config: replay.ReplayConfig) -> None:
    replay.run_replay(config)


def _run_batch_detector(output_path: Path) -> None:
    input_pattern = DETECT_BATCH_INPUT_PATTERN
    detect_batch.detect_outliers_batch(
        input_pattern,
        output_path,
        DETECT_COMMON_MIN_DURATION_S,
        DETECT_COMMON_TOLERANCE,
        DETECT_COMMON_MIN_RATE_L_MIN,
        DETECT_COMMON_MIN_FLOW_FRACTION,
        DETECT_COMMON_CONSTANT_FRACTION,
        DETECT_LIVE_END_GAP_S,
        batch_start_day=_normalize_day(RANGE_START_DAY),
        batch_end_day=_normalize_day(RANGE_END_DAY),
        alert_to_console=DETECT_COMMON_ALERT_TO_CONSOLE,
        alert_to_log=DETECT_COMMON_ALERT_TO_LOG,
        alert_log_path=DETECT_COMMON_ALERT_LOG_PATH,
        alert_include_stop=DETECT_COMMON_ALERT_INCLUDE_STOP,
        show_progress=DETECT_BATCH_SHOW_PROGRESS,
    )


def _run_live_detector(
    output_path: Path,
    *,
    stop_event: threading.Event | None = None,
    stop_after_idle_s: float | None = None,
) -> None:
    input_pattern = DETECT_LIVE_INPUT_PATTERN
    effective_lookback_s = _compute_live_lookback_s()
    detect_live.detect_outliers_live(
        input_pattern,
        output_path,
        DETECT_COMMON_MIN_DURATION_S,
        DETECT_COMMON_TOLERANCE,
        DETECT_COMMON_MIN_RATE_L_MIN,
        DETECT_COMMON_MIN_FLOW_FRACTION,
        DETECT_COMMON_CONSTANT_FRACTION,
        DETECT_LIVE_END_GAP_S,
        DETECT_LIVE_UPDATE_INTERVAL_S,
        DETECT_LIVE_POLL_INTERVAL_S,
        effective_lookback_s,
        DETECT_LIVE_FOLLOW_LATEST,
        DETECT_LIVE_STATE_PATH,
        alert_to_console=DETECT_COMMON_ALERT_TO_CONSOLE,
        alert_to_log=DETECT_COMMON_ALERT_TO_LOG,
        alert_log_path=DETECT_COMMON_ALERT_LOG_PATH,
        alert_include_stop=DETECT_COMMON_ALERT_INCLUDE_STOP,
        stop_event=stop_event,
        stop_after_idle_s=stop_after_idle_s,
    )


def _load_report_counter(path: Path, *, status_filter: str | None = None) -> Counter:
    if not path.exists():
        return Counter()
    with path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        header = next(reader, None)
        if header is None:
            return Counter()
        status_idx = None
        if status_filter is not None:
            try:
                status_idx = header.index("status")
            except ValueError:
                status_idx = None
        rows = []
        for row in reader:
            if not row:
                continue
            if status_idx is not None and row[status_idx] != status_filter:
                continue
            rows.append(tuple(row))
        return Counter(rows)


def _compare_reports(batch_path: Path, live_path: Path) -> tuple[Counter, Counter, int, int]:
    batch_rows = _load_report_counter(batch_path, status_filter="final")
    live_rows = _load_report_counter(live_path, status_filter="final")
    missing_in_live = batch_rows - live_rows
    extra_in_live = live_rows - batch_rows
    return missing_in_live, extra_in_live, sum(batch_rows.values()), sum(live_rows.values())


def _write_compare_report(
    report_path: Path,
    missing_in_live: Counter,
    extra_in_live: Counter,
    total_batch: int,
    total_live: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total_missing = sum(missing_in_live.values())
    total_extra = sum(extra_in_live.values())
    if total_batch == 0 and total_live == 0:
        status = "empty"
    elif total_missing == 0 and total_extra == 0:
        status = "match"
    else:
        status = "mismatch"

    with report_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["section", "count", "row"])
        writer.writerow(["summary_status", status, ""])
        writer.writerow(["summary_total_batch", total_batch, ""])
        writer.writerow(["summary_total_live", total_live, ""])
        writer.writerow(["summary_missing_in_live", total_missing, ""])
        writer.writerow(["summary_extra_in_live", total_extra, ""])

        if total_missing:
            for row, count in list(missing_in_live.items())[:DETECT_DEBUG_COMPARE_MAX_DIFF]:
                writer.writerow(["missing_in_live", count, " | ".join(row)])
        if total_extra:
            for row, count in list(extra_in_live.items())[:DETECT_DEBUG_COMPARE_MAX_DIFF]:
                writer.writerow(["extra_in_live", count, " | ".join(row)])


def run_detector() -> None:
    if DETECT_MODE == 1:
        _run_batch_detector(DETECT_BATCH_OUTPUT_PATH)
    elif DETECT_MODE == 2:
        _run_live_detector(DETECT_LIVE_OUTPUT_PATH)
    elif DETECT_MODE != 0:
        raise ValueError("DETECT_MODE must be 0 (off), 1 (batch), or 2 (live).")


def main() -> None:
    replay_config = build_replay_config()
    _acquire_lock(Path(".launch.lock"))
    if REPLAY_MODE not in (0, 1):
        raise ValueError("REPLAY_MODE must be 0 (off) or 1 (on).")
    if DETECT_MODE not in (0, 1, 2):
        raise ValueError("DETECT_MODE must be 0 (off), 1 (batch), or 2 (live).")
    if DEBUG_MODE not in (0, 1):
        raise ValueError("DEBUG_MODE must be 0 (off) or 1 (on).")

    if CLEAR_PREVIOUS_LIVE_DATA == 1:
        clear_previous.clear_live_data(REPLAY_COMMON_OUTPUT_DIR)
    if CLEAR_PREVIOUS_OUTPUT == 1:
        clear_previous.clear_output_dir(DETECT_LIVE_OUTPUT_PATH.parent)

    if DEBUG_MODE == 1:
        _run_batch_detector(DETECT_BATCH_OUTPUT_PATH)

        replay_thread = threading.Thread(
            target=run_replay, args=(replay_config,), name="replay", daemon=True
        )
        replay_thread.start()

        stop_event = threading.Event()
        live_thread = threading.Thread(
            target=_run_live_detector,
            kwargs={
                "output_path": DETECT_LIVE_OUTPUT_PATH,
                "stop_event": stop_event,
                "stop_after_idle_s": DETECT_DEBUG_IDLE_S,
            },
            name="detector-live-debug",
            daemon=True,
        )
        live_thread.start()

        replay_thread.join()
        stop_event.set()
        live_thread.join()
        missing_in_live, extra_in_live, total_batch, total_live = _compare_reports(
            DETECT_BATCH_OUTPUT_PATH, DETECT_LIVE_OUTPUT_PATH
        )
        _write_compare_report(
            DETECT_DEBUG_COMPARE_REPORT_PATH,
            missing_in_live,
            extra_in_live,
            total_batch,
            total_live,
        )
        print(f"Debug compare report saved to {DETECT_DEBUG_COMPARE_REPORT_PATH}")
        return

    threads = []
    if DETECT_MODE != 0:
        detector_thread = threading.Thread(target=run_detector, name="detector", daemon=True)
        detector_thread.start()
        threads.append(detector_thread)
    if REPLAY_MODE == 1:
        replay_thread = threading.Thread(
            target=run_replay, args=(replay_config,), name="replay", daemon=True
        )
        replay_thread.start()
        threads.append(replay_thread)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
