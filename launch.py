#!/usr/bin/env python3
"""Launch replay + live detector together."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import detect_modbus_abnormal as detect
import replay_modbus_csv as replay

# =========================
# Combined Config
# =========================
# Replay settings.
REPLAY_INPUT_PATTERN = "static_data/Modbus_readings_*.csv"  # Source CSV glob.
REPLAY_OUTPUT_DIR = Path("live_data")  # Output folder for live files.
REPLAY_SPEEDUP = 1000.0  # 1.0 = real-time, 10.0 = 10x; max 9999.
REPLAY_START_DAY = "20260120"  # Inclusive YYYYMMDD (None = no lower bound).
REPLAY_END_DAY = None  # Inclusive YYYYMMDD (None = no upper bound).
REPLAY_LOOP = True  # Loop over the date range forever.
REPLAY_CLEAR_OUTPUT_ON_START = True  # Delete old live_data files at startup.
REPLAY_SHOW_PROGRESS = True  # Print single-line progress while replaying.
REPLAY_PROGRESS_INTERVAL_SEC = 1.0  # Progress update interval in seconds.
REPLAY_OUTPUT_TIME_MODE = "raw"  # "raw" (epoch), "clock" (HHMMSS), "elapsed" (HHMMSS).
REPLAY_OUTPUT_TIME_FORMAT = "%H%M%S"  # Only used when OUTPUT_TIME_MODE="clock".

# Detector settings (live).
DETECT_ENABLED = True  # Set False to run replay only.
DETECT_INPUT_PATTERN = "live_data/Modbus_readings_*.csv"  # Live file glob.
DETECT_OUTPUT_PATH = Path("output") / "abnormal_report.csv"  # CSV report output.
DETECT_DETECTION_MODE = "minute_refined"  # "minute_refined" (fast) or "second" (slow/precise).
DETECT_MIN_DURATION_S = 60 * 60  # Minimum suspicious duration in seconds.
DETECT_TOLERANCE = 0.20  # +/- tolerance for constant/linear detection.
DETECT_MIN_RATE_L_MIN = 1  # Minimum flow rate to count as flowing (L/min).
DETECT_MIN_FLOW_FRACTION = 0.9  # Fraction of window minutes that must show flow.
DETECT_CONSTANT_FRACTION = 0.9  # Fraction of flowing minutes that must be "constant".
DETECT_SECOND_WINDOW_STEP_S = 10  # Sliding step in seconds for "second" mode.
DETECT_HYSTERESIS_TIME_S = 60  # Allowed short gaps (seconds) in flow.
DETECT_LIVE_POLL_INTERVAL_S = 10  # How often to poll the live file(s).
DETECT_LIVE_LOOKBACK_S = DETECT_MIN_DURATION_S * 2  # Window size for live analysis.
DETECT_LIVE_FOLLOW_LATEST = True  # Follow newest live file if multiple exist.
DETECT_LIVE_STATE_PATH = Path("output") / "abnormal_state.json"  # Resume state file.
DETECT_ALERT_LOG_PATH = Path("output") / "abnormal_alerts.log"  # Alert log file.
DETECT_ALERT_TO_CONSOLE = True  # Print alerts to console.
DETECT_ALERT_TO_LOG = True  # Write alerts to log file.
DETECT_ALERT_INCLUDE_STOP = True  # Emit stop alerts as well as start alerts.


def configure_replay() -> None:
    replay.INPUT_PATTERN = REPLAY_INPUT_PATTERN
    replay.OUTPUT_DIR = REPLAY_OUTPUT_DIR
    replay.SPEEDUP = REPLAY_SPEEDUP
    replay.START_DAY = REPLAY_START_DAY
    replay.END_DAY = REPLAY_END_DAY
    replay.LOOP = REPLAY_LOOP
    replay.CLEAR_OUTPUT_ON_START = REPLAY_CLEAR_OUTPUT_ON_START
    replay.SHOW_PROGRESS = REPLAY_SHOW_PROGRESS
    replay.PROGRESS_INTERVAL_SEC = REPLAY_PROGRESS_INTERVAL_SEC
    replay.OUTPUT_TIME_MODE = REPLAY_OUTPUT_TIME_MODE
    replay.OUTPUT_TIME_FORMAT = REPLAY_OUTPUT_TIME_FORMAT


def configure_detector() -> None:
    detect.ALERT_LOG_PATH = DETECT_ALERT_LOG_PATH
    detect.ALERT_TO_CONSOLE = DETECT_ALERT_TO_CONSOLE
    detect.ALERT_TO_LOG = DETECT_ALERT_TO_LOG
    detect.ALERT_INCLUDE_STOP = DETECT_ALERT_INCLUDE_STOP


def run_replay() -> None:
    configure_replay()
    replay.main()


def run_detector() -> None:
    configure_detector()
    detect.detect_outliers_live(
        DETECT_INPUT_PATTERN,
        DETECT_OUTPUT_PATH,
        DETECT_MIN_DURATION_S,
        DETECT_TOLERANCE,
        DETECT_MIN_RATE_L_MIN,
        DETECT_MIN_FLOW_FRACTION,
        DETECT_CONSTANT_FRACTION,
        DETECT_DETECTION_MODE,
        DETECT_HYSTERESIS_TIME_S,
        DETECT_SECOND_WINDOW_STEP_S,
        DETECT_LIVE_POLL_INTERVAL_S,
        DETECT_LIVE_LOOKBACK_S,
        DETECT_LIVE_FOLLOW_LATEST,
        DETECT_LIVE_STATE_PATH,
    )


def main() -> None:
    threads = []
    if DETECT_ENABLED:
        detector_thread = threading.Thread(target=run_detector, name="detector", daemon=True)
        detector_thread.start()
        threads.append(detector_thread)
    replay_thread = threading.Thread(target=run_replay, name="replay", daemon=True)
    replay_thread.start()
    threads.append(replay_thread)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
