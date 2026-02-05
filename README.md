# WLCD Data Analysis - Usage

All runtime parameters live in `launch.py`. Edit that file first, then run:

```bash
python launch.py
```

## Folders

- `static_data/` - historical CSV files (input for replay + batch).
- `live_data/` - live simulation output (replay writes here).
- `output/` - reports, logs, and state files.

## CSV naming

Input files are expected to be named:

```
Modbus_readings_YYYYMMDD.csv
```

## Quick start (most common)

1. Drop historical CSVs into `static_data/`.
2. Set `REPLAY_MODE = 1` and `DETECT_MODE = 2` in `launch.py`.
3. Run `python launch.py`.

This replays static files into `live_data/` and runs the live detector.

## Mode switches (top of launch.py)

- `VERIFY_MODE`
  - `0` = off
  - `1` = on (runs batch, then live, then compares reports)
- `REPLAY_MODE`
  - `0` = off
  - `1` = on
- `DETECT_MODE`
  - `0` = off
  - `1` = batch
  - `2` = live

## Housekeeping

- `CLEAR_PREVIOUS_LIVE_DATA` - clears `live_data/` at startup.
- `CLEAR_PREVIOUS_OUTPUT` - clears `output/` at startup.

These are handled by `clear_previous.py`, which prints status and verifies
cleanup. Locked files may be truncated and scheduled for deletion on reboot.

## Date range (shared by replay + batch)

- `RANGE_START_DAY`, `RANGE_END_DAY` (YYYYMMDD, inclusive).

These limits apply to *both* replay and batch to keep results aligned.

## Replay settings (simulation)

Key parameters:

- `REPLAY_COMMON_INPUT_PATTERN` - glob for static input files.
- `REPLAY_COMMON_OUTPUT_DIR` - live output folder (default `live_data/`).
- `REPLAY_SPEED_VALUE` - speedup factor (1.0 = real time).
- Speedup is clamped to hard limits in `replay_modbus_csv.py` (0.1..9999.0).
- Replay preserves raw epoch timestamps. If you need a different time format,
  change it at the sensor/source side, not in the replay pipeline.
- `REPLAY_PROGRESS_ENABLED` - single-line progress in terminal.

## Detector settings (common)

Key parameters:

- `DETECT_COMMON_MIN_DURATION_S` - minimum suspicious duration (seconds).
- `DETECT_COMMON_TOLERANCE` - **lower-side** tolerance (high spikes allowed).
- `DETECT_COMMON_MIN_RATE_L_MIN` - minimum flow to count as flowing.
- `DETECT_COMMON_MIN_FLOW_FRACTION` - fraction of minutes with flow required.
- `DETECT_COMMON_CONSTANT_FRACTION`
  - `0.0` = skip constancy check (detect continuous usage only)
  - `> 0` = require a stable flow pattern

Example (with your current style)
- `DETECT_COMMON_MIN_DURATION_S = 10 * 60` -> window is 10 minutes
- `DETECT_COMMON_MIN_FLOW_FRACTION = 0.8`
- `DETECT_COMMON_MIN_RATE_L_MIN = 1`
- `DETECT_COMMON_CONSTANT_FRACTION = 0.8`
- `DETECT_COMMON_TOLERANCE = 0.10`
- Then a suspicious window needs **>= 8 of those 10 minutes** to have avg flow >= 1 L/min,
  and within those flowing minutes, **>= 80% must be >= (median * 0.9)**.

## Detector settings (live)

- `DETECT_LIVE_INPUT_PATTERN` - glob for live files.
- `DETECT_LIVE_END_GAP_S` - end event if no suspicious/continuous flow for this long.
- `DETECT_LIVE_UPDATE_INTERVAL_S` - emit ongoing rows at this interval.
- `DETECT_LIVE_POLL_INTERVAL_S` - polling interval for file updates.
- `DETECT_LIVE_LOOKBACK_S` - lookback buffer for live analysis.
- `DETECT_LIVE_FOLLOW_LATEST` - follow newest file when multiple exist.
- `DETECT_LIVE_STATE_PATH` - resume state file for live mode.

## Detector settings (batch)

- `DETECT_BATCH_INPUT_PATTERN` - glob for static files.
- `DETECT_BATCH_SHOW_PROGRESS` - progress output.

## Diagnosis / verification

Verification mode writes a compare report to:

- `output/abnormal_report_compare.csv`

You can also enable diagnostics to generate plots per detected event:

- `DIAGNO_MODE` (0 = off, 1 = plot existing report once, 2 = follow live report, 3 = both)
- `DIAG_SOURCE` ("batch", "live", "both")
- `DIAG_OUTPUT_DIR`
- `DIAG_PADDING_MIN`
- `DIAG_Y_MAX`

## Outputs

- `output/abnormal_report_batch.csv` - batch report.
- `output/abnormal_report_live.csv` - live report.
- `output/abnormal_report_compare.csv` - verification compare report (when enabled).
- `output/abnormal_state.json` - live detector resume state.
- `output/abnormal_alerts.log` - optional alert log.

### Report columns

```
file,event_id,status,update_time,start_day_num,start_date,start_time,
end_day_num,end_date,end_time,duration_hours,note,avg_flow_rate_L_min,total_volume_L
```

- `status` is `ongoing` or `final` (live mode).
- `event_id` increments per detected event in live mode.

## Stop / exit

Press `Ctrl+C` to stop. In verification mode, live stops automatically after the
idle timeout configured by `VERIFY_IDLE_S`.
