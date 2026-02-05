#!/usr/bin/env python3
"""Generate diagnostic plots for detected Modbus events."""
from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import detect_modbus_core as core
from modbus_io import load_series


@dataclass
class DiagnosisConfig:
    report_path: Path
    output_dir: Path
    static_data_dir: Path
    live_data_dir: Path
    min_rate: float
    tolerance: float
    min_flow_fraction: float
    constant_fraction: float
    min_duration_s: float
    padding_min: int
    y_max: float | None
    show_points: bool
    legend_loc: str
    text_box_loc: tuple[float, float]
    text_box_outside: bool
    legend_outside: bool
    legend_anchor: tuple[float, float]
    right_margin: float
    x_major_min: int | None
    x_minor_min: int | None
    x_time_format: str
    follow: bool
    poll_interval_s: float


def _find_source_file(
    filename: str, static_data_dir: Path, live_data_dir: Path
) -> Path | None:
    static_path = static_data_dir / filename
    if static_path.exists():
        return static_path
    live_path = live_data_dir / filename
    if live_path.exists():
        return live_path
    return None


def _parse_event_row(row: List[str], header: Dict[str, int]) -> dict | None:
    try:
        status = row[header["status"]]
        if status != "final":
            return None
        start = datetime.fromisoformat(f"{row[header['start_date']]}T{row[header['start_time']]}")
        end = datetime.fromisoformat(f"{row[header['end_date']]}T{row[header['end_time']]}")
        return {
            "file": row[header["file"]],
            "event_id": row[header["event_id"]],
            "status": status,
            "start": start,
            "end": end,
            "avg_flow": row[header.get("avg_flow_rate_L_min", -1)]
            if "avg_flow_rate_L_min" in header
            else None,
            "total_volume": row[header.get("total_volume_L", -1)]
            if "total_volume_L" in header
            else None,
        }
    except (KeyError, IndexError, ValueError):
        return None


def _iter_report_events(report_path: Path) -> Tuple[Iterable[dict], Dict[str, int]]:
    with report_path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        header_row = next(reader, None)
        if not header_row:
            return [], {}
        header = {name: idx for idx, name in enumerate(header_row)}
        events = []
        for row in reader:
            if not row:
                continue
            event = _parse_event_row(row, header)
            if event:
                events.append(event)
        return events, header


def _slice_by_time(
    timestamps: List[datetime],
    flows: List[float],
    volumes: List[float],
    start: datetime,
    end: datetime,
) -> Tuple[List[datetime], List[float], List[float]]:
    xs: List[datetime] = []
    ys: List[float] = []
    vs: List[float] = []
    for ts, flow, vol in zip(timestamps, flows, volumes):
        if ts < start:
            continue
        if ts > end:
            break
        xs.append(ts)
        ys.append(flow)
        vs.append(vol)
    return xs, ys, vs


def _compute_metrics(
    timestamps: List[datetime],
    flows: List[float],
    volumes: List[float],
    *,
    min_rate: float,
    tolerance: float,
) -> dict:
    minute_times, minute_flows, _, _ = core.resample_minutes(timestamps, flows, volumes)
    total_minutes = len(minute_flows)
    flow_minutes = [flow for flow in minute_flows if flow >= min_rate]
    flow_fraction = None
    constant_fraction = None
    median_flow = None
    lower_bound = None

    if total_minutes:
        flow_fraction = len(flow_minutes) / total_minutes
    if flow_minutes:
        median_flow = statistics.median(flow_minutes)
        lower_bound = median_flow * (1 - tolerance)
        within = sum(1 for flow in flow_minutes if flow >= lower_bound)
        constant_fraction = within / len(flow_minutes)

    return {
        "flow_fraction": flow_fraction,
        "constant_fraction": constant_fraction,
        "median_flow": median_flow,
        "lower_bound": lower_bound,
        "window_minutes": total_minutes,
    }


def _plot_event(
    event: dict,
    config: DiagnosisConfig,
    series_cache: Dict[str, Tuple[List[datetime], List[float], List[float]]],
) -> bool:
    source_path = _find_source_file(
        event["file"], config.static_data_dir, config.live_data_dir
    )
    if source_path is None:
        print(f"Diagnosis skipped: missing source file {event['file']}")
        return False

    if source_path.name not in series_cache:
        series_cache[source_path.name] = load_series(source_path)
    timestamps, flow_rates, volumes = series_cache[source_path.name]

    pad = timedelta(minutes=config.padding_min)
    window_start = event["start"] - pad
    window_end = event["end"] + pad

    xs: List[datetime] = []
    ys: List[float] = []
    vs: List[float] = []
    for ts, flow, vol in zip(timestamps, flow_rates, volumes):
        if ts < window_start:
            continue
        if ts > window_end:
            break
        xs.append(ts)
        ys.append(flow)
        vs.append(vol)

    if not xs:
        print(f"Diagnosis skipped: no data for {event['file']} event {event['event_id']}")
        return False

    trigger_end = event["start"] + timedelta(seconds=config.min_duration_s)
    if trigger_end > event["end"]:
        trigger_end = event["end"]
    trigger_xs, trigger_ys, trigger_vs = _slice_by_time(
        timestamps, flow_rates, volumes, event["start"], trigger_end
    )
    event_xs, event_ys, event_vs = _slice_by_time(
        timestamps, flow_rates, volumes, event["start"], event["end"]
    )

    trigger_metrics = _compute_metrics(
        trigger_xs,
        trigger_ys,
        trigger_vs,
        min_rate=config.min_rate,
        tolerance=config.tolerance,
    )
    event_metrics = _compute_metrics(
        event_xs,
        event_ys,
        event_vs,
        min_rate=config.min_rate,
        tolerance=config.tolerance,
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    if config.show_points:
        ax.plot(xs, ys, linewidth=1, marker="o", markersize=2)
    else:
        ax.plot(xs, ys, linewidth=1)
    ax.axvline(event["start"], color="green", linestyle="--", linewidth=1)
    ax.axvline(event["end"], color="red", linestyle="--", linewidth=1)
    ax.set_xlim(window_start, window_end)
    ax.margins(x=0)

    if config.y_max is not None:
        ax.set_ylim(0, config.y_max)
    else:
        max_flow = max(ys) if ys else 0.0
        ax.set_ylim(0, max_flow * 1.05 if max_flow > 0 else 1.0)

    ax.set_title(
        f"{event['file']} | event {event['event_id']} | "
        f"{event['start']:%Y-%m-%d %H:%M:%S} to {event['end']:%H:%M:%S}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow Rate (L/min)")
    if config.x_major_min:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=config.x_major_min))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(config.x_time_format))
    if config.x_minor_min:
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=config.x_minor_min))
    ax.grid(True, which="major", alpha=0.3)
    if config.x_minor_min:
        ax.grid(True, which="minor", alpha=0.15)
    if config.legend_loc:
        if config.legend_outside:
            ax.legend(
                loc=config.legend_loc,
                bbox_to_anchor=config.legend_anchor,
                bbox_transform=fig.transFigure,
            )
        else:
            ax.legend(loc=config.legend_loc)
    fig.autofmt_xdate()

    duration_h = (event["end"] - event["start"]).total_seconds() / 3600
    trigger_window_minutes = trigger_metrics["window_minutes"]
    trigger_flow_fraction = trigger_metrics["flow_fraction"]
    trigger_constant_fraction = trigger_metrics["constant_fraction"]
    trigger_median_flow = trigger_metrics["median_flow"]
    trigger_lower_bound = trigger_metrics["lower_bound"]
    event_flow_fraction = event_metrics["flow_fraction"]
    event_constant_fraction = event_metrics["constant_fraction"]

    lines = [
        f"duration_hours: {duration_h:.2f} (min {config.min_duration_s/3600:.2f})",
        f"min_rate_L_min: {config.min_rate:.2f}",
        f"trigger_window_minutes: {trigger_window_minutes}",
    ]
    if trigger_flow_fraction is None:
        lines.append("trigger_flow_fraction: n/a")
    else:
        lines.append(
            f"trigger_flow_fraction: {trigger_flow_fraction:.2f} (min {config.min_flow_fraction:.2f})"
        )
    if config.constant_fraction <= 0:
        lines.append("trigger_constant_fraction: disabled")
    elif (
        trigger_constant_fraction is None
        or trigger_median_flow is None
        or trigger_lower_bound is None
    ):
        lines.append("trigger_constant_fraction: n/a")
    else:
        lines.append(
            f"trigger_constant_fraction: {trigger_constant_fraction:.2f} (min {config.constant_fraction:.2f})"
        )
        lines.append(f"trigger_median_flow: {trigger_median_flow:.2f}")
        lines.append(
            f"trigger_lower_bound: {trigger_lower_bound:.2f} (tol {config.tolerance:.2f})"
        )
    if event_flow_fraction is None:
        lines.append("event_flow_fraction: n/a")
    else:
        lines.append(f"event_flow_fraction: {event_flow_fraction:.2f}")
    if config.constant_fraction <= 0:
        lines.append("event_constant_fraction: disabled")
    elif event_constant_fraction is None:
        lines.append("event_constant_fraction: n/a")
    else:
        lines.append(f"event_constant_fraction: {event_constant_fraction:.2f}")

    if config.text_box_outside or config.legend_outside:
        fig.subplots_adjust(right=config.right_margin)
    if config.text_box_outside:
        fig.text(
            config.text_box_loc[0],
            config.text_box_loc[1],
            "\n".join(lines),
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    else:
        ax.text(
            config.text_box_loc[0],
            config.text_box_loc[1],
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    safe_start = event["start"].strftime("%H%M%S")
    safe_end = event["end"].strftime("%H%M%S")
    output_name = f"{Path(event['file']).stem}_event{event['event_id']}_{safe_start}_{safe_end}.png"
    fig.savefig(config.output_dir / output_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def run_diagnosis(config: DiagnosisConfig) -> None:
    seen = set()
    series_cache: Dict[str, Tuple[List[datetime], List[float], List[float]]] = {}
    file_pos = 0
    header: Dict[str, int] = {}

    while True:
        if not config.report_path.exists():
            if not config.follow:
                print(f"Diagnosis skipped: missing report {config.report_path}")
                return
            time.sleep(config.poll_interval_s)
            continue

        if not config.follow:
            events, _ = _iter_report_events(config.report_path)
            for event in events:
                key = (
                    event["file"],
                    event["event_id"],
                    event["start"],
                    event["end"],
                )
                if key in seen:
                    continue
                _plot_event(event, config, series_cache)
                seen.add(key)
            return

        with config.report_path.open(newline="") as input_file:
            input_file.seek(file_pos)
            reader = csv.reader(input_file)
            if file_pos == 0:
                header_row = next(reader, None)
                if not header_row:
                    time.sleep(config.poll_interval_s)
                    continue
                header = {name: idx for idx, name in enumerate(header_row)}
            for row in reader:
                if not row:
                    continue
                event = _parse_event_row(row, header)
                if not event:
                    continue
                key = (
                    event["file"],
                    event["event_id"],
                    event["start"],
                    event["end"],
                )
                if key in seen:
                    continue
                _plot_event(event, config, series_cache)
                seen.add(key)
            file_pos = input_file.tell()

        time.sleep(config.poll_interval_s)


if __name__ == "__main__":
    raise SystemExit("This module is not runnable directly. Use launch.py.")
