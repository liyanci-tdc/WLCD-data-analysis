#!/usr/bin/env python3
"""Streaming detector for suspicious long-duration usage."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, List, Tuple

import detect_modbus_core as core


@dataclass
class MinuteSample:
    minute_start: datetime
    avg_flow: float
    last_volume: float


class StreamingDetector:
    def __init__(
        self,
        *,
        source_file: str,
        min_duration_s: float,
        tolerance: float,
        min_rate: float,
        min_flow_fraction: float,
        constant_fraction: float,
        end_gap_s: float,
        update_interval_s: float,
        lookback_s: float,
        event_id_start: int = 1,
        last_reported_end: datetime | None = None,
    ) -> None:
        self.source_file = source_file
        self.min_duration_s = float(min_duration_s)
        self.tolerance = float(tolerance)
        self.min_rate = float(min_rate)
        self.min_flow_fraction = float(min_flow_fraction)
        self.constant_fraction = float(constant_fraction)
        self.end_gap_s = float(end_gap_s)
        self.update_interval_s = float(update_interval_s)
        self.lookback_s = float(lookback_s)

        self.window_minutes = max(2, int(math.ceil(self.min_duration_s / 60.0)))
        self.max_minutes = max(self.window_minutes + 2, int(math.ceil(self.lookback_s / 60.0)) + 2)

        self.event_id = max(1, int(event_id_start))
        self.last_reported_end = last_reported_end

        self.raw_buffer: Deque[Tuple[datetime, float, float]] = deque()
        self.minute_buffer: Deque[MinuteSample] = deque()
        self.current_minute: datetime | None = None
        self.minute_sum = 0.0
        self.minute_count = 0
        self.minute_last_volume: float | None = None

        self.last_seen_ts: datetime | None = None
        self.active_event: dict | None = None
        # Track the last time the rolling window still looked suspicious.
        self.last_suspicious_time: datetime | None = None
        self.last_suspicious_volume: float | None = None

    def reset(self, *, keep_event_id: bool = True) -> None:
        if not keep_event_id:
            self.event_id = 1
        self.raw_buffer.clear()
        self.minute_buffer.clear()
        self.current_minute = None
        self.minute_sum = 0.0
        self.minute_count = 0
        self.minute_last_volume = None
        self.last_seen_ts = None
        self.active_event = None
        self.last_suspicious_time = None
        self.last_suspicious_volume = None

    def set_source(self, source_file: str) -> None:
        self.source_file = source_file

    def _trim_raw_buffer(self, latest_ts: datetime) -> None:
        while self.raw_buffer and (latest_ts - self.raw_buffer[0][0]).total_seconds() > self.lookback_s:
            self.raw_buffer.popleft()

    def _append_minute(self, minute_start: datetime, avg_flow: float, last_volume: float) -> None:
        self.minute_buffer.append(MinuteSample(minute_start, avg_flow, last_volume))
        while len(self.minute_buffer) > self.max_minutes:
            self.minute_buffer.popleft()

    def _finalize_current_minute(self) -> None:
        if self.current_minute is None:
            return
        if self.minute_count > 0:
            avg_flow = self.minute_sum / self.minute_count
            last_volume = self.minute_last_volume if self.minute_last_volume is not None else 0.0
        else:
            avg_flow = 0.0
            last_volume = self.minute_last_volume if self.minute_last_volume is not None else 0.0
        self._append_minute(self.current_minute, avg_flow, last_volume)
        self.minute_sum = 0.0
        self.minute_count = 0
        self.minute_last_volume = None

    def _fill_minute_gap(self, next_minute: datetime) -> None:
        if self.current_minute is None:
            return
        minute = self.current_minute + timedelta(minutes=1)
        while minute < next_minute:
            self._append_minute(minute, 0.0, self.minute_last_volume or 0.0)
            minute += timedelta(minutes=1)

    def _window_is_suspicious(self) -> Tuple[bool, datetime | None]:
        if len(self.minute_buffer) < self.window_minutes:
            return False, None
        window = list(self.minute_buffer)[-self.window_minutes :]
        flow_values = [m.avg_flow for m in window if m.avg_flow >= self.min_rate]
        flow_fraction = len(flow_values) / self.window_minutes
        if flow_fraction < self.min_flow_fraction or not flow_values:
            return False, None
        if self.constant_fraction > 0.0:
            if not core.is_constant_flow(flow_values, self.tolerance, self.constant_fraction):
                return False, None
        return True, window[0].minute_start

    def _find_start_time(self, start_minute: datetime) -> datetime:
        for ts, flow, _ in self.raw_buffer:
            if ts < start_minute:
                continue
            if flow >= self.min_rate:
                return ts
        return start_minute

    def _init_active_event(self, start_time: datetime) -> dict | None:
        start_volume = None
        last_volume = None
        last_flow_time = None
        last_flow_volume = None
        for ts, flow, vol in self.raw_buffer:
            if ts < start_time:
                continue
            if start_volume is None:
                start_volume = vol
            last_volume = vol
            if flow >= self.min_rate:
                last_flow_time = ts
                last_flow_volume = vol
        if start_volume is None or last_volume is None:
            return None
        if last_flow_time is None:
            last_flow_time = start_time
            last_flow_volume = start_volume
        return {
            "event_id": self.event_id,
            "source_file": self.source_file,
            "start_time": start_time,
            "start_volume": start_volume,
            "last_volume": last_volume,
            "last_flow_time": last_flow_time or start_time,
            "last_flow_volume": last_flow_volume if last_flow_volume is not None else last_volume,
            "last_update_time": start_time,
        }

    def _calc_stats(self, end_time: datetime, end_volume: float) -> Tuple[float, float, float]:
        duration_s = (end_time - self.active_event["start_time"]).total_seconds()
        total_volume = end_volume - self.active_event["start_volume"]
        if duration_s <= 0:
            avg_flow = 0.0
        else:
            avg_flow = total_volume / (duration_s / 60.0)
        return avg_flow, total_volume, duration_s

    def _emit_row(
        self,
        *,
        status: str,
        end_time: datetime,
        end_volume: float,
    ) -> List[str]:
        avg_flow, total_volume, duration_s = self._calc_stats(end_time, end_volume)
        return core.build_report_row(
            source_name=self.active_event["source_file"],
            event_id=str(self.active_event["event_id"]),
            status=status,
            update_time=end_time,
            start_time=self.active_event["start_time"],
            end_time=end_time,
            duration_s=duration_s,
            avg_flow=avg_flow,
            total_volume=total_volume,
        )

    def process_sample(
        self,
        ts: datetime,
        flow: float,
        volume: float,
    ) -> Tuple[List[List[str]], dict | None, dict | None]:
        rows: List[List[str]] = []
        started_event: dict | None = None
        ended_event: dict | None = None

        if self.last_seen_ts is not None and ts < self.last_seen_ts:
            self.reset(keep_event_id=True)

        self.raw_buffer.append((ts, flow, volume))
        self._trim_raw_buffer(ts)
        self.last_seen_ts = ts

        minute = ts.replace(second=0, microsecond=0)
        if self.current_minute is None:
            self.current_minute = minute
        if minute != self.current_minute:
            self._finalize_current_minute()
            self._fill_minute_gap(minute)
            self.current_minute = minute

        self.minute_sum += flow
        self.minute_count += 1
        self.minute_last_volume = volume

        # Suspicion is evaluated from completed minute buckets (current minute is partial).
        suspicious, start_minute = self._window_is_suspicious()

        if self.active_event:
            if ts >= self.active_event["start_time"]:
                self.active_event["last_volume"] = volume
                if flow >= self.min_rate:
                    self.active_event["last_flow_time"] = ts
                    self.active_event["last_flow_volume"] = volume
            if suspicious:
                # Extend "suspiciousness" based on the rolling window outcome, not just raw flow.
                self.last_suspicious_time = self.active_event["last_flow_time"] or ts
                self.last_suspicious_volume = (
                    self.active_event.get("last_flow_volume") or self.active_event["last_volume"]
                )

            if self.update_interval_s > 0:
                last_update = self.active_event["last_update_time"]
                if (ts - last_update).total_seconds() >= self.update_interval_s:
                    rows.append(
                        self._emit_row(
                            status="ongoing",
                            end_time=ts,
                            end_volume=self.active_event["last_volume"],
                        )
                    )
                    self.active_event["last_update_time"] = ts

            last_suspicious = self.last_suspicious_time
            # End the event only after the suspicious window has been absent for long enough.
            if last_suspicious and (ts - last_suspicious).total_seconds() >= self.end_gap_s:
                end_time = last_suspicious
                end_volume = self.last_suspicious_volume or self.active_event["last_flow_volume"]
                avg_flow, _, duration_s = self._calc_stats(end_time, end_volume)
                rows.append(
                    self._emit_row(
                        status="final",
                        end_time=end_time,
                        end_volume=end_volume,
                    )
                )
                ended_event = {
                    "source_file": self.active_event["source_file"],
                    "start_time": self.active_event["start_time"],
                    "end_time": end_time,
                    "duration_s": duration_s,
                    "avg_flow": avg_flow,
                }
                self.last_reported_end = end_time
                self.active_event = None
                self.last_suspicious_time = None
                self.last_suspicious_volume = None

        if self.active_event is None:
            if suspicious and start_minute:
                start_time = self._find_start_time(start_minute)
                if self.last_reported_end and start_time <= self.last_reported_end:
                    return rows, started_event, ended_event
                event = self._init_active_event(start_time)
                if event:
                    self.active_event = event
                    self.last_suspicious_time = event["last_flow_time"]
                    self.last_suspicious_volume = event["last_flow_volume"]
                    avg_flow, _, _ = self._calc_stats(ts, volume)
                    started_event = {
                        "source_file": event["source_file"],
                        "start_time": event["start_time"],
                        "avg_flow": avg_flow,
                    }
                    self.event_id += 1

        return rows, started_event, ended_event

    def finalize_active(self) -> Tuple[List[List[str]], dict | None]:
        if not self.active_event:
            return [], None
        end_time = self.last_suspicious_time or self.active_event["last_flow_time"]
        end_volume = self.last_suspicious_volume or self.active_event["last_flow_volume"]
        avg_flow, _, duration_s = self._calc_stats(end_time, end_volume)
        row = self._emit_row(
            status="final",
            end_time=end_time,
            end_volume=end_volume,
        )
        ended_event = {
            "source_file": self.active_event["source_file"],
            "start_time": self.active_event["start_time"],
            "end_time": end_time,
            "duration_s": duration_s,
            "avg_flow": avg_flow,
        }
        self.last_reported_end = end_time
        self.active_event = None
        self.last_suspicious_time = None
        self.last_suspicious_volume = None
        return [row], ended_event
