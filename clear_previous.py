#!/usr/bin/env python3
"""Clear previous outputs and live data with verification."""
from __future__ import annotations

import os
import ctypes
from pathlib import Path
from typing import Literal

if os.name == "nt":
    from ctypes import windll, wintypes

    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_BEGIN = 0
    MOVEFILE_DELAY_UNTIL_REBOOT = 0x00000004
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    CreateFileW = windll.kernel32.CreateFileW
    CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    CreateFileW.restype = wintypes.HANDLE

    SetFilePointer = windll.kernel32.SetFilePointer
    SetFilePointer.argtypes = [wintypes.HANDLE, wintypes.LONG, ctypes.c_void_p, wintypes.DWORD]
    SetFilePointer.restype = wintypes.DWORD

    SetEndOfFile = windll.kernel32.SetEndOfFile
    SetEndOfFile.argtypes = [wintypes.HANDLE]
    SetEndOfFile.restype = wintypes.BOOL

    CloseHandle = windll.kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype = wintypes.BOOL

    MoveFileExW = windll.kernel32.MoveFileExW
    MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    MoveFileExW.restype = wintypes.BOOL


def _truncate_in_use_windows(path: Path) -> bool:
    handle = CreateFileW(
        str(path),
        GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        return False
    SetFilePointer(handle, 0, None, FILE_BEGIN)
    success = bool(SetEndOfFile(handle))
    CloseHandle(handle)
    return success


def _schedule_delete_on_reboot_windows(path: Path) -> bool:
    return bool(MoveFileExW(str(path), None, MOVEFILE_DELAY_UNTIL_REBOOT))


def _force_clear_file(path: Path) -> Literal["deleted", "truncated", "scheduled", "failed"]:
    try:
        path.unlink()
        return "deleted"
    except PermissionError:
        if os.name == "nt":
            if _truncate_in_use_windows(path):
                try:
                    path.unlink()
                    return "deleted"
                except PermissionError:
                    if _schedule_delete_on_reboot_windows(path):
                        return "scheduled"
                    return "truncated"
            if _schedule_delete_on_reboot_windows(path):
                return "scheduled"
        return "failed"


def clear_live_data(live_dir: Path, pattern: str = "Modbus_readings_*.csv") -> None:
    print(f"Clearing live data in {live_dir} ...")
    if not live_dir.exists():
        print("Live data folder not found; nothing to clear.")
        return

    removed = 0
    truncated = 0
    scheduled = 0
    failed = 0
    for path in live_dir.glob(pattern):
        result = _force_clear_file(path)
        if result == "deleted":
            removed += 1
        elif result == "truncated":
            truncated += 1
            print(f"Truncated in-use file: {path.name}")
        elif result == "scheduled":
            scheduled += 1
            print(f"Scheduled delete on reboot: {path.name}")
        else:
            failed += 1
            print(f"Failed to clear file: {path.name}")

    remaining = list(live_dir.glob(pattern))
    print(
        "Live data clear done. "
        f"removed={removed}, truncated={truncated}, scheduled={scheduled}, "
        f"failed={failed}, remaining={len(remaining)}"
    )
    if remaining:
        print("Warning: some live data files remain (likely in use).")


def clear_output_dir(output_dir: Path) -> None:
    print(f"Clearing output in {output_dir} ...")
    if not output_dir.exists():
        print("Output folder not found; nothing to clear.")
        return

    removed = 0
    truncated = 0
    scheduled = 0
    failed = 0
    for path in output_dir.iterdir():
        if path.is_dir():
            continue
        result = _force_clear_file(path)
        if result == "deleted":
            removed += 1
        elif result == "truncated":
            truncated += 1
            print(f"Truncated in-use file: {path.name}")
        elif result == "scheduled":
            scheduled += 1
            print(f"Scheduled delete on reboot: {path.name}")
        else:
            failed += 1
            print(f"Failed to clear file: {path.name}")

    remaining_files = [p for p in output_dir.iterdir() if p.is_file()]
    print(
        "Output clear done. "
        f"removed={removed}, truncated={truncated}, scheduled={scheduled}, "
        f"failed={failed}, remaining={len(remaining_files)}"
    )
    if remaining_files:
        print("Warning: some output files remain (likely in use).")


if __name__ == "__main__":
    raise SystemExit("This module is not runnable directly. Use launch.py.")
