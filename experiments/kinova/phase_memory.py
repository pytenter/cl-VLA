"""Read-only, stdlib phase telemetry; never imports or initializes a GPU runtime.

The existing WindowsMemoryMonitor remains the sole Windows producer. This
sampler only consumes its fresh cached snapshot (no PowerShell fallback).
Counters in different accounting domains must not be added together.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
import uuid

SCHEMA = "kinova-phase-memory-v1"
PROCESS_KEYS = ("VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile", "RssShmem")
SMAPS_KEYS = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Anonymous",
              "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty", "Swap")
SYSTEM_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SReclaimable",
               "Shmem", "AnonPages", "Dirty", "Writeback", "SwapTotal", "SwapFree")


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("Incomplete phase telemetry append")
    finally:
        os.close(descriptor)


def _atomic(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _bytes(text, keys):
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in keys:
            fields = value.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError(f"Expected Linux kB counter for {key}")
            result[key + "_bytes"] = int(fields[0]) * 1024
    return result


def _identity(proc_root, pid):
    # The comm field can itself contain spaces and parentheses.
    suffix = (Path(proc_root) / str(pid) / "stat").read_text().rpartition(")")[2].split()
    return int(suffix[19])  # Field 22, relative to state (field 3).


def _cached_windows(root, max_age):
    from windows_memory import snapshot
    return snapshot(root, max_age=max_age, refresh=False)


def mark_phase(root, phase, *, pid=None, **fields):
    """Record a lightweight barrier from the target process or its loader.

    Call immediately before/after imports, dataset construction, eval_shape,
    restore, device transfer and native initialization. Pass only small scalar
    metadata (dtype, source/target bytes, leaf name); never tensor values.
    """
    if not isinstance(phase, str) or not phase or len(phase) > 160:
        raise ValueError("phase must be a nonempty name of at most 160 characters")
    pid = os.getpid() if pid is None else int(pid)
    row = dict(schema=SCHEMA, kind="phase", phase=phase, pid=pid,
               recorded_at=_utc(), recorded_at_unix=time.time(),
               monotonic_seconds=time.monotonic(), fields=fields)
    _append(Path(root) / "logs/phase_memory.jsonl", row)
    _atomic(Path(root) / "reports" / f"phase_memory_phase_{pid}.json", row)
    return row


class MemorySampler:
    """Observe one existing Linux PID; creates one stoppable Python thread.

    Telemetry failures are logged as missing/degraded measurements, never as
    zero memory or resource eligibility. Resource guards remain independent.
    The target start time is pinned so PID reuse cannot silently change owners.
    """

    def __init__(self, root, pid=None, *, label="process", interval=1.0, windows_max_age=5.0,
                 proc_root="/proc", windows_reader=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        if not math.isfinite(windows_max_age) or windows_max_age <= 0:
            raise ValueError("windows_max_age must be finite and positive")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", label):
            raise ValueError("label must be 1..64 filename-safe characters")
        self.root = Path(root)
        self.pid = os.getpid() if pid is None else int(pid)
        if self.pid <= 0:
            raise ValueError("pid must be positive")
        self.label, self.interval = label, float(interval)
        self.windows_max_age = windows_max_age
        self.proc_root = Path(proc_root)
        self.windows_reader = windows_reader or _cached_windows
        self.run_id = uuid.uuid4().hex[:12]
        self.process_start_ticks = None
        self._stop = threading.Event()
        self._thread = None
        self._samples = self._degraded_samples = 0
        self._peaks = {}
        self._minima = {}
        self._last_error = None
        self._started_at = None
        self._terminal_status = None

    def sample_once(self):
        row = dict(schema=SCHEMA, kind="sample", run_id=self.run_id, label=self.label,
                   pid=self.pid, observer_pid=os.getpid(), sampled_at=_utc(),
                   sampled_at_unix=time.time(), monotonic_seconds=time.monotonic(),
                   process_start_ticks=self.process_start_ticks, linux={}, windows=None, errors=[])
        try:
            identity = _identity(self.proc_root, self.pid)
            if self.process_start_ticks is None:
                self.process_start_ticks = identity
            elif identity != self.process_start_ticks:
                row["target_status"] = "pid_reused"
                row["errors"].append("Target PID start time changed; sampling stopped")
                return row
            row["process_start_ticks"] = identity
            directory = self.proc_root / str(self.pid)
            row["linux"]["process"] = _bytes((directory / "status").read_text(), PROCESS_KEYS)
            row["target_status"] = "present"
            try:
                row["linux"]["smaps_rollup"] = _bytes((directory / "smaps_rollup").read_text(), SMAPS_KEYS)
            except (OSError, ValueError) as error:
                row["errors"].append(f"smaps_rollup: {type(error).__name__}: {error}")
        except (OSError, ValueError, IndexError) as error:
            row["target_status"] = "unavailable"
            row["errors"].append(f"target: {type(error).__name__}: {error}")
        try:
            row["linux"]["system"] = _bytes((self.proc_root / "meminfo").read_text(), SYSTEM_KEYS)
        except (OSError, ValueError) as error:
            row["errors"].append(f"meminfo: {type(error).__name__}: {error}")
        try:
            row["windows"] = self.windows_reader(self.root, self.windows_max_age)
            row["windows_status"] = "fresh_cached_snapshot"
        except Exception as error:
            row["windows_status"] = "unavailable"
            row["errors"].append(f"windows: {type(error).__name__}: {error}")
        try:
            phase = json.loads((self.root / "reports" / f"phase_memory_phase_{self.pid}.json").read_text())
            row["phase"] = {key: phase[key] for key in ("phase", "recorded_at_unix", "monotonic_seconds", "fields")}
        except (OSError, ValueError, KeyError):
            row["phase"] = None
        return row

    def _record(self, row):
        _append(self.root / "logs/phase_memory.jsonl", row)
        _atomic(self.root / "reports" / f"phase_memory_current_{self.pid}.json", row)
        self._samples += 1
        self._degraded_samples += bool(row["errors"])
        for group in ("process", "smaps_rollup", "system"):
            for key, value in row["linux"].get(group, {}).items():
                name = f"linux.{group}.{key}"
                self._peaks[name] = max(self._peaks.get(name, 0), value)
                self._minima[name] = min(self._minima.get(name, value), value)
        if row.get("windows"):
            for key in ("committed_bytes", "physical_available_bytes", "commit_free_bytes"):
                if key in row["windows"]:
                    name = f"windows.{key}"
                    value = row["windows"][key]
                    self._peaks[name] = max(self._peaks.get(name, 0), value)
                    self._minima[name] = min(self._minima.get(name, value), value)

    def _run(self):
        try:
            while not self._stop.is_set():
                row = self.sample_once()
                self._record(row)
                if row["target_status"] != "present":
                    self._terminal_status = row["target_status"]
                    break
                self._stop.wait(self.interval)
        except Exception as error:
            self._last_error = f"{type(error).__name__}: {error}"
        finally:
            self._write_summary()

    def _write_summary(self):
        report = dict(schema=SCHEMA, run_id=self.run_id, label=self.label, pid=self.pid,
                      observer_pid=os.getpid(), started_at=self._started_at, finished_at=_utc(),
                      interval_seconds=self.interval, samples=self._samples,
                      degraded_samples=self._degraded_samples, sampled_maxima_bytes=self._peaks,
                      sampled_minima_bytes=self._minima,
                      terminal_target_status=self._terminal_status, error=self._last_error,
                      windows_process_counters_collected=False,
                      gpu_runtime_imported_by_sampler=False, robot_commands_sent=0)
        _atomic(self.root / "reports" / f"phase_memory_{self.label}_{self.pid}_{self.run_id}.json", report)
        return report

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Sampler can only be started once")
        self.process_start_ticks = _identity(self.proc_root, self.pid)
        self._started_at = _utc()
        self._thread = threading.Thread(target=self._run, name=f"phase-memory-{self.pid}", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval + 1.0))
            if self._thread.is_alive():
                raise RuntimeError("Phase memory sampler thread did not stop")
        return self._write_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--label", default="process")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    sampler = MemorySampler(args.root, args.pid, label=args.label, interval=args.interval)
    def stop(_number, _frame):
        sampler._stop.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    sampler.start()
    try:
        while sampler._thread.is_alive():
            sampler._thread.join(timeout=0.5)
    finally:
        print(json.dumps(sampler.close(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
