"""Read-only Windows commit telemetry for WSL Kinova jobs; no GPU/system changes.

All memory fields are integer BYTES. GetPerformanceInfo reports pages, which
are multiplied by its PageSize. The monitor owns one persistent PowerShell
process; closing stdin stops it, including when the Python producer exits.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
SCHEMA = "kinova-windows-memory-v1"


class WindowsMemoryError(RuntimeError):
    pass


class WindowsHeadroomError(WindowsMemoryError):
    pass


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _command(source):
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand",
            base64.b64encode(source.encode("utf-16le")).decode("ascii")]


def _source(marker, *, once=False, interval_ms=1000):
    # The source marker also permits exact ownership verification if cooperative
    # stdin shutdown fails. Never terminate a process just because it is PowerShell.
    source = r'''$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)
$marker='__MARKER__'
[Console]::Out.WriteLine((@{kind='started';windows_pid=$PID;marker=$marker}|ConvertTo-Json -Compress))
[Console]::Out.Flush()
Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
namespace KinovaCommitTelemetry {
 [StructLayout(LayoutKind.Sequential)] public struct PerformanceInfo {
  public uint cb;
  public UIntPtr CommitTotal, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable;
  public UIntPtr SystemCache, KernelTotal, KernelPaged, KernelNonpaged, PageSize;
  public uint HandleCount, ProcessCount, ThreadCount;
 }
 public static class Native {
  [DllImport("psapi.dll", SetLastError=true)]
  private static extern bool GetPerformanceInfo(ref PerformanceInfo info, uint size);
  public static ulong[] Read() {
   var p = new PerformanceInfo(); p.cb=(uint)Marshal.SizeOf(typeof(PerformanceInfo));
   if(!GetPerformanceInfo(ref p,p.cb)) throw new Win32Exception(Marshal.GetLastWin32Error());
   ulong s=p.PageSize.ToUInt64();
   return new ulong[] {checked(p.CommitTotal.ToUInt64()*s),checked(p.CommitLimit.ToUInt64()*s),
    checked(p.PhysicalTotal.ToUInt64()*s),checked(p.PhysicalAvailable.ToUInt64()*s),s};
  }
  public static Task<string> WaitForStop() { return Task.Run(()=>Console.ReadLine()); }
 }
}
'@
$once=__ONCE__
$stop=[KinovaCommitTelemetry.Native]::WaitForStop()
$sequence=0
while($once -or -not $stop.IsCompleted) {
 $v=[KinovaCommitTelemetry.Native]::Read()
 $now=[DateTimeOffset]::UtcNow
 $row=@{kind='sample';source='GetPerformanceInfo';windows_pid=$PID;marker=$marker;sequence=$sequence;
  sampled_at=$now.ToString('o');sampled_at_unix=$now.ToUnixTimeMilliseconds()/1000.0;
  committed_bytes=$v[0];commit_limit_bytes=$v[1];commit_free_bytes=([long]$v[1]-[long]$v[0]);
  physical_total_bytes=$v[2];physical_available_bytes=$v[3];page_size_bytes=$v[4]}
 $guest=Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue|Select-Object -First 1
 if($null -ne $guest) {
  $row.vmmem_pid=[int]$guest.Id
  $row.vmmem_private_bytes=[long]$guest.PrivateMemorySize64
  $row.vmmem_working_set_bytes=[long]$guest.WorkingSet64
 }
 [Console]::Out.WriteLine(($row|ConvertTo-Json -Compress))
 [Console]::Out.Flush()
 if($once){break}
 $sequence++
 [System.Threading.Thread]::Sleep(__INTERVAL_MS__)
}
'''
    return source.replace("__MARKER__", marker).replace("__ONCE__", "$true" if once else "$false").replace("__INTERVAL_MS__", str(int(interval_ms)))


def _validate(sample):
    if sample.get("source") != "GetPerformanceInfo":
        raise WindowsMemoryError("Unexpected Windows memory telemetry source")
    for key in ("committed_bytes", "commit_limit_bytes", "commit_free_bytes", "physical_total_bytes",
                "physical_available_bytes", "page_size_bytes"):
        if type(sample.get(key)) is not int:
            raise WindowsMemoryError(f"Windows byte counter {key} must be an integer")
    if (sample["committed_bytes"] < 0 or sample["commit_limit_bytes"] <= 0
            or sample["physical_total_bytes"] <= 0 or sample["page_size_bytes"] <= 0
            or not 0 <= sample["physical_available_bytes"] <= sample["physical_total_bytes"]
            or sample["commit_free_bytes"] != sample["commit_limit_bytes"] - sample["committed_bytes"]):
        raise WindowsMemoryError("Inconsistent Windows memory counters")
    if not math.isfinite(float(sample["sampled_at_unix"])):
        raise WindowsMemoryError("Invalid Windows sampling time")
    return sample


def _publish(root, sample, mode):
    sample = dict(_validate(sample), schema=SCHEMA, sampling_mode=mode, producer_pid=os.getpid(),
                  received_at_unix=time.time(), received_monotonic=time.monotonic(), linux_boot_id=_boot_id())
    root = Path(root)
    _atomic(root / "reports/windows_current.json", sample)
    path = root / "logs/windows_commit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # One O_APPEND write per record keeps concurrent fallback snapshots intact.
    data = (json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    return sample


def _fresh(sample, max_age):
    _validate(sample)
    if sample.get("schema") != SCHEMA or sample.get("linux_boot_id") != _boot_id():
        raise WindowsMemoryError("Cached Windows sample belongs to another schema/WSL boot")
    age = time.monotonic() - float(sample["received_monotonic"])
    # Reject delayed stdout and significant Windows/WSL clock disagreement too.
    windows_age = time.time() - float(sample["sampled_at_unix"])
    if age < 0 or age > max_age or windows_age < -5 or windows_age > max_age:
        raise WindowsMemoryError(f"Windows telemetry stale or clock inconsistent: receipt_age={age:.3f}s, sample_age={windows_age:.3f}s")
    return dict(sample, age_seconds=max(age, windows_age, 0.0))


def snapshot(root, max_age=5, refresh=True):
    """Get a fresh cached sample, or read-only one-shot PowerShell fallback.

    Any failure raises WindowsMemoryError; missing/stale data never means free
    capacity. This function imports no JAX, Torch or GPU runtime.
    """
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("max_age must be finite and positive")
    root = Path(root)
    try:
        return _fresh(json.loads((root / "reports/windows_current.json").read_text()), max_age)
    except (OSError, ValueError, KeyError, WindowsMemoryError) as error:
        if not refresh:
            raise WindowsMemoryError(f"No fresh Windows commit sample: {error}") from error
    marker = "KinovaWindowsMemoryMonitor-oneshot-" + uuid.uuid4().hex
    try:
        result = subprocess.run(_command(_source(marker, once=True)), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", timeout=20, check=False)
        if result.returncode:
            raise WindowsMemoryError(f"Windows snapshot exited {result.returncode}: {result.stderr[-1200:]}")
        rows = [json.loads(line.lstrip("\ufeff")) for line in result.stdout.splitlines() if line.strip()]
        samples = [row for row in rows if row.get("kind") == "sample" and row.get("marker") == marker]
        if len(samples) != 1:
            raise WindowsMemoryError("Windows one-shot returned no unique matching sample")
        return _fresh(_publish(root, samples[0], "one_shot"), max_age)
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        raise WindowsMemoryError(f"Cannot read Windows commit capacity: {error}") from error


def assert_headroom(root, reserve_gib=1.5):
    """Before a heavy leaf, enforce fresh Windows commit headroom (bytes)."""
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("reserve_gib must be finite and nonnegative")
    sample = snapshot(root, max_age=5, refresh=True)
    reserve = math.ceil(reserve_gib * 2**30)
    if sample["commit_free_bytes"] < reserve:
        raise WindowsHeadroomError(f"Windows commit headroom {sample['commit_free_bytes']} bytes below reserve {reserve} bytes; no system settings changed")
    return sample


class WindowsMemoryMonitor:
    """One persistent, 1-second PowerShell producer per experiment root."""

    def __init__(self, root):
        self.root = Path(root)
        self.marker = "KinovaWindowsMemoryMonitor-" + uuid.uuid4().hex
        self.process = None
        self.windows_pid = None
        self._closed = False
        self._sample_ready = threading.Event()
        self._stderr = deque(maxlen=20)
        self._error = None
        self._lock_file = None
        self._reader = self._error_reader = None
        self._history = deque()

    def _status(self, status, **fields):
        _atomic(self.root / "reports/windows_monitor_status.json",
                {"status": status, "marker": self.marker, "windows_pid": self.windows_pid,
                 "python_pid": os.getpid(), "updated_at_unix": time.time(), **fields})

    def _read(self):
        try:
            for line in self.process.stdout:
                row = json.loads(line.lstrip("\ufeff"))
                if row.get("marker") != self.marker:
                    raise WindowsMemoryError("Unexpected monitor ownership marker")
                self.windows_pid = int(row["windows_pid"])
                if row["kind"] == "sample":
                    self._history.append((row['sampled_at_unix'], row['commit_free_bytes']))
                    while len(self._history) > 1 and self._history[0][0] < row['sampled_at_unix']-240:
                        self._history.popleft()
                    times = [item[0] for item in self._history]
                    row['rolling_commit_window_seconds'] = times[-1]-times[0]
                    row['rolling_min_commit_free_bytes'] = min(item[1] for item in self._history)
                    row['rolling_max_sample_gap_seconds'] = max((b-a for a,b in zip(times,times[1:])), default=0.)
                    _publish(self.root, row, "persistent_monitor")
                    self._sample_ready.set()
                    self._status("running", last_sample_unix=row["sampled_at_unix"])
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
        finally:
            if not self._closed:
                self._error = self._error or "Windows monitor stdout ended unexpectedly"
                self._status("failed", error=self._error, stderr=list(self._stderr))

    def _read_error(self):
        for line in self.process.stderr:
            self._stderr.append(line.rstrip()[:1000])

    def start(self):
        if self.process is not None or self._closed:
            raise WindowsMemoryError("Monitor instances can only be started once")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / ".windows_monitor.lock").open("a")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.process = subprocess.Popen(_command(_source(self.marker)), stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, encoding="utf-8", errors="replace", bufsize=1)
            self._reader = threading.Thread(target=self._read, name="windows-commit-reader", daemon=True)
            self._error_reader = threading.Thread(target=self._read_error, name="windows-commit-stderr", daemon=True)
            self._reader.start()
            self._error_reader.start()
            if not self._sample_ready.wait(20):
                raise WindowsMemoryError(f"Windows monitor has no initial sample: {self._error}; stderr={list(self._stderr)}")
            return self
        except BaseException:
            self.close()
            raise

    def assert_healthy(self):
        if self._closed or self.process is None or self.process.poll() is not None or self._error:
            raise WindowsMemoryError(f'Persistent Windows monitor is not healthy: {self._error}')
        return snapshot(self.root, max_age=5, refresh=False)

    def _cleanup_windows(self):
        # Decode the actual command line and require our unique source marker.
        # This is safe even if the numeric PID has been reused by another job.
        source = r'''$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$marker='__MARKER__'
$matches=@(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
 if($_.ProcessId -eq $PID){return $false}
 $cmd=$_.CommandLine
 if($cmd -match '(?i)-EncodedCommand\s+"?([A-Za-z0-9+/=]+)') {
  try { [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($Matches[1])).Contains($marker) } catch {$false}
 } else {$false}
})
foreach($p in $matches){Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop}
@{owned_windows_processes_stopped=$matches.Count}|ConvertTo-Json -Compress
'''
        source = source.replace("__MARKER__", self.marker)
        result = subprocess.run(_command(source), capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=10)
        if result.returncode:
            raise WindowsMemoryError(f"Cannot clean up owned Windows monitor: {result.stderr[-1000:]}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        forced = False
        try:
            if self.process is not None:
                try:
                    if self.process.stdin and not self.process.stdin.closed:
                        self.process.stdin.write("stop\n")
                        self.process.stdin.flush()
                        self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    forced = True
                    self._cleanup_windows()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()  # Only our local WSL interop child, after verified Windows cleanup.
                        self.process.wait(timeout=3)
                for thread in (self._reader, self._error_reader):
                    if thread is not None:
                        thread.join(timeout=2)
                self._status("closed", forced_windows_cleanup=forced, process_returncode=self.process.returncode,
                             error=self._error, stderr=list(self._stderr))
        finally:
            if self._lock_file is not None:
                self._lock_file.close()


def self_test():
    """Synthetic CPU-only freshness, units and fail-closed checks."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="kinova_windows_memory_cpu_") as directory:
        root = Path(directory)
        sample = {"source": "GetPerformanceInfo", "sampled_at_unix": time.time(),
                  "committed_bytes": 8*2**30, "commit_limit_bytes": 10*2**30, "commit_free_bytes": 2*2**30,
                  "physical_total_bytes": 32*2**30, "physical_available_bytes": 4*2**30, "page_size_bytes": 4096}
        _publish(root, sample, "synthetic_cpu_test")
        assert assert_headroom(root)["commit_free_bytes"] == 2*2**30
        stale = json.loads((root / "reports/windows_current.json").read_text())
        stale["received_monotonic"] -= 60
        _atomic(root / "reports/windows_current.json", stale)
        try:
            snapshot(root, refresh=False)
        except WindowsMemoryError:
            pass
        else:
            raise AssertionError("Stale samples must fail closed")
        sample.update(sampled_at_unix=time.time(), committed_bytes=9*2**30, commit_free_bytes=2**30)
        _publish(root, sample, "synthetic_cpu_test")
        try:
            assert_headroom(root)
        except WindowsHeadroomError:
            pass
        else:
            raise AssertionError("Low commit headroom must raise before a heavy leaf")
        malformed = dict(sample, commit_free_bytes=123)
        try:
            _validate(malformed)
        except WindowsMemoryError:
            pass
        else:
            raise AssertionError("Inconsistent byte counters must fail")
    return {"status": "passed", "checks": ["integer_bytes", "fresh_cache", "stale_rejected", "low_commit_rejected", "counter_consistency"],
            "gpu_used": False, "windows_settings_changed": False, "actual_windows_sampling": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--live-test", action="store_true")
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        _atomic(args.root / "reports/windows_memory_cpu_test.json", result)
    elif args.once:
        result = snapshot(args.root, refresh=True)
    else:
        initial = snapshot(args.root, refresh=True)
        monitor = WindowsMemoryMonitor(args.root).start()
        try:
            time.sleep(min(max(args.seconds, 1), 15))
            latest = snapshot(args.root, refresh=False)
            if latest["sampling_mode"] != "persistent_monitor" or latest["sequence"] < 1:
                raise WindowsMemoryError("Persistent monitor did not advance")
        finally:
            monitor.close()
        result = {"status": "passed", "initial_one_shot": initial, "latest_persistent": latest,
                  "monitor_windows_pid": monitor.windows_pid, "monitor_closed": True,
                  "gpu_used": False, "windows_settings_changed": False}
        _atomic(args.root / "reports/windows_memory_live_test.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
