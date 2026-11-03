"""CPU-only tests. No model, JAX, CUDA or Windows subprocess is started."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from phase_memory import MemorySampler, _bytes, mark_phase


def _fixture(proc, pid=123, identity=456):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stat").write_text(f"{pid} (cpu fixture) " + " ".join(["S"] + ["0"] * 18 + [str(identity)] + ["0"] * 8))
    (directory / "status").write_text("Name:\tcpu_fixture\nVmRSS:\t4096 kB\nVmHWM:\t5120 kB\nVmSize:\t8192 kB\nRssAnon:\t3072 kB\nRssFile:\t1024 kB\nRssShmem:\t0 kB\n")
    (directory / "smaps_rollup").write_text("000000-ffffff ---p 00000000 00:00 0 [rollup]\nRss: 4096 kB\nPss: 3584 kB\nAnonymous: 3072 kB\nPrivate_Dirty: 3072 kB\nPss_File: 512 kB\n")
    (proc / "meminfo").write_text("MemTotal: 32768000 kB\nMemFree: 10000000 kB\nMemAvailable: 20000000 kB\nCached: 8000000 kB\nAnonPages: 1000000 kB\nDirty: 32 kB\n")


def _windows(_root, _max_age):
    return {"source": "CPU_fixture_not_live_Windows", "committed_bytes": 10 * 2**30,
            "commit_limit_bytes": 42 * 2**30, "commit_free_bytes": 32 * 2**30,
            "physical_available_bytes": 24 * 2**30, "age_seconds": 0.0}


def run_tests(output):
    checks = {}
    with tempfile.TemporaryDirectory(prefix="kinova-phase-memory-cpu-") as temporary:
        root = Path(temporary)
        proc = root / "proc"
        _fixture(proc)
        sample = MemorySampler(root, 123, proc_root=proc, windows_reader=_windows).sample_once()
        checks["linux_kib_converts_to_integer_bytes"] = sample["linux"]["process"]["VmRSS_bytes"] == 4096 * 1024
        checks["separate_rss_pss_anon_file_and_system_cache"] = (
            sample["linux"]["smaps_rollup"]["Pss_bytes"] == 3584 * 1024
            and sample["linux"]["process"]["RssFile_bytes"] == 1024 * 1024
            and sample["linux"]["system"]["Cached_bytes"] == 8000000 * 1024)
        checks["windows_snapshot_preserved_without_refresh"] = sample["windows"] == _windows(None, None)
        mark_phase(root, "cpu.dataset_open.before", pid=123, split="train", model_loaded=False)
        sampler = MemorySampler(root, 123, proc_root=proc, windows_reader=_windows, interval=0.02)
        sampler.start()
        time.sleep(0.08)
        summary = sampler.close()
        rows = [json.loads(line) for line in (root / "logs/phase_memory.jsonl").read_text().splitlines()]
        checks["phase_and_samples_share_parseable_append_log"] = (
            rows[0]["kind"] == "phase" and summary["samples"] >= 2
            and all(row["phase"]["phase"] == "cpu.dataset_open.before" for row in rows if row["kind"] == "sample"))
        checks["sampler_thread_stops_on_close"] = not sampler._thread.is_alive()
        def stale(_root, _max_age):
            raise RuntimeError("CPU fixture: cached Windows telemetry is stale")
        rejected = MemorySampler(root, 123, proc_root=proc, windows_reader=stale).sample_once()
        checks["stale_windows_is_missing_not_zero_capacity"] = (
            rejected["windows"] is None and rejected["windows_status"] == "unavailable"
            and any("stale" in error for error in rejected["errors"]))
        pinned = MemorySampler(root, 123, proc_root=proc, windows_reader=_windows)
        pinned.sample_once()
        _fixture(proc, identity=789)
        checks["pid_reuse_rejected"] = pinned.sample_once()["target_status"] == "pid_reused"
        (proc / "123" / "smaps_rollup").unlink()
        missing_smaps = MemorySampler(root, 123, proc_root=proc, windows_reader=_windows).sample_once()
        checks["unavailable_smaps_retains_status_counters"] = (
            missing_smaps["target_status"] == "present"
            and missing_smaps["linux"]["process"]["VmRSS_bytes"] > 0 and bool(missing_smaps["errors"]))
        checks["missing_pid_reported"] = MemorySampler(root, 999, proc_root=proc, windows_reader=_windows).sample_once()["target_status"] == "unavailable"
        bad_unit_rejected = False
        try:
            _bytes("VmRSS: 123 MB", ("VmRSS",))
        except ValueError:
            bad_unit_rejected = True
        checks["unexpected_counter_units_rejected"] = bad_unit_rejected

        live = {"status": "skipped", "reason": "Real /proc is unavailable on this CPU test host"}
        if Path("/proc/self/status").is_file():
            child = subprocess.Popen([sys.executable, "-c", "import time; data=bytearray(2*1024*1024); print('READY',flush=True); time.sleep(10)"], stdout=subprocess.PIPE, text=True)
            try:
                if child.stdout.readline().strip() != "READY":
                    raise RuntimeError("CPU fixture child did not become ready")
                real = MemorySampler(root, child.pid, label="cpu_live_child", windows_reader=_windows, interval=0.05).start()
                time.sleep(0.2)
                result = real.close()
                current = json.loads((root / "reports" / f"phase_memory_current_{child.pid}.json").read_text())
                checks["live_linux_cpu_child_rss_pss_collected"] = (
                    result["samples"] >= 2 and current["linux"]["process"]["VmRSS_bytes"] >= 2 * 1024**2
                    and current["linux"]["smaps_rollup"]["Pss_bytes"] > 0 and not real._thread.is_alive())
                live = {"status": "passed" if checks["live_linux_cpu_child_rss_pss_collected"] else "failed",
                        "samples": result["samples"], "rss_bytes": current["linux"]["process"]["VmRSS_bytes"],
                        "pss_bytes": current["linux"]["smaps_rollup"]["Pss_bytes"],
                        "windows_source": "injected CPU fixture; no live Windows claim"}
            finally:
                child.terminate()
                child.wait(timeout=5)
                child.stdout.close()
        gpu_modules = sorted(name for name in sys.modules if name.split(".")[0] in ("jax", "jaxlib", "torch", "tensorflow", "cupy"))
        checks["no_gpu_runtime_imported"] = not gpu_modules
    report = {"schema": "kinova-phase-memory-cpu-v1", "status": "passed" if all(checks.values()) else "failed",
              "checks": checks, "check_count": len(checks), "live_linux_cpu_child": live,
              "gpu_modules_imported": gpu_modules, "model_loaded": False, "gpu_work_started": False,
              "windows_process_started": False, "system_settings_changed": False, "robot_commands_sent": 0}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run_tests(parser.parse_args().output)
