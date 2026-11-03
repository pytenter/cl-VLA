"""Read-only exclusive-GPU admission and runtime protection; stdlib until admission."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', project_path('kinova', '')))
LIBERO_STATUS_PATHS = (Path(project_path('libero_fourway', 'status.json')),
                       Path(project_path('libero_plain_prototype', 'status.json')))
GPU_MODES = frozenset(('exclusive_gpu', 'exclusive_idle', 'shared_gpu'))
DROP_CACHES_PATH = Path('/proc/sys/vm/drop_caches')


def load_policy():
    value = json.loads((ROOT/'configs/resource_policy.json').read_text())
    if not isinstance(value, dict):
        raise ValueError('Resource policy must be a JSON object')
    return value


def allocator_environment(policy):
    actual = {key: os.environ.get(key) for key in ('XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                                                  'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_CLIENT_MEM_FRACTION')}
    try:
        fraction, approved = float(actual['XLA_PYTHON_CLIENT_MEM_FRACTION'] or 0), float(policy.get('jax_memory_fraction', 0))
    except (ValueError, TypeError):
        fraction = approved = float('nan')
    valid = (policy.get('mode') == 'exclusive_gpu' and policy.get('user_authorized') is True
             and math.isfinite(fraction) and math.isfinite(approved) and abs(fraction-.85) < 1e-12 and abs(approved-.85) < 1e-12
             and actual['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false' and actual['XLA_PYTHON_CLIENT_ALLOCATOR'] == 'bfc'
             and not actual['XLA_CLIENT_MEM_FRACTION'])
    return actual, valid


def gpu_snapshot():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = []
    for values in csv.reader(raw.strip().splitlines()):
        if len(values) != 6:
            raise ValueError('Unexpected nvidia-smi GPU row')
        v = [x.strip() for x in values]
        rows.append(dict(index=v[0], name=v[1], total_mib=int(v[2]), used_mib=int(v[3]), free_mib=int(v[4]), utilization_percent=int(v[5])))
    if len(rows) != 1:
        raise ValueError('Expected one explicitly identified GPU')
    return rows


def host_available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            result = int(line.split()[1])*1024
            if result <= 0:
                raise ValueError('Invalid Linux MemAvailable')
            return result
    raise ValueError('Linux MemAvailable is missing')


def linux_memory_snapshot():
    wanted = {'MemAvailable': 'available_bytes', 'Cached': 'cached_bytes', 'AnonPages': 'anon_bytes',
              'Dirty': 'dirty_bytes', 'Writeback': 'writeback_bytes'}
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in wanted:
            result[wanted[key]] = int(value.split()[0])*1024
    if set(result) != set(wanted.values()) or any(value < 0 for value in result.values()):
        raise ValueError('Linux idle-cache counters are missing or invalid')
    return result


def _idle_reclaim_snapshot(monitor):
    windows = monitor.assert_healthy()
    statuses = {}
    for path in LIBERO_STATUS_PATHS:
        status = json.loads(path.read_text())
        statuses[str(path)] = {key: status.get(key) for key in ('status', 'pid', 'server_pid', 'updated_at')}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
                                  text=True, timeout=15).strip()
    return {'windows_memory': windows, 'linux_memory': linux_memory_snapshot(),
            'existing_cuda_processes': raw.splitlines() if raw else [], 'libero_experiments': statuses,
            'libero_complete': all(value.get('status') == 'complete' for value in statuses.values())}


def _drop_clean_pagecache():
    # Bit 1 drops clean file pagecache only; never use bits 2/3 (slab caches).
    with DROP_CACHES_PATH.open('w') as stream:
        stream.write('1\n')


class IdlePagecacheReclaimer:
    """One optional idle-cache maintenance attempt per queued GPU phase.

    Never called by a model or training path. The caller retains this object
    across all pre-JAX deferrals for the same phase. Admission floors stay intact.
    """

    def __init__(self, wait_phase):
        self.wait_phase = wait_phase
        self.attempted = False
        self.operation_id = uuid.uuid4().hex

    @staticmethod
    def _conditions(state):
        return (state['libero_complete'] is True and not state['existing_cuda_processes']
                and state['windows_memory']['commit_free_bytes'] < 16*2**30
                and state['linux_memory']['cached_bytes'] > 8*2**30
                and state['linux_memory']['anon_bytes'] < 2*2**30)

    def _append(self, value):
        path = ROOT/'reports/idle_reclaim.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def maybe_reclaim(self, monitor, *, gate, own_child):
        skipped = {'attempted': False, 'performed': False, 'wait_phase': self.wait_phase}
        if self.attempted:
            return dict(skipped, status='budget_spent')
        if own_child is not None or not isinstance(gate, dict) or gate.get('eligible') is not False:
            return dict(skipped, status='not_idle_or_not_blocked')
        try:
            policy = load_policy()
            _, environment_valid = allocator_environment(policy)
            if (policy.get('mode') != 'exclusive_gpu' or policy.get('user_authorized') is not True
                    or policy.get('idle_clean_pagecache_reclaim') is not True or not environment_valid
                    or policy.get('gpu_branch_disabled_after_failed_retry')):
                return dict(skipped, status='not_authorized_by_exclusive_policy')
            before = _idle_reclaim_snapshot(monitor)
            if not self._conditions(before):
                return dict(skipped, status='conditions_not_met')
        except Exception as error:
            return dict(skipped, status='unverified', error=f'{type(error).__name__}: {error}')
        self.attempted = True  # Even a failed sync/write cannot spin into another attempt.
        started = time.monotonic()
        record = {'operation_id': self.operation_id, 'wait_phase': self.wait_phase, 'attempted': True,
                  'performed': False, 'before': before, 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'action': 'os.sync then drop_caches=1 while no own child or CUDA process exists',
                  'persistent_system_configuration_changed': False, 'gpu_work_started_by_reclaim': False,
                  'optimizer_updates_performed_by_reclaim': 0}
        self._append(dict(record, event='started'))
        try:
            os.sync()
            # Recheck the full idle conditions after sync: never discard caches
            # if a CUDA job appeared or the memory pressure already recovered.
            current_policy = load_policy()
            _, environment_valid = allocator_environment(current_policy)
            synchronized = _idle_reclaim_snapshot(monitor)
            if (not environment_valid or current_policy.get('idle_clean_pagecache_reclaim') is not True
                    or current_policy.get('gpu_branch_disabled_after_failed_retry') or not self._conditions(synchronized)):
                record.update(event='cancelled_after_sync', status='conditions_changed', after=synchronized)
            else:
                _drop_clean_pagecache()
                record['performed'] = True
                action_finished = time.monotonic()
                deadline = action_finished+5
                while True:
                    after = _idle_reclaim_snapshot(monitor)
                    fresh_after = after['windows_memory']['received_monotonic'] >= action_finished
                    if fresh_after or time.monotonic() >= deadline:
                        break
                    time.sleep(.1)
                record.update(event='completed', status='completed', after=after, post_action_sample_fresh=fresh_after)
        except Exception as error:
            record.update(event='failed', status='failed', error=f'{type(error).__name__}: {error}')
        record.update(finished_at=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        self._append(record)
        return record


def windows_launch_window(latest, seconds=30, max_gap=3):
    """Read the actual trailing window, avoiding a stale 240-second minimum."""
    if latest.get('sampling_mode') != 'persistent_monitor':
        raise ValueError('Exclusive admission requires the persistent Windows monitor')
    rows = []
    path = ROOT/'logs/windows_commit.jsonl'
    if path.is_file():
        with path.open('rb') as stream:
            start = max(0, path.stat().st_size-512*1024)
            stream.seek(start)
            if start:
                stream.readline()
            for line in stream:
                try:
                    row = json.loads(line)
                    if (row.get('marker') == latest['marker'] and row.get('linux_boot_id') == latest['linux_boot_id']
                            and row.get('sampling_mode') == 'persistent_monitor'):
                        rows.append(row)
                except (ValueError, UnicodeError):
                    continue  # Any resulting gap still has to pass the check below.
    rows.append(latest)  # current.json may be atomically published just before its JSONL append.
    by_sequence = {int(row['sequence']): row for row in rows}
    rows = [by_sequence[key] for key in sorted(by_sequence)]
    cutoff = float(latest['sampled_at_unix'])-seconds
    boundary = [i for i, row in enumerate(rows) if float(row['sampled_at_unix']) <= cutoff]
    if boundary:
        rows = rows[boundary[-1]:]
    sampled = [float(row['sampled_at_unix']) for row in rows]
    received = [float(row['received_monotonic']) for row in rows]
    if not all(math.isfinite(x) for x in sampled+received):
        raise ValueError('Invalid Windows history timestamps')
    gaps = [b-a for times in (sampled, received) for a,b in zip(times,times[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError('Windows history timestamps must increase strictly')
    observed = min(sampled[-1]-sampled[0], received[-1]-received[0])
    gap = max(gaps, default=0.)
    return dict(required_seconds=seconds, observed_seconds=observed, samples=len(rows), maximum_sample_gap_seconds=gap,
                minimum_commit_free_bytes=min(int(row['commit_free_bytes']) for row in rows),
                continuous=observed >= seconds and gap <= max_gap)


def inspect_resources():
    """No model imports; unknown or missing measurements always block admission."""
    reasons = []
    result = dict(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(), resource_mode='exclusive_gpu')
    try:
        policy = load_policy()
        result['policy'] = policy
    except (OSError, ValueError) as error:
        return dict(result, eligible=False, blocking_reasons=[f'Cannot read resource policy: {error}'])
    if policy.get('mode') != 'exclusive_gpu' or policy.get('user_authorized') is not True:
        reasons.append('This run requires explicit authorized exclusive_gpu policy')
    if policy.get('gpu_branch_disabled_after_failed_retry'):
        reasons.append('This branch was explicitly disabled after an actual failed attempt')
    actual, valid = allocator_environment(policy)
    result['allocator_environment'] = actual
    if not valid:
        reasons.append('Exclusive GPU requires BFC, preallocate=false and fraction=.85 before model import')
    completed = {}
    for path in LIBERO_STATUS_PATHS:
        try:
            status = json.loads(path.read_text())
            completed[str(path)] = {key: status.get(key) for key in ('status', 'pid', 'server_pid', 'updated_at')}
            if status.get('status') != 'complete':
                reasons.append(f'Required LIBERO experiment is not complete: {path}')
        except (OSError, ValueError, AttributeError) as error:
            reasons.append(f'Cannot confirm required LIBERO experiment: {path}: {error}')
    result['libero_experiments'] = completed
    result['libero'] = completed.get(str(LIBERO_STATUS_PATHS[0]), {})
    try:
        result['gpus'] = gpu_snapshot()
        gpu = result['gpus'][0]
        minimum = max(40000, int(policy.get('min_free_vram_mib_before_launch', 40000)))
        if gpu['free_mib'] < minimum:
            reasons.append(f'Less than {minimum}MiB free VRAM')
        if gpu['utilization_percent'] > 5:
            reasons.append('GPU utilization is above 5%')
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'],
                                      text=True, timeout=15).strip()
        result['existing_cuda_processes'] = raw.splitlines() if raw else []
        if raw:
            reasons.append('Existing CUDA process present; no process will be stopped or allowed alongside this run')
        available = host_available_bytes()
        result.update(host_available_bytes=available, host_available_mib=available/2**20)
        minimum_host = max(16384, int(policy.get('min_host_available_mib_before_launch', 16384)))
        if available < minimum_host*2**20:
            reasons.append(f'Linux available memory is below {minimum_host}MiB')
    except (subprocess.SubprocessError, OSError, ValueError, TypeError) as error:
        reasons.append(f'Cannot verify GPU/Linux memory resources: {error}')
    try:
        from windows_memory import snapshot
        latest = snapshot(ROOT, max_age=min(5., float(policy.get('windows_sample_max_age_seconds', 5))), refresh=False)
        result['windows_memory'] = latest
        minimum = max(16., float(policy.get('min_windows_commit_free_gib_before_launch', 16)))*2**30
        if latest['commit_free_bytes'] < minimum:
            reasons.append('Windows commit headroom is below the 16GiB-or-stricter launch floor')
        window = windows_launch_window(latest, max(30., float(policy.get('windows_stable_window_seconds', 30))))
        result['windows_admission_window'] = window
        if not window['continuous'] or window['minimum_commit_free_bytes'] < minimum:
            reasons.append('Windows commit lacks a full 30-second stable window with sample gaps <=3 seconds')
    except Exception as error:
        reasons.append(f'Cannot verify current Windows commit headroom/window: {error}')
    return dict(result, eligible=not reasons, blocking_reasons=reasons)


def runtime_resource_snapshot(monitor):
    """Apply health and reserve checks to every GPU mode, not just shared work."""
    policy = load_policy()
    if policy.get('mode') not in GPU_MODES:
        raise RuntimeError('Unrecognized GPU resource mode during active child')
    windows = monitor.assert_healthy()  # Enforces fresh <=5-second persistent data.
    if not isinstance(windows, dict):
        raise RuntimeError('Persistent Windows monitor returned no sample')
    free = gpu_snapshot()[0]['free_mib']
    available = host_available_bytes()
    reasons = []
    if free < max(2048, int(policy.get('reserve_free_vram_mib', 2048))): reasons.append('GPU memory reserve exhausted')
    if available < max(2048, int(policy.get('abort_host_available_mib', 2048)))*2**20: reasons.append('Linux memory reserve exhausted')
    if windows['commit_free_bytes'] < max(1.5, float(policy.get('abort_windows_commit_free_gib', 1.5)))*2**30: reasons.append('Windows commit reserve exhausted')
    return dict(eligible=not reasons, blocking_reasons=reasons, mode=policy['mode'], free_vram_mib=free,
                host_available_bytes=available, host_available_mib=available/2**20,
                windows_commit_free_gib=windows['commit_free_bytes']/2**30, windows_memory=windows)


def verify_allocator_limit():
    """After admission, check .85 of total VRAM in the JAX BFC pool, not a hardware cap."""
    policy = load_policy()
    actual, valid = allocator_environment(policy)
    if not valid:
        raise RuntimeError('Exclusive GPU allocation environment disagrees with approved .85 policy')
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu': raise RuntimeError('Expected one actual GPU')
    jax.block_until_ready(jnp.zeros((), jnp.float32))
    stats = devices[0].memory_stats()
    expected = gpu_snapshot()[0]['total_mib']*2**20*.85
    limit = stats.get('bytes_limit') if stats else None
    if limit is None or not .90*expected <= limit <= 1.05*expected:
        raise RuntimeError(f'BFC allocator fraction unverified: bytes_limit={limit}, expected~{expected}')
    return dict(status='passed', jax_memory_fraction=.85, allocator_bytes_limit=limit, memory_stats=stats,
                allocator_environment=actual, scope='JAX BFC pool only; total-card, Windows and Linux headroom monitored separately')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = inspect_resources()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['eligible'] else 2)


if __name__ == '__main__': main()
