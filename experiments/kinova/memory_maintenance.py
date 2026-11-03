"""Authorized clean-file-cache reclaim between native CPU restore and GPU transfer.

This stdlib-only helper changes no persistent setting. Its only maintenance
write is literal 1 to drop_caches, after ownership/policy checks and sync. The
caller must be the frozen native-loader hook with the complete CPU tree ready.
Existing resource guards remain independent; this helper starts no monitor.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

GIB = 2**30
DROP_CACHES_PATH = Path('/proc/sys/vm/drop_caches')
MEMINFO_PATH = Path('/proc/meminfo')
LIBERO_ROOTS = (Path(project_path('libero_fourway', '')),
                Path(project_path('libero_plain_prototype', '')))
MODEL_JOBS = frozenset(('E0_wrist_frozen', 'smoke_train32', 'smoke_full_state_restore',
                        'smoke_prediction_restore', 'E1_wrist_pilot_train2000',
                        'E1_validation_500', 'E1_validation_1000',
                        'E1_validation_1500', 'E1_validation_2000'))
_LOCK = threading.Lock()
_ATTEMPTED = set()


class CacheMaintenanceError(RuntimeError):
    pass


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise CacheMaintenanceError(f'Expected JSON object: {path}')
    return value


def _policy(root):
    return _read(root / 'configs/resource_policy.json')


def _fingerprint(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _context(root, *, policy_sha=None, job=None, policy_flag="drop_clean_cache_after_native_cpu_restore"):
    policy = _policy(root)
    if (policy.get(policy_flag) is not True
            or policy.get('mode') != 'exclusive_gpu'
            or policy.get('user_authorized') is not True):
        raise CacheMaintenanceError('Clean-cache reclaim lacks explicit authorized exclusive policy')
    if policy_sha is not None and _fingerprint(policy) != policy_sha:
        raise CacheMaintenanceError('Resource policy changed during cache maintenance')
    required = policy.get('required_completed_roots', [])
    if not isinstance(required, list) or not all(str(path) in required for path in LIBERO_ROOTS):
        raise CacheMaintenanceError('Policy must require both completed LIBERO experiments')
    completed = {}
    for path in LIBERO_ROOTS:
        status = _read(path / 'status.json')
        if status.get('status') != 'complete':
            raise CacheMaintenanceError(f'Required LIBERO experiment is not complete: {path}')
        expected = status.get('expected_evaluations')
        if expected is not None and status.get('completed_evaluations') != expected:
            raise CacheMaintenanceError(f'LIBERO completion count disagrees: {path}')
        completed[str(path)] = {'status': 'complete', 'completed_evaluations': status.get('completed_evaluations')}
    workflow = _read(root / 'workflow.json')
    if (workflow.get('status') != 'running' or type(workflow.get('child_pid')) is not int
            or workflow['child_pid'] != os.getpid()
            or workflow.get('current_job') not in MODEL_JOBS
            or (job is not None and workflow.get('current_job') != job)):
        raise CacheMaintenanceError('Caller is not the current owned running model child')
    return policy, workflow['current_job'], completed


def _linux():
    wanted = {'Cached', 'AnonPages', 'MemAvailable', 'Dirty', 'Writeback'}
    counters = {}
    for line in MEMINFO_PATH.read_text().splitlines():
        key, separator, value = line.partition(':')
        if separator and key in wanted:
            fields = value.split()
            if len(fields) != 2 or fields[1] != 'kB' or int(fields[0]) < 0:
                raise CacheMaintenanceError(f'Invalid Linux memory counter: {key}')
            counters[key + '_bytes'] = int(fields[0]) * 1024
    if not {'Cached_bytes', 'AnonPages_bytes', 'MemAvailable_bytes'} <= counters.keys():
        raise CacheMaintenanceError('Required Linux memory counters are missing')
    return counters


def _windows(root, max_age):
    from windows_memory import snapshot
    # Never compile/start an additional Windows producer or one-shot fallback.
    return snapshot(root, max_age=max_age, refresh=False)


def _capture(root, max_age):
    return {'at': _now(), 'time_unix': time.time(), 'monotonic_seconds': time.monotonic(),
            'linux': _linux(), 'windows': _windows(root, max_age)}


def _drop_clean_pagecache():
    # 1 releases clean page cache only. No slab drop (2), combined drop (3),
    # remount, reboot, persistent sysctl or Windows paging change is performed.
    DROP_CACHES_PATH.write_text('1\n')


def _record(root, report, phase):
    record = dict(report, phase=phase, recorded_at=_now())
    directory = root / 'reports'
    directory.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode()
    descriptor = os.open(directory / (report.get('record_stem', 'device_transfer_reclaim') + '.jsonl'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError('Incomplete cache-maintenance log append')
    finally:
        os.close(descriptor)
    path = directory / f"{report.get('record_stem', 'device_transfer_reclaim')}_{os.getpid()}.json"
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)
    try:
        from phase_memory import mark_phase
        mark_phase(root, report.get('phase_scope', 'maintenance') + '.' + phase, status=report['status'], report=str(path),
                   drop_performed=report.get('drop_performed', False))
    except ImportError:
        pass


def _reclaim(root=None, *, policy_flag="drop_clean_cache_after_native_cpu_restore",
             phase_scope="maintenance", record_stem="device_transfer_reclaim",
             caller_contract="complete native CPU tree, before GPU transfer/native init",
             default_floor_gib=16, default_wait_seconds=15):
    """Return a JSON-safe report, or abort before transfer if safety checks fail.

    Disabled/unconfigured contexts are no-ops, preserving CPU tests and older
    protocols. Enabled contexts require the supervisor's exact owned child.
    One process may attempt one drop; failures are never automatically retried.
    """
    configured = str(root) if root is not None else os.environ.get('KINOVA_EXPERIMENT_ROOT')
    if not configured:
        return {'status': 'skipped', 'reason': 'no_experiment_root', 'drop_performed': False}
    root = Path(configured)
    if not root.is_absolute():
        raise CacheMaintenanceError('Experiment root must be absolute')
    policy = _policy(root)
    if policy.get(policy_flag) is not True:
        return {'status': 'skipped', 'reason': 'policy_flag_disabled', 'drop_performed': False}
    with _LOCK:
        report = {'schema': 'kinova-device-transfer-reclaim-v1', 'status': 'checking',
                  'pid': os.getpid(), 'root': str(root), 'started_at': _now(),
                  'drop_performed': False, 'drop_caches_value': 1,
                  'persistent_settings_changed': False, 'robot_commands_sent': 0,
                  'caller_contract': caller_contract}
        if record_stem != 'device_transfer_reclaim':
            report.update(schema='kinova-resumed-training-cache-reclaim-v1', phase_scope=phase_scope, record_stem=record_stem)
        try:
            policy, job, completed = _context(root, policy_flag=policy_flag)
            policy_sha = _fingerprint(policy)
            # The complete CPU tree is already resident at this barrier. Keep
            # its phase budget separate from the unchanged pre-CPU launch gate.
            floor = max(12 * GIB, math.ceil(float(policy.get('native_cache_reclaim_min_windows_commit_free_gib', default_floor_gib)) * GIB))
            max_age = min(5.0, float(policy.get('windows_sample_max_age_seconds', 5)))
            wait_seconds = float(policy.get('native_cache_reclaim_wait_seconds', default_wait_seconds))
            if not (math.isfinite(max_age) and max_age > 0 and math.isfinite(wait_seconds) and 0 < wait_seconds <= 120):
                raise CacheMaintenanceError('Invalid cache-maintenance wait/freshness policy')
            report.update(job=job, policy_sha256=policy_sha, completed_libero=completed,
                          required_windows_commit_free_bytes=floor, max_wait_seconds=wait_seconds)
            report['before'] = _capture(root, max_age)
            if str(root) in _ATTEMPTED:
                raise CacheMaintenanceError('This process already attempted cache reclaim; refusing another drop')
            _ATTEMPTED.add(str(root))
            report['status'] = 'syncing'
            _record(root, report, 'before_sync')
            os.sync()
            _context(root, policy_sha=policy_sha, job=job, policy_flag=policy_flag)
            report['before_drop'] = _capture(root, max_age)
            _context(root, policy_sha=policy_sha, job=job, policy_flag=policy_flag)
            _drop_clean_pagecache()
            report['drop_performed'] = True
            dropped_at = time.monotonic()
            report.update(status='waiting_windows_headroom', dropped_monotonic=dropped_at,
                          dropped_at=_now())
            _record(root, report, 'after_drop')
            deadline = dropped_at + wait_seconds
            polls = 0
            while True:
                _context(root, policy_sha=policy_sha, job=job, policy_flag=policy_flag)
                polls += 1
                try:
                    current = _capture(root, max_age)
                    report['after'] = current
                    sample = current['windows']
                    # Receipt monotonic is shared within the WSL boot. Require
                    # a post-drop monitor update, not merely a fresh old row.
                    post_drop = float(sample['received_monotonic']) > dropped_at
                    if post_drop and sample['commit_free_bytes'] >= floor:
                        report.update(status='ready_after_reclaim', polls=polls,
                                      finished_at=_now(), wait_seconds=time.monotonic()-dropped_at)
                        _record(root, report, 'ready')
                        return report
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    report['last_sample_error'] = f'{type(error).__name__}: {error}'
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CacheMaintenanceError(f'Fresh post-reclaim Windows commit headroom did not reach {floor} bytes before ' + ('GPU transfer' if record_stem == 'device_transfer_reclaim' else 'first resumed optimizer compilation'))
                time.sleep(min(0.5, remaining))
        except Exception as error:
            report.update(status='aborted_before_gpu_transfer' if record_stem == 'device_transfer_reclaim' else 'aborted_before_resumed_training_compile', finished_at=_now(),
                          error=f'{type(error).__name__}: {error}')
            _record(root, report, 'aborted')
            if isinstance(error, CacheMaintenanceError):
                raise
            raise CacheMaintenanceError(f'Cache maintenance could not establish safe headroom: {error}') from error


def reclaim_before_device_transfer():
    """Original CPU-restore/pre-transfer API; policy and defaults unchanged."""
    return _reclaim()


def reclaim_before_resumed_training(root=None):
    """Guarded once-per-process barrier after GPU restore AND sampler replay.

    This is before the first resumed optimizer JIT/compile, not CPU parameter
    restoration or GPU transfer. It changes no restore/update mathematics.
    """
    return _reclaim(root, policy_flag="resume_training_clean_cache_reclaim",
                    phase_scope="resumed_training.precompile_cache",
                    record_stem="resumed_training_cache_reclaim",
                    caller_contract="GPU checkpoint state restored and seeded sampler replay complete; before first resumed optimizer compile",
                    default_floor_gib=12, default_wait_seconds=60)
