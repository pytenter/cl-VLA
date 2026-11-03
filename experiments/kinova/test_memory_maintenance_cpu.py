"""CPU fixtures only: sync is mocked; drop_caches is a temporary ordinary file."""
import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import memory_maintenance as maintenance


class Clock:
    def __init__(self):
        self.value = 100.0
    def monotonic(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='kinova-cache-maintenance-cpu-') as temporary, ExitStack() as stack:
        root = Path(temporary)
        (root/'configs').mkdir()
        libero = (root/'libero_four', root/'libero_plain_prototype')
        for directory in libero:
            directory.mkdir()
            (directory/'status.json').write_text(json.dumps({'status': 'complete'}))
        policy = {'mode': 'exclusive_gpu', 'user_authorized': True,
                  'drop_clean_cache_after_native_cpu_restore': True,
                  'required_completed_roots': list(map(str, libero)),
                  'min_windows_commit_free_gib_before_launch': 16,
                  'windows_sample_max_age_seconds': 5, 'native_cache_reclaim_wait_seconds': 15}
        workflow = {'status': 'running', 'child_pid': os.getpid(), 'current_job': 'smoke_train32'}
        def save_policy():
            (root/'configs/resource_policy.json').write_text(json.dumps(policy))
        def save_workflow():
            (root/'workflow.json').write_text(json.dumps(workflow))
        save_policy(); save_workflow()
        meminfo = root/'meminfo'
        meminfo.write_text('Cached: 17825792 kB\nAnonPages: 7340032 kB\nMemAvailable: 22020096 kB\nDirty: 0 kB\nWriteback: 0 kB\n')
        drop = root/'drop_caches_fixture'
        clock = Clock()
        state = {'headroom_gib': 17, 'stale': False, 'pre_drop_only': False}
        def windows(_root, max_age):
            assert max_age <= 5
            if state['stale']:
                raise RuntimeError('CPU fixture: Windows cached monitor is stale')
            return {'commit_free_bytes': int(state['headroom_gib'] * maintenance.GIB),
                    'received_monotonic': 99.0 if state['pre_drop_only'] else clock.value,
                    'source': 'CPU_fixture_not_live_Windows', 'age_seconds': 0.0}
        stack.enter_context(patch.dict(os.environ, {'KINOVA_EXPERIMENT_ROOT': str(root)}))
        stack.enter_context(patch.object(maintenance, 'LIBERO_ROOTS', libero))
        stack.enter_context(patch.object(maintenance, 'MEMINFO_PATH', meminfo))
        stack.enter_context(patch.object(maintenance, 'DROP_CACHES_PATH', drop))
        stack.enter_context(patch.object(maintenance.time, 'monotonic', clock.monotonic))
        stack.enter_context(patch.object(maintenance.time, 'sleep', clock.sleep))
        sync = stack.enter_context(patch.object(maintenance.os, 'sync'))
        read_windows = stack.enter_context(patch.object(maintenance, '_windows', side_effect=windows))
        drop_call = stack.enter_context(patch.object(maintenance, '_drop_clean_pagecache', wraps=maintenance._drop_clean_pagecache))
        yield dict(root=root, policy=policy, workflow=workflow, libero=libero, drop=drop,
                   clock=clock, state=state, sync=sync, windows=read_windows, drop_call=drop_call,
                   save_policy=save_policy, save_workflow=save_workflow)


def refused(context):
    try:
        maintenance.reclaim_before_device_transfer()
    except maintenance.CacheMaintenanceError:
        return True
    return False


def run(output):
    checks = {}
    with fixture() as f:
        f['policy']['drop_clean_cache_after_native_cpu_restore'] = False; f['save_policy']()
        result = maintenance.reclaim_before_device_transfer()
        checks['disabled_policy_noop_no_sync_drop_or_windows'] = (result['status'] == 'skipped' and not f['sync'].called and not f['drop_call'].called and not f['windows'].called)
    for key, value in [('mode', 'shared_gpu'), ('user_authorized', False)]:
        with fixture() as f:
            f['policy'][key] = value; f['save_policy']()
            checks['reject_policy_' + key] = refused(f) and not f['sync'].called and not f['drop_call'].called
    for key, value in [('child_pid', os.getpid()+1), ('status', 'waiting_resources'), ('current_job', 'prepare_cpu')]:
        with fixture() as f:
            f['workflow'][key] = value; f['save_workflow']()
            checks['reject_workflow_' + key] = refused(f) and not f['drop_call'].called
    for index in range(2):
        with fixture() as f:
            (f['libero'][index]/'status.json').write_text(json.dumps({'status': 'running'}))
            checks[f'require_libero_{index}_complete'] = refused(f) and not f['drop_call'].called
    with fixture() as f:
        f['state']['stale'] = True
        checks['stale_cached_windows_blocks_before_sync_drop'] = refused(f) and not f['sync'].called and not f['drop_call'].called
    with fixture() as f:
        def race():
            f['workflow']['child_pid'] += 1; f['save_workflow']()
        f['sync'].side_effect = race
        checks['recheck_ownership_after_sync'] = refused(f) and f['sync'].call_count == 1 and not f['drop_call'].called
    with fixture() as f:
        def race():
            f['policy']['user_authorized'] = False; f['save_policy']()
        f['sync'].side_effect = race
        checks['recheck_policy_after_sync'] = refused(f) and not f['drop_call'].called
    with fixture() as f:
        result = maintenance.reclaim_before_device_transfer()
        checks['success_only_literal_one_after_sync_with_json_report'] = (
            result['status'] == 'ready_after_reclaim' and f['drop'].read_text() == '1\n'
            and f['sync'].call_count == 1 and f['drop_call'].call_count == 1
            and result['after']['windows']['commit_free_bytes'] >= 16 * maintenance.GIB
            and result['after']['windows']['received_monotonic'] > result['dropped_monotonic']
            and (f['root']/'reports/device_transfer_reclaim.jsonl').is_file()
            and json.loads(json.dumps(result))['drop_performed'] is True)
        checks['same_process_second_call_never_drops_again'] = refused(f) and f['drop_call'].call_count == 1
    with fixture() as f:
        f['state']['headroom_gib'] = 15.9
        checks['below_16gib_aborts_before_gpu_after_at_most_15s'] = refused(f) and f['clock'].value == 115 and f['drop_call'].call_count == 1
    with fixture() as f:
        f['policy']['min_windows_commit_free_gib_before_launch'] = 8; f['save_policy']()
        f['state']['headroom_gib'] = 12
        checks['policy_cannot_lower_required_16gib_floor'] = refused(f) and f['drop_call'].call_count == 1
    with fixture() as f:
        f['state']['pre_drop_only'] = True
        checks['fresh_but_pre_drop_windows_row_cannot_release_transfer'] = refused(f) and f['clock'].value == 115
    with fixture() as f:
        f['policy']['native_cache_reclaim_min_windows_commit_free_gib'] = 12; f['save_policy']()
        original_policy = (f['root']/'configs/resource_policy.json').read_bytes()
        f['state']['headroom_gib'] = 12.5
        result = maintenance.reclaim_before_device_transfer()
        checks['phase_12gib_accepts_fresh_12_5gib'] = result['status'] == 'ready_after_reclaim' and result['required_windows_commit_free_bytes'] == 12 * maintenance.GIB
        checks['startup_16gib_policy_unchanged'] = (f['root']/'configs/resource_policy.json').read_bytes() == original_policy and f['policy']['min_windows_commit_free_gib_before_launch'] == 16
    with fixture() as f:
        f['policy']['native_cache_reclaim_min_windows_commit_free_gib'] = 8; f['save_policy']()
        f['state']['headroom_gib'] = 11.9
        checks['phase_policy_cannot_lower_12gib_absolute_floor'] = refused(f) and f['drop_call'].call_count == 1
    with fixture() as f:
        f['policy']['native_cache_reclaim_wait_seconds'] = 60; f['save_policy']()
        f['state']['headroom_gib'] = 12
        checks['60second_wait_allowed_single_drop_and_times_out'] = refused(f) and f['clock'].value == 160 and f['drop_call'].call_count == 1
    with patch.dict(os.environ, {}, clear=True):
        checks['unconfigured_cpu_context_noop'] = maintenance.reclaim_before_device_transfer()['status'] == 'skipped'
    gpu_modules = sorted(name for name in sys.modules if name.split('.')[0] in ('jax', 'jaxlib', 'torch', 'tensorflow', 'cupy'))
    checks['no_gpu_framework_imported'] = not gpu_modules
    report = {'schema': 'kinova-memory-maintenance-cpu-v1', 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'check_count': len(checks), 'scope': 'Temporary Linux/Windows fixtures; os.sync mocked; only literal1 written to a temporary ordinary file. No real cache reclaim/headroom effectiveness claim.',
              'actual_sync_executed': False, 'actual_global_drop_executed': False, 'gpu_modules_imported': gpu_modules,
              'model_loaded': False, 'gpu_work_started': False, 'windows_process_started': False,
              'persistent_settings_changed': False, 'robot_commands_sent': 0}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
