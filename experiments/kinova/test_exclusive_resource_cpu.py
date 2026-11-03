"""Stdlib mocks of exclusive admission, telemetry and own-child cleanup; no GPU."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import types
from unittest import mock

import resource_guard as guard
import run_workflow as workflow


def run_tests():
    checks = {}
    policy = {'mode': 'exclusive_gpu', 'user_authorized': True, 'jax_memory_fraction': .85,
              'windows_stable_window_seconds': 30, 'windows_sample_max_age_seconds': 5,
              'resource_stable_checks': 2, 'resource_eligible_poll_seconds': 10, 'resource_blocked_poll_seconds': 30}
    environment = {'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '.85',
                   'XLA_PYTHON_CLIENT_ALLOCATOR': 'bfc'}
    with tempfile.TemporaryDirectory(prefix='kinova_exclusive_resources_cpu_') as temporary:
        root = Path(temporary)
        for relative in ['configs', 'reports', 'logs', 'manifests']:
            (root/relative).mkdir()
        status_files = (root/'old_fourway_status.json', root/'plain_prototype_status.json')
        gpu = {'free': 47590, 'util': 0, 'cuda': '', 'host': 24*2**30}
        end = time.time()
        rows = [dict(marker='cpu_monitor', linux_boot_id='cpu_boot', sampling_mode='persistent_monitor',
                     sequence=i, sampled_at_unix=end-40+i, received_monotonic=1000.+i,
                     commit_free_bytes=31*2**30) for i in range(41)]
        latest = [rows[-1]]
        def write_rows(values):
            (root/'logs/windows_commit.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
        def reset():
            (root/'configs/resource_policy.json').write_text(json.dumps(policy))
            for path in status_files: path.write_text('{"status":"complete"}')
            gpu.update(free=47590, util=0, cuda='', host=24*2**30)
            latest[0] = copy.deepcopy(rows[-1])
            write_rows(rows)
        def command(arguments, **kwargs):
            if '--query-compute-apps=pid,process_name' in arguments: return gpu['cuda']
            return f"0, CPU_TEST_GPU, 49140, {49140-gpu['free']}, {gpu['free']}, {gpu['util']}\n"
        def snapshot(*args, **kwargs):
            assert kwargs == {'max_age': 5., 'refresh': False}
            if isinstance(latest[0], Exception): raise latest[0]
            return latest[0]
        windows = types.ModuleType('windows_memory'); windows.snapshot = snapshot
        with mock.patch.object(guard, 'ROOT', root), mock.patch.object(guard, 'LIBERO_STATUS_PATHS', status_files), \
             mock.patch.object(guard.subprocess, 'check_output', side_effect=command), \
             mock.patch.object(guard, 'host_available_bytes', side_effect=lambda: gpu['host']), \
             mock.patch.dict(os.environ, environment, clear=True), mock.patch.dict(sys.modules, {'windows_memory': windows}):
            reset()
            assert guard.inspect_resources()['eligible'] is True
            checks['both_completed_plus_exclusive_resources_admitted'] = True
            for index in (0, 1):
                for state in ('missing', 'failure', 'running'):
                    reset()
                    if state == 'missing': status_files[index].unlink()
                    else: status_files[index].write_text(json.dumps({'status': state}))
                    assert guard.inspect_resources()['eligible'] is False
            checks['either_required_LIBERO_missing_failed_or_running_blocks'] = True
            for field, value in [('free',39999), ('util',6), ('cuda','123, other_job'), ('host',15*2**30)]:
                reset(); gpu[field] = value
                assert guard.inspect_resources()['eligible'] is False
            checks['low_VRAM_busy_GPU_any_CUDA_or_low_Linux_memory_blocks'] = True
            for field, value in [('XLA_PYTHON_CLIENT_MEM_FRACTION','.40'), ('XLA_PYTHON_CLIENT_PREALLOCATE','true'),
                                 ('XLA_PYTHON_CLIENT_ALLOCATOR','platform'), ('XLA_CLIENT_MEM_FRACTION','.85')]:
                reset()
                with mock.patch.dict(os.environ, {field:value}): assert guard.inspect_resources()['eligible'] is False
            checks['exclusive_point85_false_BFC_environment_enforced'] = True
            reset(); latest[0] = dict(rows[-1], commit_free_bytes=15*2**30)
            assert guard.inspect_resources()['eligible'] is False
            reset(); latest[0] = RuntimeError('mock stale Windows sample')
            assert guard.inspect_resources()['eligible'] is False
            reset(); write_rows(rows[-10:])
            assert guard.inspect_resources()['eligible'] is False
            reset(); write_rows([row for row in rows if row['sequence'] not in range(15,20)])
            assert guard.inspect_resources()['eligible'] is False
            reset(); values=copy.deepcopy(rows); values[-4]['commit_free_bytes']=15*2**30; write_rows(values)
            assert guard.inspect_resources()['eligible'] is False
            reset(); values=copy.deepcopy(rows); values[0]['commit_free_bytes']=1; write_rows(values)
            latest[0]['rolling_min_commit_free_bytes'] = 1  # Old 240-second minimum must not prolong the new 30-second gate.
            assert guard.inspect_resources()['eligible'] is True
            checks['Windows_floor_staleness_short_window_gap_and_recent_dip_block'] = True
            checks['actual_last30_seconds_ignore_older_240_second_dip'] = True
            class Monitor:
                fail = False
                def assert_healthy(self):
                    if self.fail: raise RuntimeError('mock unhealthy or stale monitor')
                    return latest[0]
            monitor = Monitor()
            for mode in guard.GPU_MODES:
                reset(); mode_policy=dict(policy,mode=mode)
                (root/'configs/resource_policy.json').write_text(json.dumps(mode_policy))
                assert guard.runtime_resource_snapshot(monitor)['eligible'] is True
                for field, value in [('free',2047), ('host',2047*2**20)]:
                    original=gpu[field];gpu[field]=value
                    assert guard.runtime_resource_snapshot(monitor)['eligible'] is False
                    gpu[field]=original
                latest[0]=dict(rows[-1],commit_free_bytes=2**30)
                assert guard.runtime_resource_snapshot(monitor)['eligible'] is False
                monitor.fail=True
                try: guard.runtime_resource_snapshot(monitor)
                except RuntimeError: pass
                else: raise AssertionError('Unhealthy monitor must abort active child')
                monitor.fail=False
            checks['every_GPU_mode_monitors_health_and_all_runtime_reserves'] = True
            reset()
            expected=49140*2**20*.85
            device=types.SimpleNamespace(platform='gpu',memory_stats=lambda:{'bytes_limit':int(expected)})
            fake_jax=types.ModuleType('jax');fake_jax.__path__=[]
            fake_jax.devices=lambda:[device];fake_jax.block_until_ready=lambda x:x
            fake_numpy=types.ModuleType('jax.numpy');fake_numpy.float32='mock_float32';fake_numpy.zeros=lambda *args:0
            with mock.patch.dict(sys.modules, {'jax':fake_jax, 'jax.numpy':fake_numpy}):
                assert guard.verify_allocator_limit()['jax_memory_fraction'] == .85
                device.memory_stats=lambda:{'bytes_limit':int(49140*2**20*.4)}
                try: guard.verify_allocator_limit()
                except RuntimeError: pass
                else: raise AssertionError('A .40 allocator must not pass as .85')
            checks['point85_allocator_verified_with_mock_stats_wrong_cap_rejected'] = True
        # Exercise the current wait_idle AST: exactly two stable checks at 10s spacing.
        tree=ast.parse(Path(workflow.__file__).read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        wait=copy.deepcopy(next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='wait_idle'))
        wrapper=ast.fix_missing_locations(ast.Module(body=[wait],type_ignores=[]))
        sequence=iter([{'eligible':False},{'eligible':True},{'eligible':True}]);sleeps=[]
        scope=dict(workflow.__dict__, inspect_resources=lambda:next(sequence), load_policy=lambda:policy,
                   monitor=types.SimpleNamespace(assert_healthy=lambda:None), update=lambda **kw:None, rebuild_report=lambda:None,
                   args=types.SimpleNamespace(wait=True,max_wait_hours=1),
                   time=types.SimpleNamespace(monotonic=lambda:0,sleep=lambda value:sleeps.append(value)))
        exec(compile(wrapper,str(workflow.__file__),'exec'),scope);scope['wait_idle']()
        assert sleeps == [30.,10.],sleeps
        checks['two_stable_checks_ten_seconds_apart_after_window'] = True
        # Only a fake Popen exists: prove runtime rejection targets its PGID alone.
        (root/'reports/cpu_preflight.json').write_text('{"status":"passed"}')
        (root/'manifests/workflow_code.json').write_text('{"test":"frozen"}')
        spawned=[];killed=[]
        class FakeProcess:
            pid=987654321
            returncode=None
            polls=0
            def __init__(self,*args,**kwargs): spawned.append(self)
            def poll(self):
                self.polls+=1
                return self.returncode if self.polls<3 else 0
            def wait(self,timeout=None): self.returncode=-15;return -15
        with mock.patch.object(workflow,'ROOT',root),mock.patch.object(workflow,'SCRIPTS',root/'scripts'), \
             mock.patch.object(workflow,'manifest',return_value={'test':'frozen'}),mock.patch.object(workflow,'rebuild_report'), \
             mock.patch.object(workflow,'load_policy',return_value=policy),mock.patch.object(workflow,'inspect_resources',return_value={'eligible':True}), \
             mock.patch.object(workflow,'runtime_resource_snapshot',return_value={'eligible':False,'blocking_reasons':['mock reserve']}), \
             mock.patch.object(workflow.subprocess,'Popen',FakeProcess),mock.patch.object(workflow.os,'killpg',side_effect=lambda pid,sig:killed.append((pid,sig))), \
             mock.patch.object(workflow.signal,'signal'),mock.patch.object(workflow.time,'sleep'):
            try: workflow.run(types.SimpleNamespace(wait=True,max_wait_hours=1),types.SimpleNamespace(assert_healthy=lambda:None))
            except RuntimeError as error: assert 'ResourceHeadroomAbort' in str(error)
            else: raise AssertionError('Runtime reserve rejection must stop branch')
        assert len(spawned)==1 and killed==[(FakeProcess.pid,signal.SIGTERM)],(spawned,killed)
        checks['runtime_abort_signals_only_own_fake_child_group'] = True
    assert 'jax' not in sys.modules and 'torch' not in sys.modules
    return dict(status='passed',scope='Stdlib synthetic status/telemetry/environment and fake allocator/Popen tests; no real workflow, GPU, model, or optimizer process',
                checks=checks,gpu_used=False,actual_JAX_imported=False,optimizer_steps=0,
                source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('resource_guard.py','run_workflow.py')})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run_tests();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
