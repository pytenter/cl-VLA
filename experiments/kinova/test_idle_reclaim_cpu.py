"""CPU mocks only: never sync the host or write a real kernel control file."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import resource_guard as guard


def run_tests():
    checks = {}
    policy = {'mode': 'exclusive_gpu', 'user_authorized': True, 'jax_memory_fraction': .85,
              'idle_clean_pagecache_reclaim': True}
    environment = {'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '.85',
                   'XLA_PYTHON_CLIENT_ALLOCATOR': 'bfc'}
    initial = {'windows_memory': {'commit_free_bytes': 14*2**30},
               'linux_memory': {'cached_bytes': 9*2**30, 'anon_bytes': 2**30,
                                'dirty_bytes': 0, 'writeback_bytes': 0, 'available_bytes': 20*2**30},
               'existing_cuda_processes': [], 'libero_complete': True,
               'libero_experiments': {'first': {'status': 'complete'}, 'second': {'status': 'complete'}}}
    with tempfile.TemporaryDirectory(prefix='kinova_idle_cache_reclaim_cpu_') as temporary:
        root = Path(temporary)
        current = [copy.deepcopy(initial)]
        active_policy = [copy.deepcopy(policy)]
        dropped = [False]
        def snapshot(monitor):
            state = copy.deepcopy(current[0])
            if dropped[0]:
                state['linux_memory']['cached_bytes'] = 58*2**20
                state['windows_memory']['commit_free_bytes'] = 31*2**30
            state['windows_memory']['received_monotonic'] = time.monotonic()
            return state
        def drop(): dropped[0] = True
        monitor = SimpleNamespace(assert_healthy=lambda: None)
        with mock.patch.object(guard, 'ROOT', root), mock.patch.object(guard, 'load_policy', side_effect=lambda:active_policy[0]), \
             mock.patch.object(guard, '_idle_reclaim_snapshot', side_effect=snapshot), \
             mock.patch.object(guard.os, 'sync') as sync, mock.patch.object(guard, '_drop_clean_pagecache', side_effect=drop) as write, \
             mock.patch.dict(os.environ, environment, clear=True):
            reclaim = guard.IdlePagecacheReclaimer('mock_wait_phase')
            result = reclaim.maybe_reclaim(monitor, gate={'eligible':False}, own_child=None)
            assert result['performed'] is True and result['attempted'] is True
            assert sync.call_count == write.call_count == 1
            assert result['before']['linux_memory']['cached_bytes'] == 9*2**30
            assert result['after']['linux_memory']['cached_bytes'] == 58*2**20
            assert result['after']['windows_memory']['commit_free_bytes'] == 31*2**30
            reclaim.maybe_reclaim(monitor, gate={'eligible':False}, own_child=None)
            assert sync.call_count == write.call_count == 1
            entries=[json.loads(line) for line in (root/'reports/idle_reclaim.jsonl').read_text().splitlines()]
            assert [row['event'] for row in entries] == ['started','completed']
            assert len({row['operation_id'] for row in entries}) == 1
            checks['all_conditions_sync_then_drop_once_with_before_after_evidence'] = True
            checks['same_wait_phase_cannot_reclaim_twice'] = True
            cases = ['no_flag','not_authorized','shared','disabled_branch','own_child','eligible','cuda',
                     'libero_incomplete','win_not_below_16','cache_not_above_8','anon_not_below_2','wrong_env']
            for case in cases:
                active_policy[0]=copy.deepcopy(policy);current[0]=copy.deepcopy(initial);dropped[0]=False
                arguments={'gate':{'eligible':False},'own_child':None}
                if case=='no_flag':active_policy[0].pop('idle_clean_pagecache_reclaim')
                if case=='not_authorized':active_policy[0]['user_authorized']=False
                if case=='shared':active_policy[0]['mode']='shared_gpu'
                if case=='disabled_branch':active_policy[0]['gpu_branch_disabled_after_failed_retry']=True
                if case=='own_child':arguments['own_child']=object()
                if case=='eligible':arguments['gate']={'eligible':True}
                if case=='cuda':current[0]['existing_cuda_processes']=['100, other_process']
                if case=='libero_incomplete':current[0]['libero_complete']=False
                if case=='win_not_below_16':current[0]['windows_memory']['commit_free_bytes']=16*2**30
                if case=='cache_not_above_8':current[0]['linux_memory']['cached_bytes']=8*2**30
                if case=='anon_not_below_2':current[0]['linux_memory']['anon_bytes']=2*2**30
                before=(sync.call_count,write.call_count)
                with mock.patch.dict(os.environ, {'XLA_PYTHON_CLIENT_MEM_FRACTION':'.4'} if case=='wrong_env' else {}):
                    outcome=guard.IdlePagecacheReclaimer(case).maybe_reclaim(monitor,**arguments)
                assert outcome['attempted'] is False and outcome['performed'] is False,case
                assert (sync.call_count,write.call_count)==before,case
            checks['every_missing_condition_prevents_sync_and_kernel_write'] = True
            active_policy[0]=copy.deepcopy(policy);current[0]=copy.deepcopy(initial);dropped[0]=False
            def after_sync_race():current[0]['existing_cuda_processes']=['200, arrived_during_sync']
            sync.side_effect=after_sync_race
            before=write.call_count
            recheck=guard.IdlePagecacheReclaimer('race_during_sync')
            outcome=recheck.maybe_reclaim(monitor,gate={'eligible':False},own_child=None)
            assert outcome['attempted'] is True and outcome['performed'] is False and write.call_count==before
            count=sync.call_count
            recheck.maybe_reclaim(monitor,gate={'eligible':False},own_child=None)
            assert sync.call_count==count
            checks['sync_followed_by_fresh_condition_recheck_and_no_second_attempt'] = True
            sync.side_effect=None;current[0]=copy.deepcopy(initial)
            write.side_effect=PermissionError('synthetic kernel write denied')
            failed=guard.IdlePagecacheReclaimer('failed_write')
            outcome=failed.maybe_reclaim(monitor,gate={'eligible':False},own_child=None)
            assert outcome['attempted'] is True and outcome['performed'] is False
            count=write.call_count;failed.maybe_reclaim(monitor,gate={'eligible':False},own_child=None)
            assert write.call_count==count
            checks['failed_reclaim_keeps_admission_floor_and_does_not_retry'] = True
            before=(sync.call_count,write.call_count)
            with mock.patch.object(guard,'_idle_reclaim_snapshot',side_effect=RuntimeError('synthetic unknown telemetry')):
                outcome=guard.IdlePagecacheReclaimer('unknown').maybe_reclaim(monitor,gate={'eligible':False},own_child=None)
            assert outcome['attempted'] is False and (sync.call_count,write.call_count)==before
            checks['unknown_or_stale_observation_fails_closed'] = True
        # Run the actual write helper only against an in-memory file object.
        output=mock.mock_open()
        assert str(guard.DROP_CACHES_PATH)=='/proc/sys/vm/drop_caches'
        with mock.patch.object(guard,'DROP_CACHES_PATH',SimpleNamespace(open=output)):
            guard._drop_clean_pagecache()
        output.assert_called_once_with('w');output().write.assert_called_once_with('1\n')
        checks['writer_emits_only_pagecache_bit1_never_slab_bits2_or3'] = True
    assert 'jax' not in sys.modules and 'torch' not in sys.modules
    return dict(status='passed',scope='CPU mocks only: no actual os.sync, kernel drop_caches write, GPU allocation, model or workflow launch',
                checks=checks,actual_cache_reclaims=0,gpu_used=False,optimizer_steps=0,
                source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in ('resource_guard.py','run_workflow.py')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run_tests();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
