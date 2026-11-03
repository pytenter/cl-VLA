"""Tiny actual native roundtrip; no GPU, original checkpoints, or cache drop."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
from openpi.shared import array_typing as at
from openpi.training import checkpoints, utils
from openpi.models import model as model_lib
import checkpoint_io as io


def tree_hashes(tree):
    return {jax.tree_util.keystr(path): (list(value.shape), str(value.dtype),
            hashlib.sha256(np.ascontiguousarray(np.asarray(value)).view(np.uint8)).hexdigest())
            for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]}


def files(path):
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest()) for p in path.rglob('*') if p.is_file()}


class Loader:
    def data_config(self):
        from types import SimpleNamespace
        return SimpleNamespace(norm_stats=None, asset_id=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert jax.default_backend() == 'cpu'
    # No maintenance hook is enabled by a tiny CPU fixture.
    os.environ.pop('KINOVA_EXPERIMENT_ROOT', None)
    checks = {}
    original_global_handler = io.type_handlers.get_type_handler(jax.Array)
    with tempfile.TemporaryDirectory(prefix='kinova-native-save-cpu-') as temporary:
        root = Path(temporary)
        model = nnx.Linear(3, 4, rngs=nnx.Rngs(42))
        params = nnx.state(model)
        params['kernel'] = params['kernel'].replace(params['kernel'].value.astype(jnp.bfloat16))
        tx = optax.adam(.001)
        with at.disable_typechecking():
            state = utils.TrainState(step=jnp.asarray(32, jnp.int32), params=params,
                model_def=nnx.graphdef(model), opt_state=tx.init(params), tx=tx, ema_decay=None)
        expected = tree_hashes(state)
        manager, resumed = io.initialize_checkpoint_dir(root/'checkpoints', keep_period=500)
        checks['new_manager_not_resumed'] = not resumed
        checks['native_global_array_handler_unchanged'] = io.type_handlers.get_type_handler(jax.Array) is original_global_handler
        checks['synchronous_manager'] = manager._options.enable_async_checkpointing is False
        checkpoints.save_state(manager, state, Loader(), 32)
        manager.wait_until_finished()
        reader = manager.serial_array_handler
        checkpoint = root/'checkpoints/32'
        checks['native_committed_marker'] = json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())['commit_timestamp_nsecs'] is not None
        checks['native_three_items'] = all((checkpoint/name).is_dir() for name in ['params','train_state','assets'])
        checks['one_native_transfer_per_batch'] = reader.native_transfer_batch_sizes and set(reader.native_transfer_batch_sizes) == {1}
        checks['one_native_write_per_batch'] = reader.native_write_batch_sizes and set(reader.native_write_batch_sizes) == {1}
        checks['one_array_snapshot_in_flight_across_items'] = reader.max_active_snapshots == 1
        checks['host_snapshot_bound_largest_existing_leaf'] = reader.max_host_snapshot_bytes == max(x.nbytes for x in jax.tree.leaves(state))
        checks['temporary_device_copy_when_no_pinned_memory'] = reader.fallback_device_copy_count > 0
        checks['every_array_uses_disjoint_temporary_device_copy'] = (
            reader.forced_device_copy_count == reader.disjoint_device_copy_count == reader.completed_arrays)
        restored = checkpoints.restore_state(manager, state, None, step=32)
        jax.block_until_ready(restored)
        checks['native_complete_state_roundtrip_exact'] = tree_hashes(restored) == expected
        params_only = model_lib.restore_params(checkpoint/'params', restore_type=np.ndarray)
        checks['native_inference_restore_value_suffix_supported'] = (
            set(params_only) == {'bias', 'kernel'} and np.array_equal(params_only['kernel'], np.asarray(state.params['kernel'].value)))
        diagnostics = {'completed_arrays': reader.completed_arrays,
            'max_active_snapshots': reader.max_active_snapshots,
            'max_host_snapshot_bytes': reader.max_host_snapshot_bytes,
            'native_transfer_batch_sizes': reader.native_transfer_batch_sizes,
            'native_write_batch_sizes': reader.native_write_batch_sizes,
            'fallback_device_copy_count': reader.fallback_device_copy_count}
        diagnostics['forced_device_copy_count'] = reader.forced_device_copy_count
        diagnostics['disjoint_device_copy_count'] = reader.disjoint_device_copy_count
        manager.close()
        pinned_manager, _ = io.initialize_checkpoint_dir(root/'pinned_fixture', keep_period=500)
        native_transfer = io.replica_slices.transfer_arrays_to_host
        requested_pinned = []
        def cpu_transfer(values, replica_id, use_replica_parallel, **kwargs):
            requested_pinned.append(kwargs.get('enable_pinned_host_transfer'))
            return native_transfer(values, replica_id, use_replica_parallel,
                                   enable_pinned_host_transfer=False)
        # Simulate the handler's pinned-capability decision, while retaining an
        # actual native CPU D2H/write. This does not test CUDA pinned allocation.
        with patch.object(io.replica_slices, '_use_pinned_host_transfer', return_value=True), \
             patch.object(io.replica_slices, 'transfer_arrays_to_host', side_effect=cpu_transfer):
            checkpoints.save_state(pinned_manager, state, Loader(), 32)
        pinned_reader = pinned_manager.serial_array_handler
        checks['pinned_capability_path_also_uses_disjoint_temporary_copy'] = (
            pinned_reader.completed_arrays == pinned_reader.forced_device_copy_count
            == pinned_reader.disjoint_device_copy_count > 0 and pinned_reader.fallback_device_copy_count == 0)
        pinned_restored = checkpoints.restore_state(pinned_manager, state, None, step=32)
        jax.block_until_ready(pinned_restored)
        checks['simulated_pinned_path_native_state_roundtrip_exact'] = tree_hashes(pinned_restored) == expected
        checks['native_default_transfer_requested_even_when_pinned_capable'] = (
            bool(requested_pinned) and all(value is False for value in requested_pinned))
        pinned_manager.close()
        cadence, _ = io.initialize_checkpoint_dir(root/'cadence_fixture', keep_period=500)
        cadence_reader = cadence.serial_array_handler
        cadence_reader.reclaim_written_bytes = 16
        maintenance_calls = []
        def record_maintenance(phase):
            maintenance_calls.append((phase, cadence_reader.active_snapshots,
                                      io._SAVE_LOCK.locked(), cadence_reader.bytes_since_cache_reclaim))
        with patch.object(io, '_checkpoint_cache_maintenance', side_effect=record_maintenance):
            checkpoints.save_state(cadence, state, Loader(), 32)
        inter = [row for row in maintenance_calls if row[0] == 'between_arrays']
        checks['cache_reclaimed_between_committed_leaves_at_byte_threshold'] = (
            len(inter) >= 2 and cadence_reader.inter_array_reclaims == len(inter)
            and all(row[3] >= 16 for row in inter))
        checks['inter_array_reclaim_holds_shared_lock_after_snapshot_release'] = (
            bool(inter) and all(row[1] == 0 and row[2] for row in inter))
        cadence_state = checkpoints.restore_state(cadence, state, None, step=32)
        jax.block_until_ready(cadence_state)
        checks['inter_array_reclaim_native_roundtrip_exact'] = tree_hashes(cadence_state) == expected
        cadence.close()
        before = files(root/'checkpoints')
        readonly = io.initialize_readonly_checkpoint_dir(root/'checkpoints')
        restored = checkpoints.restore_state(readonly, state, None, step=32)
        jax.block_until_ready(restored)
        checks['read_only_complete_state_roundtrip_exact'] = tree_hashes(restored) == expected
        checks['read_only_save_refused'] = readonly.save(33, {'params': {'params': state.params}, 'train_state': state}) is False
        readonly.close()
        checks['read_only_all_source_files_and_mtimes_unchanged'] = files(root/'checkpoints') == before
        try:
            io.initialize_readonly_checkpoint_dir(root/'not_present')
        except FileNotFoundError:
            checks['read_only_missing_path_not_created'] = not (root/'not_present').exists()
        else:
            checks['read_only_missing_path_not_created'] = False
        try:
            io.initialize_checkpoint_dir(root/'checkpoints', keep_period=500)
        except FileExistsError:
            checks['existing_checkpoint_not_overwritten'] = True
        else:
            checks['existing_checkpoint_not_overwritten'] = False
        try:
            io.initialize_checkpoint_dir(root/'overwrite', keep_period=500, overwrite=True)
        except ValueError:
            checks['overwrite_disabled'] = not (root/'overwrite').exists()
        else:
            checks['overwrite_disabled'] = False
        fixture = root/'maintenance_fixture'
        (fixture/'configs').mkdir(parents=True)
        policy = {'checkpoint_clean_cache_reclaim': True}
        (fixture/'configs/resource_policy.json').write_text(json.dumps(policy))
        calls = []
        def capture(*args):
            calls.append(1)
            # First before-sync snapshot and first post-drop read are old.
            fresh = len(calls) >= 3
            return {'windows': {'commit_free_bytes': 13*2**30,
                    'sampled_at_unix': time.time()+1 if fresh else 0,
                    'received_monotonic': time.monotonic()+1 if fresh else 0}}
        fake = SimpleNamespace(_context=lambda *a, **k: (policy, 'E1_wrist_pilot_train2000', {}),
            _fingerprint=lambda p: 'fixture', _capture=capture,
            _drop_clean_pagecache=Mock())
        with patch.dict('sys.modules', {'memory_maintenance': fake}), \
             patch.dict(os.environ, {'KINOVA_EXPERIMENT_ROOT': str(fixture)}), \
             patch.object(io.os, 'sync') as sync, patch.object(io.time, 'sleep') as sleep:
            io._checkpoint_cache_maintenance('before')
            checks['high_but_pre_drop_windows_sample_cannot_release_save'] = len(calls) == 3 and sleep.call_count == 1
            checks['maintenance_CPU_fixture_no_global_sync_or_drop'] = sync.call_count == fake._drop_clean_pagecache.call_count == 1
        for phase, floor, headroom, exit_after, expected_success, label in (
            ('before', 8, 8.5, False, True, 'before_save_phase_accepts_fresh_8_5GiB'),
            ('before', 6, 6.4, False, True, 'save_probe_phase_accepts_fresh_6_4GiB'),
            ('before', 1, 5.5, False, False, 'before_save_cannot_lower_absolute_6GiB_floor'),
            ('before', 8, 7.5, False, False, 'explicit_higher_8GiB_policy_floor_is_preserved'),
            ('after', 8, 3, True, True, 'post_commit_exit_policy_does_not_wait_for_new_model_headroom'),
            ('after', 8, 3, False, False, 'post_commit_non_exit_policy_still_requires_phase_floor'),
        ):
            policy.update(checkpoint_min_windows_commit_free_gib=floor,
                          checkpoint_process_exit_after_save=exit_after,
                          checkpoint_cache_reclaim_wait_seconds=.001)
            (fixture/'configs/resource_policy.json').write_text(json.dumps(policy))
            def fresh_capture(*args):
                return {'windows': {'commit_free_bytes': int(headroom*2**30),
                        'sampled_at_unix': time.time()+1,
                        'received_monotonic': time.monotonic()+1}}
            fake._capture = fresh_capture
            succeeded = True
            with patch.dict('sys.modules', {'memory_maintenance': fake}), \
                 patch.dict(os.environ, {'KINOVA_EXPERIMENT_ROOT': str(fixture)}), \
                 patch.object(io.os, 'sync'), patch.object(io.time, 'sleep'):
                try:
                    io._checkpoint_cache_maintenance(phase)
                except RuntimeError:
                    succeeded = False
            checks[label] = succeeded == expected_success
    result = {'schema': 'kinova-native-serial-checkpoint-cpu-v1',
        'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
        'check_count': len(checks), 'diagnostics': diagnostics,
        'scope': 'Tiny native OCDBT complete-state save/restore, native inference reader, read-only restore/no-write/source-file checks, instrumentation of per-leaf native scheduling',
        'gpu_used': False, 'full_model_loaded': False, 'global_cache_drop_executed': False,
        'original_training_checkpoint_opened': False, 'optimizer_updates_executed': 0,
        'source_module_sha256': hashlib.sha256(Path(io.__file__).read_bytes()).hexdigest()}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
