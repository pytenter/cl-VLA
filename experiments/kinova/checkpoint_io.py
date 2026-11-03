"""Native Orbax schema with one array snapshot at a time and read-only restore.

Pinned Orbax 0.11.1 snapshots every JAX array before its save byte limiter is
used. This private handler changes scheduling only: it delegates transfer and
OCDBT/sharding serialization to the installed native implementations.
"""
from __future__ import annotations

import copy
import ctypes
import dataclasses
import gc
import json
import math
import os
from pathlib import Path
import threading
import time

from etils import epath
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.serialization import replica_slices, type_handlers
import tensorstore as ts

from openpi.training.checkpoints import CallbackHandler


IO_CONCURRENT_GB = 3
CACHE_BYTES_PER_POOL = 16 * 2**20
CONTEXT_CONFIG = {
    'cache_pool': {'total_bytes_limit': CACHE_BYTES_PER_POOL},
    'cache_pool#ocdbt': {'total_bytes_limit': CACHE_BYTES_PER_POOL},
    'file_io_concurrency': {'limit': 1},
    'data_copy_concurrency': {'limit': 1},
}
_SAVE_LOCK = threading.Lock()


def _release():
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


def _log(phase, **fields):
    root = os.environ.get('KINOVA_EXPERIMENT_ROOT')
    if root:
        path = Path(root)/'logs/checkpoint_serial_io.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps({'phase': phase, 'time_unix': time.time(),
                                     'pid': os.getpid(), **fields}, allow_nan=False)+'\n')


def _checkpoint_cache_maintenance(phase):
    """Optional owned-job clean-pagecache maintenance; no persistent setting.

    New resume policies explicitly opt in. CPU fixtures have no root or flag.
    Unlike loading, a pilot saves several times; this is once per save boundary.
    """
    configured = os.environ.get('KINOVA_EXPERIMENT_ROOT')
    if not configured:
        return
    root = Path(configured)
    policy = json.loads((root/'configs/resource_policy.json').read_text())
    if policy.get('checkpoint_clean_cache_reclaim') is not True:
        return
    if phase == 'between_arrays' and policy.get('checkpoint_reclaim_during_serialization') is not True:
        return
    import memory_maintenance as maintenance
    policy, job, _ = maintenance._context(root)
    fingerprint = maintenance._fingerprint(policy)
    # This is a measured-probe admission threshold, not a process peak bound:
    # 2.25GiB largest host snapshot + possible2.25GiB native write staging +
    # 1.5GiB unchanged runtime reserve. The supervisor still enforces runtime
    # Windows/Linux/VRAM floors throughout the actual save.
    floor_key = ('checkpoint_inter_array_min_windows_commit_free_gib'
                 if phase == 'between_arrays' else 'checkpoint_min_windows_commit_free_gib')
    floor = max(6*2**30, math.ceil(float(policy.get(floor_key, 6))*2**30))
    exit_after_commit = phase == 'after' and policy.get('checkpoint_process_exit_after_save') is True
    wait_seconds = float(policy.get('checkpoint_cache_reclaim_wait_seconds', 60))
    if not math.isfinite(wait_seconds) or not 0 < wait_seconds <= 120:
        raise ValueError('Invalid checkpoint cache-maintenance timeout')
    before = maintenance._capture(root, 5)
    _log('cache.'+phase+'.before', snapshot=before, job=job)
    os.sync()
    maintenance._context(root, policy_sha=fingerprint, job=job)
    maintenance._drop_clean_pagecache()
    dropped_at_unix, dropped_monotonic = time.time(), time.monotonic()
    _release()
    deadline = time.monotonic()+wait_seconds
    last = None
    while True:
        maintenance._context(root, policy_sha=fingerprint, job=job)
        last = maintenance._capture(root, 5)
        sample = last['windows']
        post_drop = (sample['sampled_at_unix'] > dropped_at_unix
                     and sample['received_monotonic'] > dropped_monotonic)
        if post_drop and (exit_after_commit or sample['commit_free_bytes'] >= floor):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Checkpoint boundary deferred: clean-cache reclaim did not establish required Windows headroom')
        time.sleep(.5)
    _log('cache.'+phase+'.after', snapshot=last, job=job,
         drop_caches_value=1, dropped_at_unix=dropped_at_unix,
         required_windows_commit_free_bytes=None if exit_after_commit else floor,
         process_exit_after_commit=exit_after_commit,
         persistent_settings_changed=False)


class SerialArrayHandler(type_handlers.ArrayHandler):
    """Native per-leaf transfer/write under a lock shared by both items.

    Futures retain device arrays, not a full host tree. The manager is strictly
    synchronous, so device buffers cannot be donated before serialization ends.
    The optional ArrayMetadataStore defaults to None in pinned Orbax too.
    """

    def __init__(self):
        super().__init__(array_metadata_store=None)
        self.context = ts.Context(CONTEXT_CONFIG)
        self.native_transfer_batch_sizes = []
        self.native_write_batch_sizes = []
        self.max_host_snapshot_bytes = 0
        self.active_snapshots = self.max_active_snapshots = 0
        self.completed_arrays = 0
        self.fallback_device_copy_count = 0
        self.forced_device_copy_count = 0
        self.disjoint_device_copy_count = 0
        self.bytes_since_cache_reclaim = 0
        self.inter_array_reclaims = 0
        self.reclaim_written_bytes = 512 * 1024 * 1024

    async def serialize(self, values, infos, args=None):
        args = args or [ocp.SaveArgs()] * len(values)
        if not (len(values) == len(infos) == len(args)):
            raise ValueError('Checkpoint array/info/save-argument lengths differ')
        # Native contracted signals preserve directory creation ordering. All
        # data transfer occurs in the future, after the creation signal arrives.
        return [future.CommitFutureAwaitingContractedSignals(
            self._serialize_sequence(values, infos, args),
            name='serial_native_array_handler')]

    async def _serialize_sequence(self, values, infos, args):
        order = sorted(range(len(values)), key=lambda i: -values[i].nbytes)
        for index in order:
            value, info, arg = values[index], infos[index], args[index]
            # Each commit future has its own native worker event loop. A thread
            # lock (not an asyncio lock) safely serializes params/train_state.
            with _SAVE_LOCK:
                self.active_snapshots += 1
                self.max_active_snapshots = max(self.max_active_snapshots, self.active_snapshots)
                scratch = host = bounded_info = None
                try:
                    pinned = all(replica_slices._use_pinned_host_transfer(s.device)
                                 for s in value.addressable_shards)
                    # Always isolate the native D2H source. Even the pinned path
                    # must not attach a host cache to an original model shard.
                    # One process per checkpoint additionally releases PJRT's
                    # pinned-host allocator pool; this copy is not proof that
                    # the allocator itself immediately returns host memory.
                    scratch = value.copy()
                    jax.block_until_ready(scratch)
                    source = scratch
                    self.forced_device_copy_count += 1
                    if value.size and len(value.addressable_shards) == 1:
                        original_pointer = value.addressable_shards[0].data.unsafe_buffer_pointer()
                        scratch_pointer = scratch.addressable_shards[0].data.unsafe_buffer_pointer()
                        if original_pointer == scratch_pointer:
                            raise RuntimeError('Temporary checkpoint device copy aliases the original model shard')
                        self.disjoint_device_copy_count += 1
                    if not pinned:
                        self.fallback_device_copy_count += 1
                    bounded_info = dataclasses.replace(info, ts_context=self.context,
                                                       enable_pinned_host_transfer=False)
                    _log('array.before', name=info.name, bytes=value.nbytes,
                         pinned_host_capable=pinned, pinned_host_transfer=False,
                         temporary_device_copy=True)
                    self.native_transfer_batch_sizes.append(1)
                    host = replica_slices.transfer_arrays_to_host(
                        [source], self._replica_id, self._use_replica_parallel,
                        enable_pinned_host_transfer=False)
                    self.max_host_snapshot_bytes = max(self.max_host_snapshot_bytes, value.nbytes)
                    self.native_write_batch_sizes.append(1)
                    # Sole array-data writer: pinned native Orbax implementation.
                    # Its configured byte limiter commits this leaf before the
                    # host snapshot and optional scratch buffer can be released.
                    await super()._background_serialize(host, [bounded_info], [arg])
                    self.completed_arrays += 1
                    self.bytes_since_cache_reclaim += value.nbytes
                    _log('array.after', name=info.name, bytes=value.nbytes,
                         completed_arrays=self.completed_arrays)
                finally:
                    source = scratch = host = bounded_info = None
                    self.active_snapshots -= 1
                    _release()
                # Native writes still populate the Linux filesystem cache.
                # Reclaim only after this leaf is committed and its temporary
                # host/device references are gone, while retaining the shared
                # lock so another handler cannot snapshot during maintenance.
                if self.bytes_since_cache_reclaim >= self.reclaim_written_bytes:
                    _checkpoint_cache_maintenance('between_arrays')
                    self.bytes_since_cache_reclaim = 0
                    self.inter_array_reclaims += 1


def _private_registry(array_handler):
    # Shallow-copy handlers, but give the registry independent containers. This
    # retains native scalar/string/NumPy support without global registration.
    original = type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY
    registry = copy.copy(original)
    registry._type_registry = list(original._type_registry)
    registry._typestr_registry = dict(original._typestr_registry)
    registry.add(jax.Array, array_handler, override=True, ignore_warnings=True)
    return registry


class SerialCheckpointManager(ocp.CheckpointManager):
    def save(self, *args, **kwargs):
        self.wait_until_finished()
        _checkpoint_cache_maintenance('before')
        result = super().save(*args, **kwargs)
        self.wait_until_finished()
        _checkpoint_cache_maintenance('after')
        return result


def _handlers(registry=None):
    kwargs = {'save_concurrent_gb': IO_CONCURRENT_GB,
              'restore_concurrent_gb': IO_CONCURRENT_GB}
    if registry is not None:
        kwargs['type_handler_registry'] = registry
    return {'assets': CallbackHandler(),
            'train_state': ocp.PyTreeCheckpointHandler(**kwargs),
            'params': ocp.PyTreeCheckpointHandler(**kwargs)}


def initialize_checkpoint_dir(path, *, keep_period, overwrite=False, resume=False):
    if overwrite:
        raise ValueError('This independent experiment never overwrites checkpoints')
    path = epath.Path(path).resolve()
    existed = path.exists()
    if existed and not resume:
        raise FileExistsError(f'Checkpoint directory already exists: {path}')
    path.mkdir(parents=True, exist_ok=True)
    array_handler = SerialArrayHandler()
    registry = _private_registry(array_handler)
    manager = SerialCheckpointManager(path, item_handlers=_handlers(registry),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, keep_period=keep_period, create=False,
            enable_async_checkpointing=False, cleanup_tmp_directories=False))
    manager.serial_array_handler = array_handler
    resuming = existed and resume and tuple(manager.all_steps()) not in [(), (0,)]
    return manager, resuming


def initialize_readonly_checkpoint_dir(path):
    """Open an existing native manager without mkdir, cleanup, save or retention."""
    path = epath.Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return ocp.CheckpointManager(path, item_handlers=_handlers(),
        options=ocp.CheckpointManagerOptions(
            read_only=True, create=False, cleanup_tmp_directories=False,
            enable_async_checkpointing=False, save_root_metadata=False,
            save_interval_steps=0, max_to_keep=None))
