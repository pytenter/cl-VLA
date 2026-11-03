"""CPU-first native Orbax restore with target dtypes and one array in flight.

All array reads/casts use Orbax's installed NumpyHandler. The local wrapper
only schedules its calls and supplies a bounded TensorStore context/spec.
The complete target CPU tree is retained; this is not a total RSS bound.
"""
from __future__ import annotations

import asyncio
import ctypes
import dataclasses
import gc
import json
import math
import os
from pathlib import Path
import time

from etils import epath
from flax import traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint._src.serialization import type_handlers
import tensorstore as ts


RESTORE_CONCURRENT_GB = 6
CACHE_BYTES = 16 * 2**20
CONTEXT_CONFIG = {'cache_pool': {'total_bytes_limit': CACHE_BYTES},
                  'file_io_concurrency': {'limit': 1},
                  'data_copy_concurrency': {'limit': 1}}
LAST_RESTORE_DIAGNOSTICS = None


def _release():
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


def _dtype(value):
    return np.dtype(value.dtype if not isinstance(value, type) and hasattr(value, 'dtype') else value)


def _progress_path(path):
    if path is not None:
        return Path(path)
    root = os.environ.get('KINOVA_EXPERIMENT_ROOT')
    return Path(root)/'logs'/f'native_restore_{os.getpid()}.jsonl' if root else None


def _emit(progress_file, phase, **fields):
    record = {'phase': phase, 'time_unix': time.time(), 'pid': os.getpid(), **fields}
    if progress_file is not None:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with progress_file.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps({'bounded_native_restore': record}, allow_nan=False), flush=True)


class _SequentialNumpyHandler(type_handlers.NumpyHandler):
    """Delegate native reads one leaf at a time, including across batches.

Pinned NumpyHandler ignores ParamInfo.byte_limiter. Merely setting the
PyTree handler's restore_concurrent_gb therefore does not bound NumPy reads.
"""

    def __init__(self, metadata_by_name, progress_path):
        super().__init__()
        self.context = ts.Context(CONTEXT_CONFIG)
        self.metadata_by_name = metadata_by_name
        self.progress_path = progress_path
        self.lock = asyncio.Lock()
        self.active = self.max_active = self.completed = self.completed_bytes = 0
        self.native_batch_sizes = []
        self.events = []
        self.read_specs = []

    def _get_json_tspec_read(self, info, use_ocdbt):
        spec = super()._get_json_tspec_read(info, use_ocdbt)
        if use_ocdbt:
            spec['kvstore']['cache_pool'] = 'cache_pool'
            spec['kvstore']['experimental_read_coalescing_threshold_bytes'] = 0
            spec['kvstore']['experimental_read_coalescing_merged_bytes'] = CACHE_BYTES
            spec['kvstore']['experimental_read_coalescing_interval'] = '0ms'
        self.read_specs.append({'name': info.name, 'use_ocdbt': use_ocdbt,
                                'kvstore_controls': {key: value for key, value in spec.get('kvstore', {}).items()
                                     if key in ('cache_pool', 'experimental_read_coalescing_threshold_bytes',
                                                'experimental_read_coalescing_merged_bytes', 'experimental_read_coalescing_interval')}})
        return spec

    async def deserialize(self, infos, args=None):
        args = args or [ocp.RestoreArgs()] * len(infos)
        if len(infos) != len(args):
            raise ValueError('Mismatched restore info/argument lengths')
        values = [None] * len(infos)
        # Large leaves are decoded before the retained target tree grows.
        order = sorted(range(len(infos)), key=lambda i: -self.metadata_by_name.get(infos[i].name, {}).get('target_bytes', 0))
        for index in order:
            info, arg = infos[index], args[index]
            async with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                meta = self.metadata_by_name.get(info.name, {})
                _emit(self.progress_path, 'native.leaf.before', name=info.name,
                      dtype=str(arg.dtype), **meta)
                try:
                    bounded_info = dataclasses.replace(info, ts_context=self.context)
                    self.native_batch_sizes.append(1)
                    # This is the sole array-data read: installed native handler.
                    restored = await super().deserialize([bounded_info], [arg])
                    value = restored[0]
                    if not isinstance(value, np.ndarray):
                        raise TypeError(f'Native NumPy handler returned {type(value)}')
                    if arg.dtype is not None and value.dtype != np.dtype(arg.dtype):
                        raise ValueError(f'Target dtype not applied at read: {info.name}')
                    values[index] = value
                    self.completed += 1
                    self.completed_bytes += value.nbytes
                    event = {'name': info.name, 'dtype': str(value.dtype), 'shape': list(value.shape),
                             'target_bytes': value.nbytes, 'completed_leaves': self.completed,
                             'retained_target_bytes': self.completed_bytes}
                    self.events.append(event)
                    del restored, value, bounded_info
                    _release()
                    _emit(self.progress_path, 'native.leaf.after', **event)
                finally:
                    self.active -= 1
        return values


def _new_checkpointer(metadata_by_name, progress_path):
    reader = _SequentialNumpyHandler(metadata_by_name, progress_path)
    registry = type_handlers.create_type_handler_registry(
        (np.ndarray, reader), (jax.Array, type_handlers.get_type_handler(jax.Array)))
    handler = ocp.PyTreeCheckpointHandler(restore_concurrent_gb=RESTORE_CONCURRENT_GB,
                                         type_handler_registry=registry)
    # PyTreeCheckpointer(handler) is INVALID: its positional arg is primary_host.
    return ocp.Checkpointer(handler), reader, handler


def restore_native_params(params_path, *, restore_type=np.ndarray, dtype=None,
                          sharding=None, expected_dtypes=None, progress_path=None):
    """Restore pure CPU target-dtype params, optionally transfer the result once.

The return tree occupies its complete target size on CPU. Only one native
leaf read is in flight, with16MiB cache/coalescing and IO/copy concurrency1.
Source storage chunks, codec buffers and total process memory are not bounded
by16MiB or by the6GB handler configuration. JAX output is CPU-first and may
temporarily coexist with the host tree; training requests NumPy output.
"""
    global LAST_RESTORE_DIAGNOSTICS
    if restore_type not in (np.ndarray, jax.Array):
        raise TypeError('Expected numpy.ndarray or jax.Array')
    directory = epath.Path(params_path)
    if not str(params_path).startswith('gs://'):
        directory = directory.resolve()
    progress = _progress_path(progress_path)
    _emit(progress, 'native.restore.before', checkpoint=str(directory), restore_type=restore_type.__name__)
    with ocp.PyTreeCheckpointer() as metadata_reader:
        metadata = metadata_reader.metadata(directory)
    flat = traverse_util.flatten_dict(metadata['params'])
    if not flat:
        raise ValueError('Checkpoint has no parameter arrays')
    strip_value = all(path[-1] == 'value' for path in flat)
    if expected_dtypes is not None:
        if hasattr(expected_dtypes, 'to_pure_dict'):
            expected_dtypes = expected_dtypes.to_pure_dict()
        if isinstance(expected_dtypes, dict) and set(expected_dtypes) == {'params'}:
            expected_dtypes = expected_dtypes['params']
        expected = traverse_util.flatten_dict(expected_dtypes)
    else:
        expected = None
    flat_args, metadata_by_name = {}, {}
    for path, info in flat.items():
        pure = path[:-1] if strip_value else path
        target_dtype = _dtype(dtype) if dtype is not None else _dtype(info.dtype)
        if expected is not None:
            if pure not in expected:
                raise ValueError(f'Checkpoint leaf absent from expected tree: {pure}')
            reference = expected[pure]
            if hasattr(reference, 'shape') and tuple(reference.shape) != tuple(info.shape):
                raise ValueError(f'Checkpoint shape mismatch: {pure}')
            target_dtype = _dtype(reference)
        flat_args[path] = ocp.RestoreArgs(restore_type=np.ndarray, dtype=target_dtype)
        metadata_by_name[info.name] = {'path': '/'.join(map(str, pure)), 'shape': list(info.shape),
            'source_dtype': str(info.dtype), 'target_dtype': str(target_dtype),
            'source_bytes': math.prod(info.shape) * _dtype(info.dtype).itemsize,
            'target_bytes': math.prod(info.shape) * target_dtype.itemsize}
    item = {'params': metadata['params']}
    restore_args = {'params': traverse_util.unflatten_dict(flat_args)}
    checkpointer, reader, handler = _new_checkpointer(metadata_by_name, progress)
    _emit(progress, 'native.restore.plan', leaf_count=len(flat),
          target_tree_bytes=sum(row['target_bytes'] for row in metadata_by_name.values()),
          restore_concurrent_bytes=handler._restore_concurrent_bytes,
          numpy_schedule='one_native_leaf_in_flight', context_config=CONTEXT_CONFIG)
    with checkpointer:
        loaded = checkpointer.restore(directory, args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args))['params']
    loaded = traverse_util.flatten_dict(loaded)
    if strip_value:
        loaded = {path[:-1]: value for path, value in loaded.items()}
    loaded = traverse_util.unflatten_dict(loaded)
    LAST_RESTORE_DIAGNOSTICS = {'handler_restore_concurrent_bytes': handler._restore_concurrent_bytes,
        'handler_impl_restore_concurrent_bytes': handler._handler_impl._restore_concurrent_bytes,
        'max_in_flight_native_reads': reader.max_active,
        'native_read_batch_sizes': reader.native_batch_sizes, 'events': reader.events,
        'read_specs': reader.read_specs, 'context_config': CONTEXT_CONFIG,
        'target_tree_bytes': reader.completed_bytes, 'native_numpy_uses_byte_limiter': False}
    _release()
    _emit(progress, 'native.restore.after', completed_leaves=reader.completed,
          target_tree_bytes=reader.completed_bytes, cpu_arrays=True)
    # The helper requires explicit production policy and owns OS-level checks.
    # Tiny CPU tests replace this helper; they never reclaim global caches.
    from memory_maintenance import reclaim_before_device_transfer
    _emit(progress, 'native.maintenance.before', target_tree_bytes=reader.completed_bytes)
    maintenance = reclaim_before_device_transfer()
    _emit(progress, 'native.maintenance.after', target_tree_bytes=reader.completed_bytes,
          maintenance=maintenance)
    if restore_type is jax.Array:
        if sharding is None:
            sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ('x',)), jax.sharding.PartitionSpec())
        _emit(progress, 'native.device_transfer.before', target_tree_bytes=reader.completed_bytes)
        result = jax.device_put(loaded, device=sharding)
        jax.block_until_ready(result)
        del loaded
        _release()
        _emit(progress, 'native.device_transfer.after', target_tree_bytes=reader.completed_bytes)
        return result
    return loaded


@dataclasses.dataclass(frozen=True)
class CompleteWeightLoader:
    params_path: str

    def load(self, params):
        from openpi.training import weight_loaders
        progress = _progress_path(None)
        loaded = restore_native_params(self.params_path, restore_type=np.ndarray,
                                       expected_dtypes=params, progress_path=progress)
        _emit(progress, 'native.merge.before')
        merged = weight_loaders._merge_params(loaded, params, missing_regex=r'.*(?:lora|cl_block_weights).*')
        del loaded
        _release()
        _emit(progress, 'native.merge.after', target_tree_bytes=sum(value.nbytes for value in
              traverse_util.flatten_dict(merged).values() if isinstance(value, np.ndarray)))
        return merged
