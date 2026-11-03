"""Resource-gated, wrist-only single-task π0.5 ordinary-LoRA training.

The real optimizer smoke is 32 updates. E1 starts independently from pi05_base,
uses microbatch1 x accumulation4, and is capped at 2000 optimizer updates.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from phase_memory import mark_phase


ROOT = Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', project_path('kinova', '')))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


class MemoryMonitor:
    """Sample whole-card usage; allocator peak is recorded separately by JAX."""
    def __init__(self, path):
        self.path, self.stop_event = Path(path), threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open('a') as stream:
            while not self.stop_event.is_set():
                try:
                    value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                                     '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
                    used, util = map(int, (x.strip() for x in value.split(',')))
                    row = {'time': time.time(), 'used_mib': used, 'gpu_utilization_percent': util}
                    self.samples.append(row)
                    stream.write(json.dumps(row) + '\n'); stream.flush()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    stream.write(json.dumps({'time': time.time(), 'monitor_error': str(error)}) + '\n'); stream.flush()
                self.stop_event.wait(.5)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=6)

    def peak(self, start=None, end=None):
        values = [row['used_mib'] for row in self.samples
                  if (start is None or row['time'] >= start) and (end is None or row['time'] <= end)]
        return max(values) if values else None


def digest_params(params):
    """Hash bounded leading-axis slices, avoiding full2.25GiB host leaves."""
    from array_hash import hash_array
    result = {}
    for path, variable in params.flat_state().items():
        result['/'.join(map(str, path))] = hash_array(variable.value)
    return result


def optimizer_counters(state):
    import jax
    import numpy as np
    counters = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        if getattr(leaf, 'ndim', None) == 0 and np.issubdtype(leaf.dtype, np.integer):
            counters.append({'path': jax.tree_util.keystr(path), 'value': int(leaf)})
    return counters


def verify_initial_parameter_identity(captured_reference, root=None):
    """Check all BF16 inference parameters before comparing fixed outputs."""
    root = Path(root or ROOT)
    summary = json.loads((root/'evaluations/E0_wrist_frozen/summary.json').read_text())
    metadata = summary.get('config_metadata', {})
    hashes = metadata.get('parameter_hashes', {})
    base, adapter = hashes.get('base', {}), hashes.get('adapter', {})
    expected = {**base, **adapter}
    captured = json.loads(Path(captured_reference['parameter_hashes_path']).read_text())
    actual = captured.get('hashes', {})
    missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
    mismatched = sorted(name for name in set(expected)&set(actual) if expected[name] != actual[name])
    checks = {
        'same_graph_zero_update_E0': metadata.get('baseline_kind') == 'E0_same_graph_zero_update',
        'E0_parameter_hashes_recorded': metadata.get('parameter_hashes_recorded') is True,
        'E0_base51_adapter20': len(base) == 51 and len(adapter) == 20 and not set(base)&set(adapter),
        'all71_parameter_keys_equal': len(expected) == len(actual) == 71 and not missing and not extra,
        'all71_parameter_hashes_equal': not missing and not extra and not mismatched and len(actual) == 71,
        'captured_BF16_view': captured.get('view') == 'BF16 inference parameters',
        'captured_step0': captured.get('optimizer_step') == 0,
    }
    report = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'matched_parameter_count': len(set(expected)&set(actual))-len(mismatched),
              'missing_keys': missing, 'extra_keys': extra, 'mismatched_keys': mismatched,
              'actual_hashes': captured_reference['parameter_hashes_path'],
              'E0_summary': str(root/'evaluations/E0_wrist_frozen/summary.json'),
              'scope': 'Exact SHA comparison of all71 BF16 inference parameter arrays before strict output comparison'}
    write_json(root/'reports/initial_parameter_identity.json', report)
    if report['status'] != 'passed':
        raise RuntimeError('Same-graph initial BF16 parameter identity check failed')
    return report


def optimizer_hashes(state):
    import jax
    from array_hash import hash_array
    result = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        result[jax.tree_util.keystr(path)] = hash_array(leaf)
    return result


def validate_microbatch(batch, episode_lengths):
    import numpy as np
    obs, actions = batch
    if tuple(actions.shape) != (1, 50, 32) or tuple(obs.state.shape) != (1, 32):
        raise ValueError(f'Unexpected actual batch shape: {actions.shape}/{obs.state.shape}')
    ep = np.asarray(obs.source_episode_index).reshape(-1)
    frame = np.asarray(obs.source_frame_index).reshape(-1)
    expected = np.stack([np.arange(50) < (episode_lengths[int(e)] - int(f)) for e, f in zip(ep, frame)])
    actual = np.asarray(obs.action_valid_mask)
    if not np.array_equal(expected, actual) or not actual.any():
        raise ValueError('Actual loader mask disagrees with source episode/frame boundaries')
    for camera, enabled in [('base_0_rgb', False), ('left_wrist_0_rgb', True), ('right_wrist_0_rgb', False)]:
        if tuple(obs.images[camera].shape) != (1, 224, 224, 3) or bool(np.asarray(obs.image_masks[camera])[0]) != enabled:
            raise ValueError(f'Wrong wrist-only image shape/mask for {camera}')
    return {'episode_index': ep.tolist(), 'frame_index': frame.tolist(),
            'valid_steps': actual.sum(axis=-1).tolist(), 'action_shape': list(actions.shape),
            'state_shape': list(obs.state.shape), 'token_shape': list(obs.tokenized_prompt.shape),
            'camera_masks': {k: np.asarray(v).tolist() for k, v in obs.image_masks.items()}}


def capture(state, config, prefix, *, step):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from kinova_config import build_dataset
    from evaluate_offline import prepare_sample
    from verify_checkpoint import capture_reference
    plan = json.loads((ROOT/'configs/validation_plan.json').read_text())
    block = plan['episodes'][0]['blocks'][0]
    obs, actions = prepare_sample(build_dataset(config, 'validation')[block['global_index']], block)
    # Match E0/evaluator/restorer BF16 parameter views, while preserving training
    # LoRA/head parameters and optimizer states in their original precision.
    inference_state = jax.tree.map(lambda value: value.astype(jnp.bfloat16), state.params)
    model = nnx.merge(state.model_def, inference_state)
    model.eval()
    parameter_hashes_path = Path(str(prefix)+'.parameter_hashes.json')
    write_json(parameter_hashes_path, {'view': 'BF16 inference parameters',
               'optimizer_step': step, 'hashes': digest_params(inference_state)})
    result = capture_reference(model, obs, actions, prefix, seed=314159,
                               checkpoint_step=step, restore_dtype='bfloat16',
                               provenance={'experiment': config.name, 'episode_id': plan['episodes'][0]['episode_id'],
                                           'start_frame': 0, 'plan_sha256': plan['plan_sha256']})
    return {**result, 'prefix': str(prefix), 'parameter_hashes_path': str(parameter_hashes_path)}


def full_state_verify(config, checkpoint_step, expected_path, output):
    import jax
    from scripts.train import init_train_state
    from openpi.training import checkpoints, sharding
    mesh = sharding.make_mesh(1)
    from checkpoint_io import initialize_checkpoint_dir
    manager, _ = initialize_checkpoint_dir(config.checkpoint_dir, keep_period=500, overwrite=False, resume=True)
    template, _ = init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
    state = checkpoints.restore_state(manager, template, None, step=checkpoint_step)
    jax.block_until_ready(state)
    expected = json.loads(Path(expected_path).read_text())
    hashes = digest_params(state.params)
    counters = optimizer_counters(state)
    checks = {'step_matches': int(state.step) == checkpoint_step,
              'all_parameter_hashes_match': hashes == expected['final_parameter_hashes'],
              'optimizer_counters_match': counters == expected['optimizer_counters'],
              'all_optimizer_state_hashes_match': optimizer_hashes(state) == expected['final_optimizer_state_hashes']}
    result = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'optimizer_step': int(state.step), 'optimizer_counters': counters, 'new_process': True,
              'optimizer_steps_executed_by_verifier': 0, 'robot_commands_sent': 0,
              'checkpoint': str(config.checkpoint_dir/str(checkpoint_step))}
    write_json(output, result)
    manager.wait_until_finished()
    if not all(checks.values()):
        raise RuntimeError('Full optimizer/parameter checkpoint restoration verification failed')
    print(json.dumps(result), flush=True)


def train(args, gate):
    mark_phase(ROOT, 'train.imports.before', mode=args.mode, verify_full_state=args.verify_full_state)
    import dataclasses
    import random
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from kinova_config import make_config, install_data_hooks, CANDIDATE_ROOT
    from kinova_model import adapter_initialization_summary
    from accumulated_update import make_update
    from scripts.train import init_train_state
    from openpi.training import data_loader, checkpoints, sharding
    mark_phase(ROOT, 'train.imports.after', mode=args.mode)

    config = make_config(experiment=args.mode, resume=args.resume)
    stop_at = getattr(args, 'stop_at', None)
    if stop_at is not None and (args.mode != 'pilot' or not args.resume
                               or stop_at not in (1000, 1500, 2000) or config.num_train_steps != 2000):
        raise ValueError('Segmented pilot requires --resume, stop-at1000|1500|2000, and the unchanged2000-step configuration')
    loop_stop = config.num_train_steps if stop_at is None else stop_at
    if args.verify_full_state:
        full_state_verify(config, 32, ROOT/'reports/smoke_training_evidence.json', ROOT/'reports/full_state_restore.json')
        return
    if config.batch_size != 1 or config.gradient_accumulation_steps != 4:
        raise ValueError('This declared pilot fixes actual batch1/accumulation4; changes need a new experiment configuration')
    attempt_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + str(os.getpid())
    stage_dir = ROOT/'reports'/args.mode/('attempt_'+attempt_id)
    stage_dir.mkdir(parents=True, exist_ok=False)
    status_path = ROOT/'reports'/f'{args.mode}_training_status.json'
    evidence_path = ROOT/'reports'/f'{args.mode}_training_evidence.json'
    completed_marker = ROOT/'reports'/f'{args.mode}_complete.json'
    if completed_marker.exists():
        raise FileExistsError('Stage is already complete; refusing an accidental duplicate')
    write_json(ROOT/'configs'/f'{args.mode}_effective.json', dataclasses.asdict(config))
    random.seed(config.seed); np.random.seed(config.seed)
    install_data_hooks()
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    from checkpoint_io import initialize_checkpoint_dir
    manager, resuming = initialize_checkpoint_dir(config.checkpoint_dir, keep_period=500,
                                                overwrite=False, resume=args.resume)
    if stop_at is not None and not resuming:
        raise RuntimeError('Segmented pilot requires an existing full native checkpoint; refusing fresh-base fallback')
    mark_phase(ROOT, 'train.dataset_open.before', mode=args.mode, split='train')
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    iterator = iter(loader)
    mark_phase(ROOT, 'train.dataset_open.after', mode=args.mode, split='train')
    episode_file = CANDIDATE_ROOT/'lerobot/kinova_candidate/train/meta/episodes.jsonl'
    episode_lengths = {int(row['episode_index']): int(row['length'])
                       for row in map(json.loads, episode_file.read_text().splitlines())}
    monitor = MemoryMonitor(ROOT/'logs'/f'{args.mode}_gpu_memory.jsonl'); monitor.start()
    started = time.monotonic()
    state = None
    first_step_start = first_step_end = None
    optimizer_times = []
    evidence = {'stage': args.mode, 'status': 'initializing', 'resource_gate': gate,
                'optimizer_steps_executed': 0, 'robot_commands_sent': 0, 'test_split_used_for_training': False,
                'actual_batch': 1, 'gradient_accumulation': 4, 'effective_batch': 4,
                'valid_element_weighted_accumulation': True, 'seed': config.seed,
                'data_version': 'D0; manual review pending; wrist/external synchronization UNVERIFIED',
                'base_params': str(config.weight_loader.params_path), 'checkpoint_dir': str(config.checkpoint_dir)}
    if stop_at is not None:
        evidence.update(segmented_process=True, segment_stop_at=stop_at,
                        total_target_steps=config.num_train_steps)
    stop_requested = False

    def interrupted(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        mark_phase(ROOT, 'train.init_train_state.before', mode=args.mode, resuming=resuming,
                   jax_allocator=jax.devices()[0].memory_stats())
        state, _ = init_train_state(config, jax.random.key(config.seed), mesh, resume=resuming)
        jax.block_until_ready(state)
        mark_phase(ROOT, 'train.init_train_state.after', mode=args.mode, resuming=resuming,
                   jax_allocator=jax.devices()[0].memory_stats())
        if resuming:
            state = checkpoints.restore_state(manager, state, loader)
            jax.block_until_ready(state)
        first = int(state.step)
        if first > config.num_train_steps:
            raise ValueError('Restored step exceeds declared training target')
        if stop_at is not None and first != stop_at-500:
            raise ValueError(f'Segment stop-at{stop_at} requires native full-state resume at{stop_at-500}; got{first}')
        if stop_at is not None:
            resume_counters = optimizer_counters(state)
            if not resume_counters or any(row['value'] != first for row in resume_counters):
                raise RuntimeError('Resumed Adam/schedule counters do not match the segment start')
            evidence['resume_optimizer_counters'] = resume_counters
            evidence['resume_optimizer_step'] = first
        # Native seeded Torch sampler is replayed to recover the same data offset.
        # No validation/test batches are ever supplied to the gradient update.
        for _ in range(first * config.gradient_accumulation_steps):
            next(iterator)
        if args.mode == 'pilot' and resuming and stop_at is not None:
            from memory_maintenance import reclaim_before_resumed_training
            evidence['resumed_training_cache_maintenance'] = reclaim_before_resumed_training(root=ROOT)
        initial_train_hash = initial_frozen_hash = None
        if args.mode == 'smoke':
            initial_path = ROOT/'reports/smoke_initial_parameter_hashes.json'
            if initial_path.exists():
                original = json.loads(initial_path.read_text())
                initial_train_hash, initial_frozen_hash = original['trainable'], original['frozen']
            elif first:
                raise RuntimeError('Cannot verify a resumed smoke without its original initial-parameter hashes')
            else:
                initial_train_hash = digest_params(state.params.filter(config.trainable_filter))
                initial_frozen_hash = digest_params(state.params.filter(config.freeze_filter))
                write_json(initial_path, {'trainable': initial_train_hash, 'frozen': initial_frozen_hash,
                                          'optimizer_step': 0, 'attempt_id': attempt_id})
        evidence['initial_optimizer_step'] = first
        evidence['adapter_initialization'] = adapter_initialization_summary(state.params) if not resuming else {'resumed': True}
        if not resuming and not evidence['adapter_initialization']['all_lora_b_zero']:
            raise RuntimeError('Ordinary LoRA does not start at zero contribution')
        if args.mode == 'smoke' and not resuming:
            evidence['reference_initial'] = capture(state, config, stage_dir/'initial_reference', step=0)
            evidence['initial_parameter_identity'] = verify_initial_parameter_identity(evidence['reference_initial'])
            from verify_checkpoint import compare_references
            equivalence = compare_references(ROOT/'reports/E0/reference', stage_dir/'initial_reference', comparison='same_graph_zero_lora_baseline_vs_train_initial')
            write_json(ROOT/'reports/base_zero_lora_equivalence.json', equivalence)
            if equivalence['status'] != 'passed':
                raise RuntimeError('Loaded ordinary-LoRA model failed fixed-input same-graph zero-update baseline equivalence')
        update = jax.jit(make_update(config), donate_argnums=(1,))
        log = (ROOT/'logs'/f'{args.mode}_optimizer.jsonl').open('a')
        for _ in range(first, loop_stop):
            if stop_requested:
                checkpoints.save_state(manager, state, loader, int(state.step)); manager.wait_until_finished()
                raise InterruptedError('Own training interrupted; current completed optimizer state saved')
            micros = [next(iterator) for _ in range(config.gradient_accumulation_steps)]
            provenance = [validate_microbatch(batch, episode_lengths) for batch in micros]
            if int(state.step) == first:
                evidence['actual_first_microbatches'] = provenance
            batch = jax.tree.map(lambda *values: jnp.stack(values, axis=0), *micros)
            if first_step_start is None:
                mark_phase(ROOT, 'train.first_optimizer.before', mode=args.mode,
                           optimizer_step=int(state.step), jax_allocator=jax.devices()[0].memory_stats())
            step_started = time.monotonic()
            if first_step_start is None:
                first_step_start = time.time()
            with sharding.set_mesh(mesh):
                state, info = update(jax.random.key(config.seed), state, batch)
            jax.block_until_ready((state.step, info))
            seconds = time.monotonic()-step_started
            optimizer_times.append(seconds)
            if first_step_end is None:
                first_step_end = time.time()
                evidence['jax_allocator_after_first_update'] = jax.devices()[0].memory_stats()
                evidence['first_update_sampled_whole_card_peak_mib'] = monitor.peak(first_step_start, first_step_end)
                mark_phase(ROOT, 'train.first_optimizer.after', mode=args.mode,
                           optimizer_step=int(state.step), jax_allocator=evidence['jax_allocator_after_first_update'])
            record = {'optimizer_step': int(state.step), 'optimizer_seconds': seconds,
                      **{key: np.asarray(value).item() for key, value in jax.device_get(info).items()},
                      'microbatches': provenance}
            if not record['update_applied']:
                import math
                diagnostic = {key: (str(value) if isinstance(value, float) and not math.isfinite(value) else value)
                              for key, value in record.items()}
                log.write(json.dumps(diagnostic, allow_nan=False)+'\n'); log.flush()
                raise FloatingPointError('Empty or nonfinite gradient update rejected; parameters/Adam/step were not advanced')
            log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
            if int(state.step) == first+1 or int(state.step) % config.log_interval == 0:
                print(json.dumps({k: v for k,v in record.items() if k != 'microbatches'}), flush=True)
            write_json(status_path, {'status': 'training', 'optimizer_step': int(state.step), 'target_steps': config.num_train_steps,
                                     'segment_stop_at': stop_at,
                                     'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'last_update': record,
                                     'sampled_whole_card_peak_mib': monitor.peak()})
            if int(state.step) % config.save_interval == 0 or int(state.step) == config.num_train_steps:
                if args.mode == 'pilot' and stop_at is not None and int(state.step) == stop_at:
                    # Native save_state uses the loader only for normalization
                    # assets. The segment has no further batches; release all
                    # Arrow mappings and input buffers before saving weights.
                    from types import SimpleNamespace
                    from checkpoint_io import _release
                    asset_config = loader.data_config()
                    loader = SimpleNamespace(data_config=lambda value=asset_config: value)
                    del iterator, micros, batch
                    _release()
                    evidence['checkpoint_inputs_released_before_save'] = True
                    mark_phase(ROOT, 'train.segment_inputs_released', optimizer_step=int(state.step))
                checkpoints.save_state(manager, state, loader, int(state.step)); manager.wait_until_finished()
        log.close()
        manager.wait_until_finished()
        counters = optimizer_counters(state)
        if not counters or any(row['value'] != int(state.step) for row in counters):
            raise RuntimeError('Adam/schedule counters do not match the actual optimizer step')
        evidence.update(status='trained', optimizer_step=int(state.step), optimizer_steps_executed=int(state.step)-first,
                        optimizer_counters=counters, elapsed_seconds=time.monotonic()-started,
                        mean_optimizer_seconds=float(np.mean(optimizer_times)) if optimizer_times else None,
                        first_optimizer_seconds=optimizer_times[0] if optimizer_times else None,
                        mean_warm_optimizer_seconds=float(np.mean(optimizer_times[1:])) if len(optimizer_times)>1 else None,
                        sampled_whole_card_peak_mib=monitor.peak(), jax_allocator_final=jax.devices()[0].memory_stats(),
                        memory_measurement='whole-card nvidia-smi 0.5s samples may miss short peaks; JAX allocator peak is separate',
                        checkpoint=str(config.checkpoint_dir/str(int(state.step))))
        if stop_at is not None:
            # Saving is synchronous. Read only the small native commit marker
            # after save; no parameter hashing or inference before exiting.
            commit = json.loads((config.checkpoint_dir/str(stop_at)/'_CHECKPOINT_METADATA').read_text())
            if int(state.step) != stop_at or not commit.get('commit_timestamp_nsecs'):
                raise RuntimeError('Segment checkpoint has no completed native commit marker')
            segment_report = {'status': 'segment_complete', 'initial_optimizer_step': first,
                'optimizer_step': int(state.step), 'optimizer_steps_executed': int(state.step)-first,
                'total_target_steps': config.num_train_steps, 'checkpoint': evidence['checkpoint'],
                'commit_timestamp_nsecs': commit['commit_timestamp_nsecs'],
                'optimizer_counters': counters, 'resume_full_train_state': True,
                'sampler_microbatches_replayed': first*config.gradient_accumulation_steps,
                'robot_commands_sent': 0, 'test_split_used_for_training': False}
            manager.close()
            if stop_at < config.num_train_steps:
                evidence['status'] = 'segment_complete'
                write_json(stage_dir/'evidence.json', evidence)
                write_json(evidence_path, evidence)
                write_json(ROOT/'reports/pilot_segments'/f'step_{stop_at}_complete.json', segment_report)
                write_json(status_path, {'status': 'segment_complete', 'optimizer_step': int(state.step),
                    'target_steps': config.num_train_steps, 'segment_stop_at': stop_at,
                    'checkpoint': evidence['checkpoint']})
                print('KINOVA_SEGMENT_COMPLETE', first, stop_at, flush=True)
                return
            write_json(ROOT/'reports/pilot_segments'/f'step_{stop_at}_complete.json', segment_report)
        if args.mode == 'smoke':
            final_train_hash = digest_params(state.params.filter(config.trainable_filter))
            final_frozen_hash = digest_params(state.params.filter(config.freeze_filter))
            changed = [name for name,value in final_train_hash.items() if initial_train_hash.get(name) != value]
            evidence.update(frozen_parameters_unchanged=initial_frozen_hash == final_frozen_hash,
                            changed_trainable_parameters=changed,
                            changed_lora_tensors=sum('lora' in name for name in changed),
                            changed_head_tensors=sum('lora' not in name for name in changed),
                            final_optimizer_state_hashes=optimizer_hashes(state),
                            final_parameter_hashes={**final_frozen_hash, **final_train_hash})
            evidence['reference_final'] = capture(state, config, stage_dir/'final_reference', step=int(state.step))
            if not evidence['frozen_parameters_unchanged'] or not evidence['changed_lora_tensors'] or not evidence['changed_head_tensors']:
                raise RuntimeError('Smoke parameter preservation/update verification failed')
        write_json(stage_dir/'evidence.json', evidence)
        write_json(evidence_path, evidence)
        write_json(completed_marker, {'status': 'training_complete', 'optimizer_step': int(state.step),
                                     'evidence': str(evidence_path), 'checkpoint': evidence['checkpoint'],
                                     'new_process_restore_still_required': args.mode == 'smoke', 'robot_commands_sent': 0})
        write_json(status_path, {'status': 'training_complete', 'optimizer_step': int(state.step), 'target_steps': config.num_train_steps})
        print('KINOVA_TRAINING_COMPLETE', args.mode, int(state.step), flush=True)
    except BaseException as error:
        evidence.update(status='failed', error=f'{type(error).__name__}: {error}',
                        optimizer_step=int(state.step) if state is not None else 0,
                        sampled_whole_card_peak_mib=monitor.peak(), elapsed_seconds=time.monotonic()-started)
        write_json(stage_dir/'evidence.json', evidence)
        write_json(evidence_path, evidence); write_json(status_path, evidence)
        raise
    finally:
        monitor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'pilot'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-at', type=int, choices=[1000, 1500, 2000],
                        help='Pilot only: restore the previous500-step full state, train one segment, save and exit; total schedule remains2000')
    parser.add_argument('--verify-full-state', action='store_true', help='New-process restore check; no optimizer updates')
    args = parser.parse_args()
    from resource_guard import inspect_resources
    gate = inspect_resources()
    if not gate['eligible']:
        write_json(ROOT/'reports/last_gpu_gate.json', gate)
        print('KINOVA_RESOURCE_DEFERRED '+json.dumps(gate, ensure_ascii=False, allow_nan=False), flush=True)
        raise SystemExit(75)
    from resource_guard import verify_allocator_limit
    mark_phase(ROOT, 'train.allocator_init.before', mode=args.mode)
    gate['allocator_verification'] = verify_allocator_limit()
    mark_phase(ROOT, 'train.allocator_init.after', mode=args.mode,
               allocator_verification=gate['allocator_verification'])
    train(args, gate)


if __name__ == '__main__':
    main()
