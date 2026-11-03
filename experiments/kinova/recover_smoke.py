"""Recover committed smoke32 without updates; independently verify its snapshot.

The first final-state snapshot is read from the committed checkpoint. No
pre-save final RAM hash survived the original process; this script never
claims that such a comparison was made.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback


SOURCE_ROOT = Path(project_path('archive', 'run_20261102_205332_deterministic'))
STEP = 32
EXPECTED_ENV = {
    'XLA_FLAGS': '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest',
    'PYTHONHASHSEED': '42', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.85', 'XLA_PYTHON_CLIENT_ALLOCATOR': 'bfc',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_json(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_json(item) for item in value)
    return True


def checkpoint_stat_identity(checkpoint):
    return hashlib.sha256(json.dumps({str(path.relative_to(checkpoint)):
        [path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted(checkpoint.rglob('*')) if path.is_file()},
        sort_keys=True).encode()).hexdigest()


def audit_source(source):
    source = Path(source).resolve()
    if source != SOURCE_ROOT.resolve():
        raise ValueError('Recovery is restricted to the original committed deterministic smoke32')
    checkpoint = source/'checkpoints/E1_wrist_smoke/seed42/32'
    marker = read_json(checkpoint/'_CHECKPOINT_METADATA')
    initial = read_json(source/'reports/smoke_initial_parameter_hashes.json')
    identity = read_json(source/'reports/initial_parameter_identity.json')
    equivalence = read_json(source/'reports/base_zero_lora_equivalence.json')
    e0 = read_json(source/'evaluations/E0_wrist_frozen/summary.json')
    baseline = e0.get('config_metadata', {})
    expected_bf16 = {**baseline.get('parameter_hashes', {}).get('base', {}),
                     **baseline.get('parameter_hashes', {}).get('adapter', {})}
    captured_initial = read_json(identity['actual_hashes'])
    rows = [json.loads(line) for line in (source/'logs/smoke_optimizer.jsonl').read_text().splitlines()]
    checks = {
        'committed_checkpoint': bool(marker.get('commit_timestamp_nsecs')),
        'native_items_present': all((checkpoint/item/'_METADATA').is_file() for item in ('params', 'train_state')),
        'original_initial_hashes': initial.get('optimizer_step') == 0 and len(initial.get('frozen', {})) == 43
                                   and len(initial.get('trainable', {})) == 28,
        'original_initial_identity_passed': identity.get('status') == 'passed' and identity.get('matched_parameter_count') == 71,
        'original_initial_BF16_hashes_equal_E0': len(expected_bf16) == 71 and captured_initial.get('hashes') == expected_bf16,
        'original_E0_protocol': e0.get('evaluation_valid') is True and e0.get('records_completed') == 60
                               and e0.get('test_split_used') is False and baseline.get('parameter_hashes_recorded') is True,
        'original_strict_output_passed': equivalence.get('status') == 'passed'
             and equivalence.get('comparison') == 'same_graph_zero_lora_baseline_vs_train_initial'
             and equivalence.get('atol') == equivalence.get('rtol') == 1e-5
             and all(item.get('passed') is True for item in equivalence.get('checks', {}).values())
             and set(equivalence.get('checks', {})) == {'prediction_normalized', 'fm_numerator', 'fm_count'},
        'exact32_sequential_optimizer_records': len(rows) == STEP and [row.get('optimizer_step') for row in rows] == list(range(1, STEP+1)),
        'all_logged_updates_applied': all(row.get('update_applied') is True for row in rows),
        'all_optimizer_records_finite': all(finite_json(row) for row in rows),
        'positive_valid_elements': all(row.get('valid_elements', 0) > 0 for row in rows),
        'positive_finite_gradients': all(row.get('grad_norm', 0) > 0 for row in rows),
        'some_nonzero_parameter_updates': any(row.get('update_norm', 0) > 0 for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError('Source smoke evidence failed: '+', '.join(key for key, valid in checks.items() if not valid))
    return {'status': 'passed', 'source_root': str(source), 'checkpoint': str(checkpoint),
            'checks': checks, 'checkpoint_marker': marker,
            'checkpoint_stat_sha256': checkpoint_stat_identity(checkpoint),
            'original_initial_hashes': initial, 'original_initial_identity': identity,
            'original_strict_output_comparison': equivalence,
            'optimizer_log_record_count': len(rows), 'first_optimizer_record': rows[0],
            'last_optimizer_record': rows[-1],
            'original_post_training_RAM_hashes_available': False,
            'evidence_boundary': '32 updates are proven by original logs; recovered hashes are first final checkpoint snapshots, not original final RAM hashes'}


def check_environment():
    actual = {key: os.environ.get(key) for key in EXPECTED_ENV}
    if actual != EXPECTED_ENV:
        raise RuntimeError('Recovery must use the unchanged deterministic/memory environment: '+str(actual))
    return actual


def resource_gate():
    from resource_guard import inspect_resources, verify_allocator_limit
    gate = inspect_resources()
    if not gate['eligible']:
        print('KINOVA_RESOURCE_DEFERRED '+json.dumps(gate, allow_nan=False), flush=True)
        raise SystemExit(75)
    gate['allocator_verification'] = verify_allocator_limit()
    return gate


def restore_state(checkpoint):
    import jax
    from kinova_config import make_config
    from scripts.train import init_train_state
    from openpi.training import checkpoints, sharding
    from checkpoint_io import initialize_readonly_checkpoint_dir
    config = make_config('smoke', resume=True)
    mesh = sharding.make_mesh(1)
    template, _ = init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
    # No source mkdir/save/deletion. No base weight load or training loader.
    manager = initialize_readonly_checkpoint_dir(Path(checkpoint).parent)
    try:
        state = checkpoints.restore_state(manager, template, None, step=STEP)
        jax.block_until_ready(state)
    finally:
        manager.close()
    return state, config


def full_state_finite(state):
    import jax
    import jax.numpy as jnp
    finite = jax.jit(lambda value: jnp.all(jnp.isfinite(value)))
    results = {}
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        if hasattr(value, 'dtype') and hasattr(value, 'shape'):
            results[jax.tree_util.keystr(path)] = bool(jax.device_get(finite(value)))
    if not results or not all(results.values()):
        raise RuntimeError('Recovered full state contains nonfinite arrays')
    return {'all_finite': True, 'array_count': len(results)}


def snapshot_state(state):
    from train_kinova import digest_params, optimizer_hashes, optimizer_counters
    counters = optimizer_counters(state)
    if int(state.step) != STEP or len(counters) != 4 or any(item['value'] != STEP for item in counters):
        raise RuntimeError('Recovered step or Adam/schedule counters are not all32')
    return {'optimizer_step': int(state.step), 'optimizer_counters': counters,
            'final_parameter_hashes': digest_params(state.params),
            'final_optimizer_state_hashes': optimizer_hashes(state),
            'finite_full_state': full_state_finite(state)}


def collect(root, source_evidence, gate):
    from phase_memory import mark_phase
    from train_kinova import capture
    destination = root/'reports/recovered_smoke'
    snapshot_path = destination/'final_checkpoint_snapshot.json'
    prefix = destination/'final_reference'
    if snapshot_path.exists() or prefix.with_suffix('.json').exists() or prefix.with_suffix('.npz').exists():
        raise FileExistsError('Prior recovery snapshot/reference exists; refusing overwrite')
    mark_phase(root, 'recovery.restore.before', source=source_evidence['checkpoint'])
    state, config = restore_state(source_evidence['checkpoint'])
    mark_phase(root, 'recovery.restore.after', step=int(state.step))
    snapshot = snapshot_state(state)
    initial = source_evidence['original_initial_hashes']
    # Reuse the full mixed-state digests; inspect filter keys without another
    # complete device-to-host hash pass.
    frozen = {'/'.join(map(str, key)): snapshot['final_parameter_hashes']['/'.join(map(str, key))]
              for key in state.params.filter(config.freeze_filter).flat_state()}
    trainable = {'/'.join(map(str, key)): snapshot['final_parameter_hashes']['/'.join(map(str, key))]
                 for key in state.params.filter(config.trainable_filter).flat_state()}
    changed = sorted(name for name in trainable if initial['trainable'].get(name) != trainable[name])
    changed_lora = [name for name in changed if 'lora' in name]
    changed_heads = [name for name in changed if 'lora' not in name]
    if len(frozen) != 43 or frozen != initial['frozen'] or not changed_lora or not changed_heads:
        raise RuntimeError('Recovered frozen43/changed LoRA/changed heads checks failed')
    reference = capture(state, config, prefix, step=STEP)
    if checkpoint_stat_identity(Path(source_evidence['checkpoint'])) != source_evidence['checkpoint_stat_sha256']:
        raise RuntimeError('Read-only source checkpoint stat identity changed during recovery')
    snapshot.update({'process_id': os.getpid(), 'source_checkpoint': source_evidence['checkpoint'],
                     'source_checkpoint_stat_sha256': source_evidence['checkpoint_stat_sha256'],
                     'snapshot_origin': 'first final-state snapshot restored from committed checkpoint',
                     'original_final_RAM_hashes_available': False, 'reference_final': reference})
    write_json(snapshot_path, snapshot)
    evidence = {**snapshot, 'status': 'recovered_pending_independent_verification',
                'source_root': source_evidence['source_root'], 'checkpoint': source_evidence['checkpoint'],
                'base_params': str(config.weight_loader.params_path),
                'optimizer_steps_executed_in_original_training': 32, 'optimizer_steps_executed_by_recovery': 0,
                'optimizer_step': STEP, 'frozen_parameters_unchanged': True,
                'frozen_parameter_count': len(frozen), 'changed_lora_tensors': changed_lora,
                'changed_head_tensors': changed_heads, 'original_initial_parameter_hashes': initial,
                'source_evidence': source_evidence, 'resource_gate': gate,
                'robot_commands_sent': 0, 'test_split_used_for_training': False,
                'independent_full_restore_verified': False,
                'evidence_boundary': source_evidence['evidence_boundary']}
    write_json(root/'reports/smoke_training_evidence.json', evidence)
    print(json.dumps({'status': evidence['status'], 'optimizer_step': STEP,
                      'snapshot': str(snapshot_path), 'recovery_updates': 0}), flush=True)


def verify(root, source_evidence, gate):
    from train_kinova import capture
    from verify_checkpoint import compare_references
    snapshot = read_json(root/'reports/recovered_smoke/final_checkpoint_snapshot.json')
    if snapshot['process_id'] == os.getpid():
        raise RuntimeError('Independent verification requires a different process')
    if (root/'reports/full_state_restore.json').exists() or (root/'reports/checkpoint_prediction_restore.json').exists():
        raise FileExistsError('Prior independent verification evidence exists; refusing overwrite')
    state, config = restore_state(source_evidence['checkpoint'])
    actual = snapshot_state(state)
    checks = {key: actual[key] == snapshot[key] for key in
              ('optimizer_step', 'optimizer_counters', 'final_parameter_hashes', 'final_optimizer_state_hashes')}
    checks['finite_full_state'] = actual['finite_full_state']['all_finite'] is True
    checks['source_checkpoint_readonly'] = (checkpoint_stat_identity(Path(source_evidence['checkpoint']))
        == source_evidence['checkpoint_stat_sha256'] == snapshot['source_checkpoint_stat_sha256'])
    result = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'optimizer_step': STEP, 'optimizer_counters': actual['optimizer_counters'],
              'new_process': True, 'process_id': os.getpid(), 'collect_process_id': snapshot['process_id'],
              'checkpoint': source_evidence['checkpoint'], 'optimizer_steps_executed_by_verifier': 0,
              'reference_origin': 'checkpoint-restored first snapshot; original final RAM hashes did not survive',
              'resource_gate': gate, 'robot_commands_sent': 0}
    write_json(root/'reports/full_state_restore.json', result)
    if result['status'] != 'passed':
        raise RuntimeError('Independent full parameter/Adam checkpoint restore failed')
    reference = capture(state, config, root/'reports/recovered_smoke/verified_reference', step=STEP)
    comparison = compare_references(snapshot['reference_final']['prefix'], reference['prefix'],
                                    comparison='independent_checkpoint32_restore')
    write_json(root/'reports/checkpoint_prediction_restore.json', comparison)
    if comparison['status'] != 'passed':
        raise RuntimeError('Independent checkpoint prediction restore failed strict1e-5 comparison')
    evidence_path = root/'reports/smoke_training_evidence.json'
    evidence = read_json(evidence_path)
    evidence.update(status='complete', independent_full_restore_verified=True,
                    independent_prediction_restore_verified=True,
                    optimizer_steps_executed_by_verifier=0, restored_reference=reference)
    write_json(evidence_path, evidence)
    write_json(root/'reports/smoke_complete.json', {'status': 'complete', 'optimizer_step': STEP,
               'source_checkpoint': source_evidence['checkpoint'],
               'original_training_updates': STEP, 'recovery_updates': 0,
               'original_final_RAM_hashes_available': False,
               'independent_full_restore_verified': True, 'independent_prediction_restore_verified': True})
    print(json.dumps({'status': 'passed', 'optimizer_step': STEP, 'recovery_updates': 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'collect', 'verify'))
    parser.add_argument('--source-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--resource-approved', action='store_true')
    args = parser.parse_args()
    root = Path(os.environ['KINOVA_EXPERIMENT_ROOT']).resolve()
    if root == SOURCE_ROOT.resolve():
        raise ValueError('Recovery output must use a new independent root')
    source = audit_source(args.source_root)
    if args.mode == 'audit':
        write_json(root/'reports/smoke_recovery_source_audit.json', source)
        print(json.dumps({'status': 'passed', 'original_updates': STEP,
                          'checkpoint': source['checkpoint'], 'model_loaded': False}), flush=True)
        return
    check_environment()
    gate = resource_gate()
    try:
        {'collect': collect, 'verify': verify}[args.mode](root, source, gate)
    except BaseException as error:
        write_json(root/'reports/recovered_smoke'/f'{args.mode}_failure.json',
                   {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                    'traceback': traceback.format_exc(), 'optimizer_updates_executed_by_recovery': 0,
                    'source_checkpoint': source['checkpoint'], 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
        raise


if __name__ == '__main__':
    main()
