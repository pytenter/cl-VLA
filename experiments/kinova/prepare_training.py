"""CPU-only effective-config export and preflight; does not run π0.5 training."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import dataclasses
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ROOT = Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', project_path('kinova', '')))


def streamed_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def numerical_source_identity(inherited_e0):
    """Check preserved math independently of explicitly changed loader functions."""
    previous = Path(inherited_e0['source_root'])
    rows = []
    for name in ['kinova_model.py', 'validation_plan.py', 'verify_checkpoint.py', 'array_hash.py']:
        old, new = previous/'scripts'/name, ROOT/'scripts'/name
        a, b = old.read_bytes(), new.read_bytes()
        equal = a.replace(str(previous).encode(), b'<RUN_ROOT>') == b.replace(str(ROOT).encode(), b'<RUN_ROOT>')
        rows.append({'name': name, 'comparison': 'whole file, only run-root literals normalized',
                     'source_sha256': hashlib.sha256(a).hexdigest(), 'copy_sha256': hashlib.sha256(b).hexdigest(),
                     'identical': equal})
    functions = ['action_stats', 'unnormalize', 'error_metrics', 'score_prediction', 'mean_metrics', 'add_rmse',
                 'aggregate_records', 'prepare_sample', 'make_inference_functions', 'evaluate_model',
                 'canonical_array_hash', 'sample_hash', 'select_checkpoint']
    for name, selected in [('kinova_config.py', ['make_config']), ('evaluate_offline.py', functions)]:
        trees = []
        paths = [previous/'scripts'/name, ROOT/'scripts'/name]
        for path, root in zip(paths, [previous, ROOT], strict=True):
            tree = ast.parse(path.read_text().replace(str(root), '<RUN_ROOT>'))
            definitions = {node.name: copy.deepcopy(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for function in selected:
                node = definitions[function]
                if name == 'kinova_config.py':
                    for call in ast.walk(node):
                        if isinstance(call, ast.Call):
                            call.keywords = [kw for kw in call.keywords if kw.arg != 'weight_loader']
                definitions[function] = ast.dump(node, include_attributes=False)
            trees.append({function: definitions[function] for function in selected})
        rows.append({'name': name, 'comparison': 'numerical function AST, ignoring locations/run-root literals',
                     'functions': selected, 'excluded': ['make_config weight_loader keyword only'] if name == 'kinova_config.py' else ['load_model native restore replacement; resource/CLI helpers'],
                     'source_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                     'copy_sha256': hashlib.sha256(paths[1].read_bytes()).hexdigest(), 'identical': trees[0] == trees[1]})
    result = {'status': 'passed' if all(row['identical'] for row in rows) else 'failed',
              'scope': 'Read-only source/AST identity only; changed native loader requires tiny CPU equivalence and actual zero-LoRA fixed-input GPU prediction gate',
              'E0_source_root': str(previous), 'current_root': str(ROOT), 'files': rows}
    (ROOT/'reports/native_protocol_identity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    import jax
    from kinova_config import make_config
    from validation_plan import validate_plan
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Preparation must remain CPU only')
    checks = {}
    required = ['masked_loss_cpu.json', 'native_loader_validation.json', 'accumulation_unit_tests.json',
                'offline_metric_cpu_test_v2.json', 'restore_comparison_cpu_test_v3.json',
                'evaluation_data_interface_cpu_test.json', 'evaluation_resource_gate_cpu_test.json',
                'evaluation_runtime_cpu_toy_test.json', 'training_entry_cpu_test.json',
                'trainer_structured_cpu_review.json', 'bounded_native_weights_cpu.json', 'cacheguard_native_cpu.json', 'bounded_inference_cpu.json', 'same_graph_evaluation_cpu.json', 'phase_memory_cpu.json', 'memory_maintenance_cpu.json',
                'windows_memory_live_test.json', 'continuation_data_check.json', 'idle_reclaim_cpu.json',
                'monitor_health_cpu_live.json', 'window_guard_cpu.json',
                'resource_defer_cpu.json', 'resource_race_workflow_cpu.json', 'exclusive_resource_cpu.json', 'tiny_lora_cpu_diagnosis.json', 'same_graph_training_entry_cpu.json',
                'deterministic_evaluation_cpu.json', 'deterministic_training_entry_cpu.json', 'same_graph_cross_process_nondeterminism_cpu.json',
                'checkpoint_io_cpu.json', 'recover_smoke_cpu.json', 'resume_commit32_workflow_cpu.json', 'smoke_recovery_source_audit.json',
                'segmented_workflow_cpu.json', 'segmented_training_cpu.json', 'resumed_training_cache_cpu.json']
    for filename in required:
        path = ROOT/'reports'/filename
        data = json.loads(path.read_text())
        checks[filename] = data.get('status') == 'passed'
    io_proof = json.loads((ROOT/'reports/checkpoint_io_cpu.json').read_text())
    checks['current_checkpoint_IO_source_tested'] = (
        io_proof.get('source_module_sha256') == hashlib.sha256((ROOT/'scripts/checkpoint_io.py').read_bytes()).hexdigest()
        and io_proof.get('gpu_used') is False and io_proof.get('full_model_loaded') is False
    )
    resumed_proof = json.loads((ROOT/'reports/resumed_training_cache_cpu.json').read_text())
    checks['current_resumed_precompile_cache_helper_tested'] = (
        resumed_proof.get('source_module_sha256') == hashlib.sha256((ROOT/'scripts/memory_maintenance.py').read_bytes()).hexdigest()
        and resumed_proof.get('actual_sync_executed') is False
        and resumed_proof.get('actual_global_drop_executed') is False
        and all(resumed_proof.get('checks', {}).values())
    )
    assets_proof = json.loads((ROOT/'reports/segment_assets_cpu.json').read_text())
    checks['end_segment_inputs_released_and_native_assets_unchanged'] = (
        assets_proof.get('status') == 'passed' and all(assets_proof.get('checks', {}).values())
        and assets_proof.get('trainer_sha256') == hashlib.sha256((ROOT/'scripts/train_kinova.py').read_bytes()).hexdigest())
    trainer_proof = json.loads((ROOT/'reports/segmented_training_cpu.json').read_text())
    checks['current_trainer_AST_identity_to_tested_segmented_source'] = (
        trainer_proof.get('whole_module_AST_identical_except_ROOT_and_save_branch') is True
        and trainer_proof.get('trainer_sha256') == hashlib.sha256((ROOT/'scripts/train_kinova.py').read_bytes()).hexdigest())
    probe_root = Path(project_path('archive', 'run_20261103_110800_writecache_probe'))
    probe = json.loads((probe_root/'reports/probe_complete.json').read_text())
    probe_state = json.loads((probe_root/'reports/probe_independent_CPU_restore.json').read_text())
    checks['actual_full_model_save_and_independent_restore_probe_passed'] = (
        json.loads((probe_root/'workflow.json').read_text()).get('status') == 'probe_complete'
        and probe.get('status') == 'checkpoint_save_probe_complete'
        and probe.get('optimizer_steps_executed') == 1 and probe.get('optimizer_step') == 501
        and bool(probe.get('commit_timestamp_nsecs'))
        and probe_state.get('status') == 'passed' and probe_state.get('optimizer_step') == 501
        and all(probe_state.get('checks', {}).values()))
    recovery_proof = json.loads((ROOT/'reports/recover_smoke_cpu.json').read_text())
    checks['recovery_CPU_proof_keeps_missing_oldRAM_boundary'] = (
        recovery_proof.get('source_audit_passed') is True
        and recovery_proof.get('strict_environment_guard_passed') is True
        and recovery_proof.get('original_final_RAM_hash_missing_preserved') is True
        and recovery_proof.get('complete_written_only_after_state_and_prediction_checks') is True
        and recovery_proof.get('source_checkpoint_stat_unchanged') is True
        and recovery_proof.get('actual_recovery_updates_executed') == 0
    )
    routing_proof = json.loads((ROOT/'reports/segmented_workflow_cpu.json').read_text())
    checks['current_workflow_routes_three_segments_without_repeating_E0_smoke'] = (
        routing_proof.get('source_sha256') == hashlib.sha256((ROOT/'scripts/run_workflow.py').read_bytes()).hexdigest()
        and routing_proof.get('no_E0_or_smoke_GPU_jobs') is True
        and routing_proof.get('optimizer_updates_executed') == 0
        and all(case.get('passed') is True for case in routing_proof.get('cases', []))
    )
    checks['deterministic_execution_environment'] = (
        os.environ.get('XLA_FLAGS') == '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
        and os.environ.get('PYTHONHASHSEED') == '42'
        and jax.config.jax_default_matmul_precision == 'highest'
    )
    flags_proof = json.loads((ROOT/'reports/deterministic_training_entry_cpu.json').read_text())
    checks['pinned_CPU_flags_highest_and_strict_hash_gate_proof'] = (
        all(flags_proof.get('runtime_environment', {}).get(key) == os.environ[key] for key in (
            'XLA_FLAGS', 'CUBLAS_WORKSPACE_CONFIG', 'JAX_DEFAULT_MATMUL_PRECISION'))
        and flags_proof.get('jax_default_matmul_precision') == 'highest'
        and flags_proof.get('training_function_ast_unchanged_except_parameter_identity_gate') is True
        and flags_proof.get('workflow_E0_requires_recorded_51base20adapter_hashes') is True
        and flags_proof.get('strict_prediction_comparator_tolerance_unchanged') is True
    )
    discrepancy = json.loads((ROOT/'reports/same_graph_cross_process_nondeterminism_cpu.json').read_text())
    checks['recorded_same71params_same14inputs_different_outputs'] = (
        discrepancy.get('schema') == 'kinova-same-graph-cross-process-output-cpu-v1'
        and all(discrepancy.get('checks', {}).get(key) is True for key in (
            'all71_mixed_parameter_hashes_match', 'both_optimizer_step0', 'all14_fixed_inputs_bitwise_identical',
            'reference_conditions_identical', 'capture_and_inference_function_AST_identical', 'model_config_identical',
            'freeze_filter_identical', 'predictions_differ', 'FM_numerator_differs', 'FM_count_identical'))
    )
    data = json.loads((ROOT/'reports/data_summary.json').read_text())
    checks['data_numeric'] = data['numerical_checks_pass'] and not data['issues']
    checks['data_identity'] = data['candidate_episodes'] == 107 and data['candidate_steps'] == 14523
    checks['train_only_stats'] = data['norm_stats']['only_train_source'] is True
    plan = validate_plan(json.loads((ROOT/'configs/validation_plan.json').read_text()))
    checks['fixed_validation_plan'] = plan['episode_count'] == 20 and plan['test_split_used'] is False
    policy = json.loads((ROOT/'configs/resource_policy.json').read_text())
    checks['new_exclusive_resource_policy'] = (
        policy.get('mode') == 'exclusive_gpu'
        and policy.get('jax_memory_fraction') == .85 and policy.get('jax_preallocate') is False
        and policy.get('min_free_vram_mib_before_launch') == 40000
        and policy.get('min_host_available_mib_before_launch') == 16384
        and policy.get('min_windows_commit_free_gib_before_launch') == 16
        and policy.get('windows_stable_window_seconds') == 30
        and policy.get('idle_clean_pagecache_reclaim') is True
        and policy.get('drop_clean_cache_after_native_cpu_restore') is True
        and policy.get('checkpoint_min_windows_commit_free_gib') == 8
        and policy.get('checkpoint_process_exit_after_save') is True
        and policy.get('resume_training_clean_cache_reclaim') is True
        and policy.get('native_cache_reclaim_min_windows_commit_free_gib') == 12
        and policy.get('abort_windows_commit_free_gib') == 1.5
        and policy.get('abort_host_available_mib') == policy.get('reserve_free_vram_mib') == 2048
    )
    lineage = json.loads((ROOT/'manifests/continuation_provenance.json').read_text())
    inherited_e0 = lineage['legacy_E0']
    files = inherited_e0.get('files', [])
    summary_path = next(Path(item['copy']) for item in files if item['source'].endswith('/evaluations/E0_wrist_frozen/summary.json'))
    e0 = json.loads(summary_path.read_text())
    checks['legacy_E0_complete_60_records_historical_only'] = (
        e0.get('status') == 'complete' and e0.get('evaluation_valid') is True
        and e0.get('records_completed') == 60 and e0.get('expected_block_seed_count') == 60
        and e0.get('missing_records') == 0 and e0.get('errors') == 0
        and len(e0.get('per_episode', [])) == 20 and e0.get('test_split_used') is False
        and e0.get('optimizer_step') == 0
    )
    checks['legacy_E0_identity_historical_only'] = (
        e0.get('plan_sha256') == plan['plan_sha256'] == inherited_e0.get('plan_sha256')
        and e0.get('norm_stats_sha256') == inherited_e0.get('norm_stats_sha256')
        and e0.get('checkpoint') == inherited_e0.get('base_params')
        and e0.get('validation_input_and_noise_sha256') == inherited_e0.get('validation_input_and_noise_sha256')
        and e0.get('dataset_metrics', {}).get('selection_score') == inherited_e0.get('selection_score')
    )
    checks['legacy_E0_copied_file_hashes'] = len(files) == 64 and all(
        (Path(item['copy']).is_relative_to(ROOT/'legacy_evaluations/E0_non_lora')
         or Path(item['copy']).is_relative_to(ROOT/'legacy_reports/E0_non_lora'))
        and hashlib.sha256(Path(item['copy']).read_bytes()).hexdigest() == item['sha256']
        for item in files
    )
    checks['legacy_numerical_source_AST_identity'] = numerical_source_identity(inherited_e0)['status'] == 'passed'
    workflow = json.loads((ROOT/'workflow.json').read_text()) if (ROOT/'workflow.json').exists() else {}
    reused_e0 = lineage.get('reused_E0', lineage.get('inherited_E0', {}))
    reused_files = reused_e0.get('files', [])
    reused_summary = json.loads((ROOT/'evaluations/E0_wrist_frozen/summary.json').read_text())
    baseline = reused_summary.get('config_metadata', {})
    hashes = baseline.get('parameter_hashes', {})
    checks['deterministic_E0_explicit_verified_reuse'] = (
        reused_e0.get('source_root') == project_path('archive', 'run_20261102_205332_deterministic')
        and len(reused_files) == 64
        and all(Path(item['copy']).is_relative_to(ROOT)
                and hashlib.sha256(Path(item['copy']).read_bytes()).hexdigest() == item['sha256']
                for item in reused_files)
        and reused_summary.get('status') == 'complete' and reused_summary.get('evaluation_valid') is True
        and reused_summary.get('records_completed') == reused_summary.get('expected_block_seed_count') == 60
        and reused_summary.get('episode_count') == 20 and reused_summary.get('errors') == reused_summary.get('missing_records') == 0
        and reused_summary.get('plan_sha256') == plan['plan_sha256']
        and reused_summary.get('norm_stats_sha256') == plan['normalization']['sha256']
        and baseline.get('baseline_kind') == 'E0_same_graph_zero_update'
        and baseline.get('graph_kind') == 'zero_lora_same_as_training'
        and baseline.get('optimizer_step') == 0 and baseline.get('all_lora_b_zero') is True
        and baseline.get('parameter_hashes_recorded') is True
        and len(hashes.get('base', {})) == 51 and len(hashes.get('adapter', {})) == 20
        and any(job.get('name') == 'E0_wrist_frozen' and job.get('reused_from_root') == reused_e0['source_root']
                and job.get('reused_from_run') == project_path('archive', 'run_20261102_215627_resume_commit32')
                for job in workflow.get('completed_jobs', []))
    )
    recovery_audit = json.loads((ROOT/'reports/smoke_recovery_source_audit.json').read_text())
    checks['committed32_source_readonly_recovery_audit'] = (
        recovery_audit.get('status') == 'passed'
        and recovery_audit.get('source_root') == project_path('archive', 'run_20261102_205332_deterministic')
        and recovery_audit.get('original_post_training_RAM_hashes_available') is False
        and len(recovery_audit.get('checks', {})) >= 12
        and all(recovery_audit.get('checks', {}).values())
    )
    smoke = json.loads((ROOT/'reports/smoke_training_evidence.json').read_text())
    complete = json.loads((ROOT/'reports/smoke_complete.json').read_text())
    checks['source_smoke32_recovery_verified_and_reused_without_GPU_repeat'] = (
        smoke.get('optimizer_step') == 32 and smoke.get('status') == 'complete'
        and smoke.get('independent_full_restore_verified') is True
        and smoke.get('independent_prediction_restore_verified') is True
        and complete.get('optimizer_step') == 32 and complete.get('recovery_updates') == 0
        and all(json.loads((ROOT/'reports'/name).read_text()).get('status') == 'passed' for name in
                ['full_state_restore.json', 'checkpoint_prediction_restore.json', 'base_zero_lora_equivalence.json'])
        and all(any(job.get('name') == name and job.get('reused_from_run') == project_path('archive', 'run_20261102_215627_resume_commit32')
                    for job in workflow.get('completed_jobs', [])) for name in ['smoke_full_state_restore', 'smoke_prediction_restore'])
    )
    segment = lineage['segmented_pilot_resume']
    checkpoint = Path(segment['copied_checkpoint'])
    cp_files = segment.get('checkpoint_files', [])
    marker = json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())
    rows = [json.loads(line) for line in (ROOT/'logs/pilot_optimizer.jsonl').read_text().splitlines() if line]
    checks['complete_native500_copy_and_exact_log_prefix'] = (
        checkpoint == ROOT/'checkpoints/E1_wrist_pilot/seed42/500'
        and segment.get('optimizer_step') == 500 and segment.get('total_schedule_steps') == 2000
        and segment.get('segment_stops') == [1000, 1500, 2000]
        and segment.get('source1000_checkpoint_committed') is False
        and segment.get('source_preserved') is True
        and len(cp_files) == 277 and sum(item['bytes'] for item in cp_files) == 5628702062
        and all(Path(item['copy']).is_relative_to(checkpoint) and Path(item['copy']).stat().st_size == item['bytes']
                and streamed_sha256(item['copy']) == item['sha256'] for item in cp_files)
        and bool(marker.get('commit_timestamp_nsecs'))
        and (checkpoint/'params/_METADATA').is_file() and (checkpoint/'train_state/_METADATA').is_file()
        and len(rows) == 500 and [row['optimizer_step'] for row in rows] == list(range(1, 501))
    )
    diagnostic_root = ROOT/'reports/shared_graph_diagnostic'
    shared = json.loads((diagnostic_root/'shared_parameter_identity.json').read_text())
    diagnostic = json.loads((diagnostic_root/'summary.json').read_text())
    comparisons = {name: json.loads((diagnostic_root/(name+'.json')).read_text()) for name in (
        'shared_lora_vs_previous_initial', 'shared_frozen_vs_original_e0', 'shared_lora_vs_shared_frozen')}
    lora = comparisons['shared_lora_vs_previous_initial']
    frozen = comparisons['shared_frozen_vs_original_e0']
    cross = comparisons['shared_lora_vs_shared_frozen']
    checks['diagnostic_shared_51_base_arrays_and_10_zero_B'] = (
        shared.get('common_parameter_count') == 51 and shared.get('adapter_parameter_count') == 20
        and shared.get('all_common_arrays_shared') is True and shared.get('base_restore_calls') == 1
        and len(shared.get('common_parameters', [])) == 51
        and all(row.get('python_object_identity') or row.get('device_buffer_identity') for row in shared['common_parameters'])
        and diagnostic.get('adapter_initialization', {}).get('lora_b_tensors') == 10
        and diagnostic.get('adapter_initialization', {}).get('all_lora_b_zero') is True
        and diagnostic.get('optimizer_steps_executed') == 0
    )
    checks['same_graph_LoRA_reference_strictly_reproduced'] = (
        lora.get('status') == 'passed' and lora.get('atol') == lora.get('rtol') == 1e-5
        and all(lora['checks'][key]['passed'] and lora['checks'][key]['max_abs_error'] == 0
                for key in ['prediction_normalized', 'fm_numerator', 'fm_count'])
    )
    checks['legacy_Frozen_prediction_only_reproduced_FM_difference_retained'] = (
        frozen.get('status') == 'failed' and diagnostic.get('references_reproduced') is False
        and frozen['checks']['prediction_normalized']['max_abs_error'] == 0
        and frozen['checks']['fm_count']['max_abs_error'] == 0
        and frozen['checks']['fm_numerator']['passed'] is False
        and frozen['checks']['fm_numerator']['max_abs_error'] > 0
        and cross.get('status') == 'failed' and cross['checks']['prediction_normalized']['max_abs_error'] > 1e-5
    )
    diagnostic_gate = {'status': 'passed' if all(checks[key] for key in (
        'diagnostic_shared_51_base_arrays_and_10_zero_B', 'same_graph_LoRA_reference_strictly_reproduced',
        'legacy_Frozen_prediction_only_reproduced_FM_difference_retained')) else 'failed',
        'scope': 'Separate recorded GPU comparisons; not a claim that every legacy reference metric reproduced',
        'source_root': str(diagnostic_root), 'references_reproduced_global': diagnostic.get('references_reproduced'),
        'comparisons': comparisons, 'new_baseline': 'fresh same-training-graph zero-LoRA E0, never legacy pure Frozen'}
    (ROOT/'reports/same_graph_diagnostic_gate.json').write_text(json.dumps(diagnostic_gate, ensure_ascii=False, indent=2)+'\n')
    native_cpu = json.loads((ROOT/'reports/cacheguard_native_cpu.json').read_text())
    checks['bounded_native_loader_value_dtype_and_serialization'] = (
        native_cpu.get('non_loader_make_config_ast_identical_to_previous') is True
        and native_cpu.get('restore_type') == 'numpy.ndarray'
        and native_cpu.get('custom_tensorstore_or_gpu_first_used') is False
        and len(native_cpu.get('native_source_identities', [])) == 3
        and all(item.get('identical') is True for item in native_cpu.get('native_source_identities', []))
        and len(native_cpu.get('cases', [])) == 2
        and all(item.get('values_shapes_and_mixed_dtypes_exact') is True for item in native_cpu.get('cases', []))
        and native_cpu.get('handler_restore_concurrent_bytes') == 6000000000
        and native_cpu.get('max_native_read_batch_size') == 1
        and native_cpu.get('max_in_flight_native_reads') == 1
        and native_cpu.get('per_leaf_target_dtype_at_native_read') is True
    )
    hook = native_cpu.get('maintenance_hook_cpu', {})
    checks['fresh_native_maintenance_hook_order_and_no_OS_writes'] = (
        hook.get('status') == 'passed' and hook.get('production_helper_imported') is False
        and hook.get('os_cache_writes') == 0 and len(hook.get('maintenance_calls', [])) == 6
        and all(call.get('completed_leaves') == 3 and call.get('os_cache_writes') == 0
                for call in hook.get('maintenance_calls', []))
        and hook.get('jax_cpu_transfer_pairs') == 2
        and hook.get('non_maintenance_loader_ast_identical') is True
    )
    cache = ROOT/'cache/datasets/parquet/default-f9c7b8e1f3bff9e2/0.0.0/44f416866d3c9e08988351a4e721e2a50379e82d7ca68262da231acd39e318fe'
    cache_manifest = json.loads((ROOT/'manifests/cache_reuse.json').read_text())
    old_cache = Path(cache_manifest['source_root'])/'cache/datasets'/cache.relative_to(ROOT/'cache/datasets')
    cache_files = [f'parquet-train-{index:05d}-of-00011.arrow' for index in range(11)] + ['dataset_info.json']
    checks['reuse_completed_train_arrow_cache'] = (
        Path(os.environ.get('HF_DATASETS_CACHE', '')).absolute() == ROOT/'cache/datasets'
        and all((cache/name).is_file() and (cache/name).stat().st_size == (old_cache/name).stat().st_size for name in cache_files)
        and sum((cache/name).stat().st_size for name in cache_files) == 5659005114
        and json.loads((cache/'dataset_info.json').read_text())['splits']['train']['num_examples'] == 8011
        and not any('.incomplete' in path.name or path.suffix == '.lock' for path in cache.iterdir())
    )
    completed_cache_files = cache_manifest.get('files', [])
    checks['reuse_completed_train_and_validation_caches'] = (
        cache_manifest.get('status') == 'copied_completed_files'
        and cache_manifest.get('destination_cache') == str(ROOT/'cache/datasets')
        and sorted(cache_manifest.get('dataset_rowcounts', [])) == [2754, 8011]
        and len(completed_cache_files) == 17
        and sum(item['bytes'] for item in completed_cache_files) == 7603798421
        and cache_manifest.get('raw_data_modified') is False
        and cache_manifest.get('old_cache_modified') is False
        and cache_manifest.get('lock_or_incomplete_files_copied') is False
        and all(
            Path(item['copy']).is_relative_to(ROOT/'cache/datasets')
            and '.incomplete' not in Path(item['copy']).name and Path(item['copy']).suffix != '.lock'
            and Path(item['copy']).stat().st_size == Path(item['source']).stat().st_size == item['bytes']
            and item.get('same_size') is True for item in completed_cache_files
        )
    )
    configs = {}
    for stage in ['smoke', 'pilot']:
        config = make_config(experiment=stage)
        if stage == 'pilot':
            checks['pilot_total_schedule_remains2000'] = config.num_train_steps == 2000 and config.lr_schedule.decay_steps == 2000
        path = ROOT/'configs'/f'{stage}_declared.json'
        path.write_text(json.dumps(dataclasses.asdict(config), ensure_ascii=False, indent=2, default=str)+'\n')
        configs[stage] = str(path)
    e0_declared = {'baseline_kind': 'E0_same_graph_zero_update', 'graph_kind': 'zero_lora_same_as_training',
                   'construction': 'native make_config(smoke) init seed42 step0 then all-BF16 nnx.merge eval',
                   'model': dataclasses.asdict(make_config('smoke').model), 'optimizer_updates': 0,
                   'validation_blocks_reused_from_source': 60, 'new_E0_execution': False,
                   'source_root': reused_e0.get('source_root'), 'legacy_Frozen_used_as_baseline': False}
    (ROOT/'configs/E0_samegraph_declared.json').write_text(json.dumps(e0_declared, ensure_ascii=False, indent=2)+'\n')
    configs['E0_samegraph'] = str(ROOT/'configs/E0_samegraph_declared.json')
    protocol = {
        'experiment': 'E1_wrist_pilot_segmented_from_committed500', 'scope': 'D0 exploratory offline single-task ordinary LoRA',
        'candidate_root': project_path('candidate', ''),
        'source_data_mutation': False, 'robot_commands': False,
        'base_params': project_path('base_params', ''),
        'ordinary_lora': {'rank': 16, 'alpha': 32, 'dropout': 0, 'zero_init_B': True, 'CL_banks': False},
        'training': {'actual_batch': 1, 'accumulation': 4, 'effective_batch': 4, 'seed': 42,
                     'smoke_optimizer_steps': 32, 'smoke_steps_already_committed_in_source': 32,
                     'smoke_updates_to_execute_this_run': 0, 'pilot_optimizer_steps': 2000,
                     'pilot_starts_from': 'original base trajectory; full-state resumed committed500, never the smoke checkpoint',
                     'lora_peak_lr': 1e-4, 'head_peak_lr': 1e-5, 'warmup': 100,
                     'optimizer': 'AdamW b1=.9,b2=.95,eps=1e-8,weight_decay=0; global clip1',
                     'checkpoint_steps': [500, 1000, 1500, 2000], 'ema': False, 'teacher': False},
        'loss': {'target': 'native flow velocity noise-actions', 'valid_action_dimensions': 7,
                 'model_action_dimensions': 32, 'action_horizon': 50,
                 'invalid_GT_inputs': 'canonicalized to zero before forward; original full noise/attention retained',
                 'mask': 'real-dimension mask AND native episode-boundary time mask',
                 'accumulation': 'sum numerator gradients / sum valid elements across four microbatches',
                 'zero_or_nonfinite_update': 'skip params/Adam/step advancement and fail the training branch'},
        'camera': {'mode': 'wrist_only', 'base_0_rgb': False, 'left_wrist_0_rgb': True, 'right_wrist_0_rgb': False,
                   'resize': 'native 224x224 with padding', 'rotation': 'none', 'external_image_used': False},
        'state': {'real_dimensions': 28, 'tokens': 'train-only quantile normalized then state tokens',
                  'continuous_shape_after_padding': 32, 'token_limit': 200},
        'data_quality': {'manual_review_pending': 107, 'external_sync_status': 'UNVERIFIED',
                         'wrist_sync_status': 'UNVERIFIED', 'controller_action_API': 'UNKNOWN'},
        'validation': {'plan_sha256': plan['plan_sha256'], 'checkpoint_selection': plan['selection'],
                       'test_evaluation': 'reserved; no model scoring/selection on test in this pilot',
                       'E0_execution': 'Explicit SHA-verified reuse of source205332 same-graph deterministic E0 (60 blocks,71 BF16 hashes); not re-executed here',
                       'E0_source_root': reused_e0.get('source_root'),
                       'E0_factory': 'zero_lora_baseline.build_zero_lora_baseline(hash_parameters=True); native smoke init, no optimizer updates, full BF16 view',
                       'required_pre_prediction_parameter_identity': 'E0 and independent smoke each hash all71 BF16 inference leaves (51 base+20 adapters), exact equality before strict prediction comparison',
                       'E0_legacy_source_root_historical_only': inherited_e0.get('source_root'),
                       'legacy_input_and_noise_sha256': inherited_e0.get('validation_input_and_noise_sha256'),
                       'strict_same_graph_prediction_tolerance': {'atol': 1e-5, 'rtol': 1e-5},
                       'E1_improvement_reference': 'Only newly executed same-graph E0; legacy pure Frozen is excluded',
                       'E0_source_identity': 'Untouched source files and numerical-function AST checked; load_model and weight_loader construction explicitly changed',
                       'loader_change_equivalence': 'Bounded native CPU value/target-dtype/serialization tests plus full-base zero-LoRA fixed-input GPU prediction acceptance remain required'},
        'resource_policy': policy,
        'segmented_pilot': {'source_checkpoint': segment['source_checkpoint'],
                            'copied_checkpoint': str(checkpoint), 'initial_step': 500,
                            'segment_stops': [1000, 1500, 2000], 'total_schedule_steps': 2000,
                            'updates_to_execute_this_run': 1500, 'source_uncommitted_suffix': [501, 1000],
                            'dataset_replay': 'Every segment replays initial_optimizer_step*4 seeded microbatches before next update',
                            'checkpoint_copy_validation': 'All277 copied files SHA256 checked with bounded8MiB reads; exact first500 optimizer rows'},
        'smoke_recovery': {'source_root': recovery_audit.get('source_root'),
                          'source_checkpoint': recovery_audit.get('checkpoint'),
                          'original_training_updates': 32, 'new_smoke_updates': 0,
                          'order': 'reuse already verified source32 recovery; no repeated GPU smoke. Resume full-state pilot500 in three500-update processes',
                          'original_post_training_RAM_hashes_available': False,
                          'evidence_boundary': 'First final snapshot is recovered from committed checkpoint. Two restores can prove repeated restoration equality, not original-live-RAM-to-save equivalence.'},
        'deterministic_execution': {'XLA_FLAGS': os.environ.get('XLA_FLAGS'),
                                    'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                                    'JAX_DEFAULT_MATMUL_PRECISION': os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                                    'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
                                    'scope': 'Declared execution configuration; CPU parse/proofs do not establish full GPU determinism. Fresh E0,71 BF16 hashes and strict1e-5 prediction/restore acceptance remain mandatory'},
        'loading': {'mode': 'Native Orbax CPU restore with per-leaf RestoreArgs.dtype before NumPy read; private SequentialNumpyHandler delegates one leaf at a time to native NumpyHandler.deserialize, then native merge/init_train_state',
                    'checkpointer': 'ocp.Checkpointer(ocp.PyTreeCheckpointHandler(restore_concurrent_gb=6,type_handler_registry=private_registry))',
                    'serialization': 'wrapper lock plus one-leaf native deserialize; pinned native NumPy handler does not consume the byte limiter, so 6GB declaration is not the actual memory cap',
                    'evaluation': 'same native CPU direct-target-dtype restore, then config.model.load; no custom TensorStore data reader or GPU-first/DUS restore',
                    'tensorstore_context': '16MiB shared cache/coalescing, I/O=1, copy=1; native handler performs data reads',
                    'memory_scope': 'Full target-dtype host tree may exist; per-leaf/read limits and allocator fraction are not process/Windows commit peak guarantees',
                    'cpu_evidence': 'reports/cacheguard_native_cpu.json',
                    'prior_bounded_cpu_evidence': 'reports/bounded_native_weights_cpu.json',
                    'changed_loader_not_certified_by': 'Historical chunked/one-leaf CPU tests',
                    'legacy_restore_budget_correction': 'Historical PyTreeCheckpointer(handler) first positional argument was primary_host; old custom restore-budget claims do not certify this native path',
                    'checkpoint_save': 'CheckpointManager item_handlers save interface is separate and unchanged'},
        'pre_transfer_clean_cache_maintenance': {'enabled': True,
                     'when': 'After full native CPU restore/release, before native merge/GPU init or evaluation device_put',
                     'scope': 'Invoked only in current own exclusive model child; Linux-wide transient clean pagecache reclaim, not persistent OS/pagefile/WSL/swap changes',
                     'guards': 'No startup/runtime threshold reduction; any actual model/maintenance error preserves evidence and stops branch',
                     'cpu_evidence': 'reports/memory_maintenance_cpu.json',
                     'result_claim': 'CPU mocks/hook order are not actual reclaim, GPU peak or optimizer evidence'},
        'completed_train_arrow_cache': {'source': str(old_cache), 'copy': str(cache),
                                        'files': 12, 'bytes': 5659005114, 'train_rows': 8011,
                                        'scope': 'Independent new copy of completed Arrow+dataset_info only; no old lock/incomplete files; metadata/size checks, no full payload hash'},
        'completed_cache_reuse': {'source_root': cache_manifest.get('source_root'),
                                 'manifest': 'manifests/cache_reuse.json', 'files': 17, 'bytes': 7603798421,
                                 'rowcounts': [8011, 2754], 'validation_cache_is_not_new_model_scoring': True},
        'phase_memory': {'cpu_evidence': 'reports/phase_memory_cpu.json',
                         'measurements': 'stdlib /proc RSS/PSS and saved Windows commit/private counters; sampled maxima are observations, not theoretical phase peak bounds',
                         'system_setting_changes': False},
        'idle_clean_pagecache_reclaim': {'enabled': True, 'cpu_evidence': 'reports/idle_reclaim_cpu.json',
                                       'scope': 'One guarded idle attempt per execute phase before admission; preserve 16GiB/30s and runtime thresholds; no promised memory recovery'},
        'parameter_hashing': 'C-order leading-axis slices, target64MiB; largest individual existing scan row128MiB',
    }
    (ROOT/'configs/experiment.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    result = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'configs': configs, 'gpu_smoke_executed': False, 'real_pi05_optimizer_steps': 0,
              'scope': 'CPU preparation only. Verified E0 and smoke32 independent recovery are reused, with no repeated GPU validation. Resume complete copied pilot500 in three500-update processes to total2000; unsaved source501..1000 are not current progress. No model/optimizer executed by this script.',
              'legacy_E0_historical_only': {'source_root': inherited_e0.get('source_root'), 'records_completed': e0.get('records_completed'),
                               'selection_score': inherited_e0.get('selection_score'), 'used_as_current_baseline': False},
              'continuation_provenance': str(ROOT/'manifests/continuation_provenance.json'),
              'old_telemetry_qualification': 'Initial resource probe refreshed old windows_current cache only; old failed workflow/logs/checkpoints/config/source and protected manifests/results preserved'}
    (ROOT/'reports/cpu_preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
