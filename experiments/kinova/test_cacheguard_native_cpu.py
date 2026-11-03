"""Existing native tiny tests with a no-OS-write maintenance stub."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('OMP_NUM_THREADS', '1')

import ast
import json
from pathlib import Path
import sys
import types
from unittest import mock


def main():
    root = os.environ.pop('KINOVA_EXPERIMENT_ROOT', None)
    # Missing ROOT also makes the real helper's explicit policy disabled.
    # Additionally inject a stub so the real helper is never imported at all.
    history, maintenance_calls = [], []
    stub = types.ModuleType('memory_maintenance')
    def no_os_maintenance():
        import native_weights
        diagnostics = native_weights.LAST_RESTORE_DIAGNOSTICS
        assert diagnostics is not None and len(diagnostics['events']) == 3
        assert history[-1] == 'native.maintenance.before'
        assert history[-2] == 'native.restore.after'
        maintenance_calls.append({'completed_leaves': 3,
                                  'target_tree_bytes': diagnostics['target_tree_bytes'],
                                  'os_cache_writes': 0})
        return {'status': 'test_stub_disabled_policy', 'cache_reclaim_executed': False}
    stub.reclaim_before_device_transfer = no_os_maintenance
    with mock.patch.dict(sys.modules, {'memory_maintenance': stub}):
        import native_weights
        import test_bounded_native_cpu
        emit = native_weights._emit
        def capture(progress_file, phase, **fields):
            history.append(phase)
            return emit(progress_file, phase, **fields)
        with mock.patch.object(native_weights, '_emit', capture):
            test_bounded_native_cpu.main()
    assert len(maintenance_calls) == 6
    assert history.count('native.device_transfer.before') == 2
    assert history.count('native.device_transfer.after') == 2
    for i, phase in enumerate(history):
        if phase == 'native.device_transfer.before':
            assert history[i-1] == 'native.maintenance.after'
            assert history[i+1] == 'native.device_transfer.after'
    # Verify the entire loader AST is unchanged after removing ONLY new hooks.
    class RemoveMaintenance(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == 'memory_maintenance' else node
        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id == 'maintenance' for target in node.targets):
                return None
            return self.generic_visit(node)
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == '_emit' and len(call.args)>1:
                phase = call.args[1]
                if isinstance(phase, ast.Constant) and isinstance(phase.value, str) and (
                    phase.value.startswith('native.maintenance.') or phase.value == 'native.device_transfer.before'):
                    return None
            return self.generic_visit(node)
    current = Path(native_weights.__file__)
    prior = Path(project_path('archive', 'run_20261102_192657_bounded_native/scripts/native_weights.py'))
    assert ast.dump(RemoveMaintenance().visit(ast.parse(current.read_text())), include_attributes=False) == ast.dump(ast.parse(prior.read_text()), include_attributes=False)
    output = Path(sys.argv[sys.argv.index('--output')+1])
    result = json.loads(output.read_text())
    result['maintenance_hook_cpu'] = {'status': 'passed', 'helper': 'sys.modules stub only',
        'production_helper_imported': False, 'explicit_policy_disabled_by_missing_root': True,
        'os_cache_writes': 0, 'maintenance_calls': maintenance_calls,
        'order': 'all native CPU leaves -> release -> restore.after -> maintenance.before/stub/after -> CPU return or device_transfer.before/after',
        'jax_cpu_transfer_pairs': 2,
        'non_maintenance_loader_ast_identical': True,
        'scope': 'Hook placement and old numerical tests only; no actual cache reclaim, full checkpoint/model or GPU execution'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    if root is not None:
        os.environ['KINOVA_EXPERIMENT_ROOT'] = root
    print(json.dumps({'maintenance_hook_cpu': result['maintenance_hook_cpu'], 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
