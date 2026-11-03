"""Small CPU checkpoints only; compare against the successful LIBERO loader."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import mock

from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from openpi.models import model as native_model
from openpi.training import weight_loaders
from native_weights import CompleteWeightLoader


def old_loader_class(source):
    text = Path(source).read_text()
    tree = ast.parse(text)
    definition = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == 'CompleteWeightLoader')
    namespace = {'dataclasses': dataclasses, 'np': np,
                 'model_lib': native_model, 'weight_loaders': weight_loaders}
    # Execute the actual historical class without importing unrelated LIBERO data hooks.
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['CompleteWeightLoader'], hashlib.sha256(ast.get_source_segment(text, definition).encode()).hexdigest()


def source_identities(successful_root, used_root):
    rows = []
    for relative in ('scripts/train.py', 'src/openpi/models/model.py',
                     'src/openpi/training/weight_loaders.py'):
        successful = Path(successful_root) / relative
        used = Path(used_root) / relative
        a, b = hashlib.sha256(successful.read_bytes()).hexdigest(), hashlib.sha256(used.read_bytes()).hexdigest()
        assert a == b, f'Native source differs: {relative}'
        rows.append({'relative_path': relative, 'successful_source': str(successful),
                     'used_source': str(used), 'sha256': a, 'identical': True})
    return rows


def protocol_unchanged(previous_config, new_config):
    functions = []
    for path in (previous_config, new_config):
        definition = next(node for node in ast.parse(Path(path).read_text()).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'make_config')
        definition = copy.deepcopy(definition)
        for call in ast.walk(definition):
            if isinstance(call, ast.Call):
                call.keywords = [kw for kw in call.keywords if kw.arg != 'weight_loader']
        functions.append(ast.dump(definition, include_attributes=False))
    assert functions[0] == functions[1], 'Non-loader make_config fields changed'


def run(args):
    assert all(device.platform == 'cpu' for device in jax.devices())
    old_loader, historical_class_sha = old_loader_class(args.old_loader_source)
    identities = source_identities(args.successful_source_root, args.used_source_root)
    protocol_unchanged(args.previous_config, Path(__file__).with_name('kinova_config.py'))
    original = {'PaliGemma': {'kernel': np.arange(35, dtype=np.float32).reshape(5, 7) / 13},
                'action_in_proj': {'kernel': np.arange(15, dtype=np.float32).reshape(3, 5) / 11},
                'scalar': np.asarray(.25, np.float32)}
    expected = {'PaliGemma': {'kernel': jax.ShapeDtypeStruct((5, 7), jnp.bfloat16),
                            'lora_a': jax.ShapeDtypeStruct((3, 2), jnp.float32),
                            'lora_b': jax.ShapeDtypeStruct((2, 4), jnp.float32)},
                'action_in_proj': {'kernel': jax.ShapeDtypeStruct((3, 5), jnp.float32)},
                'scalar': jax.ShapeDtypeStruct((), jnp.bfloat16),
                'cl_block_weights_0': jax.ShapeDtypeStruct((2,), jnp.float32)}
    flat_original, flat_expected = map(traverse_util.flatten_dict, (original, expected))
    cases = []
    with tempfile.TemporaryDirectory(prefix='kinova_native_weights_cpu_') as directory:
        for wrapped in (False, True):
            path = Path(directory) / ('value_schema' if wrapped else 'pure_schema')
            item = {'params': jax.tree.map(lambda value: {'value': value}, original) if wrapped else original}
            with ocp.PyTreeCheckpointer() as saver:
                saver.save(path, item)
            with mock.patch.object(native_model, 'restore_params', wraps=native_model.restore_params) as restore:
                old = old_loader(str(path)).load(expected)
                new = CompleteWeightLoader(str(path)).load(expected)
                assert restore.call_count == 2
                for call in restore.call_args_list:
                    assert call.kwargs == {'restore_type': np.ndarray}
            old_flat, new_flat = map(traverse_util.flatten_dict, (old, new))
            assert set(old_flat) == set(new_flat) == set(flat_expected)
            loaded_names, placeholder_names = [], []
            for name, target in flat_expected.items():
                a, b = old_flat[name], new_flat[name]
                assert a.dtype == b.dtype == target.dtype and a.shape == b.shape == target.shape
                if name in flat_original:
                    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(b, flat_original[name].astype(target.dtype))
                    loaded_names.append('/'.join(name))
                else:
                    assert isinstance(a, jax.ShapeDtypeStruct) and a is target and b is target
                    placeholder_names.append('/'.join(name))
            cases.append({'schema': path.name, 'native_numpy_restore_calls': 2,
                          'loaded_names': loaded_names, 'missing_shape_placeholders': placeholder_names,
                          'values_shapes_and_mixed_dtypes_exact': True})
    result = {'status': 'passed', 'devices': [str(device) for device in jax.devices()],
              'scope': 'Tiny CPU checkpoints only; no full base restore, no model construction, no GPU, no optimizer update',
              'historical_loader_source': args.old_loader_source,
              'historical_class_source_sha256': historical_class_sha,
              'missing_regex': r'.*(?:lora|cl_block_weights).*',
              'native_source_identities': identities,
              'non_loader_make_config_ast_identical_to_previous': True,
              'restore_type': 'numpy.ndarray', 'custom_tensorstore_or_gpu_first_used': False,
              'missing_cl_placeholder_scope': 'Regex compatibility test only; Kinova architecture has no CL banks',
              'cases': cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-loader-source', default=project_path('libero_fourway', 'scripts/experiment_config.py'))
    parser.add_argument('--successful-source-root', default=project_path('libero_fourway', 'source/pi05'))
    parser.add_argument('--used-source-root', default=project_path('deploy', 'source/pi05'))
    parser.add_argument('--previous-config', default=project_path('archive', 'run_20261102_174357_chunked/scripts/kinova_config.py'))
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
