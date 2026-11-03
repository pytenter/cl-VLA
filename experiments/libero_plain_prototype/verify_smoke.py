"""CPU-only verification of the five-step real-model replay smoke evidence."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    import numpy as np
    import orbax.checkpoint as ocp
    from flax import traverse_util

    root = Path(os.environ['PI05_LIBERO_ROOT'])
    name = 'pi05_libero_plain_prototype_B_smoke'
    complete = json.loads((root/'reports'/f'{name}_complete.json').read_text())
    config = json.loads((root/'reports/configurations'/f'{name}.json').read_text())
    metrics = [json.loads(line) for line in (root/'logs'/f'{name}.metrics.jsonl').read_text().splitlines()]
    assert complete['smoke'] is True and complete['steps'] == 5 and complete['batch_size'] == 2
    assert complete['runtime'] == 'native_replay_loop_with_ordinary_compute_loss_and_initializer'
    assert [row['step'] for row in metrics] == list(range(5))
    for row in metrics:
        assert all(math.isfinite(float(row[key])) for key in ['loss_task', 'loss_rehearsal', 'loss_total', 'grad_norm'])
        assert row['grad_norm'] > 0 and row['loss_kd'] == 0 and row['lambda_kd'] == 0
        assert np.isclose(row['loss_total'], row['loss_task'] + .5 * row['loss_rehearsal'], rtol=2e-6, atol=1e-7)
        assert row['loss_rehearsal'] > 0 if row['step'] in [0, 4] else row['loss_rehearsal'] == 0
    source = Path(config['weight_loader']['params_path'])
    assert source == Path(project_path('libero_fourway', 'checkpoints/pi05_libero_plain_A/seed42/10000/params'))
    target = Path(complete['checkpoint_dir'])/'5/params'

    def nested(keys, value):
        for key in reversed(keys):
            value = {key: value}
        return value

    def read_leaf(checkpointer, folder, metadata, keys):
        item = {'params': nested(keys, metadata[keys])}
        args = {'params': nested(keys, ocp.ArrayRestoreArgs(restore_type=np.ndarray))}
        result = checkpointer.restore(folder, ocp.args.PyTreeRestore(item=item, restore_args=args, transforms={}))
        return traverse_util.flatten_dict(result['params'])[keys]

    with ocp.PyTreeCheckpointer() as checkpointer:
        before_meta = traverse_util.flatten_dict(checkpointer.metadata(source)['params'])
        after_meta = traverse_util.flatten_dict(checkpointer.metadata(target)['params'])
        assert before_meta.keys() == after_meta.keys()
        assert not any('lora_b_bank' in '/'.join(keys) or 'cl_block_weights' in '/'.join(keys) for keys in after_meta)
        lora_keys = min((keys for keys in after_meta if 'lora_b' in '/'.join(keys)),
                        key=lambda keys: math.prod(after_meta[keys].shape))
        frozen_keys = min((keys for keys in after_meta if keys[:2] == ('PaliGemma', 'img')),
                          key=lambda keys: math.prod(after_meta[keys].shape))
        before = read_leaf(checkpointer, source, before_meta, lora_keys)
        after = read_leaf(checkpointer, target, after_meta, lora_keys)
        assert np.isfinite(after).all() and not np.array_equal(before, after)
        frozen_before = read_leaf(checkpointer, source, before_meta, frozen_keys)
        frozen_after = read_leaf(checkpointer, target, after_meta, frozen_keys)
        assert np.array_equal(frozen_before, frozen_after)
        parameter_checks = {'sampled_lora_leaf': '/'.join(lora_keys),
                            'lora_before_sha256': hashlib.sha256(before.tobytes()).hexdigest(),
                            'lora_after_sha256': hashlib.sha256(after.tobytes()).hexdigest(),
                            'lora_max_abs_change': float(np.max(np.abs(after-before))),
                            'sampled_frozen_image_leaf': '/'.join(frozen_keys),
                            'sampled_frozen_leaf_unchanged': True,
                            'checkpoint_parameter_leaves': len(after_meta),
                            'parameter_schema_matches_ordinary_lora': True}
    identity = json.loads((root/'manifests/original_readonly_identity.json').read_text())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
               for path, expected in identity['files'].items())
    report = {'status': 'passed', 'actual_pi05_optimizer_steps': 5,
              'replay_enabled_zero_based_steps': [0, 4], 'metrics': metrics,
              'checkpoint': str(target), 'parameter_checks': parameter_checks,
              'original_four_groups_unchanged': True,
              'scope': 'Real B smoke initialized from the old ordinary A checkpoint only for validation; formal A starts fresh from pi05_base'}
    (root/'reports/plain_replay_smoke.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
