"""Small CPU-only optimizer tests; these are not full-model optimizer smoke steps."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import dataclasses
import json
import os
from pathlib import Path

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from accumulated_update import make_update


class Tiny(nnx.Module):
    def __init__(self):
        self.lora_weight = nnx.Param(jnp.array(1., jnp.float32))
        self.frozen = nnx.Param(jnp.array(2., jnp.float32))

    def compute_loss_terms(self, rng, observation, actions, train=False):
        del rng, train
        difference = self.lora_weight.value * observation['x'] + self.frozen.value - actions
        mask = observation['mask']
        return jnp.sum(jnp.square(difference) * mask), jnp.sum(mask)


@struct.dataclass
class State:
    step: object
    params: object
    model_def: object
    opt_state: object
    tx: object = struct.field(pytree_node=False)


def main():
    model = Tiny()
    graph, params = nnx.split(model)
    trainable_filter = nnx.All(nnx.Param, lambda path, value: 'lora' in '/'.join(map(str, path)))
    config = dataclasses.make_dataclass('Config', [('trainable_filter', object)])(trainable_filter)
    tx = optax.sgd(.01)
    state = State(jnp.int32(0), params, graph, tx.init(params.filter(trainable_filter)), tx)
    observations = {'x': jnp.array([[[1., 2.]], [[3., 4.]]]),
                    'mask': jnp.array([[[1., 1.]], [[1., 0.]]])}
    targets = jnp.zeros((2, 1, 2))
    update = jax.jit(make_update(config))
    new, info = update(jax.random.key(7), state, (observations, targets))
    # Three valid elements: predictions 3,4,5. d(sum(error^2))/dw=2*(3+8+15).
    expected = 1 - .01 * (52 / 3)
    np.testing.assert_allclose(new.params['lora_weight'].value, expected, rtol=1e-6)
    np.testing.assert_allclose(info['loss'], 50/3, rtol=1e-6)
    assert int(new.step) == 1
    assert float(new.params['frozen'].value) == 2.
    invalid = dict(observations, mask=jnp.zeros_like(observations['mask']))
    unchanged, invalid_info = update(jax.random.key(7), new, (invalid, targets))
    assert int(unchanged.step) == 1 and not bool(invalid_info['update_applied'])
    np.testing.assert_array_equal(unchanged.params['lora_weight'].value, new.params['lora_weight'].value)
    broken = targets.at[0, 0, 0].set(jnp.nan)
    rejected, broken_info = update(jax.random.key(7), new, (observations, broken))
    assert int(rejected.step) == 1 and not bool(broken_info['update_applied'])
    result = {'status': 'passed', 'backend': jax.default_backend(), 'weighted_accumulation': True,
              'all_invalid_skips_step': True, 'nonfinite_skips_step': True, 'frozen_parameter_unchanged': True,
              'scope': 'Tiny scalar CPU model only; not Kinova/pi05 optimizer smoke', 'pi05_optimizer_steps': 0}
    root = Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', project_path('kinova', '')))
    (root/'reports/accumulation_unit_tests.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
