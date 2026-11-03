"""Behavior checks for the task-bank and freezing functionality being deployed."""
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.lora import Einsum, LoRAConfig


def fixture():
    module = Einsum((8, 8), lora_config=LoRAConfig(
        rank=2, alpha=4, use_orthogonal_init=True,
        use_task_lora_b_bank=True, task_bank_size=4, freeze_lora_a=True))
    x = jax.random.normal(jax.random.key(1), (4, 8))
    params = module.init(jax.random.key(2), 'bi,ij->bj', x,
                         task_id=1, use_task_bank=True)['params']
    params['lora_b_bank'] = .1 * jax.random.normal(jax.random.key(3), params['lora_b_bank'].shape)
    return module, params, x


def test_current_task_bank_update_preserves_other_tasks_and_frozen_a():
    module, params, x = fixture()
    def predict(p, task):
        return module.apply({'params': p}, 'bi,ij->bj', x, task_id=task, use_task_bank=True)
    def loss(p):
        return jnp.mean((predict(p, 1) - 1.) ** 2)
    old_prediction = predict(params, 0)
    current_prediction = predict(params, 1)
    grads = jax.grad(loss)(params)
    np.testing.assert_array_equal(grads['lora_a'], jnp.zeros_like(grads['lora_a']))
    np.testing.assert_array_equal(grads['lora_b'], jnp.zeros_like(grads['lora_b']))
    for task in [0, 2, 3]:
        np.testing.assert_array_equal(grads['lora_b_bank'][task], jnp.zeros_like(grads['lora_b_bank'][task]))
    assert float(jnp.linalg.norm(grads['lora_b_bank'][1])) > 0
    updated = dict(params)
    updated['lora_b_bank'] = params['lora_b_bank'] - .01 * grads['lora_b_bank']
    np.testing.assert_array_equal(predict(updated, 0), old_prediction)
    assert not np.allclose(predict(updated, 1), current_prediction)


def test_frozen_replay_task_bank_blocks_bank_gradients():
    module, params, x = fixture()
    def loss(p):
        out = module.apply({'params': p}, 'bi,ij->bj', x,
                           task_id=0, use_task_bank=True, freeze_b=True)
        return jnp.mean((out - 1.) ** 2)
    grads = jax.grad(loss)(params)
    np.testing.assert_array_equal(grads['lora_b_bank'], jnp.zeros_like(grads['lora_b_bank']))
