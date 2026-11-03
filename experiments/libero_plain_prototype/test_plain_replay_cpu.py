"""Tiny CPU regression checks for ordinary-LoRA replay, never a model smoke."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import dataclasses
import functools
import json
from pathlib import Path
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training.utils import TrainState
from scripts import train as ordinary_train
from scripts import train_cl_lora as replay_train
from experiment_config import make_config, OLD_ROOT, BASE_PARAMS
from plain_replay_training import (install_training_fixes, train_step_plain_replay,
                                   uninstall_training_fixes, validate_plain_config)


class TinyFlow(model_lib.BaseModel):
    def __init__(self):
        super().__init__(action_dim=3, action_horizon=2, max_token_len=4)
        self.lora_a = nnx.Param(jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 13)
        self.action_in_proj = nnx.Param(jnp.eye(3, dtype=jnp.float32) * .2)
        self.frozen = nnx.Param(jnp.array([.5, -.25, .125], dtype=jnp.bfloat16))
        self.deterministic = True

    def compute_loss(self, rng, observation, actions, *, train=False):
        # Every RNG split influences this objective, so the test detects a
        # nested-two split or altered current-task augmentation stream.
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observed = observation.state + .05 * jax.random.normal(preprocess_rng, observation.state.shape)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001
        noisy = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        velocity = observed[:, None, :] @ self.lora_a.value + noisy @ self.action_in_proj.value + self.frozen.value
        return jnp.mean(jnp.square(velocity - (noise - actions)), axis=-1)

    def sample_actions(self, rng, observation, **kwargs):
        raise NotImplementedError('This tiny test never samples a policy')


def assert_state_equal(left, right, *, exact=True):
    assert int(left.step) == int(right.step)
    for name in ('params', 'opt_state'):
        a, a_structure = jax.tree.flatten(getattr(left, name))
        b, b_structure = jax.tree.flatten(getattr(right, name))
        assert a_structure == b_structure
        for x, y in zip(a, b):
            assert x.shape == y.shape and x.dtype == y.dtype
            if exact:
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            else:
                np.testing.assert_allclose(np.asarray(x).astype(np.float64), np.asarray(y).astype(np.float64),
                                           rtol=2e-6, atol=1e-8)


def run_test():
    assert all(device.platform == 'cpu' for device in jax.devices())
    model_config = Pi0Config(pi05=True, paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    config = config_lib.TrainConfig(name='cpu_plain_replay', exp_name='cpu', model=model_config,
        freeze_filter=nnx_utils.PathRegex('frozen'), ema_decay=None, lambda_kd=0.,
        rehearsal=config_lib.RehearsalConfig(enabled=True, loss_weight=.5))
    model = TinyFlow()
    graph, params = nnx.split(model)
    tx = optax.sgd(.01)
    state = TrainState(step=jnp.int32(7), params=params, model_def=graph,
                       opt_state=tx.init(params.filter(config.trainable_filter)), tx=tx,
                       ema_decay=None, ema_params=None)
    observation = model_lib.Observation(
        images={key: jnp.zeros((2, 2, 2, 3), jnp.float32)
                for key in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')},
        image_masks={key: jnp.ones(2, bool) for key in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')},
        state=jnp.array([[.5, -.2, .7], [.1, .4, -.3]], jnp.float32))
    batch = observation, jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 20
    replay_observation = dataclasses.replace(observation, state=-observation.state * 2)
    replay_batch = replay_observation, jnp.ones((2, 2, 3), jnp.float32) * 1.3
    rng = jax.random.key(42)
    native_step = jax.jit(functools.partial(ordinary_train.train_step, config))
    new_step = jax.jit(functools.partial(train_step_plain_replay, config, 0))
    native_state, native_info = native_step(rng, state, batch)
    off_state, off_info = new_step(rng, state, None, batch, replay_batch, jnp.bool_(False))
    assert_state_equal(native_state, off_state)
    np.testing.assert_array_equal(native_info['loss'], off_info['loss_task'])
    np.testing.assert_array_equal(native_info['grad_norm'], off_info['grad_norm'])
    assert float(off_info['loss_rehearsal']) == 0
    checks = {'replay_off_matches_actual_native_ordinary_update_bitwise': True,
              'current_loss_and_rng_match_native_split3_pipeline': True}

    on_state, on_info = new_step(rng, state, None, batch, replay_batch, jnp.bool_(True))
    train_rng = jax.random.fold_in(rng, state.step)
    replay_rng = jax.random.fold_in(train_rng, 99991)
    manual_model = nnx.merge(graph, params)
    manual_model.train()
    def manual_loss(module):
        current = jnp.mean(module.compute_loss(train_rng, batch[0], batch[1], train=True))
        replay = jnp.mean(module.compute_loss(replay_rng, replay_batch[0], replay_batch[1], train=True))
        return current + .5 * replay, (current, replay)
    (manual_total, (manual_current, manual_replay)), manual_grads = nnx.value_and_grad(
        manual_loss, argnums=nnx.DiffState(0, config.trainable_filter), has_aux=True)(manual_model)
    trainable = params.filter(config.trainable_filter)
    updates, opt_state = tx.update(manual_grads, state.opt_state, trainable)
    nnx.update(manual_model, optax.apply_updates(trainable, updates))
    manual_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(manual_model), opt_state=opt_state)
    # The independent eager manual calculation is not XLA-fused; allow only
    # float32 rounding here.  The native-vs-shim replay-off test above remains
    # a strict bitwise comparison of the actual compiled update/state.
    assert_state_equal(on_state, manual_state, exact=False)
    np.testing.assert_allclose(on_info['loss_total'], manual_total, rtol=1e-7)
    np.testing.assert_allclose(on_info['loss_task'], manual_current, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(on_info['loss_rehearsal'], manual_replay, rtol=2e-6, atol=1e-7)
    assert not np.array_equal(np.asarray(on_state.params['lora_a'].value), np.asarray(off_state.params['lora_a'].value))
    checks['replay_on_uses_current_plus_half_replay_loss_and_gradient'] = True
    checks['replay_changes_ordinary_lora_update'] = True
    other_id = jax.jit(functools.partial(train_step_plain_replay, config, 3))
    other_state, _ = other_id(rng, state, None, batch, replay_batch, jnp.bool_(True))
    assert_state_equal(on_state, other_state)
    checks['old_task_id_is_metadata_not_parameter_bank_routing'] = True
    for candidate in (native_state, off_state, on_state, other_state):
        np.testing.assert_array_equal(candidate.params['frozen'].value, state.params['frozen'].value)
    assert all('bank' not in '/'.join(map(str, path)) and 'block_weights' not in '/'.join(map(str, path))
               for path in state.params.flat_state())
    checks['frozen_parameters_unchanged_and_no_bank_gate_parameters'] = True
    checks['teacher_and_kd_losses_absent'] = bool(float(on_info['loss_kd']) == float(on_info['lambda_kd']) == 0)

    original_init, original_step = replay_train.init_train_state, replay_train.train_step_cl_lora
    sentinel_state, sentinel_sharding = object(), object()
    try:
        with mock.patch.object(ordinary_train, 'init_train_state', return_value=(sentinel_state, sentinel_sharding)) as delegate:
            installed = install_training_fixes()
            returned = installed.init_train_state(config, rng, None, resume=False)
            assert returned == (sentinel_state, sentinel_sharding, None, None)
            delegate.assert_called_once_with(config, rng, None, resume=False)
            assert installed.train_step_cl_lora is train_step_plain_replay
            assert install_training_fixes() is installed
    finally:
        uninstall_training_fixes()
    assert replay_train.init_train_state is original_init and replay_train.train_step_cl_lora is original_step
    checks['initializer_is_exact_ordinary_delegate_no_teacher_no_bank_wrapper'] = True

    cases = {}
    for stage in 'ABCD':
        planned = make_config(stage=stage, weights=BASE_PARAMS)
        validate_plain_config(planned)
        expected_buffers = tuple(str(OLD_ROOT/'buffers/prototype'/f'task_{task}') for task in 'ABCD'[:'ABCD'.index(stage)])
        assert planned.rehearsal.buffer_dirs == expected_buffers
        assert planned.rehearsal.enabled == (stage != 'A')
        assert planned.model.paligemma_variant == 'gemma_2b_lora'
        assert planned.model.action_expert_variant == 'gemma_300m_lora'
        cases[stage] = {'name': planned.name, 'replay_enabled': planned.rehearsal.enabled,
                        'buffers': planned.rehearsal.buffer_dirs, 'checkpoint_dir': str(planned.checkpoint_dir)}
    checks['configs_use_only_learned_fixed_buffers_and_new_checkpoint_namespace'] = True
    return {'status': 'passed', 'devices': [str(device) for device in jax.devices()], 'checks': checks,
            'configuration_cases': cases, 'tiny_loss_current': float(on_info['loss_task']),
            'tiny_loss_replay': float(on_info['loss_rehearsal']), 'tiny_loss_total': float(on_info['loss_total']),
            'scope': 'Tiny NNX model and mocked initialization only; no full pi05 model, real data or GPU',
            'manual_reference_fp32_tolerance': {'rtol': 2e-6, 'atol': 1e-8},
            'full_model_restored': False, 'pi05_optimizer_steps': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run_test()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
