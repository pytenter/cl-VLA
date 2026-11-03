"""Use the native replay loop with ordinary-LoRA loss and initialization.

No task-specific bank, CL gate, bank update mask, teacher or KD is introduced.
The current loss calls compute_loss with precisely the native ordinary RNG;
replay uses a separate fold_in(...,99991) stream and the same compute_loss.
"""
import dataclasses
import json
import math
import numbers
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import optax


def validate_plain_config(config):
    if float(config.lambda_kd) != 0 or config.ema_decay is not None:
        raise ValueError('Ordinary replay requires lambda_kd=0 and EMA disabled')
    model = config.model
    if (model.paligemma_variant, model.action_expert_variant) != ('gemma_2b_lora', 'gemma_300m_lora'):
        raise ValueError('This update is for ordinary LoRA, not CL-LoRA')
    for field in ('cl_pali_lora_bank', 'cl_action_lora_bank', 'cl_pali_block_weight_bank',
                  'cl_action_block_weight_bank', 'cl_freeze_lora_a', 'cl_freeze_shared_lora',
                  'cl_freeze_pali_lora_b', 'cl_freeze_action_lora_b'):
        if getattr(model, field, False):
            raise ValueError(f'Unexpected CL option {field}')


def train_step_plain_replay(config, replay_task_id, rng, state, teacher_params, batch, replay_batch, replay_on):
    """Native-loop-compatible signature; replay_task_id is dataset metadata."""
    del replay_task_id
    validate_plain_config(config)
    if teacher_params is not None:
        raise ValueError('A teacher must not be constructed for ordinary replay')
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(local_model, local_rng, observation, actions):
        # Exact same call and RNG as scripts.train.train_step, including the
        # native Pi0.compute_loss split(rng,3), preprocessing and flow target.
        loss_task = jnp.mean(local_model.compute_loss(local_rng, observation, actions, train=True))
        if config.rehearsal.enabled:
            def replay_loss(_):
                replay_rng = jax.random.fold_in(local_rng, 99991)
                replay_obs, replay_actions = replay_batch
                return jnp.mean(local_model.compute_loss(replay_rng, replay_obs, replay_actions, train=True))
            loss_rehearsal = jax.lax.cond(replay_on, replay_loss,
                                        lambda _: jnp.array(0., dtype=loss_task.dtype), operand=None)
            replay_weight = jnp.asarray(config.rehearsal.loss_weight, dtype=loss_task.dtype)
        else:
            loss_rehearsal = jnp.array(0., dtype=loss_task.dtype)
            replay_weight = jnp.array(0., dtype=loss_task.dtype)
        total = loss_task + replay_weight * loss_rehearsal
        return total, {'loss_task': loss_task, 'loss_rehearsal': loss_rehearsal,
                       'loss_total': total, 'loss_kd': jnp.array(0., dtype=loss_task.dtype),
                       'lambda_kd': jnp.array(0., dtype=loss_task.dtype)}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (_, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions)
    trainable = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable)
    nnx.update(model, optax.apply_updates(trainable, updates))
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state)
    info['grad_norm'] = optax.global_norm(grads)
    return new_state, info


_ORIGINAL_INIT = None
_ORIGINAL_STEP = None
_ORIGINAL_LOG = None
_METRICS_PATH = None
_ORIGINAL_WANDB_INIT = None


def _install_metrics_sink(replay_train, metrics_path):
    """Mirror the actual native reduced_info, then forward the original call."""
    global _ORIGINAL_LOG, _METRICS_PATH
    path = Path(metrics_path)
    if getattr(replay_train.wandb.log, '_plain_replay_log', False) is True:
        if path != _METRICS_PATH:
            raise ValueError('An ordinary-replay process already has a different metrics sink')
        return
    original_log = replay_train.wandb.log
    _ORIGINAL_LOG, _METRICS_PATH = original_log, path
    path.parent.mkdir(parents=True, exist_ok=True)

    def log_metrics(data, *args, **kwargs):
        converted = {}
        finite = True
        for key, value in data.items():
            scalar = value.item() if hasattr(value, 'item') else value
            if not isinstance(scalar, numbers.Real):
                raise TypeError(f'Expected scalar numeric native metric {key}, got {type(scalar)}')
            if isinstance(scalar, numbers.Integral):
                converted[str(key)] = int(scalar)
            elif math.isfinite(float(scalar)):
                converted[str(key)] = float(scalar)
            else:
                converted[str(key)] = str(float(scalar))
                finite = False
        step = kwargs.get('step', args[0] if args else None)
        step = None if step is None else int(step)
        record = {**converted, 'step': step, 'optimizer_step': None if step is None else step + 1,
                  'finite': finite, 'source': 'native replay loop reduced_info; step is its zero-based loop index'}
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        if not finite:
            raise FloatingPointError(f'Non-finite native metrics recorded at {path}, step={step}')
        return original_log(data, *args, **kwargs)

    log_metrics._plain_replay_log = True
    replay_train.wandb.log = log_metrics


def _install_metrics_runtime(replay_train, metrics_path):
    # wandb.init may overwrite module-level wandb.log.  Reattach AFTER native
    # init_wandb finishes, rather than silently losing the mirror at startup.
    global _ORIGINAL_WANDB_INIT
    if getattr(replay_train.init_wandb, '_plain_replay_wandb_init', False) is not True:
        original_init = replay_train.init_wandb
        _ORIGINAL_WANDB_INIT = original_init
        def init_wandb(*args, **kwargs):
            result = original_init(*args, **kwargs)
            _install_metrics_sink(replay_train, metrics_path)
            return result
        init_wandb._plain_replay_wandb_init = True
        replay_train.init_wandb = init_wandb
    _install_metrics_sink(replay_train, metrics_path)


def install_training_fixes(metrics_path=None):
    """Patch only the current process's generic replay runtime functions."""
    from scripts import train as ordinary_train
    from scripts import train_cl_lora as replay_train
    global _ORIGINAL_INIT, _ORIGINAL_STEP
    if getattr(replay_train.init_train_state, '_plain_replay_init', False) is True:
        if metrics_path is not None:
            _install_metrics_runtime(replay_train, metrics_path)
        return replay_train
    _ORIGINAL_INIT, _ORIGINAL_STEP = replay_train.init_train_state, replay_train.train_step_cl_lora

    def init_train_state(config, init_rng, mesh, *, resume):
        validate_plain_config(config)
        if replay_train._teacher_required(config):
            raise ValueError('Teacher-dependent schedules/representation KD are forbidden')
        state, shardings = ordinary_train.init_train_state(config, init_rng, mesh, resume=resume)
        return state, shardings, None, None

    init_train_state._plain_replay_init = True
    replay_train.init_train_state = init_train_state
    replay_train.train_step_cl_lora = train_step_plain_replay
    if metrics_path is not None:
        _install_metrics_runtime(replay_train, metrics_path)
    return replay_train


def uninstall_training_fixes():
    global _ORIGINAL_LOG, _METRICS_PATH, _ORIGINAL_WANDB_INIT
    from scripts import train_cl_lora as replay_train
    if _ORIGINAL_INIT is not None and getattr(replay_train.init_train_state, '_plain_replay_init', False) is True:
        replay_train.init_train_state = _ORIGINAL_INIT
        replay_train.train_step_cl_lora = _ORIGINAL_STEP
    if _ORIGINAL_LOG is not None and getattr(replay_train.wandb.log, '_plain_replay_log', False) is True:
        replay_train.wandb.log = _ORIGINAL_LOG
    if _ORIGINAL_WANDB_INIT is not None and getattr(replay_train.init_wandb, '_plain_replay_wandb_init', False) is True:
        replay_train.init_wandb = _ORIGINAL_WANDB_INIT
    _ORIGINAL_LOG, _METRICS_PATH = None, None
    _ORIGINAL_WANDB_INIT = None
