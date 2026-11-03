"""Effective-batch flow-matching update with valid-element-weighted accumulation."""
from __future__ import annotations

import dataclasses

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import optax


def group_norms(tree):
    flat = traverse_util.flatten_dict(tree.to_pure_dict())
    groups = {'lora': [], 'heads': []}
    for path, value in flat.items():
        group = 'lora' if 'lora' in '/'.join(str(p) for p in path) else 'heads'
        groups[group].append(jnp.sum(jnp.square(value.astype(jnp.float32))))
    return {group: jnp.sqrt(sum(values, jnp.array(0., dtype=jnp.float32))) for group, values in groups.items()}


def make_update(config):
    """Return a pure update; leading batch axis is the accumulation dimension.

    Gradients of each numerator are summed and divided by the combined count.
    Unequal valid horizons therefore contribute in proportion to their valid
    elements, rather than averaging differently normalized microbatch losses.
    All-invalid and nonfinite updates leave parameters, Adam state and step
    unchanged. The caller must stop if ``update_applied`` is false.
    """
    diff_state = nnx.DiffState(0, config.trainable_filter)

    def update(rng, state, batch):
        trainable = state.params.filter(config.trainable_filter)
        zeros = jax.tree.map(jnp.zeros_like, trainable)
        keys = jax.random.split(jax.random.fold_in(rng, state.step), batch[1].shape[0])

        def micro(carry, inputs):
            num_sum, count_sum, grad_sum = carry
            key, observation, actions = inputs
            model = nnx.merge(state.model_def, state.params)
            model.train()

            def objective(model, key, observation, actions):
                return model.compute_loss_terms(key, observation, actions, train=True)

            (numerator, count), grads = nnx.value_and_grad(
                objective, argnums=diff_state, has_aux=True
            )(model, key, observation, actions)
            return (num_sum + numerator, count_sum + count,
                    jax.tree.map(jnp.add, grad_sum, grads)), None

        (numerator, count, grad_sum), _ = jax.lax.scan(
            micro, (jnp.float32(0), jnp.float32(0), zeros), (keys, batch[0], batch[1])
        )
        grads = jax.tree.map(lambda g: g / jnp.maximum(count, 1), grad_sum)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = state.tx.update(grads, state.opt_state, trainable)
        update_norm = optax.global_norm(updates)
        valid = (count > 0) & jnp.isfinite(numerator) & jnp.isfinite(grad_norm) & jnp.isfinite(update_norm)

        def apply(_):
            model = nnx.merge(state.model_def, state.params)
            nnx.update(model, optax.apply_updates(trainable, updates))
            return dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=opt_state)

        next_state = jax.lax.cond(valid, apply, lambda _: state, operand=None)
        norms = group_norms(grads)
        return next_state, {'loss': numerator / jnp.maximum(count, 1), 'numerator': numerator,
                            'valid_elements': count, 'grad_norm': grad_norm, 'update_norm': update_norm,
                            'lora_grad_norm': norms['lora'], 'heads_grad_norm': norms['heads'],
                            'update_applied': valid}

    return update
