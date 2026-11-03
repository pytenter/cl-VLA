"""Process-local, low-host-memory initialization and old-task bank protection.

The flow-matching and replay losses remain native.  A parameter update mask
protects all task-specific bank rows except the current stage, while shared
parameters remain trainable according to the configured freeze filter.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import optax


def _path_parts(path: tuple[Any, ...]) -> tuple[str, ...]:
    parts = []
    for item in path:
        value = getattr(item, "key", getattr(item, "name", getattr(item, "idx", item)))
        parts.append(str(value))
    return tuple(parts)


def bank_axis(path: tuple[Any, ...], shape: tuple[int, ...], num_tasks: int) -> int | None:
    """Find the actual task axis, accounting for Gemma's scanned layer axis."""
    parts = _path_parts(path)
    joined = "/".join(parts)
    if "lora_b_bank" in joined:
        # Linen nn.scan(variable_axes={'params': 0}) prepends depth to
        # attention/FFN bank tensors, making task axis 1 inside `layers`.
        axis = 1 if "layers" in parts else 0
        if len(shape) <= axis or shape[axis] != num_tasks:
            raise ValueError(f"Unexpected LoRA bank shape {shape} at {joined}; expected task axis {axis} size {num_tasks}")
        return axis
    if "cl_block_weights_" in joined:
        # Block gates are declared outside scan.  Non-bank gates are [depth],
        # bank gates are [num_tasks, depth].
        if len(shape) == 1:
            return None
        if len(shape) != 2 or shape[0] != num_tasks:
            raise ValueError(f"Unexpected block-weight bank shape {shape} at {joined}")
        return 0
    return None


def mask_other_task_banks(tree: Any, *, task_id: int, num_tasks: int) -> Any:
    """Zero old/future bank rows in a gradient or update pytree."""
    task_id, num_tasks = int(task_id), int(num_tasks)
    if num_tasks < 1 or not 0 <= task_id < num_tasks:
        raise ValueError(f"Invalid task bank id/capacity: {task_id}/{num_tasks}")

    def apply(path, value):
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            return value
        axis = bank_axis(path, tuple(value.shape), num_tasks)
        if axis is None:
            return value
        broadcast_shape = [1] * value.ndim
        broadcast_shape[axis] = num_tasks
        keep = (jnp.arange(num_tasks) == task_id).reshape(broadcast_shape)
        return jnp.where(keep, value, jnp.zeros((), dtype=value.dtype))

    return jax.tree_util.tree_map_with_path(apply, tree)


def protect_task_banks(
    tx: optax.GradientTransformation,
    *,
    task_id: int,
    num_tasks: int,
) -> optax.GradientTransformation:
    """Keep the native optimizer state layout and prevent any old-bank update.

    Masking gradients before clipping also prevents replay-only old-bank
    gradients from consuming the shared-parameter clipping budget.  Masking
    final updates additionally protects against pre-existing optimizer momentum
    or decoupled weight decay.  The protocol resets optimizer state per stage.
    """
    task_id, num_tasks = int(task_id), int(num_tasks)

    def update(grads, opt_state, params=None, **extra_args):
        grads = mask_other_task_banks(grads, task_id=task_id, num_tasks=num_tasks)
        if extra_args:
            updates, new_state = tx.update(grads, opt_state, params, **extra_args)
        else:
            updates, new_state = tx.update(grads, opt_state, params)
        updates = mask_other_task_banks(updates, task_id=task_id, num_tasks=num_tasks)
        return updates, new_state

    return optax.GradientTransformation(tx.init, update)


_ORIGINAL_CL_INIT = None


def install_training_fixes():
    """Patch only native CL initialization in this process.

    KD must be disabled in this four-way protocol.  Native `scripts.train`'s
    initialization loads one base/previous-stage parameter tree and initializes
    the student/optimizer with JIT/device placement, avoiding the CL initializer's
    materialized CPU mask model plus CPU student plus checkpoint copies.
    """
    from scripts import train as ordinary_train
    from scripts import train_cl_lora as cl_train

    global _ORIGINAL_CL_INIT
    if getattr(cl_train.init_train_state, "_libero_stable_init", False):
        return cl_train
    _ORIGINAL_CL_INIT = cl_train.init_train_state

    def stable_init(config, init_rng, mesh, *, resume):
        if cl_train._teacher_required(config):
            raise ValueError("Low-host-memory LIBERO initialization requires lambda_kd=0 and teacher-dependent schedules disabled")
        if float(getattr(config.optimizer, "weight_decay", 0.0)) != 0.0:
            raise ValueError("The LIBERO bank-protection protocol requires optimizer.weight_decay=0")
        state, state_sharding = ordinary_train.init_train_state(config, init_rng, mesh, resume=resume)
        model_config = config.model
        tx = protect_task_banks(
            state.tx,
            task_id=int(getattr(model_config, "cl_task_id", 0)),
            num_tasks=int(getattr(model_config, "cl_num_tasks", 4)),
        )
        # tx is static pytree metadata, so actual state and its sharding tree
        # must carry the exact same replacement transformation object.
        from openpi.shared import array_typing as at
        with at.disable_typechecking():
            state = dataclasses.replace(state, tx=tx)
            state_sharding = dataclasses.replace(state_sharding, tx=tx)
        return state, state_sharding, None, None

    stable_init._libero_stable_init = True
    cl_train.init_train_state = stable_init
    return cl_train


def uninstall_training_fixes() -> None:
    from scripts import train_cl_lora as cl_train
    if _ORIGINAL_CL_INIT is not None and getattr(cl_train.init_train_state, "_libero_stable_init", False):
        cl_train.init_train_state = _ORIGINAL_CL_INIT


def self_test_bank_mask() -> dict[str, Any]:
    """Small CPU-only arrays: no robot model, dataset, training run, or checkpoint."""
    import numpy as np

    with jax.default_device(jax.devices("cpu")[0]):
        params = {
            "PaliGemma": {"llm": {
                "layers": {
                    "attention": {"lora_b_bank": jnp.ones((2, 4, 3, 2))},
                    "mlp": {"gating_einsum_lora_b_bank": jnp.ones((2, 4, 2, 3, 2))},
                },
                "cl_block_weights_1": jnp.ones((4, 2)),
                "cl_block_weights_0": jnp.ones((2,)),
            }},
            "shared_projection": jnp.ones((3, 2)),
        }
        grads = jax.tree.map(jnp.ones_like, params)
        native_tx = optax.adamw(learning_rate=0.01, weight_decay=0.0)
        initial_opt = native_tx.init(params)
        # Deliberately create stale momentum on every bank row; a gradient-only
        # mask would not protect those rows in this stronger diagnostic.
        _, stale_opt = native_tx.update(grads, initial_opt, params)
        tx = protect_task_banks(native_tx, task_id=1, num_tasks=4)
        updates, _ = jax.jit(tx.update)(grads, stale_opt, params)
        new_params = optax.apply_updates(params, updates)
        for path, old_value in jax.tree_util.tree_flatten_with_path(params)[0]:
            parts = _path_parts(path)
            new_value = new_params
            for key in parts:
                new_value = new_value[key]
            axis = bank_axis(path, tuple(old_value.shape), 4)
            old_array, new_array = np.asarray(old_value), np.asarray(new_value)
            if axis is not None:
                for task in range(4):
                    a, b = np.take(old_array, task, axis=axis), np.take(new_array, task, axis=axis)
                    assert np.array_equal(a, b) if task != 1 else not np.array_equal(a, b)
            else:
                assert not np.array_equal(old_array, new_array)
        protected_grads = mask_other_task_banks(grads, task_id=1, num_tasks=4)
        assert np.count_nonzero(np.asarray(protected_grads["PaliGemma"]["llm"]["layers"]["attention"]["lora_b_bank"])[:, 0]) == 0
    return {
        "status": "passed",
        "task_id": 1,
        "bank_count": 4,
        "scanned_lora_task_axis": 1,
        "block_gate_task_axis": 0,
        "old_and_future_rows_bitwise_unchanged": True,
        "current_row_and_shared_parameters_updated": True,
        "stale_momentum_case_checked": True,
        "scope": "small CPU optimizer diagnostic only; no full model training",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(self_test_bank_mask(), indent=2))
