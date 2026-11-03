"""Reproducible tiny CPU checks for bounded hashes and native checkpoint IO."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from array_hash import hash_array
from checkpoint_io import initialize_checkpoint_dir
from kinova_config import GroupedAdamW
from openpi.training import checkpoints
from openpi.training.utils import TrainState


def full_hash(value):
    host = np.asarray(jax.device_get(value))
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(np.ascontiguousarray(host).reshape(-1).view(np.uint8))
    return digest.hexdigest()


def compare_tree(before, after):
    left, left_structure = jax.tree_util.tree_flatten_with_path(before)
    right, right_structure = jax.tree_util.tree_flatten_with_path(after)
    assert left_structure == right_structure
    rows = []
    for (left_path, a), (right_path, b) in zip(left, right):
        assert jax.tree_util.keystr(left_path) == jax.tree_util.keystr(right_path)
        assert tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert hash_array(a, target_bytes=17) == full_hash(a) == hash_array(b, target_bytes=17)
        rows.append({"path": jax.tree_util.keystr(left_path), "shape": list(a.shape),
                     "dtype": str(a.dtype), "sha256": hash_array(a, target_bytes=17)})
    return rows


def run_test():
    assert all(device.platform == "cpu" for device in jax.devices())
    hash_rows = []
    for dtype in (jnp.float32, jnp.bfloat16):
        for shape in ((), (1,), (3, 5), (2, 7, 9), (0, 5)):
            count = int(np.prod(shape)) if shape else 1
            values = np.arange(count, dtype=np.float32).reshape(shape) / 7 - .75
            value = jnp.asarray(values, dtype=dtype)
            expected = full_hash(value)
            for budget in (1, 17, 64 * 2**20):
                actual = hash_array(value, target_bytes=budget)
                assert actual == expected
            hash_rows.append({"dtype": str(value.dtype), "shape": list(shape), "sha256": expected,
                              "budgets_bytes": [1, 17, 64 * 2**20]})
        transposed = jnp.arange(35, dtype=dtype).reshape(5, 7).T
        assert hash_array(transposed, target_bytes=3) == full_hash(transposed)
        sliced = jnp.arange(48, dtype=dtype).reshape(6, 8)[::2, 1::2]
        assert hash_array(sliced, target_bytes=3) == full_hash(sliced)
    special = jnp.asarray(np.array([0., -0., np.inf, -np.inf, np.nan], np.float32))
    assert hash_array(special, target_bytes=1) == full_hash(special)
    hash_rows.append({"dtype": str(special.dtype), "shape": list(special.shape),
                      "scope": "positive/negative zero, infinity, NaN bytes", "sha256": full_hash(special)})

    class Tiny(nnx.Module):
        def __init__(self):
            self.lora_a = nnx.Param(jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 7)
            self.action_in_proj = nnx.Dict(kernel=nnx.Param(jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 13))
            self.frozen = nnx.Param(jnp.arange(35, dtype=jnp.bfloat16).reshape(5, 7) / 17)

    model = Tiny()
    graph, params = nnx.split(model)
    trainable_filter = nnx.All(nnx.Param, nnx.Not(lambda path, value: path[0] == "frozen"))
    tx = GroupedAdamW().create(optax.linear_schedule(1e-4, 1e-5, 20))
    state = TrainState(step=jnp.int32(0), params=params, model_def=graph,
                       opt_state=tx.init(params.filter(trainable_filter)), tx=tx, ema_decay=None, ema_params=None)

    def tiny_step(current):
        trainable = current.params.filter(trainable_filter)
        grads = jax.tree.map(lambda value: jnp.ones_like(value) * (current.step + 1) * .3, trainable)
        updates, opt_state = current.tx.update(grads, current.opt_state, trainable)
        module = nnx.merge(current.model_def, current.params)
        nnx.update(module, optax.apply_updates(trainable, updates))
        return dataclasses.replace(current, step=current.step + 1, params=nnx.state(module), opt_state=opt_state)

    step = jax.jit(tiny_step)
    state = step(step(state))
    jax.block_until_ready(state)
    assert int(state.step) == 2
    assert any(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(state.opt_state) if value.ndim > 0)
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None))
    with tempfile.TemporaryDirectory(prefix="kinova_bounded_io_cpu_") as temporary:
        path = Path(temporary) / "checkpoint"
        manager, resuming = initialize_checkpoint_dir(path, keep_period=500)
        assert not resuming
        checkpoints.save_state(manager, state, loader, 2)
        manager.wait_until_finished()
        assert set((path / "2").iterdir()) >= {path / "2" / name for name in ("assets", "params", "train_state")}
        with ocp.PyTreeCheckpointer() as reader:
            metadata = reader.metadata(path / "2" / "params")
            native_paths = traverse_util.flatten_dict(metadata["params"])
            assert native_paths and all(keys[-1] == "value" for keys in native_paths)
        manager.close()
        reopened, resuming = initialize_checkpoint_dir(path, keep_period=500, resume=True)
        assert resuming
        template = jax.eval_shape(lambda current: current, state)
        restored = checkpoints.restore_state(reopened, template, None, step=2)
        jax.block_until_ready(restored)
        assert int(restored.step) == 2
        parameter_rows = compare_tree(state.params, restored.params)
        optimizer_rows = compare_tree(state.opt_state, restored.opt_state)
        follow_a, follow_b = step(state), step(restored)
        jax.block_until_ready((follow_a, follow_b))
        assert int(follow_a.step) == int(follow_b.step) == 3
        compare_tree(follow_a.params, follow_b.params)
        compare_tree(follow_a.opt_state, follow_b.opt_state)
        reopened.close()
    return {"status": "passed", "devices": [str(device) for device in jax.devices()],
            "scope": "Tiny CPU arrays and tiny native TrainState/checkpoint only; no full model/GPU or real pi05 optimizer steps",
            "checks": {"bounded_hash_matches_full_C_order_SHA256": True,
                       "bfloat16_float32_scalars_nonsquare_empty_and_special_values": True,
                       "native_schema_value_suffix_and_assets_trainstate_params": True,
                       "new_manager_resume_detected": True,
                       "all_parameter_leaves_exact": True, "all_adam_and_schedule_leaves_exact": True,
                       "restored_next_optimizer_update_exact": True},
            "hash_cases": hash_rows, "parameter_leaves": parameter_rows, "optimizer_leaves": optimizer_rows,
            "checkpoint_step": 2, "save_concurrent_gb": 3, "restore_concurrent_gb": 3,
            "full_model_restored": False, "pi05_optimizer_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_test()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "devices", "checks", "save_concurrent_gb", "restore_concurrent_gb")}, indent=2))


if __name__ == "__main__":
    main()
