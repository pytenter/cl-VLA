"""Bounded-concurrency Orbax parameter restore without shared-source edits.

The public restore interface matches native OpenPI and adds ``expected_dtypes``
for direct per-leaf conversion during restoration.  The mixed loader restores
host arrays in the dtypes of native init_train_state's abstract parameter tree,
avoiding an intermediate all-FP32 host checkpoint and subsequent BF16 copy.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
from typing import Any

from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp


# Orbax's limiter requests the whole single-device shard.  The native base's
# largest stored FP32 array is 4.5GiB; a 1GB budget rejects it before reading.
# 6GB permits one such array with margin while bounding concurrent restores.
RESTORE_CONCURRENT_GB = 6


def _dtype_of(value):
    if not isinstance(value, type) and hasattr(value, "dtype"):
        value = value.dtype
    return np.dtype(value)


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: Any | None = None,
    sharding: jax.sharding.Sharding | None = None,
    expected_dtypes: Any | None = None,
):
    """Restore pure params, with a 6GB concurrent Orbax restore budget.

    ``expected_dtypes`` is a pure parameter dict whose leaves are dtype objects,
    arrays, or ShapeDtypeStructs.  It may include new LoRA leaves absent from the
    checkpoint.  Every checkpoint leaf must have a corresponding expected dtype
    when this argument is supplied; no silent FP32 fallback is allowed.  Per-leaf
    expected dtypes take precedence over the uniform ``dtype`` argument.

    Native checkpoints wrapping every parameter in ``{'value': array}`` are
    matched against pure expected paths and have that suffix removed on return.
    """
    if restore_type not in (np.ndarray, jax.Array):
        raise TypeError("restore_type must be numpy.ndarray or jax.Array")
    path = str(params_path) if str(params_path).startswith("gs://") else pathlib.Path(params_path).resolve()
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    expected = None
    if expected_dtypes is not None:
        if hasattr(expected_dtypes, "to_pure_dict"):
            expected_dtypes = expected_dtypes.to_pure_dict()
        if isinstance(expected_dtypes, dict) and set(expected_dtypes) == {"params"}:
            expected_dtypes = expected_dtypes["params"]
        expected = traverse_util.flatten_dict(expected_dtypes)
    handler = ocp.PyTreeCheckpointHandler(restore_concurrent_gb=RESTORE_CONCURRENT_GB)
    with ocp.PyTreeCheckpointer(handler) as checkpointer:
        metadata = checkpointer.metadata(path)
        if "params" not in metadata:
            raise ValueError(f"Checkpoint has no native 'params' root: {path}")
        parameter_metadata = metadata["params"]
        flat_metadata = traverse_util.flatten_dict(parameter_metadata)
        if not flat_metadata:
            raise ValueError(f"Checkpoint has no parameter arrays: {path}")
        strip_value = all(keys[-1] == "value" for keys in flat_metadata)
        flat_args = {}
        for keys in flat_metadata:
            pure_keys = keys[:-1] if strip_value else keys
            target_dtype = dtype
            if expected is not None:
                if pure_keys not in expected:
                    raise ValueError(f"Checkpoint leaf has no expected dtype: {'/'.join(map(str, pure_keys))}")
                target_dtype = _dtype_of(expected[pure_keys])
            elif target_dtype is not None:
                target_dtype = _dtype_of(target_dtype)
            flat_args[keys] = ocp.ArrayRestoreArgs(
                sharding=sharding, restore_type=restore_type, dtype=target_dtype,
            )
        item = {"params": parameter_metadata}
        args = {"params": traverse_util.unflatten_dict(flat_args)}
        restored = checkpointer.restore(path, ocp.args.PyTreeRestore(item=item, restore_args=args))["params"]
    flat = traverse_util.flatten_dict(restored)
    if all(keys[-1] == "value" for keys in flat):
        flat = {keys[:-1]: value for keys, value in flat.items()}
    return traverse_util.unflatten_dict(flat)


@dataclasses.dataclass(frozen=True)
class MixedCheckpointWeightLoader:
    """Native loader protocol, restoring directly to init's target host dtypes."""

    params_path: str

    def load(self, params):
        from openpi.training import weight_loaders
        loaded = restore_params(self.params_path, restore_type=np.ndarray, expected_dtypes=params)
        # The native merge validates the expected tree at the trainer boundary.
        # Here the loaded dtypes already equal the abstract target dtypes, so its
        # conditional astype does not allocate a second full parameter tree.
        return weight_loaders._merge_params(loaded, params, missing_regex=".*lora.*")


def self_test():
    """Tiny real Orbax saves/restores only; never allocate a full model."""
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Run self-test with CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu")
    original = {
        "PaliGemma": {"kernel": np.arange(12, dtype=np.float32).reshape(3, 4) / 7},
        "action_in_proj": {"kernel": np.arange(8, dtype=np.float32).reshape(4, 2) / 11},
    }
    expected = {
        "PaliGemma": {"kernel": jax.ShapeDtypeStruct((3, 4), jnp.bfloat16)},
        "action_in_proj": {"kernel": jax.ShapeDtypeStruct((4, 2), jnp.float32)},
        "new_lora_a": jax.ShapeDtypeStruct((2, 2), jnp.float32),
    }
    checks = {}
    with tempfile.TemporaryDirectory(prefix="kinova_low_memory_restore_") as temporary:
        for kind in ("pure", "nnx_value"):
            path = pathlib.Path(temporary) / kind
            saved = original if kind == "pure" else jax.tree.map(lambda array: {"value": array}, original)
            with ocp.PyTreeCheckpointer() as saver:
                saver.save(path, {"params": saved})
            mixed = restore_params(path, restore_type=np.ndarray, expected_dtypes=expected)
            assert mixed["PaliGemma"]["kernel"].dtype == np.dtype(jnp.bfloat16)
            assert mixed["action_in_proj"]["kernel"].dtype == np.dtype(np.float32)
            np.testing.assert_array_equal(mixed["PaliGemma"]["kernel"], original["PaliGemma"]["kernel"].astype(jnp.bfloat16))
            np.testing.assert_array_equal(mixed["action_in_proj"]["kernel"], original["action_in_proj"]["kernel"])
            assert isinstance(mixed["PaliGemma"]["kernel"], np.ndarray)
            merged = MixedCheckpointWeightLoader(str(path)).load(expected)
            assert merged["new_lora_a"] is expected["new_lora_a"]
            assert merged["PaliGemma"]["kernel"].dtype == expected["PaliGemma"]["kernel"].dtype
            np.testing.assert_array_equal(merged["action_in_proj"]["kernel"], original["action_in_proj"]["kernel"])
            checks[kind + "_mixed_host_dtype_and_values"] = True
            checks[kind + "_native_merge_preserves_new_lora_shape"] = True
        uniform = restore_params(path, restore_type=np.ndarray, dtype=jnp.bfloat16)
        assert all(leaf.dtype == np.dtype(jnp.bfloat16) for leaf in jax.tree.leaves(uniform))
        checks["uniform_dtype_argument"] = True
        preserved = restore_params(path, restore_type=np.ndarray)
        assert all(leaf.dtype == np.dtype(np.float32) for leaf in jax.tree.leaves(preserved))
        checks["none_dtype_preserves_stored_dtype"] = True
        cpu_arrays = restore_params(path, dtype=jnp.bfloat16)
        assert all(isinstance(leaf, jax.Array) and leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(cpu_arrays))
        checks["jax_array_restore_cpu_and_default_sharding"] = True
        dtype_tree = {"PaliGemma": {"kernel": jnp.bfloat16}, "action_in_proj": {"kernel": np.dtype("float32")}}
        explicit = restore_params(path, restore_type=np.ndarray, expected_dtypes=dtype_tree)
        np.testing.assert_array_equal(explicit["PaliGemma"]["kernel"], mixed["PaliGemma"]["kernel"])
        checks["explicit_dtype_tree_leaves"] = True
        try:
            restore_params(path, restore_type=np.ndarray, expected_dtypes={"PaliGemma": expected["PaliGemma"]})
        except ValueError as error:
            assert "no expected dtype" in str(error)
        else:
            raise AssertionError("Missing expected dtypes must fail before array restore")
        checks["missing_expected_dtype_rejected"] = True
    return {"status": "passed", "scope": "Tiny CPU Orbax native-schema dtype/merge test; no full model or GPU",
            "devices": [str(device) for device in jax.devices()], "restore_concurrent_gb": RESTORE_CONCURRENT_GB,
            "checks": checks, "full_model_restore_executed": False, "optimizer_steps": 0}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = self_test()
    path = pathlib.Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
