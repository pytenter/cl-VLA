"""Draft same-architecture, zero-update E0 factory.

Importing this module does not import JAX or initialize a model. Calling the
factory restores a real model and must be done only by the existing guarded E0
job after the shared-weight graph diagnosis has justified this baseline.
No optimizer update, checkpoint save, dataset scoring or robot action occurs
here. The caller evaluates the unchanged validation plan separately.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import dataclasses
from pathlib import Path


ORIGINAL_BASE_PARAMS = Path(
    project_path('base_params', '')
)


def build_zero_lora_baseline(config_seed42: int = 42, *, hash_parameters: bool = False):
    """Return ``(model, metadata)`` from the exact seeded smoke architecture.

    Uses native initialization, including its real A initialization and zero B;
    it never calls ``model.load`` to fill missing adapter parameters. As in
    ``train_kinova.capture``, the evaluation view casts every parameter to BF16
    and merges it with the native ``state.model_def`` before ``model.eval()``.

    ``hash_parameters`` defaults to False: no large parameter arrays are copied
    to the CPU merely to construct this E0. If explicit identity evidence is
    needed, the existing bounded ``array_hash`` helper hashes the BF16 inference
    view in slices, grouped into 51 base and 20 adapter leaves.
    """
    if isinstance(config_seed42, bool) or config_seed42 != 42:
        raise ValueError("The declared zero-update baseline requires seed42")

    # Deliberately lazy: importing this draft is safe without a GPU runtime.
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from kinova_config import make_config
    from openpi.training import sharding
    from scripts.train import init_train_state

    config = make_config("smoke")
    model_config = config.model
    checkpoint = Path(config.weight_loader.params_path)
    if checkpoint != ORIGINAL_BASE_PARAMS:
        raise ValueError(f"Unexpected E0 base checkpoint: {checkpoint}")
    if config.seed != 42 or config.resume:
        raise ValueError("E0 must initialize seed42 from the original base, without resume")
    expected = {
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "lora_rank": 16,
        "lora_alpha": 32.0,
        "lora_dropout": 0.0,
        "action_dim": 32,
        "action_horizon": 50,
        "max_token_len": 200,
        "pi05": True,
        "discrete_state_input": True,
    }
    for name, value in expected.items():
        if getattr(model_config, name) != value:
            raise ValueError(f"Same-graph E0 configuration differs: {name}")
    for name in (
        "cl_pali_lora_bank", "cl_action_lora_bank", "cl_pali_block_weight_bank",
        "cl_action_block_weight_bank", "cl_freeze_shared_lora", "cl_freeze_lora_a",
        "cl_freeze_pali_lora_b", "cl_freeze_action_lora_b",
    ):
        if getattr(model_config, name):
            raise ValueError(f"Ordinary single-task E0 cannot enable {name}")

    # This is the same call made by train_kinova before its first update. The
    # existing native weight loader/maintenance helper owns restore and memory
    # policy. This factory neither changes nor bypasses any resource guard.
    mesh = sharding.make_mesh(1)
    state, _ = init_train_state(config, jax.random.key(config.seed), mesh, resume=False)
    jax.block_until_ready(state)
    if int(state.step) != 0:
        raise ValueError("Fresh baseline native state is not at optimizer step0")

    original_flat = {
        "/".join(map(str, path)): variable.value
        for path, variable in state.params.flat_state().items()
    }
    adapter_names = sorted(name for name in original_flat if "lora" in name)
    base_names = sorted(set(original_flat) - set(adapter_names))
    a_names = [name for name in adapter_names if name.endswith("lora_a")]
    b_names = [name for name in adapter_names if name.endswith("lora_b")]
    if (len(base_names), len(adapter_names), len(a_names), len(b_names)) != (51, 20, 10, 10):
        raise ValueError("Expected51 base leaves and20 ordinary adapter leaves (10 A/10 B)")
    if any("cl_block_weights" in name for name in original_flat):
        raise ValueError("Unexpected CL parameters in the ordinary LoRA baseline")

    b_checks = {}
    for name in b_names:
        # Transfer only reduction scalars, never full B matrices for this check.
        value = original_flat[name]
        finite = bool(jax.device_get(jnp.all(jnp.isfinite(value))))
        maximum = float(jax.device_get(jnp.max(jnp.abs(value))))
        b_checks[name] = {"finite": finite, "max_abs": maximum, "zero": finite and maximum == 0.0}
    if not all(item["zero"] for item in b_checks.values()):
        raise ValueError("Baseline ordinary LoRA has a nonzero/nonfinite B tensor")

    optimizer_counters = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        if getattr(leaf, "ndim", None) == 0 and jnp.issubdtype(leaf.dtype, jnp.integer):
            value = int(jax.device_get(leaf))
            optimizer_counters.append({"path": jax.tree_util.keystr(path), "value": value})
            if value != 0:
                raise ValueError("Native E0 optimizer was already advanced")

    # Keep this identical to the trainer's read-only capture view. Optimizer
    # parameters remain mixed precision; only this returned inference view is BF16.
    inference_state = jax.tree.map(lambda value: value.astype(jnp.bfloat16), state.params)
    model = nnx.merge(state.model_def, inference_state)
    model.eval()
    jax.block_until_ready(inference_state)
    inference_flat = {
        "/".join(map(str, path)): variable.value
        for path, variable in inference_state.flat_state().items()
    }
    if set(inference_flat) != set(original_flat) or any(
        value.dtype != jnp.bfloat16 for value in inference_flat.values()
    ):
        raise ValueError("BF16 inference view changed parameter keys or retained another dtype")

    metadata = {
        "graph_kind": "zero_lora_same_as_training",
        "baseline_kind": "E0_same_graph_zero_update",
        "config_name": config.name,
        "model": dataclasses.asdict(model_config),
        "base_params": str(checkpoint),
        "seed": int(config.seed),
        "optimizer_step": 0,
        "optimizer_updates_executed": 0,
        "native_init_allocated_optimizer_state": True,
        "optimizer_counters": optimizer_counters,
        "restore_method": "native init_train_state with configured CPU weight loader and maintenance",
        "inference_view": "all parameters BF16; nnx.merge(native model_def, inference_state); eval",
        "missing_adapters_filled_by_model_load": False,
        "base_parameter_keys": base_names,
        "adapter_parameter_keys": adapter_names,
        "parameter_leaf_counts": {"base": 51, "adapter": 20, "all": 71},
        "lora_b_checks": b_checks,
        "all_lora_b_zero": True,
        "parameter_hashes_recorded": bool(hash_parameters),
        "dataset_scored_by_factory": False,
        "robot_commands_sent": 0,
        "resource_policy_changed_by_factory": False,
    }
    if hash_parameters:
        from array_hash import hash_array
        metadata["parameter_hashes"] = {
            "view": "BF16 inference parameters; array_hash bounded leading-axis transfers",
            "base": {name: hash_array(inference_flat[name]) for name in base_names},
            "adapter": {name: hash_array(inference_flat[name]) for name in adapter_names},
        }

    # The returned model retains its inference buffers. Release the unused
    # optimizer/training state; no checkpoint manager or update function exists.
    del original_flat, inference_flat, inference_state, state
    return model, metadata
