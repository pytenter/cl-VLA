"""Small CPU numerical/gradient checks.  Never construct the full model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from flax import linen as nn, nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from kinova_model import KinovaObservation, KinovaPi05, _ordinary_lora_settings, masked_flow_terms, normalized_masked_loss
from kinova_config import GroupedAdamW, KinovaInputs, parameter_group, write_parameter_layout
from openpi.models import gemma, lora


def run_tests():
    assert all(device.platform == "cpu" for device in jax.devices())
    checks = {}
    actions = jnp.arange(2 * 4 * 32, dtype=jnp.float32).reshape(2, 4, 32) / 100
    noise = jnp.ones_like(actions) * 0.25
    velocity = jnp.ones_like(actions) * 0.5
    all_valid = jnp.ones((2, 4), dtype=bool)
    numerator, count = masked_flow_terms(velocity, noise, actions, all_valid)
    expected = np.mean(np.square(np.asarray(velocity[..., :7] - (noise[..., :7] - actions[..., :7]))))
    np.testing.assert_allclose(normalized_masked_loss(numerator, count), expected, rtol=1e-6)
    assert float(count) == 56
    checks["all_valid_matches_manual_real7"] = True

    valid = jnp.array([[True, True, False, False], [True, False, False, False]])
    numerator, count = masked_flow_terms(velocity, noise, actions, valid)
    expected = np.asarray(velocity - (noise - actions))[np.asarray(valid), :7]
    np.testing.assert_allclose(numerator, np.square(expected).sum(), rtol=1e-6)
    assert float(count) == 21
    mask = np.asarray(valid)[..., None] & (np.arange(32) < 7)
    # Reducer invariance: masked predictions/targets do not enter the objective.
    altered_v = jnp.where(mask, velocity, 1e6)
    altered_n = jnp.where(mask, noise, -2e6)
    altered_a = jnp.where(mask, actions, 3e6)
    np.testing.assert_array_equal(masked_flow_terms(altered_v, altered_n, altered_a, valid), (numerator, count))
    checks["padded_times_and_dimensions_excluded_from_reducer"] = True

    def loss(prediction, time_mask=valid):
        n, c = masked_flow_terms(prediction, noise, actions, time_mask)
        return normalized_masked_loss(n, c)
    gradient = jax.grad(loss)(velocity)
    np.testing.assert_array_equal(np.asarray(gradient)[~mask], 0)
    assert np.isfinite(gradient).all() and np.any(np.asarray(gradient)[mask] != 0)
    zero_mask = jnp.zeros_like(valid)
    zero_n, zero_c = masked_flow_terms(velocity, noise, actions, zero_mask)
    assert float(zero_n) == float(zero_c) == float(loss(velocity, zero_mask)) == 0
    np.testing.assert_array_equal(jax.grad(lambda v: loss(v, zero_mask))(velocity), 0)
    checks["finite_valid_gradients_and_zero_invalid_gradients"] = True
    checks["zero_valid_returns_zero_loss_zero_gradient_count0"] = True

    # End-to-end loss-method check with a deliberately cross-time/cross-dim
    # fake denoiser.  This would fail if only the final SSE were masked.
    observation = KinovaObservation(
        images={key: jnp.zeros((2, 224, 224, 3), jnp.float32)
                for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        image_masks={"base_0_rgb": jnp.zeros(2, bool), "left_wrist_0_rgb": jnp.ones(2, bool),
                     "right_wrist_0_rgb": jnp.zeros(2, bool)},
        state=jnp.zeros((2, 32)), action_valid_mask=valid)
    def end_to_end(scale, labels):
        fake = SimpleNamespace(forward_denoise=lambda obs, clean, eps, time:
                               scale * (clean + eps + jnp.mean(clean, axis=(-2, -1), keepdims=True)))
        return KinovaPi05.compute_loss_terms(fake, jax.random.key(17), observation, labels, train=False)
    changed_labels = jnp.where(mask, actions, jnp.nan)
    first = end_to_end(jnp.array(0.4), actions)
    second = end_to_end(jnp.array(0.4), changed_labels)
    np.testing.assert_array_equal(first, second)
    first_grad = jax.grad(lambda scale: normalized_masked_loss(*end_to_end(scale, actions)))(jnp.array(0.4))
    second_grad = jax.grad(lambda scale: normalized_masked_loss(*end_to_end(scale, changed_labels)))(jnp.array(0.4))
    assert np.isfinite(first_grad) and float(first_grad) != 0
    np.testing.assert_array_equal(first_grad, second_grad)
    checks["end_to_end_invalid_gt_invariance_with_cross_time_dimension_forward"] = True

    # Different valid lengths require sum-of-numerators/counts, not the average
    # of per-microbatch means.  Verify gradients against one concatenated batch.
    inputs = jnp.arange(4 * 3 * 32, dtype=jnp.float32).reshape(4, 3, 32) / 20
    targets = jnp.zeros_like(inputs); eps = jnp.ones_like(inputs)
    times = jnp.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    def concatenated(scale):
        n, c = masked_flow_terms(inputs * scale, eps, targets, times)
        return normalized_masked_loss(n, c)
    grads, counts = [], []
    for i in range(4):
        def terms(scale):
            return masked_flow_terms(inputs[i:i+1] * scale, eps[i:i+1], targets[i:i+1], times[i:i+1])
        (_, c), g = jax.value_and_grad(terms, has_aux=True)(jnp.array(0.3))
        grads.append(g); counts.append(c)
    np.testing.assert_allclose(sum(grads) / sum(counts), jax.grad(concatenated)(jnp.array(0.3)), rtol=1e-6)
    checks["accumulation_uses_total_valid_element_count"] = True

    # Exercise the deployed LeRobot boundary implementation itself, without
    # opening an episode/image or constructing a dataset object.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    boundary = SimpleNamespace(episode_data_index={"from": np.array([0, 3]), "to": np.array([3, 7])},
                               delta_indices={"action": list(range(50))})
    indices, padding = LeRobotDataset._get_query_indices(boundary, 2, 0)
    assert indices["action"] == [2] * 50
    assert padding["action_is_pad"].tolist() == [False] + [True] * 49
    indices2, padding2 = LeRobotDataset._get_query_indices(boundary, 3, 1)
    assert indices2["action"][:5] == [3, 4, 5, 6, 6]
    assert padding2["action_is_pad"].tolist() == [False] * 4 + [True] * 46
    checks["native_lerobot_no_cross_episode_action_query"] = True

    raw = {"wrist": np.zeros((3, 8, 12), dtype=np.float32), "state": np.arange(28, dtype=np.float32),
           "prompt": "夹取桌面上的物体,并将它放到盒子里面", "actions": np.ones((50, 7), np.float32),
           "action_is_pad": np.array([False] * 7 + [True] * 43),
           "source_episode_index": 2, "source_frame_index": 93}
    mapped = KinovaInputs()(raw)
    assert mapped["image_mask"] == {"base_0_rgb": False, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
    np.testing.assert_array_equal(mapped["action_valid_mask"], ~raw["action_is_pad"])
    mapped["tokenized_prompt"] = np.zeros(200, np.int32); mapped["tokenized_prompt_mask"] = np.ones(200, bool)
    mapped.pop("prompt")
    observed = KinovaObservation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None, ...], mapped))
    assert observed.action_valid_mask.shape == (1, 50)
    assert observed.source_episode_index.tolist() == [2] and observed.source_frame_index.tolist() == [93]
    checks["wrist_only_masks_and_observation_supervision_mask_retained"] = True

    # Tiny ordinary-LoRA Einsum from the real implementation.  No orthogonal
    # CL initialization or full transformer is used here.
    with _ordinary_lora_settings(rank=16, alpha=32):
        for variant in ("gemma_2b_lora", "gemma_300m_lora"):
            for options in gemma.get_config(variant).lora_configs.values():
                assert options.rank == 16 and options.alpha == 32 and not options.use_orthogonal_init
    assert gemma.get_config("gemma_300m_lora").lora_configs["attn"].rank == 32
    module = lora.Einsum(shape=(4, 3), init_fn=nn.initializers.normal(), lora_config=lora.LoRAConfig(rank=2, alpha=4))
    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    variables = module.init(jax.random.key(42), "bd,df->bf", x)
    variables["params"]["lora_b"] = jnp.zeros_like(variables["params"]["lora_b"])
    y = module.apply(variables, "bd,df->bf", x)
    np.testing.assert_allclose(y, x @ variables["params"]["w"], rtol=1e-6, atol=1e-7)
    checks["tiny_native_ordinary_lora_zero_b_equals_base"] = True

    # Use NNX's real parameter-state PyTree to catch optimizer label-layout
    # incompatibility before the full native init is attempted.
    class Tiny(nnx.Module):
        def __init__(self):
            self.lora_a = nnx.Param(jnp.ones((2, 2)))
            self.action_in_proj = nnx.Dict(kernel=nnx.Param(jnp.ones((2, 2))))
    params = nnx.state(Tiny())
    tx = GroupedAdamW().create(1e-4)
    opt_state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    updates, _ = jax.jit(tx.update)(grads, opt_state, params)
    after = optax.apply_updates(params, updates)
    la = np.asarray(after["lora_a"].value); head = np.asarray(after["action_in_proj"]["kernel"].value)
    assert np.isfinite(la).all() and np.isfinite(head).all()
    np.testing.assert_allclose((1 - la) / (1 - head), 10, rtol=0.02)
    assert parameter_group("PaliGemma/llm/layers/attn/q_einsum/lora_a") == "lora"
    checks["native_nnx_optimizer_groups_and_head_lr_ratio"] = True
    return {"status": "passed", "devices": [str(device) for device in jax.devices()], "checks": checks,
            "scope": "Small CPU tensors/native primitive modules only; no full model, checkpoint restore, or GPU",
            "loss_semantics": "Invalid GT canonicalized to zero before forward plus masked supervision; native full noise/attention retained",
            "zero_valid_policy": "Model returns (0,0); trainer must skip/reject optimizer update"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout-output")
    args = parser.parse_args()
    report = run_tests()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if args.layout_output:
        layout = write_parameter_layout(args.layout_output)
        print(json.dumps({key: layout[key] for key in ("total_parameters", "trainable_parameters", "trainable_tensors", "group_counts")}))


if __name__ == "__main__":
    main()
