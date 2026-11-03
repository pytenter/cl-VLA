"""Fixed, episode-balanced Kinova offline evaluation; never controls a robot.

GPU execution is explicit: the run command refuses to load a model without
--resource-approved. prepare-plan, self-test and select are CPU-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import math
from pathlib import Path
import time

from validation_plan import PROTOCOL, canonical_hash, validate_plan, write_json

GROUPS = {"xyz": slice(0, 3), "euler": slice(3, 6), "gripper": slice(6, 7)}
HORIZONS = (1, 5, 10, 20, 50)


def action_stats(plan):
    import numpy as np
    data = json.loads(Path(plan["normalization"]["path"]).read_text())
    stats = data.get("norm_stats", data)["actions"]
    lo, hi = np.asarray(stats["q01"], dtype=np.float64), np.asarray(stats["q99"], dtype=np.float64)
    if lo.shape != (7,) or hi.shape != (7,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
        raise ValueError("Expected finite 7D training-only action quantiles")
    return lo, hi


def unnormalize(actions, stats):
    """Exact inverse of native quantile normalization, including its epsilon."""
    lo, hi = stats
    return (actions[..., :7] + 1.0) / 2.0 * (hi - lo + 1e-6) + lo


def error_metrics(normalized_error, physical_error):
    import numpy as np
    result = {}
    for name, dims in GROUPS.items():
        norm, physical = normalized_error[..., dims], physical_error[..., dims]
        result[name + "_normalized_mse"] = float(np.mean(norm**2))
        result[name + "_mse"] = float(np.mean(physical**2))
        result[name + "_mae"] = float(np.mean(np.abs(physical)))
        result[name + "_component_mse"] = np.mean(physical**2, axis=0).tolist()
        result[name + "_component_mae"] = np.mean(np.abs(physical), axis=0).tolist()
    result["xyz_euclidean_mse"] = float(np.mean(np.sum(physical_error[..., :3]**2, axis=-1)))
    result["normalized_group_equal_mse"] = sum(result[g + "_normalized_mse"] for g in GROUPS) / 3.0
    return result


def score_prediction(prediction, target, valid_mask, stats):
    """Score adjacent increments, not accumulated end-effector positions."""
    import numpy as np
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != (50, 32) or target.shape != (50, 32) or valid_mask.shape != (50,):
        raise ValueError(f"Expected (50,32)/(50,32)/(50,), got {prediction.shape}/{target.shape}/{valid_mask.shape}")
    if not valid_mask.any():
        raise ValueError("Zero valid actions; no denominator can be formed")
    nonfinite = int((~np.isfinite(prediction)).sum())
    if nonfinite or not np.isfinite(target).all():
        raise ValueError(f"Nonfinite model prediction/target, prediction count={nonfinite}")
    error_norm = prediction[..., :7].astype(np.float64) - target[..., :7].astype(np.float64)
    error_phys = unnormalize(prediction, stats) - unnormalize(target, stats)
    short = valid_mask & (np.arange(50) < 10)
    if not short.any():
        raise ValueError("No valid short-window steps")
    result = {"selection_score": error_metrics(error_norm[short], error_phys[short])["normalized_group_equal_mse"],
              "valid_steps": int(valid_mask.sum()), "nonfinite_output_count": nonfinite, "point": {}, "prefix": {}}
    for horizon in HORIZONS:
        if valid_mask[horizon - 1]:
            result["point"][str(horizon)] = error_metrics(error_norm[horizon-1:horizon], error_phys[horizon-1:horizon])
        mask = valid_mask & (np.arange(50) < horizon)
        if mask.any():
            result["prefix"][str(horizon)] = {**error_metrics(error_norm[mask], error_phys[mask]),
                                                 "actual_valid_steps": int(mask.sum())}
    return result


def mean_metrics(rows):
    """Mean scalar/array leaves; horizon absence is reported separately, not zero-filled."""
    import numpy as np
    if not rows:
        raise ValueError("Cannot aggregate zero records")
    keys = sorted(set().union(*(row.keys() for row in rows)))
    result = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        if isinstance(values[0], dict):
            result[key] = mean_metrics(values)
        elif isinstance(values[0], (int, float, list)):
            value = np.mean(np.asarray(values, dtype=np.float64), axis=0)
            result[key] = value.tolist() if value.ndim else float(value)
    return result


def add_rmse(metrics):
    import numpy as np
    result = dict(metrics)
    for key, value in metrics.items():
        if isinstance(value, dict):
            result[key] = add_rmse(value)
        elif key.endswith("_mse") and "normalized" not in key:
            result[key[:-4] + "_rmse"] = np.sqrt(value).tolist()
    return result


def aggregate_records(records, plan):
    grouped = defaultdict(list)
    expected = {(ep["episode_id"], block["start_frame"], seed["seed"])
                for ep in plan["episodes"] for block in ep["blocks"] for seed in block["seeds"]}
    actual = [(row["episode_id"], row["start_frame"], row["seed"]) for row in records]
    if len(actual) != len(set(actual)) or set(actual) - expected:
        raise ValueError("Duplicate or unplanned validation records")
    errors = [row for row in records if row.get("status") != "complete"]
    for row in records:
        if row.get("status") == "complete":
            grouped[row["episode_id"]].append(row)
    per_episode = []
    for episode in plan["episodes"]:
        rows = grouped.get(episode["episode_id"], [])
        if not rows:
            continue
        metrics = mean_metrics([row["metrics"] for row in rows])
        per_episode.append({"episode_id": episode["episode_id"], "blocks_and_seeds_completed": len(rows),
                            "task_phase": "UNKNOWN", "metrics": add_rmse(metrics)})
    complete = set(actual) == expected and not errors and len(per_episode) == len(plan["episodes"])
    dataset_metrics = add_rmse(mean_metrics([row["metrics"] for row in per_episode])) if per_episode else None
    # Recompute RMSE from episode-averaged MSE, never average episode RMSE.
    if dataset_metrics:
        dataset_metrics = add_rmse(dataset_metrics)
    point_counts = {str(k): {"block_seed_count": sum(str(k) in row["metrics"]["point"] for row in records if row.get("status") == "complete"),
                             "episode_count": sum(any(str(k) in row["metrics"]["point"] for row in rows) for rows in grouped.values())}
                    for k in HORIZONS}
    finite = dataset_metrics is not None and math.isfinite(dataset_metrics["selection_score"])
    fm_rows = [row for row in records if row.get("status") == "complete" and "fm_numerator" in row]
    fm_sum = sum(row["fm_numerator"] for row in fm_rows)
    fm_count = sum(row["fm_valid_element_count"] for row in fm_rows)
    input_records = {f"{row['episode_id']}|{row['start_frame']}|{row['seed']}":
                     {"input": row.get("model_input_sha256"), "initial_noise": row.get("initial_noise_sha256")}
                     for row in records if row.get("status") == "complete"}
    return {"status": "complete" if complete else "incomplete", "evaluation_valid": complete and finite,
            "expected_block_seed_count": len(expected), "records_completed": len(records) - len(errors),
            "errors": len(errors), "missing_records": len(expected - set(actual)),
            "nonfinite_output_count": sum(row.get("nonfinite_output_count", 0) for row in records),
            "flow_matching_nonfinite_term_count": sum(row.get("flow_matching_nonfinite_terms", 0) for row in records),
            "abnormal_output_records": sum(row.get("nonfinite_output_count", 0) > 0 or row.get("prediction_shape") not in (None, [50, 32]) for row in records),
            "episode_count": len(per_episode), "per_episode": per_episode, "dataset_metrics": dataset_metrics,
            "validation_input_and_noise_sha256": canonical_hash(input_records) if complete else None,
            "flow_matching_element_weighted": {"numerator": fm_sum, "valid_element_count": fm_count,
                                                "loss": fm_sum/fm_count if fm_count else None,
                                                "primary_report": "dataset_metrics.flow_matching_loss is episode-balanced"},
            "point_horizon_coverage": point_counts, "task_phase": "UNKNOWN", "test_split_used": False}


def load_symbol(name):
    module, sep, function = name.partition(":")
    if not sep:
        raise ValueError("Factory must be module:function")
    return getattr(importlib.import_module(module), function)


def load_model(config, checkpoint, *, frozen_base=False, restore_dtype="bfloat16"):
    """Restore exactly the configuration requested; E0 uses engine's no-LoRA config."""
    import jax
    import jax.numpy as jnp
    from native_weights import restore_native_params
    dtype = None if restore_dtype in (None, "preserve") else getattr(jnp, restore_dtype)
    path = Path(checkpoint)
    if (path / "params").is_dir():
        path = path / "params"
    # Keep the shared native CPU-first restore, with bounded handler concurrency.
    # Global BF16 matches capture_reference's read-only inference parameter view.
    params = restore_native_params(path, restore_type=jax.Array, dtype=dtype)
    model = config.model.load(params)
    if hasattr(model, "eval"):
        model.eval()
    return model


def default_components(label, checkpoint, plan):
    import dataclasses
    from kinova_config import build_dataset, make_config
    is_baseline = label == "E0_wrist_frozen"
    config = make_config("smoke" if is_baseline else "pilot")
    if Path(config.data.candidate_root).resolve() != Path(plan["candidate_root"]).resolve():
        raise ValueError("Configuration candidate root differs from the fixed validation protocol")
    metadata = {"config_name": config.name, "model": dataclasses.asdict(config.model),
                "input_pipeline": "kinova_config.build_dataset(validation); no inference-only transform"}
    if is_baseline:
        from zero_lora_baseline import ORIGINAL_BASE_PARAMS, build_zero_lora_baseline
        if Path(checkpoint).resolve() != ORIGINAL_BASE_PARAMS.resolve():
            raise ValueError("Same-graph E0 must initialize from the exact original pi05_base params")
        model, baseline = build_zero_lora_baseline(hash_parameters=True)
        metadata = {**baseline, "input_pipeline": metadata["input_pipeline"]}
    else:
        model = load_model(config, checkpoint, restore_dtype="bfloat16")
    dataset = build_dataset(config, split="validation")
    return model, dataset, metadata


def enforce_resource_gate():
    """Must execute before importing JAX or any model configuration in GPU CLIs."""
    from resource_guard import inspect_resources, verify_allocator_limit
    result = inspect_resources()
    if not result["eligible"]:
        print("KINOVA_RESOURCE_DEFERRED " + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        raise SystemExit(75)
    result['allocator_verification'] = verify_allocator_limit()
    return result


def prepare_sample(sample, block):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from kinova_model import KinovaObservation
    if not isinstance(sample, dict):
        raise TypeError("Validation dataset must return transformed normalized dictionaries")
    if "source_frame_index" not in sample or int(sample["source_frame_index"]) != block["start_frame"]:
        raise ValueError("Loader frame index differs from fixed validation start")
    values = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], sample)
    observation = KinovaObservation.from_dict(values)
    actions = jnp.asarray(values["actions"])
    actual = np.asarray(observation.action_valid_mask)[0]
    if actual.shape != (50,) or not np.array_equal(actual, block["time_mask"]):
        raise ValueError("Loader time mask disagrees with frozen episode-boundary plan")
    if actions.shape != (1, 50, 32):
        raise ValueError(f"Expected padded actions (1,50,32), got {actions.shape}")
    if observation.state.shape != (1, 32) or not np.isfinite(np.asarray(observation.state)).all() or not np.isfinite(np.asarray(actions)).all():
        raise ValueError("Expected finite normalized state[1,32] and actions[1,50,32]")
    for key, expected in [("base_0_rgb", False), ("left_wrist_0_rgb", True), ("right_wrist_0_rgb", False)]:
        if observation.images[key].shape != (1, 224, 224, 3):
            raise ValueError(f"Wrong policy image geometry for {key}: {observation.images[key].shape}")
        if bool(np.asarray(observation.image_masks[key])[0]) != expected:
            raise ValueError(f"Wrist-only input mask mismatch for {key}")
        if not expected:
            image = np.asarray(observation.images[key])
            if not (np.all(image == 0) or np.all(image == -1)):
                raise ValueError(f"Inactive camera must use an actual zero-image placeholder: {key}")
    return observation, actions


def make_inference_functions(model):
    from openpi.shared import nnx_utils
    return (nnx_utils.module_jit(model.sample_actions),
            nnx_utils.module_jit(model.compute_loss_terms, static_argnames=("train",)))


def evaluate_model(model, dataset, plan, output_dir, *, label, optimizer_step, checkpoint,
                   config_metadata=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["summary.json", "records.jsonl"]):
        raise FileExistsError(f"Select a fresh evaluation directory: {output}")
    validate_plan(plan)
    if int(model.action_horizon) != 50 or int(model.action_dim) != 32:
        raise ValueError("Model geometry must match the frozen 50x32 validation protocol")
    if len(dataset) != plan["total_frames"]:
        raise ValueError("Validation loader frame count disagrees with fixed plan")
    stats = action_stats(plan)
    sample_fn, loss_fn = make_inference_functions(model)
    rows, start = [], time.monotonic()
    provenance = {"protocol": PROTOCOL, "label": label, "optimizer_step": optimizer_step,
                  "checkpoint": str(checkpoint), "plan_sha256": plan["plan_sha256"],
                  "norm_stats_sha256": plan["normalization"]["sha256"],
                  "inference_dtype": "bfloat16 parameters", "denoise_steps": plan["denoise_steps"],
                  "config_metadata": config_metadata or {}, "robot_actions_sent": 0}
    with (output / "records.jsonl").open("x") as stream:
        for episode in plan["episodes"]:
            for block in episode["blocks"]:
                for seed in block["seeds"]:
                    row = {"episode_id": episode["episode_id"], "episode_index": episode["episode_index"],
                           "start_frame": block["start_frame"], "block_label": block["label"],
                           **seed, "task_phase": "UNKNOWN"}
                    try:
                        sample = dataset[block["global_index"]]
                        if int(sample["source_episode_index"]) != episode["episode_index"]:
                            raise ValueError("Loader episode index disagrees with frozen validation plan")
                        obs, actions = prepare_sample(sample, block)
                        rng = jax.random.key(seed["sampling_seed"])
                        noise = jax.random.normal(rng, (1, 50, 32), dtype=jnp.float32)
                        prediction = np.asarray(sample_fn(rng, obs, noise=noise, num_steps=plan["denoise_steps"]))[0]
                        row.update(prediction_shape=list(prediction.shape), nonfinite_output_count=int((~np.isfinite(prediction)).sum()))
                        numerator, count = loss_fn(jax.random.key(seed["flow_matching_seed"]), obs, actions, train=False)
                        numerator, count = float(numerator), float(count)
                        row["flow_matching_nonfinite_terms"] = int(not math.isfinite(numerator)) + int(not math.isfinite(count))
                        expected_count = block["valid_steps"] * 7
                        if count != expected_count or count <= 0 or not math.isfinite(numerator):
                            raise ValueError(f"Invalid masked FM terms: numerator={numerator}, count={count}, expected={expected_count}")
                        metrics = score_prediction(prediction, np.asarray(actions)[0], block["time_mask"], stats)
                        metrics["flow_matching_loss"] = numerator / count
                        row.update(status="complete", metrics=metrics, fm_numerator=numerator,
                                   fm_valid_element_count=count, initial_noise_sha256=canonical_array_hash(noise),
                                   model_input_sha256=sample_hash(sample))
                        np.savez_compressed(output / f"ep{episode['episode_index']:03d}_{block['label']}_seed{seed['seed']}.npz",
                                            prediction_normalized=prediction, target_normalized=np.asarray(actions)[0],
                                            prediction_physical=unnormalize(prediction, stats),
                                            target_physical=unnormalize(np.asarray(actions)[0], stats),
                                            action_valid_mask=np.asarray(block["time_mask"], dtype=bool))
                    except Exception as error:
                        row.update(status="error", error=f"{type(error).__name__}: {error}")
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        write_json(output / "summary.json", {**provenance, **aggregate_records(rows, plan),
                                                             "elapsed_seconds": time.monotonic()-start})
                        raise
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps({"episode": episode["episode_index"], "block": block["label"],
                                      "seed": seed["seed"], "selection_score": row["metrics"]["selection_score"]}), flush=True)
    summary = {**provenance, **aggregate_records(rows, plan), "elapsed_seconds": time.monotonic()-start}
    write_json(output / "summary.json", summary)
    return summary


def canonical_array_hash(value):
    import hashlib
    import numpy as np
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.shape, str(array.dtype))).encode() + array.tobytes()).hexdigest()


def sample_hash(sample):
    return canonical_hash({key: sample_hash(value) if isinstance(value, dict) else canonical_array_hash(value)
                           for key, value in sorted(sample.items())})


def select_checkpoint(summary_paths, plan):
    rows, reference, fixed_inputs = [], None, None
    for path in summary_paths:
        data = json.loads(Path(path).read_text())
        if data["plan_sha256"] != plan["plan_sha256"] or data["norm_stats_sha256"] != plan["normalization"]["sha256"]:
            raise ValueError(f"Cannot compare different validation protocols/statistics: {path}")
        if data.get("evaluation_valid"):
            digest = data.get("validation_input_and_noise_sha256")
            if not digest:
                raise ValueError(f"Missing actual validation input/noise evidence: {path}")
            if fixed_inputs is not None and digest != fixed_inputs:
                raise ValueError(f"Validation inputs or initial noise differ between checkpoints: {path}")
            fixed_inputs = digest
        metric = data.get("dataset_metrics")
        score = metric.get("selection_score") if metric else None
        eligible = (data.get("evaluation_valid") is True and data["label"] == "E1_wrist_pilot"
                    and data["optimizer_step"] in plan["selection"]["eligible_optimizer_steps"]
                    and score is not None and math.isfinite(score))
        row = {"summary": str(path), "label": data["label"], "optimizer_step": data["optimizer_step"],
               "checkpoint": data["checkpoint"], "score": score, "eligible": eligible}
        if data["label"] == "E0_wrist_frozen" and data.get("evaluation_valid"):
            if reference is not None:
                raise ValueError("Only one frozen E0 result is allowed")
            reference = row
        rows.append(row)
    eligible_rows = [row for row in rows if row["eligible"]]
    steps = [row["optimizer_step"] for row in eligible_rows]
    if len(steps) != len(set(steps)):
        raise ValueError("Duplicate candidate results for the same optimizer step")
    selected = min(eligible_rows, key=lambda row: (row["score"], row["optimizer_step"])) if eligible_rows else None
    return {"protocol": PROTOCOL, "plan_sha256": plan["plan_sha256"], "selection_rule": plan["selection"],
            "validation_input_and_noise_sha256": fixed_inputs,
            "status": "selected" if selected else "no_eligible_checkpoint", "selected": selected, "E0": reference,
            "E1_improved_over_E0": (selected["score"] < reference["score"]) if selected and reference else None,
            "all_declared_steps_evaluated": set(steps) == set(plan["selection"]["eligible_optimizer_steps"]),
            "comparison": rows, "test_split_used": False, "interpretation": "Offline action error only; no robot success estimate"}


def self_test():
    import numpy as np
    import tempfile
    stats = (np.zeros(7), np.ones(7))
    target = np.zeros((50, 32))
    prediction = np.zeros_like(target)
    prediction[:, :3] = 1.0  # xyz-only error: group-equal score is exactly 1/3, not 3/7.
    mask = np.arange(50) < 10
    baseline = score_prediction(prediction, target, mask, stats)
    assert abs(baseline["selection_score"] - 1/3) < 1e-12
    prediction[:, 7:] = 100000
    prediction[10:, :7] = -99999
    changed = score_prediction(prediction, target, mask, stats)
    assert baseline["selection_score"] == changed["selection_score"]
    assert "20" not in changed["point"] and changed["prefix"]["50"]["actual_valid_steps"] == 10
    try:
        score_prediction(prediction, target, np.zeros(50, dtype=bool), stats)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero-valid action block must fail")
    # One episode with three blocks and another with one: dataset must weight episodes equally.
    ep_metrics = mean_metrics([mean_metrics([{"selection_score": 0.0}]*3), {"selection_score": 2.0}])
    assert ep_metrics["selection_score"] == 1.0
    with tempfile.TemporaryDirectory(prefix="kinova_selection_") as directory:
        plan = {"plan_sha256": "fixed_plan", "normalization": {"sha256": "train_stats"},
                "selection": {"eligible_optimizer_steps": [500, 1000, 1500, 2000]}}
        files = []
        for label, step, score in [("E0_wrist_frozen", 0, 0.01), ("E1_wrist_pilot", 500, 0.5),
                                   ("E1_wrist_pilot", 1000, 0.2), ("E1_wrist_pilot", 1500, 0.2),
                                   ("E1_wrist_pilot", 2000, 0.4)]:
            path = Path(directory) / f"{step}.json"
            write_json(path, {"plan_sha256": "fixed_plan", "norm_stats_sha256": "train_stats", "label": label,
                              "optimizer_step": step, "checkpoint": f"checkpoint/{step}", "evaluation_valid": True,
                              "dataset_metrics": {"selection_score": score}, "validation_input_and_noise_sha256": "same_inputs"})
            files.append(path)
        selected = select_checkpoint(files, plan)
        assert selected["selected"]["optimizer_step"] == 1000
        assert selected["E1_improved_over_E0"] is False
        assert selected["all_declared_steps_evaluated"] is True
    return {"status": "passed", "checks": ["dimension_groups_equal_weight", "padding_dimensions_ignored",
                                            "tail_time_padding_ignored", "horizon_coverage_exact", "zero_valid_rejected",
                                            "episode_equal_weight", "checkpoint_tie_selects_earlier",
                                            "E0_comparison_only", "worse_than_E0_reported"], "gpu_used": False, "optimizer_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--output")
    select = sub.add_parser("select")
    select.add_argument("--plan", required=True)
    select.add_argument("--summaries", nargs="+", required=True)
    select.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--label", choices=["E0_wrist_frozen", "E1_wrist_pilot"], required=True)
    run.add_argument("--optimizer-step", type=int, required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--reference-prefix", help="Also save first validation episode/start0 with seed314159 for step-0/restoration comparisons")
    run.add_argument("--components-factory", default="evaluate_offline:default_components",
                     help="factory(label, checkpoint, plan) -> (model, transformed_validation_dataset, metadata)")
    run.add_argument("--resource-approved", action="store_true", help="Only pass after the supervisor proves GPU is idle/available")
    args = parser.parse_args()
    if args.command == "self-test":
        result = self_test()
        if args.output:
            write_json(args.output, result)
    else:
        plan = validate_plan(json.loads(Path(args.plan).read_text()))
        if args.command == "select":
            result = select_checkpoint(args.summaries, plan)
            write_json(args.output, result)
        else:
            if not args.resource_approved:
                raise RuntimeError("Model execution deferred: require --resource-approved after the resource guard passes")
            resource_status = enforce_resource_gate()
            if args.label == "E0_wrist_frozen" and args.optimizer_step != 0:
                raise ValueError("Frozen E0 must use optimizer-step 0")
            if args.label == "E1_wrist_pilot" and args.optimizer_step not in plan["selection"]["eligible_optimizer_steps"]:
                raise ValueError("E1 checkpoint is outside the fixed 500/1000/1500/2000 selection schedule")
            model, dataset, metadata = load_symbol(args.components_factory)(args.label, args.checkpoint, plan)
            metadata["resource_gate"] = resource_status
            if args.reference_prefix:
                from verify_checkpoint import capture_reference
                block = plan["episodes"][0]["blocks"][0]
                obs, actions = prepare_sample(dataset[block["global_index"]], block)
                capture_reference(model, obs, actions, args.reference_prefix, seed=314159,
                                  checkpoint_step=args.optimizer_step, restore_dtype="bfloat16",
                                  provenance={"label": args.label, "checkpoint": args.checkpoint,
                                              "plan_sha256": plan["plan_sha256"],
                                              "episode_id": plan["episodes"][0]["episode_id"], "start_frame": 0})
            result = evaluate_model(model, dataset, plan, args.output_dir, label=args.label,
                                    optimizer_step=args.optimizer_step, checkpoint=args.checkpoint, config_metadata=metadata)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
