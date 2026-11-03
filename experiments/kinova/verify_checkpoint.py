"""Compare fixed-input Kinova predictions/FM terms across a fresh checkpoint restore.

The trainer calls capture_reference on a read-only BF16 model view. The verify
command runs in a NEW process and never creates an optimizer or sends actions.
The compare command is CPU-only and also supports E0 vs zero-LoRA step-0 checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from evaluate_offline import canonical_array_hash, enforce_resource_gate, load_model, load_symbol, make_inference_functions
from validation_plan import PROTOCOL, canonical_hash, write_json


def paths(prefix):
    prefix = str(prefix)
    return Path(prefix + ".npz"), Path(prefix + ".json")


def flatten_arrays(value, prefix="obs", arrays=None, none_paths=None):
    import numpy as np
    arrays = {} if arrays is None else arrays
    none_paths = [] if none_paths is None else none_paths
    if value is None:
        none_paths.append(prefix)
    elif isinstance(value, dict):
        for key in sorted(value):
            if "/" in key:
                raise ValueError("Observation keys containing '/' cannot be serialized")
            flatten_arrays(value[key], prefix + "/" + key, arrays, none_paths)
    else:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object array is forbidden in fixed inputs: {prefix}")
        arrays[prefix] = array
    return arrays, none_paths


def unflatten_observation(arrays, none_paths):
    import jax.numpy as jnp
    from kinova_model import KinovaObservation
    output = {}
    for key in sorted([key for key in arrays if key.startswith("obs/")] + list(none_paths)):
        names = key.split("/")[1:]
        parent = output
        for name in names[:-1]:
            parent = parent.setdefault(name, {})
        parent[names[-1]] = None if key in none_paths else jnp.asarray(arrays[key])
    return KinovaObservation.from_dict(output)


def fixed_input_hash(arrays, none_paths):
    leaves = {key: canonical_array_hash(value) for key, value in arrays.items()
              if key.startswith("obs/") or key in ("actions", "initial_noise")}
    return canonical_hash({"leaves": leaves, "none_paths": sorted(none_paths)})


def capture_reference(model, observation, actions, output_prefix, *, seed=314159,
                      checkpoint_step=0, restore_dtype="bfloat16", num_steps=10,
                      provenance=None, initial_noise=None, flow_matching_seed=None):
    """Save fixed input and output in the current process; does NOT cast/mutate model.

    Caller must supply a BF16 inference view if restore_dtype='bfloat16'. To retain
    mixed checkpoint dtypes on both sides, use restore_dtype='preserve'. Returns
    paths to reference .npz/.json for verify in a separate Python process.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    npz_path, meta_path = paths(output_prefix)
    if npz_path.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to replace restore evidence: {output_prefix}")
    if restore_dtype not in ("bfloat16", "preserve"):
        raise ValueError("restore_dtype must explicitly be bfloat16 or preserve")
    if int(model.action_horizon) != 50 or int(model.action_dim) != 32:
        raise ValueError("Expected Kinova model geometry 50x32")
    if tuple(actions.shape) != (1, 50, 32):
        raise ValueError(f"Expected one fixed action block, got {actions.shape}")
    flow_seed = int(flow_matching_seed if flow_matching_seed is not None else (seed ^ 0x5A5A5A5A))
    rng = jax.random.key(seed)
    if initial_noise is None:
        initial_noise = jax.random.normal(rng, actions.shape, dtype=jnp.float32)
    initial_noise = jnp.asarray(initial_noise, dtype=jnp.float32)
    if initial_noise.shape != actions.shape:
        raise ValueError("Initial noise shape must match the model action block")
    start = time.monotonic()
    sample_fn, loss_fn = make_inference_functions(model)
    prediction = np.asarray(sample_fn(rng, observation, noise=initial_noise, num_steps=num_steps), dtype=np.float32)
    numerator, count = loss_fn(jax.random.key(flow_seed), observation, actions, train=False)
    numerator, count = float(numerator), float(count)
    expected = int(np.asarray(observation.action_valid_mask).sum()) * 7
    if expected <= 0 or count != expected or not np.isfinite(prediction).all() or not np.isfinite(numerator):
        raise ValueError(f"Invalid prediction/FM reference: count={count}, expected={expected}, numerator={numerator}")
    arrays, none_paths = flatten_arrays(observation.to_dict())
    arrays.update(actions=np.asarray(actions), initial_noise=np.asarray(initial_noise), prediction_normalized=prediction,
                  fm_numerator=np.asarray(numerator, dtype=np.float64), fm_count=np.asarray(count, dtype=np.float64))
    input_hash = fixed_input_hash(arrays, none_paths)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive binary open prevents accidentally overwriting existing evidence.
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    meta = {"protocol": PROTOCOL, "status": "captured", "checkpoint_step": int(checkpoint_step),
            "process_id": os.getpid(),
            "restore_dtype": restore_dtype, "sampling_seed": int(seed), "flow_matching_seed": flow_seed,
            "denoise_steps": int(num_steps), "fixed_input_sha256": input_hash,
            "observation_none_paths": none_paths, "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "prediction_shape": list(prediction.shape), "expected_valid_element_count": expected,
            "flow_matching_loss": numerator/count, "capture_seconds": time.monotonic()-start,
            "provenance": provenance or {}, "optimizer_steps_executed_by_verifier": 0, "robot_actions_sent": 0}
    write_json(meta_path, meta)
    return {"npz": str(npz_path), "metadata": str(meta_path), "fixed_input_sha256": input_hash}


def read_reference(prefix):
    import numpy as np
    npz_path, meta_path = paths(prefix)
    meta = json.loads(meta_path.read_text())
    if meta["protocol"] != PROTOCOL or hashlib.sha256(npz_path.read_bytes()).hexdigest() != meta["npz_sha256"]:
        raise ValueError("Reference metadata/data digest mismatch")
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if fixed_input_hash(arrays, meta["observation_none_paths"]) != meta["fixed_input_sha256"]:
        raise ValueError("Fixed reference input changed")
    return arrays, meta


def compare_references(reference_prefix, candidate_prefix, *, atol=1e-5, rtol=1e-5, comparison="checkpoint_restore"):
    import numpy as np
    before, before_meta = read_reference(reference_prefix)
    after, after_meta = read_reference(candidate_prefix)
    matching = ["fixed_input_sha256", "restore_dtype", "sampling_seed", "flow_matching_seed", "denoise_steps"]
    for key in matching:
        if before_meta[key] != after_meta[key]:
            raise ValueError(f"Cannot compare different fixed inputs/random conditions/dtypes: {key}")
    checks = {}
    for key in ["prediction_normalized", "fm_numerator", "fm_count"]:
        a, b = np.asarray(before[key], dtype=np.float64), np.asarray(after[key], dtype=np.float64)
        shape_equal = a.shape == b.shape
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        passed = shape_equal and finite and bool(np.allclose(a, b, atol=atol, rtol=rtol))
        checks[key] = {"passed": passed, "shape_equal": shape_equal, "finite": finite,
                       "max_abs_error": float(np.max(np.abs(a-b))) if shape_equal and finite else None,
                       "max_relative_error": float(np.max(np.abs(a-b)/np.maximum(np.abs(a), atol))) if shape_equal and finite else None}
    passed = all(row["passed"] for row in checks.values())
    fresh_process = (before_meta.get("process_id") is not None and after_meta.get("process_id") is not None
                     and before_meta["process_id"] != after_meta["process_id"])
    if comparison == "checkpoint_restore":
        passed = passed and fresh_process
    return {"protocol": PROTOCOL, "status": "passed" if passed else "failed", "comparison": comparison,
            "reference": str(reference_prefix), "candidate": str(candidate_prefix), "atol": atol, "rtol": rtol,
            "fixed_input_sha256": before_meta["fixed_input_sha256"], "checks": checks,
            "checkpoint_step": after_meta["checkpoint_step"], "restore_dtype": after_meta["restore_dtype"],
            "fresh_process_verified": fresh_process,
            "test_split_used": False, "robot_actions_sent": 0,
            "scope": "Fixed-input prediction and masked FM terms; optimizer/state restoration requires separate trainer evidence"}


def verify(args):
    arrays, meta = read_reference(args.reference_prefix)
    if not args.resource_approved:
        raise RuntimeError("GPU/model work deferred until supervisor resource check passes (--resource-approved)")
    resource_status = enforce_resource_gate()
    import jax.numpy as jnp
    factory = load_symbol(args.config_factory)
    config = factory(args.config_mode)
    model = load_model(config, args.checkpoint, restore_dtype=meta["restore_dtype"])
    obs = unflatten_observation(arrays, meta["observation_none_paths"])
    capture_reference(model, obs, jnp.asarray(arrays["actions"]), args.output_prefix,
                      seed=meta["sampling_seed"], checkpoint_step=meta["checkpoint_step"],
                      restore_dtype=meta["restore_dtype"], num_steps=meta["denoise_steps"],
                      initial_noise=arrays["initial_noise"], flow_matching_seed=meta["flow_matching_seed"],
                      provenance={"restored_checkpoint": str(Path(args.checkpoint).resolve()),
                                  "config_mode": args.config_mode, "new_process": True,
                                  "resource_gate": resource_status})
    return compare_references(args.reference_prefix, args.output_prefix, atol=args.atol, rtol=args.rtol)


def self_test():
    import numpy as np
    import tempfile
    arrays = {"obs/state": np.arange(32, dtype=np.float32)[None], "actions": np.zeros((1,50,32), np.float32),
              "initial_noise": np.zeros((1,50,32), np.float32), "prediction_normalized": np.zeros((1,50,32), np.float32),
              "fm_numerator": np.asarray(7.0), "fm_count": np.asarray(7.0)}
    with tempfile.TemporaryDirectory(prefix="kinova_restore_compare_") as directory:
        prefixes = [Path(directory)/name for name in ("before", "after", "changed")]
        for i, prefix in enumerate(prefixes):
            data = {key: value.copy() for key, value in arrays.items()}
            if i == 2:
                data["prediction_normalized"][0, 0, 0] = 0.1
            npz_path, meta_path = paths(prefix)
            np.savez_compressed(npz_path, **data)
            write_json(meta_path, {"protocol": PROTOCOL, "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                                   "fixed_input_sha256": fixed_input_hash(data, []), "observation_none_paths": [],
                                   "restore_dtype": "bfloat16", "sampling_seed": 1, "flow_matching_seed": 2,
                                   "denoise_steps": 10, "checkpoint_step": 32, "process_id": 1000+i})
        assert compare_references(*prefixes[:2])["status"] == "passed"
        assert compare_references(prefixes[0], prefixes[2])["status"] == "failed"
        assert compare_references(prefixes[0], prefixes[0])["status"] == "failed"
    return {"status": "passed", "checks": ["same_reference_passes", "changed_prediction_fails", "same_process_restore_rejected", "pickle_forbidden", "fixed_input_digest_checked"],
            "gpu_used": False, "optimizer_steps": 0, "actual_checkpoint_restored": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--report")
    for name in ("verify", "compare"):
        child = sub.add_parser(name)
        child.add_argument("--reference-prefix", required=True)
        child.add_argument("--report", required=True)
        child.add_argument("--atol", type=float, default=1e-5)
        child.add_argument("--rtol", type=float, default=1e-5)
        if name == "verify":
            child.add_argument("--checkpoint", required=True)
            child.add_argument("--config-factory", default="kinova_config:make_config")
            child.add_argument("--config-mode", choices=["smoke", "pilot", "frozen"], default="smoke")
            child.add_argument("--output-prefix", required=True)
            child.add_argument("--resource-approved", action="store_true")
        else:
            child.add_argument("--candidate-prefix", required=True)
            child.add_argument("--comparison", choices=["checkpoint_restore", "base_vs_zero_lora"], default="checkpoint_restore")
    args = parser.parse_args()
    if args.command == "self-test":
        result = self_test()
    elif args.command == "compare":
        result = compare_references(args.reference_prefix, args.candidate_prefix, atol=args.atol, rtol=args.rtol, comparison=args.comparison)
    else:
        result = verify(args)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
