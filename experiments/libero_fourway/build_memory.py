"""LIBERO single-arm adaptation of the archive's multimodal Prototype Replay V2.

This builds memory assets only: it never constructs an optimizer or updates model
parameters. Prototype selection reuses the native context/action feature path,
feature fusion, segment merge, and robust temporal k-center implementation. The
single-arm physical descriptor explicitly interprets state[3:6] as a rotvec.
Uniform memory matches the reference prototype's episode list and per-episode
sample counts; it never substitutes a visual-only prototype implementation.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


STAGES = "ABCD"
FEATURE_WEIGHTS = (0.25, 0.50, 0.25)
REQUIRED_HELPERS = (
    "_extract_model_features", "_pool_local_action_features", "_fuse_behavior_features",
    "_select_coverage_representatives", "_build_kinematic_segments", "_moving_average_columns", "MotionSignals",
)


def stage_letter(value: str | int) -> str:
    value = str(value).upper()
    if value in STAGES and len(value) == 1:
        return value
    if value in ("0", "1", "2", "3"):
        return STAGES[int(value)]
    raise ValueError(f"Expected stage A..D or zero-based 0..3, got {value!r}")


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _array(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _rgb(value: Any) -> np.ndarray:
    array = _array(value)
    if array.ndim != 3:
        raise ValueError(f"Expected RGB rank 3, got {array.shape}")
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected exactly three RGB channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or array.min() < -1e-5 or array.max() > 1.00001:
            raise ValueError("Floating LeRobot images must be finite in [0,1]")
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if array.dtype != np.uint8:
        raise TypeError(f"Expected uint8/float RGB, got {array.dtype}")
    return np.ascontiguousarray(array)


def _raw_sample(sample: dict, horizon: int) -> dict:
    # These are the integration hook's raw keys, before normalization or padding.
    required = ("image", "wrist_image", "state", "actions", "prompt")
    if missing := [key for key in required if key not in sample]:
        raise KeyError(f"LIBERO raw sample missing {missing}; refusing RoboTwin schema fallback")
    state = _array(sample["state"], np.float32)
    actions = _array(sample["actions"], np.float32)
    if state.shape != (8,) or actions.shape != (horizon, 7):
        raise ValueError(f"Expected state8/actions[{horizon},7], got {state.shape}/{actions.shape}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("State/actions contain nonfinite values")
    prompt = sample["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = prompt.item()
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Empty or non-string language prompt")
    return dict(image=_rgb(sample["image"]), wrist_image=_rgb(sample["wrist_image"]),
                state=state, actions=actions, prompt=prompt)


def _column(dataset, name: str) -> np.ndarray:
    # A formatted HF column access invokes hf_transform_to_torch on its batch,
    # which can decode BOTH image columns for every row. Drop the format AND
    # select just this numeric column before reading metadata.
    values = dataset.hf_dataset.with_format(None).select_columns([name])[name]
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _episode_layout(dataset, source_mapping: dict | None = None) -> tuple[Any, list[dict]]:
    """Unwrap native transformed datasets/subsets without inferring IDs from order."""
    mapped = np.arange(len(dataset), dtype=np.int64)
    base = dataset
    while not hasattr(base, "hf_dataset"):
        if hasattr(base, "indices") and hasattr(base, "dataset"):
            mapped = np.asarray(base.indices, dtype=np.int64)[mapped]
            base = base.dataset
        elif hasattr(base, "_dataset"):
            base = base._dataset
        else:
            raise TypeError(f"Cannot locate LeRobot dataset beneath {type(base).__name__}")
    episode_ids = _column(base, "episode_index")[mapped]
    frame_ids = _column(base, "frame_index")[mapped]
    global_ids = _column(base, "index")[mapped]
    result = []
    for episode_id in dict.fromkeys(episode_ids.tolist()):
        positions = np.flatnonzero(episode_ids == episode_id)
        if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise ValueError(f"Episode {episode_id} is interleaved in the selected dataset")
        if np.any(np.diff(frame_ids[positions]) != 1):
            raise ValueError(f"Episode {episode_id} has gaps/reordering in raw frame indices")
        episode_record = base.meta.episodes.get(int(episode_id), {})
        original_id = episode_record.get("source_episode_index")
        if original_id is None:
            original_id = episode_record.get("original_episode_index")
        mapped_original = (source_mapping or {}).get((base.meta.repo_id, int(episode_id)))
        if original_id is not None and mapped_original is not None and int(original_id) != mapped_original:
            raise ValueError(f"Conflicting source episode mapping for {base.meta.repo_id}/{episode_id}")
        if original_id is None:
            original_id = mapped_original
        result.append({"episode_index": int(episode_id),
                       "source_episode_index": None if original_id is None else int(original_id),
                       "dataset_indices": positions.tolist(),
                       "source_row_indices": mapped[positions].tolist(),
                       "global_indices": global_ids[positions].tolist(),
                       "frame_indices": frame_ids[positions].tolist()})
    return base, result


def _load_native_helpers(root: Path, *, noise_samples: int = 2):
    engine = root / "source/pi05"
    if not (engine / "scripts/build_robotwin_prototype_buffer_v2.py").is_file():
        deploy = Path(os.environ.get("PI05_DEPLOY_ROOT", project_path('deploy', '')))
        engine = deploy / "source/pi05"
    path = engine / "scripts/build_robotwin_prototype_buffer_v2.py"
    if not path.is_file():
        raise FileNotFoundError(f"Native multimodal prototype implementation unavailable: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("libero_native_prototype_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if missing := [name for name in REQUIRED_HELPERS if not hasattr(module, name)]:
        raise AttributeError(f"Native Prototype V2 missing {missing}; no visual-only fallback")
    # The archive's plain @jax.jit closure captures the entire NNX model. That
    # makes multi-GB weights HLO constants and can exhaust host RAM during XLA
    # compilation. Native module_jit instead supplies frozen model state as a
    # dynamic pytree argument, exactly like the already validated inference path.
    from openpi.shared import nnx_utils
    def safe_factory(model, task_id: int):
        compiled = nnx_utils.module_jit(model.extract_behavior_features, static_argnames=("task_id",))
        calls = 0
        frames = 0
        def extract(observation, actions, noise, flow_time):
            nonlocal calls, frames
            result = compiled(observation, actions, noise, flow_time, task_id=int(task_id))
            module.jax.block_until_ready(result)
            calls += 1
            if calls % noise_samples == 0:
                frames += int(observation.state.shape[0])
                print(f"FEATURE_BATCH_COMPLETE batches={calls//noise_samples} frames={frames}", flush=True)
            return result
        return extract
    module._make_behavior_feature_fn = safe_factory
    provenance = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "reused_functions": list(REQUIRED_HELPERS),
                  "runtime_fixes": [
                      "HF numeric/metadata reads remove format and select only requested columns; no whole image decoding",
                      "_make_behavior_feature_fn uses native nnx_utils.module_jit with dynamic model-state arguments instead of full-model JAX closure constants",
                  ], "runtime_fix_source": str(Path(__file__).resolve()),
                  "runtime_fix_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return module, provenance


def _motion(states: np.ndarray, native, args, width_endpoints: tuple[float, float]):
    """One-arm xyz/relative-rotvec/gripper-width dynamics, in physical raw state."""
    length = len(states)
    future = np.minimum(np.arange(length) + args.kinematic_window, length - 1)
    translation = states[future, :3].astype(np.float64) - states[:, :3]
    rotations = Rotation.from_rotvec(states[:, 3:6].astype(np.float64))
    # Spatial relative rotation, unlike subtracting rotvec/Euler components.
    rotation = (rotations[future] * rotations.inv()).as_rotvec()
    width = np.abs(states[:, 6:8].astype(np.float64)).sum(axis=1)
    lower, upper = width_endpoints
    normalized_width = (width - lower) / (upper - lower) if upper - lower > 1e-8 else np.zeros(length)
    grip_delta = normalized_width[future] - normalized_width
    scaled = np.concatenate((translation / args.translation_threshold,
                             rotation / args.rotation_threshold,
                             grip_delta[:, None] / args.gripper_threshold), axis=1)
    smoothed = native._moving_average_columns(scaled, args.kinematic_window)
    smoothed = np.clip(smoothed, -args.descriptor_clip, args.descriptor_clip)
    strongest = np.argmax(np.abs(smoothed), axis=1)
    signed = smoothed[np.arange(length), strongest]
    modes = np.where(np.abs(signed) >= 1.0, 1 + 2 * strongest + (signed < 0), 0).astype(np.int32)
    descriptors = np.concatenate((smoothed, normalized_width[:, None],
                                  np.linalg.norm(smoothed, axis=1, keepdims=True)), axis=1).astype(np.float32)
    signals = native.MotionSignals(descriptors=descriptors, modes=modes[:, None],
                                   translation=translation[:, None].astype(np.float32),
                                   rotation=rotation[:, None].astype(np.float32),
                                   gripper_delta=grip_delta[:, None].astype(np.float32))
    return signals, width


def _save_sample(raw_dataset, episode: dict, offset: int, *, output: Path, number: int,
                 stage: str, horizon: int, selection: dict) -> dict:
    dataset_index = episode["dataset_indices"][offset]
    sample = _raw_sample(raw_dataset[dataset_index], horizon)
    relative = f"samples/sample_{number:07d}.npz"
    np.savez_compressed(output / relative, image=sample["image"], wrist_image=sample["wrist_image"],
                        state=sample["state"], actions=sample["actions"], action_chunk=sample["actions"],
                        prompt=np.asarray(sample["prompt"]), task=np.asarray(sample["prompt"]),
                        cl_task_id=np.asarray(STAGES.index(stage), np.int32),
                        task_id=np.asarray(STAGES.index(stage), np.int32))
    return {"sample_path": relative, "prompt": sample["prompt"], "task": sample["prompt"],
            "task_name": sample["prompt"], "cl_task_id": STAGES.index(stage),
            "repo_id": f"libero_fourway/task_{stage}",
            "episode_index": episode["episode_index"],
            "source_episode_index": episode["source_episode_index"],
            "episode_frame_index": episode["frame_indices"][offset],
            "episode_local_offset": offset, "source_dataset_index": dataset_index,
            "source_row_index": episode["source_row_indices"][offset],
            "global_frame_index": episode["global_indices"][offset], **selection}


def validate_buffer(output: Path, horizon: int = 10) -> dict:
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines() if line.strip()]
    meta = json.loads((output / "meta.json").read_text())
    if not records:
        raise ValueError("Empty memory buffer")
    seen = set()
    for record in records:
        identity = record["episode_index"], record["episode_frame_index"]
        if identity in seen:
            raise ValueError(f"Duplicate selected source frame {identity}")
        seen.add(identity)
        path = (output / record["sample_path"]).resolve()
        if output.resolve() not in path.parents:
            raise ValueError("Manifest sample escapes buffer directory")
        with np.load(path, allow_pickle=False) as data:
            assert data["state"].shape == (8,)
            assert data["actions"].shape == (horizon, 7)
            assert np.isfinite(data["state"]).all() and np.isfinite(data["actions"]).all()
            for key in ["image", "wrist_image"]:
                assert data[key].ndim == 3 and data[key].shape[-1] == 3 and data[key].dtype == np.uint8
            assert str(data["prompt"].item()) == record["prompt"]
            assert int(data["cl_task_id"]) == meta["cl_task_id"]
    summary = {"status": "passed", "samples": len(records), "unique_source_frames": len(seen),
               "episodes": len({record["episode_index"] for record in records}),
               "parameter_updates": 0, "actions_shape": [horizon, 7]}
    json_dump(output / "validation.json", summary)
    return summary


def build(args) -> dict:
    root = Path(os.environ["PI05_LIBERO_ROOT"]).expanduser().resolve()
    stage = stage_letter(args.stage)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Buffer exists; refusing to overwrite: {output}")
    if args.episodes != 10 or args.top_k != 3 or args.seed != 42:
        raise ValueError("This four-way protocol fixes first 10 episodes, topK=3, and seed=42")
    if not 0.0 < args.flow_time < 1.0 or args.feature_batch_size < 1 or args.noise_samples < 1:
        raise ValueError("Invalid flow time or feature extraction batch/noise count")
    if min(args.translation_threshold, args.rotation_threshold, args.gripper_threshold) <= 0:
        raise ValueError("Motion thresholds must be positive")
    from libero_integration import install_hooks
    install_hooks()
    from experiment_config import make_config
    from openpi.training import data_loader
    from openpi.shared import normalize
    config = make_config(args.group, stage)
    if config.model.action_horizon != 10:
        raise ValueError("The memory protocol requires action_horizon=10")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != f"libero_fourway/task_{stage}":
        raise ValueError(f"Unexpected source repo {data_config.repo_id!r}")
    raw_dataset = data_loader.create_torch_dataset(data_config, 10, config.model)
    mapping_path = root / "reports/episode_mapping.json"
    source_mapping = {}
    if mapping_path.is_file():
        source_mapping = {(row["repo_id"], int(row["episode_index"])): int(row["source_episode_index"])
                          for row in json.loads(mapping_path.read_text())}
    base, layout = _episode_layout(raw_dataset, source_mapping)
    if len(layout) < args.episodes:
        raise ValueError(f"Only {len(layout)} episodes, require fixed first {args.episodes}")
    episodes = layout[:args.episodes]
    if any(episode["source_episode_index"] is None for episode in episodes):
        raise ValueError("Original source episode mapping is required in metadata or reports/episode_mapping.json")
    print(f"EPISODE_LAYOUT_READY repo={data_config.repo_id} total_episodes={len(layout)} selected={len(episodes)} selected_frames={sum(len(e['dataset_indices']) for e in episodes)}", flush=True)
    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if (checkpoint / "params").is_dir():
            checkpoint /= "params"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
    if args.group == "prototype" and checkpoint is None:
        raise ValueError("Prototype extraction requires this task's trained --checkpoint params")
    reference_records = None
    reference_meta = None
    reference = None
    if args.group == "uniform":
        if not args.reference_buffer:
            raise ValueError("Uniform selection requires --reference-buffer for exact per-episode budgets")
        reference = Path(args.reference_buffer).expanduser().resolve()
        reference_meta = json.loads((reference / "meta.json").read_text())
        if reference_meta["method"] != "multimodal_prototype_v2_libero_rotvec":
            raise ValueError("Reference is not the specified multimodal prototype method")
        if reference_meta["cl_task_id"] != STAGES.index(stage) or reference_meta["repo_id"] != data_config.repo_id:
            raise ValueError("Reference buffer stage/repo mismatch")
        reference_records = [json.loads(line) for line in (reference / "manifest.jsonl").read_text().splitlines() if line.strip()]
        if reference_meta["selected_episode_indices"] != [e["episode_index"] for e in episodes]:
            raise ValueError("Uniform/reference source episode lists differ")
        originals = [e["source_episode_index"] for e in episodes]
        if reference_meta["source_episode_indices"] != originals:
            raise ValueError("Uniform/reference original-source episode mappings differ")
    output.mkdir(parents=True)
    (output / "samples").mkdir()
    records = []
    segments_manifest = []
    feature_provenance = None
    width_endpoints = None
    episode_budgets = {}
    if data_config.norm_stats is None:
        raise ValueError("Memory feature extraction/replay requires the old task's training-only norm stats")
    normalize.save(output / "assets" / data_config.repo_id, data_config.norm_stats)
    if args.group == "prototype":
        native, feature_provenance = _load_native_helpers(root, noise_samples=args.noise_samples)
        states_by_episode = {}
        state_values = base.hf_dataset.with_format(None).select_columns(["state"])["state"]
        all_states = np.asarray(state_values, dtype=np.float32)
        if all_states.shape != (len(base), 8) or not np.isfinite(all_states).all():
            raise ValueError("Raw dataset state column is not finite state8")
        for episode in episodes:
            states = all_states[episode["source_row_indices"]]
            states_by_episode[episode["episode_index"]] = states
        widths = np.concatenate([np.abs(s[:, 6:8]).sum(axis=1) for s in states_by_episode.values()])
        width_endpoints = (float(widths.min()), float(widths.max()))
        print(f"NUMERIC_STATES_READY state_shape={all_states.shape} selected_width_endpoints={width_endpoints}; no images read for numeric columns", flush=True)
        transformed = data_loader.transform_dataset(raw_dataset, data_config)
        import jax.numpy as jnp
        from openpi.models import model as model_lib
        model = config.model.load(model_lib.restore_params(checkpoint, dtype=jnp.bfloat16))
        model.eval()
        print(f"MODEL_RESTORED checkpoint={checkpoint}; dynamic-state module_jit enabled", flush=True)
        if not hasattr(model, "extract_behavior_features"):
            raise AttributeError("Native model lacks action-conditioned features; refusing visual-only fallback")
        all_frame_indices = [i for e in episodes for i in e["dataset_indices"]]
        print(f"FEATURE_EXTRACTION_START frames={len(all_frame_indices)} batch_size={args.feature_batch_size} noise_samples={args.noise_samples}", flush=True)
        context, action_tokens = native._extract_model_features(
            transformed, all_frame_indices, model=model, task_id=STAGES.index(stage),
            batch_size=args.feature_batch_size, flow_time=args.flow_time,
            noise_seed=args.seed, noise_samples=args.noise_samples)
        print(f"FEATURE_EXTRACTION_DONE context_shape={context.shape} action_tokens_shape={action_tokens.shape}", flush=True)
        if len(context) != len(all_frame_indices) or len(action_tokens) != len(all_frame_indices):
            raise ValueError("Model feature count differs from selected raw frames")
        if not np.isfinite(context).all() or not np.isfinite(action_tokens).all():
            raise ValueError("Model feature extraction returned nonfinite values")
        (output / "diagnostics").mkdir()
        feature_offset = 0
        for episode in episodes:
            ep = episode["episode_index"]
            states = states_by_episode[ep]
            length = len(states)
            signals, width = _motion(states, native, args, width_endpoints)
            segments = native._build_kinematic_segments(
                signals, persistence_frames=args.persistence_frames,
                min_boundary_gap=args.min_boundary_gap, min_segment_len=args.min_segment_len,
                max_segment_len=0)
            if not segments or segments[0][0] != 0 or segments[-1][1] != length:
                raise ValueError("Motion segments do not cover the full episode")
            if any(a[1] != b[0] for a, b in zip(segments[:-1], segments[1:])):
                raise ValueError("Motion segments have gaps or overlap")
            episode_context = context[feature_offset:feature_offset + length]
            episode_action = action_tokens[feature_offset:feature_offset + length]
            feature_offset += length
            before = len(records)
            diagnostic = {"context": episode_context, "action_tokens": episode_action,
                          "physical": signals.descriptors, "state": states, "gripper_width": width,
                          "translation_delta": signals.translation, "relative_rotation_rotvec": signals.rotation,
                          "segments": np.asarray(segments, np.int32)}
            np.savez_compressed(output / "diagnostics" / f"episode_{ep:06d}.npz", **diagnostic)
            for number, (start, end) in enumerate(segments):
                local_action = native._pool_local_action_features(episode_action, start, end, 10)
                fused, _ = native._fuse_behavior_features(
                    episode_context[start:end], local_action, signals.descriptors[start:end],
                    context_weight=FEATURE_WEIGHTS[0], action_weight=FEATURE_WEIGHTS[1],
                    physical_weight=FEATURE_WEIGHTS[2])
                choice = native._select_coverage_representatives(
                    fused, top_k=3, temporal_min_gap=args.temporal_min_gap,
                    outlier_mad_scale=args.outlier_mad_scale)
                sid = f"ep{ep:06d}_seg{number:04d}"
                for rank, local in enumerate(choice.selected_indices):
                    record = _save_sample(raw_dataset, episode, start + int(local), output=output,
                                          number=len(records), stage=stage, horizon=10,
                                          selection={"selection_method": "robust_temporal_k_center",
                                                     "segment_id": sid, "segment_start": start,
                                                     "segment_end": end, "segment_rank": rank})
                    records.append(record)
                segments_manifest.append({"segment_id": sid, "episode_index": ep, "start": start,
                    "end": end, "selected_offsets": [start + int(i) for i in choice.selected_indices],
                    "inliers": choice.inlier_count, "outliers": choice.outlier_count,
                    "effective_temporal_gap": choice.effective_temporal_gap,
                    "coverage_error": choice.coverage_error, "coverage_gains": choice.coverage_gains})
            episode_budgets[str(ep)] = len(records) - before
            print(f"prototype stage={stage} episode={ep} frames={length} segments={len(segments)} samples={len(records)-before}", flush=True)
    else:
        budgets = collections.Counter(int(record["episode_index"]) for record in reference_records)
        for episode in episodes:
            ep = episode["episode_index"]
            count = budgets[ep]
            length = len(episode["dataset_indices"])
            if not 1 <= count <= length:
                raise ValueError(f"Reference budget {count} invalid for episode {ep}, length {length}")
            # Endpoint-inclusive deterministic sampling; counts <= length gives unique offsets.
            offsets = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
            if len(np.unique(offsets)) != count:
                raise ValueError("Uniform sampling unexpectedly produced repeated source indices")
            for rank, offset in enumerate(offsets):
                records.append(_save_sample(raw_dataset, episode, int(offset), output=output,
                               number=len(records), stage=stage, horizon=10,
                               selection={"selection_method": "uniform_time", "uniform_rank": rank}))
            episode_budgets[str(ep)] = count
            print(f"uniform stage={stage} episode={ep} matched_samples={count}", flush=True)
        if len(records) != len(reference_records):
            raise ValueError("Uniform total budget differs from reference")
    with (output / "manifest.jsonl").open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "segments.jsonl").open("w") as file:
        for record in segments_manifest:
            file.write(json.dumps(record) + "\n")
    meta = {"format": "pi05_libero_memory_v1", "stage": stage, "cl_task_id": STAGES.index(stage),
            "repo_id": data_config.repo_id,
            "task_name": records[0]["prompt"],
            "method": "multimodal_prototype_v2_libero_rotvec" if args.group == "prototype" else "uniform_time_matched_per_episode",
            "source_checkpoint": None if checkpoint is None else str(checkpoint),
            "reference_buffer": None if reference is None else str(reference),
            "selected_episode_indices": [e["episode_index"] for e in episodes],
            "source_episode_indices": [e["source_episode_index"] for e in episodes],
            "source_episode_indices_known": all(e["source_episode_index"] is not None for e in episodes),
            "episode_budget": 10, "episode_selection": "first 10 in training repo order; no evaluation selection",
            "seed": args.seed, "sample_count": len(records), "saved_frame_count": len(records),
            "source_frame_count": sum(len(e["dataset_indices"]) for e in episodes),
            "episode_sample_counts": episode_budgets,
            "state_dim": 8, "action_dim": 7, "action_horizon": 10,
            "state_schema": "xyz_m[0:3], orientation_rotvec_rad[3:6], gripper_joint_qpos[6:8]",
            "chunk_boundary_policy": "native LeRobot future chunk confined to current episode and end-padded",
            "norm_assets_dir": "assets", "norm_asset_id": data_config.repo_id,
            "feature_weights": dict(zip(("context", "action", "physical"), FEATURE_WEIGHTS)),
            "feature_fusion_used_for_selection": args.group == "prototype",
            "native_helper_provenance": feature_provenance,
            "physical_descriptor": {"rotation": "Rotation.from_rotvec(r_future) * Rotation.from_rotvec(r_current).inv(); as_rotvec",
                "gripper": "abs(qpos[0])+abs(qpos[1]), empirical min/max normalized using same 10 training episodes",
                "gripper_width_endpoints": width_endpoints,
                "gripper_width_endpoints_are_physical_calibration": False,
                "dimensions": "scaled dxyz3,drotvec3,dnormalized_width1,current_normalized_width1,motion_norm1",
                "kinematic_window": args.kinematic_window, "translation_threshold_m": args.translation_threshold,
                "rotation_threshold_rad": args.rotation_threshold, "normalized_gripper_threshold": args.gripper_threshold,
                "persistence_frames": args.persistence_frames, "min_boundary_gap": args.min_boundary_gap,
                "min_segment_len": args.min_segment_len, "descriptor_clip": args.descriptor_clip},
            "selection": {"top_k": 3, "temporal_min_gap": args.temporal_min_gap,
                "outlier_mad_scale": args.outlier_mad_scale, "flow_time": args.flow_time,
                "noise_samples": args.noise_samples, "noise_seed": args.seed},
            "parameter_updates": 0}
    if reference_meta is not None:
        meta["reference_feature_weights"] = reference_meta["feature_weights"]
        meta["reference_physical_descriptor"] = reference_meta["physical_descriptor"]
        meta["reference_native_helper_provenance"] = reference_meta["native_helper_provenance"]
        meta["reference_manifest_sha256"] = hashlib.sha256((reference / "manifest.jsonl").read_bytes()).hexdigest()
    json_dump(output / "meta.json", meta)
    result = validate_buffer(output, horizon=10)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("prototype", "uniform"), required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--checkpoint", help="Trained params directory, or step directory containing params/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-buffer", help="Prototype memory for uniform per-episode budget matching")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--noise-samples", type=int, default=2)
    parser.add_argument("--flow-time", type=float, default=0.5)
    parser.add_argument("--kinematic-window", type=int, default=5)
    parser.add_argument("--translation-threshold", type=float, default=0.03)
    parser.add_argument("--rotation-threshold", type=float, default=0.05)
    parser.add_argument("--gripper-threshold", type=float, default=0.1)
    parser.add_argument("--descriptor-clip", type=float, default=5.0)
    parser.add_argument("--persistence-frames", type=int, default=5)
    parser.add_argument("--min-boundary-gap", type=int, default=5)
    parser.add_argument("--min-segment-len", type=int, default=8)
    parser.add_argument("--temporal-min-gap", type=int, default=5)
    parser.add_argument("--outlier-mad-scale", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
