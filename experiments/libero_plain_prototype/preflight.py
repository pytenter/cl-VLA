"""CPU configuration/data/frozen-memory audit; never initialize a robot model."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    os.replace(temp, path)


def require(value, message):
    if not value:
        raise ValueError(message)


def source_identity(baseline):
    rows = []
    for filename, expected in baseline["files"].items():
        path = Path(filename)
        actual = sha(path) if path.is_file() else None
        rows.append({"path": filename, "expected_sha256": expected, "actual_sha256": actual,
                     "unchanged": actual == expected})
    require(all(v["unchanged"] for v in rows), "Original experiment source/results/fixed-memory identity changed")
    return rows


def check(root):
    started = time.monotonic()
    old = Path(project_path('libero_fourway', ''))
    require(root != old.resolve() and old.resolve() not in root.parents, "Output root must be independent")
    protocol = load(root / "experiment.json")
    identity_path = root / "manifests/original_readonly_identity.json"
    identity = load(identity_path)
    before = source_identity(identity)
    require(protocol["group"] == "plain_prototype", "Wrong experiment group")
    result = load(old / "RESULTS.json")
    require(result["status"] == "complete" and result["completed_evaluations"] == 40
            and not result.get("invalid_results"), "Original four-group results are not complete/valid")

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import experiment_config as configs
    from openpi.models import gemma

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    require(configs.ROOT.resolve() == root, "PI05_LIBERO_ROOT does not point to the new experiment")
    datasets, configuration_rows, memory_rows = {}, [], {}
    for stage in "ABCD":
        index = "ABCD".index(stage)
        expected_weights = (Path(protocol["base_params"]) if index == 0 else root / "checkpoints" /
                            f"pi05_libero_plain_prototype_{'ABCD'[index-1]}" / "seed42/10000/params")
        # B+ checkpoints are intentionally planned paths before their predecessor trains.
        cfg = configs.make_config(stage=stage, weights=str(expected_weights), resume=False)
        reference = configs.baseline.make_config("plain", stage, weights=str(expected_weights), resume=False)
        model = cfg.model
        require(dataclasses.asdict(model) == dataclasses.asdict(reference.model), f"{stage}: model differs from original ordinary-LoRA config")
        require(str(cfg.freeze_filter) == str(reference.freeze_filter), f"{stage}: freeze filter differs from original plain")
        require((model.paligemma_variant, model.action_expert_variant) == ("gemma_2b_lora", "gemma_300m_lora"), "CL variants forbidden")
        flags = ("cl_pali_lora_bank", "cl_action_lora_bank", "cl_pali_block_weight_bank", "cl_action_block_weight_bank",
                 "cl_freeze_lora_a", "cl_freeze_shared_lora", "cl_freeze_pali_lora_b", "cl_freeze_action_lora_b")
        require(not any(getattr(model, field, False) for field in flags), f"{stage}: unexpected CL bank/gate/A-freezing")
        require(cfg.name == f"pi05_libero_plain_prototype_{stage}", "Checkpoint configuration name mismatch")
        require(Path(cfg.weight_loader.params_path).resolve() == expected_weights.resolve(), "Wrong previous-stage weights")
        if index == 0:
            require((expected_weights / "_METADATA").is_file(), "Original base checkpoint metadata missing")
        require(cfg.num_train_steps == 10000 and cfg.batch_size == 2 and cfg.seed == 42, "Training budget differs")
        require(cfg.lambda_kd == 0 and cfg.ema_decay is None, "KD/EMA must be disabled")
        require(model.pi05 and model.discrete_state_input and model.action_horizon == 10
                and model.action_dim == 32 and model.max_token_len == 200, "Model input/horizon differs")
        lr = dataclasses.asdict(cfg.lr_schedule)
        require(lr["warmup_steps"] == 500 and lr["decay_steps"] == 10000
                and lr["peak_lr"] == 5e-5 and lr["decay_lr"] == 5e-6, "Learning-rate protocol differs")
        require(float(cfg.optimizer.weight_decay) == 0 and float(cfg.optimizer.clip_gradient_norm) == 1, "AdamW settings differ")
        adapters = {}
        for variant, rank in ((model.paligemma_variant, 16), (model.action_expert_variant, 32)):
            adapters[variant] = {name: dataclasses.asdict(value) for name, value in gemma.get_config(variant).lora_configs.items()}
            require(all(v["rank"] == rank and v["alpha"] == rank for v in adapters[variant].values()), "Ordinary LoRA rank/alpha differs")
        buffers = tuple(str(old / "buffers/prototype" / f"task_{key}") for key in "ABCD"[:index])
        rehearsal = cfg.rehearsal
        require(rehearsal.enabled == (index > 0) and tuple(rehearsal.buffer_dirs) == buffers, "Future/current task replay or wrong buffer source")
        require(rehearsal.source == "offline_buffer" and rehearsal.loss_weight == .5
                and rehearsal.replay_every_n_steps == 4 and rehearsal.sample_strategy == "round_robin", "Replay budget differs")
        require(not cfg.data.extra_delta_transform, "Existing delta actions must not be delta-transformed again")
        data = cfg.data.create(cfg.assets_dirs, model)
        require(data.norm_stats is not None, "Missing actual task training norm assets")
        repo_id = f"libero_fourway/task_{stage}"
        data_root = Path(data.root) if data.root else Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
        require(data_root.resolve() == (old / "datasets" / repo_id).resolve(), "Data must be a read-only reference to the original task repo")
        info = load(data_root / "meta/info.json")
        episodes = [json.loads(line) for line in (data_root / "meta/episodes.jsonl").read_text().splitlines() if line]
        tasks = [json.loads(line) for line in (data_root / "meta/tasks.jsonl").read_text().splitlines() if line]
        expected = protocol["tasks"][stage]
        require(info["total_episodes"] == len(episodes) == expected["episodes"] and info["total_frames"] == expected["steps"], "Dataset episode/frame count differs")
        require(tuple(info["features"]["state"]["shape"]) == (8,) and tuple(info["features"]["actions"]["shape"]) == (7,), "Wrong LIBERO raw schema")
        require(len(tasks) == 1 and tasks[0]["task"] == expected["instruction"], "Task instruction differs")
        require(configs.TASKS[stage] == (expected["suite"], expected["task_id"], expected["instruction"]), "Evaluation task/suite differs")
        payloads = []
        for episode in episodes:
            ep = episode["episode_index"]
            path = data_root / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
            pf = pq.ParquetFile(path)
            values = pf.read(columns=["state", "actions"], use_threads=False)
            state = np.asarray(values["state"].to_pylist(), dtype=np.float32)
            action = np.asarray(values["actions"].to_pylist(), dtype=np.float32)
            require(state.shape == (episode["length"], 8) and action.shape == (episode["length"], 7)
                    and np.isfinite(state).all() and np.isfinite(action).all(), "Invalid finite raw dataset vectors")
            payloads.append({"episode_index": ep, "rows": pf.metadata.num_rows,
                             "file": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                             "numeric_sha256": hashlib.sha256(state.tobytes() + action.tobytes()).hexdigest()})
        norm_path = old / "assets" / repo_id / "norm_stats.json"
        datasets[stage] = {"repo_id": repo_id, "root": str(data_root.resolve()), "episodes": len(episodes),
                           "steps": sum(v["rows"] for v in payloads), "metadata": [stamp(data_root / "meta" / name) for name in ("info.json", "episodes.jsonl", "tasks.jsonl")],
                           "norm_assets": stamp(norm_path), "numeric_payloads": payloads}
        configuration_rows.append({"stage": stage, "config_name": cfg.name, "model_matches_original_plain": True,
                                   "freeze_filter_matches_original_plain": True, "ordinary_lora": adapters,
                                   "expected_weight_source": str(expected_weights), "weight_source_exists_at_preflight": expected_weights.is_dir(),
                                   "weight_source_role": "original base; stageA fully retrains" if index == 0 else "planned NEW-group preceding-stage path; existence is not a training completion claim",
                                   "replay_buffers": list(buffers), "teacher_KD_enabled": False,
                                   "replay_updates_if_10000steps": 2500 if index else 0,
                                   "old_sample_draws_if_batch2": 5000 if index else 0,
                                   "model_initialized": False})
        write(root / "reports/configurations" / f"{cfg.name}_preflight.json", dataclasses.asdict(cfg))

    for stage in "ABC":
        folder = old / "buffers/prototype" / f"task_{stage}"
        expected = protocol["replay"]["buffers"][stage]
        meta = load(folder / "meta.json")
        manifest = folder / "manifest.jsonl"
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        require(sha(manifest) == expected["manifest_sha256"], "Frozen sample manifest hash differs")
        require(meta["sample_count"] == len(records) == expected["sample_count"], "Frozen replay sample count differs")
        require(meta["selected_episode_indices"] == list(range(10)) and meta["source_episode_indices"] == expected["source_episode_indices"], "Frozen source episode selection differs")
        require(meta["stage"] == stage and meta["repo_id"] == f"libero_fourway/task_{stage}"
                and meta["cl_task_id"] == "ABCD".index(stage), "Replay task identity differs")
        require(meta["method"] == "multimodal_prototype_v2_libero_rotvec"
                and meta["feature_weights"] == protocol["replay"]["feature_weights"], "Prototype selector provenance differs")
        selector = old / "checkpoints" / f"pi05_libero_prototype_{stage}" / "seed42/10000/params"
        require(Path(meta["source_checkpoint"]).resolve() == selector.resolve() and (selector / "_METADATA").is_file(), "Expected old CL-selector checkpoint metadata missing")
        norm_path = folder / meta["norm_assets_dir"] / meta["norm_asset_id"] / "norm_stats.json"
        reference_norm = old / "assets" / meta["repo_id"] / "norm_stats.json"
        require(load(norm_path) == load(reference_norm), "Replay statistics differ from matching old task")
        seen, per_episode, samples = set(), collections.Counter(), []
        for row in records:
            path = (folder / row["sample_path"]).resolve()
            require(folder.resolve() in path.parents, "Replay sample escapes frozen buffer")
            pair = (row["episode_index"], row["episode_frame_index"])
            require(pair not in seen and row["episode_index"] in range(10), "Duplicate or outside-budget replay frame")
            seen.add(pair)
            per_episode[str(row["episode_index"])] += 1
            require(row["source_episode_index"] == meta["source_episode_indices"][row["episode_index"]], "Original episode mapping differs")
            with np.load(path, allow_pickle=False) as sample:
                require(sample["state"].shape == (8,) and sample["actions"].shape == (10, 7), "Replay shape differs")
                require(np.isfinite(sample["state"]).all() and np.isfinite(sample["actions"]).all(), "Nonfinite replay data")
                require(str(sample["prompt"].item()) == protocol["tasks"][stage]["instruction"], "Replay prompt differs")
                require(int(sample["cl_task_id"]) == "ABCD".index(stage), "Replay ID differs")
                for view in ("image", "wrist_image"):
                    image = sample[view]
                    require(image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8, "Wrong raw replay RGB format")
            samples.append({**stamp(path), "episode_index": row["episode_index"],
                            "source_episode_index": row["source_episode_index"], "episode_frame_index": row["episode_frame_index"]})
        require(dict(per_episode) == meta["episode_sample_counts"], "Per-episode replay count differs")
        memory_rows[stage] = {"sample_count": len(records), "meta": stamp(folder / "meta.json"),
                              "manifest": stamp(manifest), "selected_episode_indices": meta["selected_episode_indices"],
                              "source_episode_indices": meta["source_episode_indices"], "episode_sample_counts": dict(per_episode),
                              "source_checkpoint": str(selector), "selector_type": "old CL-LoRA prototype model; not new ordinary model",
                              "selector_checkpoint_metadata": stamp(selector / "_METADATA"), "norm_assets": stamp(norm_path),
                              "samples": samples, "parameter_updates_by_preflight": 0}
    fixed_identity = root / "manifests/fixed_replay_memory_identity.json"
    compact_memory = {s: {"manifest_sha256": v["manifest"]["sha256"], "meta_sha256": v["meta"]["sha256"],
                          "norm_sha256": v["norm_assets"]["sha256"], "samples": v["samples"]} for s, v in memory_rows.items()}
    if fixed_identity.exists():
        require(load(fixed_identity) == compact_memory, "Frozen replay sample identity changed since first preflight")
    else:
        write(fixed_identity, compact_memory)
    after = source_identity(identity)
    require(before == after, "Old experiment changed during CPU preflight")
    output = {"status": "passed", "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "protocol": protocol["protocol"], "group": "plain_prototype", "root": str(root),
              "configurations": configuration_rows, "datasets": datasets, "fixed_prototype_memories": memory_rows,
              "old_source_identity_manifest": stamp(identity_path), "old_source_checks": after,
              "old_source_results_unchanged": True, "old_results_sha256": sha(old / "RESULTS.json"),
              "dataset_episodes": sum(v["episodes"] for v in datasets.values()),
              "dataset_steps": sum(v["steps"] for v in datasets.values()),
              "replay_sample_count": sum(v["sample_count"] for v in memory_rows.values()),
              "A_fully_retrained_from_original_base": "configuration verified; training not executed by preflight",
              "model_initializations": 0, "optimizer_steps": 0, "gpu_used": False, "robot_commands_sent": 0,
              "scope": "CPU config exports, raw numerical columns, frozen NPZ samples/norms, protected source/results SHA; no model initialization/training/evaluation",
              "elapsed_seconds": time.monotonic() - started}
    write(root / "reports/preflight.json", output)
    print(json.dumps({k: output[k] for k in ("status", "dataset_episodes", "dataset_steps", "replay_sample_count", "old_source_results_unchanged", "model_initializations", "optimizer_steps", "elapsed_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PI05_LIBERO_ROOT", project_path('libero_plain_prototype', ''))))
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        check(root)
    except Exception as error:
        write(root / "reports/preflight_failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
              "model_initializations": 0, "optimizer_steps": 0, "gpu_used": False})
        raise


if __name__ == "__main__":
    main()
