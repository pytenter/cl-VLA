"""Freeze the Kinova validation protocol using metadata only (no model/GPU/test reads)."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL = "kinova-wrist-offline-v1"
DEFAULT_CANDIDATE = project_path('candidate', '')
DEFAULT_ROOT = project_path('kinova', '')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() == text:
            return
        raise FileExistsError(f"Refusing to replace a fixed protocol/result: {path}")
    with path.open("x") as stream:
        stream.write(text)


def seed_for(episode_id, start, seed, purpose):
    """Stable across Python processes, checkpoint order and model type."""
    payload = f"{PROTOCOL}|{episode_id}|{start}|{seed}|{purpose}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def build_plan(candidate_root, seeds=(1729,)):
    root = Path(candidate_root).resolve()
    repo = root / "lerobot/kinova_candidate/validation"
    episode_file = repo / "meta/episodes.jsonl"
    mapping_file = root / "reports/trajectory_mapping.json"
    norm_file = root / "assets/norm_stats.json"
    provenance_file = root / "assets/statistics_provenance.json"
    episodes = [json.loads(line) for line in episode_file.read_text().splitlines() if line.strip()]
    mappings = [row for row in json.loads(mapping_file.read_text()) if row["split"] == "validation"]
    by_index = {int(row["lerobot_episode_index"]): row for row in mappings}
    if len(episodes) != 20 or len(by_index) != 20 or len(mappings) != 20:
        raise ValueError("This predeclared experiment expects exactly 20 validation episodes")
    provenance = json.loads(provenance_file.read_text())
    if provenance["split"] != "train":
        raise ValueError("Normalization must come only from training episodes")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Provide a nonempty unique list of uint32 seeds before any evaluation")
    rows = []
    offset = 0
    seen_ids = set()
    for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
        index, length = int(episode["episode_index"]), int(episode["length"])
        mapping = by_index[index]
        eid = str(mapping["original_episode_id"])
        if length <= 0 or length != int(mapping["steps"]) or eid in seen_ids:
            raise ValueError(f"Episode length/identity mismatch: {index}/{eid}")
        seen_ids.add(eid)
        # Predeclared temporal coverage; no image, action or performance-based selection.
        starts = [("first", 0), ("mid", (length - 1) // 2), ("late", max(0, length - 10))]
        if len({start for _, start in starts}) != 3:
            raise ValueError("Episode too short for the fixed three distinct validation blocks")
        blocks = []
        for label, start in starts:
            valid = min(50, length - start)
            blocks.append({"label": label, "start_frame": start, "global_index": offset + start,
                           "valid_steps": valid, "time_mask": [i < valid for i in range(50)],
                           "point_horizons_present": [k for k in [1, 5, 10, 20, 50] if k <= valid],
                           "seeds": [{"seed": seed,
                                      "sampling_seed": seed_for(eid, start, seed, "sample"),
                                      "flow_matching_seed": seed_for(eid, start, seed, "flow")}
                                     for seed in seeds]})
        rows.append({"episode_index": index, "episode_id": eid, "candidate_episode_id": mapping["candidate_episode_id"],
                     "split": "validation", "length": length, "global_offset": offset,
                     "parquet_path": mapping["parquet_path"], "task_phase": "UNKNOWN", "blocks": blocks})
        offset += length
    plan = {
        "protocol": PROTOCOL, "status": "fixed_before_evaluation", "split": "validation",
        "candidate_root": str(root), "repo_root": str(repo), "episode_count": len(rows), "total_frames": offset,
        "action_horizon": 50, "real_action_dim": 7, "model_action_dim": 32,
        "start_rule": {"first": "0", "mid": "floor((episode_length-1)/2)", "late": "max(0,episode_length-10)",
                       "reason": "deterministic beginning/middle/trailing-10-action coverage; no selection by performance"},
        "point_horizons": [1, 5, 10, 20, 50], "prefix_horizons": [1, 5, 10, 20, 50],
        "seeds": list(seeds), "denoise_steps": 10,
        "randomization": "Each block/seed uses fixed SHA256-derived sampling and FM RNG keys; same keys and explicit initial noise for E0/E1; average all seeds, never pick best noise.",
        "camera_protocol": {"base_0_rgb": "zero image; mask=false", "left_wrist_0_rgb": "wrist RGB; mask=true",
                            "right_wrist_0_rgb": "zero image; mask=false", "rotation": "none",
                            "resize": "native resize_with_pad to 224x224; identical train/eval", "external_image_used": False},
        "normalization": {"path": str(norm_file), "sha256": sha256(norm_file), "source_split": "train",
                          "method": "native quantile: 2*(x-q01)/(q99-q01+1e-6)-1; no clipping",
                          "provenance_sha256": sha256(provenance_file)},
        "selection": {
            "metric": "first10_valid_normalized_group_equal_mse", "window": 10,
            "formula": "mean_episode(mean_block_seed(mean_valid_t<10((mean_xyz(error_norm^2)+mean_euler(error_norm^2)+gripper_error_norm^2)/3)))",
            "dimension_groups": {"xyz": [0, 1, 2], "euler": [3, 4, 5], "gripper": [6]},
            "weights": {"xyz": 1/3, "euler": 1/3, "gripper": 1/3},
            "episode_weight": "equal", "block_seed_weight_within_episode": "equal",
            "eligible_optimizer_steps": [500, 1000, 1500, 2000], "tie_break": "smaller optimizer step",
            "E0_role": "comparison only; never an E1 checkpoint candidate", "incomplete_or_nonfinite": "ineligible; no silent sample dropping",
            "no_valid_elements": "error; no division by zero"},
        "physical_reporting": {"xyz": "per-component RMSE/MAE in meters plus Euclidean RMSE",
                               "euler": "per-component increment RMSE/MAE in radians; no cumulative orientation or geodesic claim",
                               "gripper": "feedback-position RMSE/MAE in candidate units; not a controller command",
                               "aggregation": "MSE/MAE averaged within block then block/seed then episode then dataset; RMSE=sqrt(dataset MSE)",
                               "mixed_unit_sum": False},
        "task_phase_status": "UNKNOWN; temporal block labels are not identified grasp/place stages",
        "test_split_used": False, "ground_truth": "offline feedback-derived adjacent action increments; not robot success",
        "source_hashes": {"validation_episodes": sha256(episode_file), "trajectory_mapping": sha256(mapping_file)},
        "episodes": rows,
    }
    plan["plan_sha256"] = canonical_hash(plan)
    return plan


def validate_plan(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["protocol"] != PROTOCOL or canonical_hash(body) != plan["plan_sha256"]:
        raise ValueError("Validation plan content/hash mismatch")
    if plan["split"] != "validation" or plan["test_split_used"] is not False:
        raise ValueError("Only the predeclared validation split is allowed")
    if sha256(plan["normalization"]["path"]) != plan["normalization"]["sha256"]:
        raise ValueError("Training normalization statistics changed after protocol freeze")
    root = Path(plan["candidate_root"])
    if sha256(root / "lerobot/kinova_candidate/validation/meta/episodes.jsonl") != plan["source_hashes"]["validation_episodes"]:
        raise ValueError("Validation episode metadata changed after protocol freeze")
    if sha256(root / "reports/trajectory_mapping.json") != plan["source_hashes"]["trajectory_mapping"]:
        raise ValueError("Episode source mapping changed after protocol freeze")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", default=DEFAULT_ROOT + "/configs/validation_plan.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729])
    args = parser.parse_args()
    plan = build_plan(args.candidate_root, args.seeds)
    write_json(args.output, plan)
    print(json.dumps({"status": "prepared_not_evaluated", "plan": args.output, "plan_sha256": plan["plan_sha256"],
                      "episodes": plan["episode_count"], "blocks": sum(len(ep["blocks"]) for ep in plan["episodes"]),
                      "optimizer_steps": 0, "gpu_used": False, "test_split_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
