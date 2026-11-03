"""CPU audit of real transformed samples and native batches; no model arrays."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import gc
import json
from pathlib import Path
import time

import jax
import numpy as np
import pyarrow.parquet as pq
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import data_loader
from kinova_config import CANDIDATE_ROOT, ROOT, build_dataset, install_data_hooks, make_config
from kinova_model import KinovaObservation


def run_audit(splits):
    assert all(device.platform == "cpu" for device in jax.devices())
    started = time.monotonic()
    mapping = json.loads((CANDIDATE_ROOT / "reports/trajectory_mapping.json").read_text())
    config = make_config("smoke")
    tokenizer = PaligemmaTokenizer(200)
    report = {"status": "running", "scope": "Real native dataset/transform/batch format audit, not model evaluation",
              "test_usage": "format/boundary audit only; no parameter selection, gradients or predictions",
              "devices": [str(x) for x in jax.devices()], "splits": {}, "samples": [], "native_batches": [],
              "peak_cached_numeric_episodes": 1, "model_parameters_materialized": False}
    max_action_error = 0.0
    max_state_error = 0.0
    for split in splits:
        dataset = build_dataset(config, split=split)
        base = dataset
        while hasattr(base, "_dataset"):
            base = base._dataset
        rows = [row for row in mapping if row["split"] == split]
        assert len(dataset) == sum(row["steps"] for row in rows)
        stats = dataclasses.replace(config.data, split=split).create(config.assets_dirs, config.model).norm_stats
        offset = 0
        for row in rows:
            episode = int(row["lerobot_episode_index"])
            length = int(row["steps"])
            path = Path(row["parquet_path"])
            table = pq.read_table(path, columns=["action", "observation.state", "episode_index", "frame_index", "task_index"])
            assert table.num_rows == length
            raw_actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            raw_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            raw_episodes = np.asarray(table["episode_index"].to_pylist())
            raw_frames = np.asarray(table["frame_index"].to_pylist())
            task_ids = np.asarray(table["task_index"].to_pylist())
            assert np.all(raw_episodes == episode)
            np.testing.assert_array_equal(raw_frames, np.arange(length))
            for start in (0, length // 2, length - 1):
                item = dataset[offset + start]
                actual_episode = int(np.asarray(item["source_episode_index"]))
                actual_frame = int(np.asarray(item["source_frame_index"]))
                assert (actual_episode, actual_frame) == (episode, start)
                query = np.minimum(np.arange(start, start + 50), length - 1)
                expected_mask = np.arange(50) < length - start
                actual_mask = np.asarray(item["action_valid_mask"], dtype=bool)
                np.testing.assert_array_equal(actual_mask, expected_mask)
                np.testing.assert_array_equal(item["action_is_pad"], ~expected_mask)
                action_stats = stats["actions"]
                expected_actions = (raw_actions[query] - action_stats.q01) / (action_stats.q99 - action_stats.q01 + 1e-6) * 2 - 1
                actual_actions = np.asarray(item["actions"])
                np.testing.assert_allclose(actual_actions[:, :7], expected_actions, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(actual_actions[:, 7:], 0)
                assert actual_actions.shape == (50, 32) and np.isfinite(actual_actions).all()
                state_stats = stats["state"]
                expected_state = (raw_states[start] - state_stats.q01) / (state_stats.q99 - state_stats.q01 + 1e-6) * 2 - 1
                np.testing.assert_allclose(item["state"][:28], expected_state, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(item["state"][28:], 0)
                prompt = base.meta.tasks[int(task_ids[start])]
                expected_tokens, expected_token_mask = tokenizer.tokenize(prompt, expected_state)
                np.testing.assert_array_equal(item["tokenized_prompt"], expected_tokens)
                np.testing.assert_array_equal(item["tokenized_prompt_mask"], expected_token_mask)
                assert item["image_mask"] == {"base_0_rgb": False, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
                for camera, pixels in item["image"].items():
                    assert pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                    if camera != "left_wrist_0_rgb":
                        np.testing.assert_array_equal(pixels, 0)
                max_action_error = max(max_action_error, float(np.max(np.abs(actual_actions[:, :7] - expected_actions))))
                max_state_error = max(max_state_error, float(np.max(np.abs(item["state"][:28] - expected_state))))
                report["samples"].append({"split": split, "original_episode_id": row["original_episode_id"],
                                          "global_index": offset + start, "source_episode_index": actual_episode,
                                          "source_frame_index": actual_frame, "episode_length": length,
                                          "valid_time_count": int(actual_mask.sum()), "valid_element_count": int(actual_mask.sum()) * 7,
                                          "tokens": int(np.asarray(item["tokenized_prompt_mask"]).sum()),
                                          "action_query_min": int(query.min()), "action_query_max": int(query.max()),
                                          "passed": True})
            offset += length
            del table, raw_actions, raw_states, raw_episodes, raw_frames, task_ids
            gc.collect()
        report["splits"][split] = {"episodes": len(rows), "steps": len(dataset), "actual_samples_checked": len(rows) * 3}
        del dataset, base
        gc.collect()
        print(split, report["splits"][split], flush=True)

    # Also exercise DataLoaderImpl's real structured output, not just a custom
    # from_dict conversion.  Batches are deterministic train frame0/frame1.
    install_data_hooks()
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=2)
    for number, (observation, actions) in enumerate(loader):
        assert isinstance(observation, KinovaObservation)
        assert actions.shape == (1, 50, 32) and observation.action_valid_mask.shape == (1, 50)
        episodes = np.asarray(observation.source_episode_index).tolist()
        frames = np.asarray(observation.source_frame_index).tolist()
        assert episodes == [0] and frames == [number]
        report["native_batches"].append({"batch": number, "source_episode_index": episodes,
                                          "source_frame_index": frames, "actions_shape": list(actions.shape),
                                          "state_shape": list(observation.state.shape),
                                          "valid_time_count": np.asarray(observation.action_valid_mask).sum(-1).tolist(),
                                          "image_masks": {key: np.asarray(value).tolist() for key, value in observation.image_masks.items()}})
    report.update(status="passed", max_normalized_action_abs_error=max_action_error,
                  max_normalized_state_abs_error=max_state_error, seconds=time.monotonic() - started)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "reports/native_loader_validation.json"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"], choices=["train", "validation", "test"])
    args = parser.parse_args()
    report = run_audit(args.splits)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "splits", "max_normalized_action_abs_error", "max_normalized_state_abs_error", "seconds")}))


if __name__ == "__main__":
    main()
