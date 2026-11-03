"""Evaluate one fixed LIBERO task against a pi05 websocket policy.

This process belongs in the simulator environment. It does not import JAX/OpenPI
training code, load checkpoints, train models, or access physical robots.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np


TASKS = {
    "A": ("libero_spatial", 6, "pick up the black bowl next to the cookie box and place it on the plate"),
    "B": ("libero_spatial", 0, "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    "C": ("libero_object", 8, "pick up the chocolate pudding and place it in the basket"),
    "D": ("libero_goal", 7, "turn on the stove"),
}
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
NOOP = [0.0] * 6 + [-1.0]


def quaternion_to_axis_angle(quaternion):
    """LIBERO xyzw quaternion -> three axis-angle components, without mutating obs."""
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"Invalid quaternion shape/value: {quat}")
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(quat[:3] * (2.0 * math.acos(w)) / denominator, dtype=np.float32)


def prepare_observation(obs, language, resize_size=224):
    from openpi_client import image_tools

    images = {}
    for output_key, source_key in [("observation/image", "agentview_image"), ("observation/wrist_image", "robot0_eye_in_hand_image")]:
        image = np.asarray(obs[source_key])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"Unexpected {source_key}: {image.shape}, {image.dtype}")
        # Both views must use the same 180-degree rotation as the RLDS conversion.
        image = np.ascontiguousarray(image[::-1, ::-1])
        images[output_key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    state = np.concatenate((np.asarray(obs["robot0_eef_pos"]), quaternion_to_axis_angle(obs["robot0_eef_quat"]), np.asarray(obs["robot0_gripper_qpos"]))).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite 8D LIBERO state, got {state.shape}")
    return {**images, "observation/state": state, "prompt": str(language)}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    tmp.replace(path)


def create_client(host, port, timeout, inference_timeout):
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import websockets.sync.client

    class BoundedClient(WebsocketClientPolicy):
        def _wait_for_server(self):
            deadline = time.monotonic() + timeout
            while True:
                try:
                    connection = websockets.sync.client.connect(self._uri, compression=None, max_size=None, open_timeout=min(10, timeout))
                    return connection, msgpack_numpy.unpackb(connection.recv(timeout=inference_timeout))
                except (OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"No policy server at {self._uri} within {timeout}s")
                    time.sleep(1)

        def infer(self, obs):
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv(timeout=inference_timeout)
            if isinstance(response, str):
                raise RuntimeError("Policy server error:\n" + response)
            return msgpack_numpy.unpackb(response)

        def close(self):
            self._ws.close()

    return BoundedClient(host, port)


def resolve_task(args):
    if args.task_label:
        suite, task_id, language = TASKS[args.task_label]
        for supplied, expected, name in [(args.suite, suite, "suite"), (args.task_id, task_id, "task-id"), (args.expected_language, language, "expected-language")]:
            if supplied is not None and supplied != expected:
                raise ValueError(f"{name} disagrees with task {args.task_label}: {supplied!r} != {expected!r}")
        return suite, task_id, language
    if args.suite is None or args.task_id is None or args.expected_language is None:
        raise ValueError("Supply --task-label A/B/C/D or all of --suite, --task-id, --expected-language")
    return args.suite, args.task_id, args.expected_language


def evaluate(args):
    # Must be configured before importing robosuite / PyOpenGL / MuJoCo.
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
        os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
    if os.environ["MUJOCO_GL"] == "osmesa":
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio

    suite_name, task_id, language = resolve_task(args)
    if args.trials <= 0 or args.replan_steps <= 0 or args.num_steps_wait < 0:
        raise ValueError("trials/replan-steps must be positive; num-steps-wait must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    task = suite.get_task(task_id)
    if task.language != language:
        raise ValueError(f"Task identity mismatch: expected {language!r}, got {task.language!r}")
    initial_states = suite.get_task_init_states(task_id)
    if len(initial_states) < args.trials:
        raise ValueError(f"Only {len(initial_states)} fixed initial states for {args.trials} trials")
    max_steps = MAX_STEPS[suite_name] if args.max_steps is None else args.max_steps
    if max_steps <= 0:
        raise ValueError("max-steps must be positive")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    episode_file = output / "episodes.jsonl"
    if episode_file.exists():
        raise FileExistsError(f"Refusing to mix evaluations: {episode_file}. Select a new output-dir.")
    protocol = {"status": "connecting", "suite": suite_name, "task_id": task_id, "language": language,
                "trials": args.trials, "seed": args.seed, "evaluation_valid": False}
    write_json(output / "evaluation.json", protocol)
    client = create_client(args.policy_host, args.policy_port, args.connect_timeout, args.inference_timeout)
    env = None
    records = []
    try:
        server_metadata = client.get_server_metadata()
        if server_metadata.get("protocol") != "pi05-libero-v1" or server_metadata.get("model_type") != "pi05":
            raise ValueError(f"Expected the explicit pi05 LIBERO server, got metadata {server_metadata}")
        if args.expected_bank_id is not None and server_metadata.get("task_bank_id") != args.expected_bank_id:
            raise ValueError(f"Task-bank mismatch: expected {args.expected_bank_id}, got {server_metadata.get('task_bank_id')}")
        if server_metadata.get("expected_language") not in (None, language):
            raise ValueError("Server expected language disagrees with evaluation task")
        if int(server_metadata["action_horizon"]) < args.replan_steps:
            raise ValueError("replan-steps exceeds model action horizon")
        protocol = {
            "status": "running", "task_label": args.task_label, "suite": suite_name, "task_id": task_id,
            "language": language, "seed": args.seed, "trials": args.trials, "max_policy_steps": max_steps,
            "num_steps_wait": args.num_steps_wait, "replan_steps": args.replan_steps,
            "render_resolution": 256, "policy_resolution": args.resize_size, "mujoco_gl": os.environ["MUJOCO_GL"],
            "initial_state_indices": list(range(args.trials)), "server_metadata": server_metadata,
            "image_rotation": "180 degrees, both cameras", "state": "xyz + axis-angle + 2 gripper qpos (8D)",
            "action_processing": "native pi05 7D action passed directly; no OpenVLA gripper flip or extra delta transform",
        }
        write_json(output / "evaluation.json", protocol)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=args.render_gpu_device_id)
        env.seed(args.seed)
        for episode_idx in range(args.trials):
            started = time.monotonic()
            success = False
            error = None
            steps = 0
            queries = 0
            executed_actions = []
            query_steps = []
            queue = deque()
            video = None
            state_vector = np.asarray(initial_states[episode_idx])
            init_hash = hashlib.sha256(np.ascontiguousarray(state_vector).tobytes()).hexdigest()
            try:
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                client.reset()
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(NOOP)
                if args.video_mode == "all" or (args.video_mode == "first" and episode_idx == 0):
                    video = imageio.get_writer(output / f"episode_{episode_idx:03d}.mp4", fps=args.video_fps)
                for step_index in range(max_steps):
                    if video is not None:
                        video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    if not queue:
                        request = prepare_observation(obs, language, args.resize_size)
                        if queries == 0 and episode_idx < args.save_input_samples:
                            np.savez_compressed(output / f"episode_{episode_idx:03d}_input.npz", image=request['observation/image'], wrist_image=request['observation/wrist_image'], state=request['observation/state'], prompt=np.asarray(language))
                        actions = np.asarray(client.infer(request)["actions"], dtype=np.float32)
                        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < args.replan_steps or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid policy output: {actions.shape}, finite={np.isfinite(actions).all()}")
                        queue.extend(actions[:args.replan_steps])
                        queries += 1
                        query_steps.append(step_index)
                    action = np.array(queue.popleft(), copy=True)
                    executed_actions.append(action)
                    obs, _, done, _ = env.step(action.tolist())
                    steps += 1
                    if done:
                        success = True
                        if video is not None:
                            video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                        break
            except Exception:
                error = traceback.format_exc()
            finally:
                if video is not None:
                    video.close()
            actions = np.asarray(executed_actions, dtype=np.float32).reshape(-1, 7)
            np.savez_compressed(output / f"episode_{episode_idx:03d}_actions.npz", actions=actions, query_policy_steps=np.asarray(query_steps, dtype=np.int64))
            record = {"episode_index": episode_idx, "initial_state_index": episode_idx, "initial_state_sha256": init_hash,
                      "success": success, "policy_steps": steps, "queries": queries, "seconds": time.monotonic() - started,
                      "error": error, "action_min": actions.min(0).tolist() if len(actions) else None,
                      "action_max": actions.max(0).tolist() if len(actions) else None}
            records.append(record)
            with episode_file.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            successes = sum(r["success"] for r in records)
            errors = sum(r["error"] is not None for r in records)
            summary = {**protocol, "status": "failed" if error else ("complete" if len(records) == args.trials else "running"),
                       "episodes_completed": len(records), "successes": successes, "errors": errors,
                       "success_rate": successes / len(records), "evaluation_valid": errors == 0}
            write_json(output / "evaluation.json", summary)
            logging.info("Episode %d/%d: success=%s, %d/%d, errors=%d", episode_idx + 1, args.trials, success, successes, len(records), errors)
            if error:
                raise RuntimeError(f"Evaluation infrastructure/policy error in episode {episode_idx}; not a valid task-failure score.\n{error}")
        return summary
    except BaseException:
        failure = {**protocol, "status": "failed", "evaluation_valid": False,
                   "episodes_completed": len(records), "error": traceback.format_exc()}
        write_json(output / "evaluation.json", failure)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logging.exception("LIBERO environment cleanup failed")
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-label", choices=sorted(TASKS))
    parser.add_argument("--suite", choices=sorted(MAX_STEPS))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--expected-language")
    parser.add_argument("--expected-bank-id", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glx"], default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--connect-timeout", type=float, default=180)
    parser.add_argument("--inference-timeout", type=float, default=300)
    parser.add_argument("--video-mode", choices=["all", "first", "none"], default="first")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--save-input-samples", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
