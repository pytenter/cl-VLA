"""Explicit pi05 LIBERO policy server; load one checkpoint/task bank per process."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
from pathlib import Path
import time


def load_factory(spec):
    module_name, separator, name = spec.partition(":")
    if not separator or not module_name or not name:
        raise ValueError("config-factory must be module_name:callable_name")
    return getattr(importlib.import_module(module_name), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-factory", default="openpi.training.config:get_config", help="Callable taking config-name and returning a TrainConfig")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats-dir", help="Directory containing eval task's norm_stats.json; else checkpoint assets are used")
    parser.add_argument("--bank-id", type=int)
    parser.add_argument("--expected-language")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--metadata-out", help="Optional JSON file written after checkpoint and transforms load successfully")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.shared import normalize

    config = load_factory(args.config_factory)(args.config_name)
    if not bool(getattr(config.model, "pi05", False)):
        raise ValueError("Refusing to serve a non-pi05 config for the pi05 experiment")
    if args.denoise_steps <= 0:
        raise ValueError("denoise-steps must be positive")
    horizon = int(config.model.action_horizon)
    if horizon != 10:
        raise ValueError(f"This reproduction protocol expects action_horizon=10, got {horizon}")
    if args.bank_id is not None and not 0 <= args.bank_id < int(getattr(config.model, "cl_num_tasks", 1)):
        raise ValueError("bank-id is outside configured task bank capacity")
    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Native JAX checkpoint params missing: {checkpoint / 'params'}")
    norm_stats = None
    norm_file = None
    if args.norm_stats_dir:
        norm_dir = Path(args.norm_stats_dir).resolve()
        norm_file = norm_dir / "norm_stats.json"
        norm_stats = normalize.load(norm_dir)
    kwargs = {"num_steps": args.denoise_steps}
    if args.bank_id is not None:
        kwargs["task_id"] = args.bank_id
    started = time.monotonic()
    policy = policy_config.create_trained_policy(config, checkpoint, norm_stats=norm_stats, sample_kwargs=kwargs)
    # Native Policy has one deterministic PRNG stream for the evaluation session.
    # No per-episode reseeding: matches the archived benchmark's persistent client/server.
    policy._rng = jax.random.key(args.seed)
    bank_enabled = bool(getattr(config.model, "cl_pali_lora_bank", False) or getattr(config.model, "cl_action_lora_bank", False)
                        or getattr(config.model, "cl_pali_block_weight_bank", False) or getattr(config.model, "cl_action_block_weight_bank", False))
    metadata = {"protocol": "pi05-libero-v1", "model_type": "pi05", "config_name": args.config_name,
                "config_factory": args.config_factory, "checkpoint": str(checkpoint), "task_bank_id": args.bank_id,
                "task_bank_enabled": bank_enabled, "action_horizon": horizon, "returned_action_dim": 7,
                "model_action_dim": int(config.model.action_dim), "input_state_dim": 8,
                "discrete_state_input": bool(config.model.discrete_state_input), "max_token_len": int(config.model.max_token_len),
                "seed": args.seed, "seed_policy": "one persistent native Policy RNG stream", "denoise_steps": args.denoise_steps,
                "expected_language": args.expected_language, "norm_stats_file": str(norm_file) if norm_file else "checkpoint/assets resolved by native factory",
                "norm_stats_sha256": hashlib.sha256(norm_file.read_bytes()).hexdigest() if norm_file else None,
                "load_seconds": time.monotonic() - started}

    class ValidatedPolicy:
        def infer(self, obs):
            for key in ["observation/image", "observation/wrist_image"]:
                image = np.asarray(obs[key])
                if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Expected RGB uint8 224x224 input {key}, got {image.shape}/{image.dtype}")
            state = np.asarray(obs["observation/state"])
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError(f"Expected finite 8D LIBERO state, got {state.shape}")
            if args.expected_language and str(obs.get("prompt")) != args.expected_language:
                raise ValueError("Prompt does not match this evaluation task")
            out = policy.infer(obs)
            actions = np.asarray(out["actions"])
            if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite action chunk {(horizon, 7)}, got {actions.shape}")
            return out

        def reset(self):
            return None

    if args.metadata_out:
        p = Path(args.metadata_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    logging.info("pi05 LIBERO server ready: %s", json.dumps(metadata, ensure_ascii=False))
    server = websocket_policy_server.WebsocketPolicyServer(ValidatedPolicy(), host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
