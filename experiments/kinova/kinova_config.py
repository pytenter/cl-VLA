"""Dedicated candidate-D0 wrist-only Kinova configuration, without source edits."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi import transforms
from openpi.models import model as model_lib
from openpi.shared import normalize
from openpi.training import config as cfg, data_loader as loader_lib, optimizer, weight_loaders

from kinova_model import FrozenKinovaPi05Config, KinovaObservation, KinovaPi05Config
from native_weights import CompleteWeightLoader


ROOT = Path(os.environ.get("KINOVA_EXPERIMENT_ROOT", project_path('kinova', '')))
CANDIDATE_ROOT = Path(os.environ.get("KINOVA_CANDIDATE_ROOT", project_path('candidate', '')))
BASE_PARAMS = Path(os.environ.get("KINOVA_BASE_PARAMS", project_path('base_params', '')))
ASSET_ID = "kinova_gen3_candidate_v1"
HEAD_NAMES = {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}


@dataclasses.dataclass(frozen=True)
class KinovaInputs:
    """Preserve measured labels and a reliable LeRobot boundary mask."""

    def __call__(self, data):
        def rgb(value):
            value = np.asarray(value)
            if value.ndim != 3:
                raise ValueError(f"Expected 3D RGB image, got {value.shape}")
            if value.shape[0] == 3:
                value = np.moveaxis(value, 0, -1)
            if np.issubdtype(value.dtype, np.floating):
                if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                    raise ValueError("LeRobot floating images must be finite and in [0,1]")
                value = np.rint(value * 255).astype(np.uint8)
            if value.shape[-1] != 3 or value.dtype != np.uint8:
                raise ValueError("Expected uint8 RGB image")
            return value

        wrist = rgb(data["wrist"])
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (28,) or not np.isfinite(state).all():
            raise ValueError("Kinova state must have 28 finite coordinates")
        # Empty cameras use the native image masks.  Do not duplicate wrist
        # pixels into a fabricated base-camera observation.
        result = {
            "state": state, "prompt": data["prompt"],
            "image": {"base_0_rgb": np.zeros_like(wrist), "left_wrist_0_rgb": wrist,
                      "right_wrist_0_rgb": np.zeros_like(wrist)},
            "image_mask": {"base_0_rgb": np.False_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            is_pad = np.asarray(data["action_is_pad"], dtype=bool)
            if actions.shape != (50, 7) or is_pad.shape != (50,):
                raise ValueError(f"Expected action[50,7]/is_pad[50], got {actions.shape}/{is_pad.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("Non-finite Kinova actions")
            if np.any(is_pad[:-1] & ~is_pad[1:]):
                raise ValueError("Expected suffix-only episode-boundary action padding")
            result.update(actions=actions, action_is_pad=is_pad, action_valid_mask=~is_pad)
            for key in ("source_episode_index", "source_frame_index"):
                if key not in data:
                    raise ValueError(f"Missing batch provenance {key}")
                result[key] = np.asarray(data[key], dtype=np.int32).reshape(())
        return result


@dataclasses.dataclass(frozen=True)
class KinovaOutputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :7]}


@dataclasses.dataclass(frozen=True)
class KinovaDataConfig(cfg.DataConfig):
    root: str | None = None
    split: str = "train"
    source_candidate_root: str = str(CANDIDATE_ROOT)
    view_mode: str = "wrist_only"


@dataclasses.dataclass(frozen=True)
class KinovaDataFactory(cfg.DataConfigFactory):
    repo_id: str = "kinova_candidate/train"
    split: str = "train"
    candidate_root: str = str(CANDIDATE_ROOT)
    view_mode: str = "wrist_only"

    def __post_init__(self):
        if self.split not in ("train", "validation", "test"):
            raise ValueError("Unknown Kinova split")
        object.__setattr__(self, "repo_id", f"kinova_candidate/{self.split}")

    def create(self, assets_dirs: Path, model_config: Any) -> KinovaDataConfig:
        if self.split not in ("train", "validation", "test") or self.view_mode != "wrist_only":
            raise ValueError("This experiment only supports split-isolated wrist-only data")
        if not model_config.pi05 or not model_config.discrete_state_input:
            raise ValueError("Kinova requires π0.5 discrete state tokens")
        if (model_config.action_horizon, model_config.action_dim) != (50, 32):
            raise ValueError("This candidate protocol fixes horizon=50/action_dim=32")
        root = Path(self.candidate_root)
        repo = f"kinova_candidate/{self.split}"
        stats = normalize.load(root / "assets")
        if tuple(stats["state"].mean.shape) != (28,) or tuple(stats["actions"].mean.shape) != (7,):
            raise ValueError("Wrong normalization assets: expected Kinova state28/actions7")
        # These read-only assets were computed on train once, regardless of the
        # split being evaluated.  Never recompute from validation/test.
        return KinovaDataConfig(
            repo_id=repo, asset_id=ASSET_ID, root=str(root / "lerobot" / repo), split=self.split,
            source_candidate_root=str(root), view_mode=self.view_mode,
            norm_stats=stats, use_quantile_norm=True, action_sequence_keys=("action",), prompt_from_task=True,
            repack_transforms=transforms.Group(inputs=(transforms.RepackTransform({
                "wrist": "observation.images.wrist", "state": "observation.state",
                "actions": "action", "action_is_pad": "action_is_pad", "prompt": "prompt",
                "source_episode_index": "episode_index", "source_frame_index": "frame_index",
            }),)),
            data_transforms=transforms.Group(inputs=(KinovaInputs(),), outputs=(KinovaOutputs(),)),
            model_transforms=cfg.ModelTransformFactory()(model_config),
        )


_ORIGINAL_CREATE = None
_ORIGINAL_ITER = None


def install_hooks():
    """Process-local loader hooks; unrelated native configurations delegate."""
    global _ORIGINAL_CREATE, _ORIGINAL_ITER
    if getattr(loader_lib.create_torch_dataset, "_kinova_hook", False):
        return
    _ORIGINAL_CREATE = loader_lib.create_torch_dataset
    _ORIGINAL_ITER = loader_lib.DataLoaderImpl.__iter__

    def create_torch_dataset(data_config, action_horizon, model_config):
        if not isinstance(data_config, KinovaDataConfig):
            return _ORIGINAL_CREATE(data_config, action_horizon, model_config)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(data_config.repo_id, root=Path(data_config.root))
        for key, expected in (("observation.state", (28,)), ("action", (7,))):
            if tuple(meta.features[key]["shape"]) != expected:
                raise ValueError(f"Unexpected feature shape for {key}: {meta.features[key]}")
        dataset = LeRobotDataset(data_config.repo_id, root=Path(data_config.root), delta_timestamps={
            "action": [step / meta.fps for step in range(action_horizon)]})
        return loader_lib.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(meta.tasks)])

    def iterate(self):
        if not isinstance(self._data_config, KinovaDataConfig):
            yield from _ORIGINAL_ITER(self)
            return
        for batch in self._data_loader:
            yield KinovaObservation.from_dict(batch), batch["actions"]

    create_torch_dataset._kinova_hook = True
    loader_lib.create_torch_dataset = create_torch_dataset
    loader_lib.DataLoaderImpl.__iter__ = iterate


install_data_hooks = install_hooks


def build_dataset(config, split="validation"):
    """Random-access normalized samples, with actions[50,32] and valid mask."""
    install_hooks()
    factory = dataclasses.replace(config.data, split=split)
    data = factory.create(config.assets_dirs, config.model)
    raw = loader_lib.create_torch_dataset(data, config.model.action_horizon, config.model)
    return loader_lib.transform_dataset(raw, data)


def model_batch(sample):
    """Convert one normalized sample to a batch of one; no GPU work implied."""
    batch = jax.tree.map(lambda value: np.asarray(value)[None, ...], sample)
    return KinovaObservation.from_dict(batch), batch["actions"]


def model_observation(sample):
    return model_batch(sample)[0]


def _path_string(path):
    return "/".join(str(getattr(part, "key", getattr(part, "idx", getattr(part, "name", part))))
                    for part in path)


def parameter_group(path):
    name = path if isinstance(path, str) else _path_string(path)
    if "lora" in name:
        if "bank" in name:
            raise ValueError("Unexpected CL bank in ordinary-LoRA training")
        return "lora"
    if name.split("/")[0] in HEAD_NAMES:
        return "heads"
    raise ValueError(f"Unexpected trainable parameter: {name}")


@dataclasses.dataclass(frozen=True)
class GroupedAdamW:
    """One global gradient clip, separate LoRA/head AdamW learning rates."""

    head_lr_ratio: float = 0.1
    clip_gradient_norm: float = 1.0
    weight_decay: float = 0.0
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8

    def create(self, lr, weight_decay_mask=None):
        if weight_decay_mask is not None:
            raise ValueError("No decay mask is used in this explicit weight_decay=0 protocol")
        head_lr = (lambda step: lr(step) * self.head_lr_ratio) if callable(lr) else lr * self.head_lr_ratio
        def labels(params):
            return jax.tree_util.tree_map_with_path(lambda path, value: parameter_group(path), params)
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), optax.multi_transform({
            "lora": optax.adamw(lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay),
            "heads": optax.adamw(head_lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay),
        }, labels))


@dataclasses.dataclass(frozen=True)
class KinovaTrainConfig(cfg.TrainConfig):
    gradient_accumulation_steps: int = 4
    protocol: str = "single_task_ordinary_lora_wrist_D0_exploratory"


def make_config(stage="smoke", *, experiment=None, root=None, steps=None, weights=None, resume=False,
                micro_batch=1, accumulation=4):
    stage = experiment or stage
    if stage not in ("smoke", "pilot", "frozen") or micro_batch < 1 or accumulation < 1:
        raise ValueError("Expected smoke|pilot|frozen and positive micro batch/accumulation")
    root = Path(root or os.environ.get("KINOVA_EXPERIMENT_ROOT", str(ROOT)))
    model = FrozenKinovaPi05Config() if stage == "frozen" else KinovaPi05Config()
    total_steps = steps if steps is not None else (32 if stage == "smoke" else 2000)
    if total_steps < 1 or (stage == "pilot" and total_steps > 2000):
        raise ValueError("Positive steps required; pilot capped at 2000 optimizer updates")
    return KinovaTrainConfig(
        name={"smoke": "E1_wrist_smoke", "pilot": "E1_wrist_pilot", "frozen": "E0_wrist_frozen"}[stage], exp_name="seed42",
        model=model, data=KinovaDataFactory(),
        weight_loader=CompleteWeightLoader(str(weights or BASE_PARAMS)),
        freeze_filter=nnx.Param if stage == "frozen" else model.get_balanced_lora_freeze_filter(),
        optimizer=GroupedAdamW(),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=1e-4,
                                                 decay_steps=2000, decay_lr=1e-5),
        ema_decay=None, lambda_kd=0.0, rehearsal=cfg.RehearsalConfig(enabled=False),
        seed=42, batch_size=micro_batch, gradient_accumulation_steps=accumulation, num_workers=0,
        num_train_steps=total_steps, log_interval=1 if stage == "smoke" else 10,
        save_interval=total_steps if stage == "smoke" else 500, keep_period=500,
        assets_base_dir=str(root / "config_assets"), checkpoint_base_dir=str(root / "checkpoints"),
        overwrite=False, resume=resume, wandb_enabled=False, fsdp_devices=1,
    )


def input_transforms(split="train"):
    data = KinovaDataFactory(split=split).create(ROOT / "config_assets", KinovaPi05Config())
    return (*data.repack_transforms.inputs, *data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=True), *data.model_transforms.inputs)


def output_transforms():
    return (transforms.Unnormalize(normalize.load(CANDIDATE_ROOT / "assets"), use_quantiles=True), KinovaOutputs())


def write_parameter_layout(output):
    """Abstract CPU shape audit: never allocate the full parameter arrays."""
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Parameter audit must run with JAX_PLATFORMS=cpu")
    config = make_config()
    state = jax.eval_shape(lambda key: nnx.state(config.model.create(key)), jax.random.key(0))
    trainable = state.filter(config.trainable_filter)
    train_paths = set(trainable.flat_state())
    rows = []
    for path, variable in state.flat_state().items():
        name = "/".join(map(str, path)); active = path in train_paths
        rows.append({"path": name, "shape": list(variable.value.shape),
                     "dtype_before_frozen_cast": str(variable.value.dtype),
                     "count": math.prod(variable.value.shape), "trainable": active,
                     "optimizer_group": parameter_group(name) if active else None})
    result = {"scope": "CPU abstract shapes; full model arrays not materialized", "parameters": rows,
              "total_parameters": sum(row["count"] for row in rows),
              "trainable_parameters": sum(row["count"] for row in rows if row["trainable"]),
              "trainable_tensors": sum(row["trainable"] for row in rows),
              "group_counts": {group: sum(row["count"] for row in rows if row["optimizer_group"] == group)
                               for group in ("lora", "heads")},
              "model": dataclasses.asdict(config.model),
              "frozen_cast": "Native train.init_train_state casts only frozen parameters to bfloat16"}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
