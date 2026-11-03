"""Process-local LIBERO integration for the deployed native OpenPI engine.

The four task repositories retain the raw ``image/wrist_image/state/actions``
schema.  This module does not import a trainer, construct an optimizer, or send
robot commands.  Call ``install_hooks()`` before creating native data loaders.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

from contextlib import contextmanager
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch

from openpi import transforms
from openpi.policies import libero_policy
from openpi.training import config as config_lib
from openpi.training import data_loader as loader_lib

LOGGER = logging.getLogger(__name__)
ROOT = Path(os.environ.get("PI05_LIBERO_ROOT", project_path('libero_fourway', '')))


@dataclasses.dataclass(frozen=True)
class LiberoDataConfig(config_lib.DataConfig):
    """Native DataConfig plus read-only local-repository metadata."""

    root: str | None = None
    target_task_name: str | None = None
    strict_single_task: bool = True
    expected_state_dim: int = 8
    expected_action_dim: int = 7


@dataclasses.dataclass(frozen=True)
class LiberoDataFactory(config_lib.DataConfigFactory):
    """Map a split-isolated LIBERO task repo into native π0.5 transforms.

    ``assets.assets_dir / assets.asset_id`` must contain ``norm_stats.json``.
    When asset_id is omitted it defaults to repo_id, matching native OpenPI.
    Missing statistics are allowed during statistics construction; the native
    training transform rejects their absence when normalization is required.
    """

    root: str | None = None
    target_task_name: str | None = None
    strict_single_task: bool = True
    expected_state_dim: int = 8
    expected_action_dim: int = 7
    extra_delta_transform: bool = False

    def create(self, assets_dirs: Path, model_config: Any) -> LiberoDataConfig:
        if self.extra_delta_transform:
            raise ValueError("LIBERO actions are already deltas; extra_delta_transform must remain False")
        base = self.create_base_config(assets_dirs, model_config)
        values = {field.name: getattr(base, field.name) for field in dataclasses.fields(config_lib.DataConfig)}
        values.update(
            root=self.root,
            target_task_name=self.target_task_name,
            strict_single_task=self.strict_single_task,
            expected_state_dim=self.expected_state_dim,
            expected_action_dim=self.expected_action_dim,
            action_sequence_keys=("actions",),
            prompt_from_task=True,
            repack_transforms=transforms.Group(inputs=(transforms.RepackTransform({
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            }),)),
            data_transforms=transforms.Group(
                inputs=(libero_policy.LiberoInputs(model_type=model_config.model_type),),
                outputs=(libero_policy.LiberoOutputs(),),
            ),
            model_transforms=config_lib.ModelTransformFactory()(model_config),
        )
        return LiberoDataConfig(**values)


def _text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("Expected a scalar prompt")
        return _text(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class LiberoReplayDataset:
    """Read two-view, 8D-state, 7D-action replay without a second delta transform."""

    def __init__(
        self,
        buffer_dir: str | Path,
        *,
        action_horizon: int,
        manifest_filename: str = "manifest.jsonl",
        state_dim: int = 8,
        action_dim: int = 7,
    ) -> None:
        self.root = Path(buffer_dir).resolve()
        self.action_horizon = int(action_horizon)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        manifest = self.root / manifest_filename
        self.records = []
        with manifest.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid replay manifest {manifest}:{line_no}") from exc
                if not isinstance(record, dict) or "sample_path" not in record:
                    raise ValueError(f"Replay manifest entry lacks sample_path: {manifest}:{line_no}")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"Replay manifest contains no samples: {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        path = Path(record["sample_path"])
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if self.root not in path.parents:
            raise ValueError(f"Replay sample escapes its buffer directory: {path}")
        with np.load(path, allow_pickle=False) as saved:
            for key in ("image", "wrist_image", "state"):
                if key not in saved:
                    raise KeyError(f"Replay sample {path} lacks {key}")
            result = {
                "image": np.asarray(saved["image"]),
                "wrist_image": np.asarray(saved["wrist_image"]),
                "state": np.asarray(saved["state"], dtype=np.float32),
            }
            action_key = next((key for key in ("actions", "action_chunk", "action") if key in saved), None)
            if action_key is None:
                raise KeyError(f"Replay sample {path} lacks actions/action_chunk")
            result["actions"] = np.asarray(saved[action_key], dtype=np.float32)
            prompt = saved["prompt"] if "prompt" in saved else record.get("prompt")
            if prompt is None:
                raise ValueError(f"Replay sample has no prompt: {path}")
            result["prompt"] = _text(prompt)
        if result["state"].shape != (self.state_dim,):
            raise ValueError(f"Expected state[{self.state_dim}] in {path}, got {result['state'].shape}")
        expected_actions = (self.action_horizon, self.action_dim)
        if result["actions"].shape != expected_actions:
            raise ValueError(f"Expected actions{expected_actions} in {path}, got {result['actions'].shape}")
        if not np.isfinite(result["state"]).all() or not np.isfinite(result["actions"]).all():
            raise ValueError(f"Replay sample has non-finite state/action: {path}")
        for name in ("image", "wrist_image"):
            image = result[name]
            if image.ndim != 3 or (image.shape[-1] != 3 and image.shape[0] != 3):
                raise ValueError(f"Expected RGB HWC/CHW {name} in {path}, got {image.shape}")
            if image.dtype != np.uint8 and not np.issubdtype(image.dtype, np.floating):
                raise ValueError(f"Unsupported replay image dtype in {path}: {image.dtype}")
        return result


_ORIGINAL_CREATE_TORCH_DATASET = loader_lib.create_torch_dataset
_ORIGINAL_CREATE_REPLAY_LOADER = loader_lib.create_replay_buffer_data_loader


def _check_feature(meta: Any, name: str, expected_dim: int) -> None:
    feature = meta.features.get(name)
    if feature is None or tuple(feature.get("shape", ())) != (expected_dim,):
        raise ValueError(f"LIBERO repository {meta.repo_id} must have {name}[{expected_dim}]; got {feature}")


def create_torch_dataset(data_config: Any, action_horizon: int, model_config: Any) -> Any:
    if not isinstance(data_config, LiberoDataConfig):
        return _ORIGINAL_CREATE_TORCH_DATASET(data_config, action_horizon, model_config)
    repo_id = data_config.repo_id
    if not repo_id or repo_id == "fake":
        raise ValueError("LIBERO integration requires an actual task repository")
    meta = loader_lib.lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_config.root)
    _check_feature(meta, "state", data_config.expected_state_dim)
    _check_feature(meta, "actions", data_config.expected_action_dim)
    for name in ("image", "wrist_image"):
        if name not in meta.features:
            raise ValueError(f"LIBERO repository lacks required camera: {repo_id}:{name}")
    tasks = {int(key): _text(value) for key, value in meta.tasks.items()}
    target = data_config.target_task_name
    if data_config.strict_single_task and len(set(tasks.values())) != 1:
        raise ValueError(f"Expected a split-isolated, single-task repo {repo_id}; found {tasks}")
    if target is not None and target not in tasks.values():
        raise ValueError(f"Expected task {target!r} absent from repo {repo_id}: {tasks}")
    dataset = loader_lib.lerobot_dataset.LeRobotDataset(
        repo_id,
        root=data_config.root,
        delta_timestamps={key: [step / meta.fps for step in range(action_horizon)]
                          for key in data_config.action_sequence_keys},
    )
    # Filtering happens outside the LeRobot object so its original episode
    # boundaries and action-chunk indexing remain intact.
    if target is not None and len(set(tasks.values())) > 1:
        selected = []
        for episode_index, episode in meta.episodes.items():
            if target not in episode.get("tasks", ()):
                continue
            begin = int(dataset.episode_data_index["from"][int(episode_index)])
            end = int(dataset.episode_data_index["to"][int(episode_index)])
            selected.extend(range(begin, end))
        if not selected:
            raise ValueError(f"No episode metadata matches task {target!r} in {repo_id}")
        dataset = torch.utils.data.Subset(dataset, selected)
    if data_config.prompt_from_task:
        dataset = loader_lib.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(tasks)])
    return dataset


def _replay_factory(config: Any, buffer_dir: Path) -> LiberoDataFactory:
    factory = config.data
    metadata = buffer_dir / "meta.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Replay metadata is required to resolve the old task's statistics: {metadata}")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if "repo_id" not in meta or "cl_task_id" not in meta:
        raise ValueError(f"Replay meta must include repo_id and cl_task_id: {metadata}")
    repo_id = str(meta["repo_id"])
    asset_id = str(meta.get("norm_asset_id") or repo_id)
    assets_dir = meta.get("norm_assets_dir")
    if assets_dir is not None:
        assets_path = Path(assets_dir)
        if not assets_path.is_absolute():
            assets_path = (buffer_dir / assets_path).resolve()
        assets = config_lib.AssetsConfig(assets_dir=str(assets_path), asset_id=asset_id)
    else:
        # Existing shared assets tree, but ALWAYS the replay task's asset_id.
        # Reusing current task's statistics would silently corrupt rehearsal.
        assets = dataclasses.replace(factory.assets, asset_id=asset_id)
    old_root = meta.get("dataset_root")
    return dataclasses.replace(factory, repo_id=repo_id, assets=assets,
                               root=str(old_root) if old_root is not None else None,
                               target_task_name=meta.get("task_name"))


def create_replay_buffer_data_loader(
    config: Any,
    *,
    replay_buffer_dir: str | Path,
    sharding: Any = None,
    shuffle: bool = True,
    framework: str = "jax",
    manifest_filename: str | None = None,
) -> Any:
    if not isinstance(config.data, LiberoDataFactory):
        return _ORIGINAL_CREATE_REPLAY_LOADER(
            config, replay_buffer_dir=replay_buffer_dir, sharding=sharding,
            shuffle=shuffle, framework=framework, manifest_filename=manifest_filename,
        )
    buffer_dir = Path(replay_buffer_dir).resolve()
    factory = _replay_factory(config, buffer_dir)
    data_config = factory.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError(f"Old-task normalization statistics not found for {factory.repo_id}")
    dataset = LiberoReplayDataset(
        buffer_dir, action_horizon=config.model.action_horizon,
        manifest_filename=manifest_filename or config.rehearsal.buffer_manifest_filename,
        state_dim=factory.expected_state_dim, action_dim=factory.expected_action_dim,
    )
    dataset = loader_lib.transform_dataset(dataset, data_config, skip_norm_stats=False)
    if framework == "pytorch" and torch.distributed.is_initialized():
        local_batch_size = config.batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = config.batch_size if framework == "pytorch" else config.batch_size // jax.process_count()
    if local_batch_size < 1:
        raise ValueError("Local replay batch size must be positive")
    if len(dataset) < local_batch_size:
        dataset = loader_lib.RepeatDataset(dataset, local_batch_size)
    batches = loader_lib.TorchDataLoader(
        dataset, local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=shuffle, num_batches=None, num_workers=config.num_workers,
        seed=config.seed, framework=framework,
    )
    return loader_lib.DataLoaderImpl(data_config, batches)


def install_hooks() -> None:
    """Install only these two factories in this Python process; source files stay untouched."""
    loader_lib.create_torch_dataset = create_torch_dataset
    loader_lib.create_replay_buffer_data_loader = create_replay_buffer_data_loader


def uninstall_hooks() -> None:
    """Restore native factories when these integration hooks still own the entries."""
    if loader_lib.create_torch_dataset is create_torch_dataset:
        loader_lib.create_torch_dataset = _ORIGINAL_CREATE_TORCH_DATASET
    if loader_lib.create_replay_buffer_data_loader is create_replay_buffer_data_loader:
        loader_lib.create_replay_buffer_data_loader = _ORIGINAL_CREATE_REPLAY_LOADER


@contextmanager
def installed_hooks():
    already_installed = loader_lib.create_torch_dataset is create_torch_dataset
    install_hooks()
    try:
        yield
    finally:
        if not already_installed:
            uninstall_hooks()
