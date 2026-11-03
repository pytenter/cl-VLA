"""Isolated ordinary-LoRA π0.5 and element-count-normalized Kinova loss.

No source files are patched.  The scoped Gemma configuration override is used
only while constructing this model; the created graph stores those configs.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import threading
from typing import Any

from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma, model as model_lib
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils


VALID_ACTION_DIM = 7
_CONFIG_LOCK = threading.RLock()


@struct.dataclass
class KinovaObservation(model_lib.Observation):
    """Time supervision mask, kept separate from the image attention masks."""

    action_valid_mask: Any = None
    source_episode_index: Any = None
    source_frame_index: Any = None

    @classmethod
    def from_dict(cls, data):
        if "action_valid_mask" not in data:
            raise ValueError("Kinova training requires action_valid_mask; refusing an unmasked fallback")
        observation = model_lib.Observation.from_dict(data)
        values = {field.name: getattr(observation, field.name)
                  for field in dataclasses.fields(model_lib.Observation)}
        return cls(**values, action_valid_mask=data["action_valid_mask"],
                   source_episode_index=data.get("source_episode_index"),
                   source_frame_index=data.get("source_frame_index"))


def canonicalize_actions(actions, valid_times, *, valid_action_dim=VALID_ACTION_DIM):
    """Zero invalid GT before forward; retain full native noise/attention.

    Changing invalid labels (even to NaN) cannot change the noisy action input
    or valid predictions under the same RNG.
    """
    actions = jnp.asarray(actions, dtype=jnp.float32)
    valid_times = jnp.asarray(valid_times, dtype=bool)
    if valid_times.shape != actions.shape[:-1]:
        raise ValueError(f"Expected time mask {actions.shape[:-1]}, got {valid_times.shape}")
    mask = valid_times[..., None] & (jnp.arange(actions.shape[-1]) < valid_action_dim)
    return jnp.where(mask, actions, 0.0)


def masked_flow_terms(predicted_velocity, noise, actions, valid_times, *, valid_action_dim=VALID_ACTION_DIM):
    """Return (SSE, count), preserving the native target ``noise - actions``.

    Axes are [..., horizon, model_action_dim].  Invalid rows and padded action
    coordinates contribute exactly zero to this reducer.  With zero valid
    elements it returns (0, 0); the trainer must skip the optimizer update.
    The model canonicalizes invalid GT before forward as well.  The native
    attention pattern and noise at all coordinates remain unchanged.
    """
    prediction = jnp.asarray(predicted_velocity, dtype=jnp.float32)
    noise = jnp.asarray(noise, dtype=jnp.float32)
    actions = jnp.asarray(actions, dtype=jnp.float32)
    valid_times = jnp.asarray(valid_times, dtype=jnp.bool_)
    if prediction.shape != actions.shape or noise.shape != actions.shape:
        raise ValueError(f"Velocity/noise/action shape mismatch: {prediction.shape}/{noise.shape}/{actions.shape}")
    if actions.ndim < 2 or valid_times.shape != actions.shape[:-1]:
        raise ValueError(f"Expected time mask {actions.shape[:-1]}, got {valid_times.shape}")
    if not 0 < valid_action_dim <= actions.shape[-1]:
        raise ValueError("Invalid number of real action coordinates")
    dimension_mask = jnp.arange(actions.shape[-1]) < valid_action_dim
    mask = valid_times[..., None] & dimension_mask
    residual = jnp.where(mask, prediction - (noise - actions), 0.0)
    numerator = jnp.sum(jnp.square(residual), dtype=jnp.float32)
    count = jnp.sum(mask, dtype=jnp.float32)
    return numerator, count


def normalized_masked_loss(numerator, count):
    """Finite, zero-gradient result for an empty supervision mask."""
    return numerator / jnp.maximum(count, jnp.asarray(1.0, dtype=jnp.float32))


@contextmanager
def _ordinary_lora_settings(*, rank, alpha):
    """Override supported ordinary variants only, then restore the function."""
    with _CONFIG_LOCK:
        original = gemma.get_config

        def get_config(variant):
            config = original(variant)
            if variant in ("gemma_2b_lora", "gemma_300m_lora"):
                config = dataclasses.replace(config, lora_configs={
                    key: dataclasses.replace(value, rank=rank, alpha=alpha,
                                             rslora=False, use_orthogonal_init=False,
                                             use_task_lora_b_bank=False, freeze_lora_a=False)
                    for key, value in config.lora_configs.items()
                })
            return config

        gemma.get_config = get_config
        try:
            yield
        finally:
            gemma.get_config = original


class KinovaPi05(Pi0):
    def compute_loss_terms(self, rng, observation, actions, *, train=False):
        """Native flow matching with only real actions/time rows supervised."""
        valid_times = getattr(observation, "action_valid_mask", None)
        if valid_times is None:
            raise ValueError("Missing action_valid_mask in KinovaObservation")
        actions = canonicalize_actions(actions, valid_times)
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_lib.preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        predicted_velocity = self.forward_denoise(observation, actions, noise, time)
        return masked_flow_terms(predicted_velocity, noise, actions, valid_times)

    def compute_loss(self, rng, observation, actions, *, train=False):
        numerator, count = self.compute_loss_terms(rng, observation, actions, train=train)
        return normalized_masked_loss(numerator, count)


@dataclasses.dataclass(frozen=True)
class KinovaPi05Config(Pi0Config):
    """Ordinary LoRA on both Gemma experts; no task banks or CL gates."""

    pi05: bool = True
    dtype: str = "bfloat16"
    action_horizon: int = 50
    action_dim: int = 32
    max_token_len: int = 200
    discrete_state_input: bool = True
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or not self.discrete_state_input:
            raise ValueError("Kinova π0.5 requires discrete state tokens")
        if (self.action_dim, self.action_horizon, self.max_token_len) != (32, 50, 200):
            raise ValueError("This candidate protocol fixes action_dim=32, horizon=50, tokens=200")
        if self.lora_rank < 1 or self.lora_alpha <= 0 or self.lora_dropout != 0.0:
            raise ValueError("Positive rank/alpha and dropout=0 are required; native LoRA has no dropout option")
        if (self.paligemma_variant, self.action_expert_variant) != ("gemma_2b_lora", "gemma_300m_lora"):
            raise ValueError("Only the two ordinary LoRA variants are supported")
        if any((self.cl_pali_lora_bank, self.cl_action_lora_bank,
                self.cl_pali_block_weight_bank, self.cl_action_block_weight_bank,
                self.cl_freeze_shared_lora, self.cl_freeze_lora_a,
                self.cl_freeze_pali_lora_b, self.cl_freeze_action_lora_b)):
            raise ValueError("CL options are forbidden in the single-task ordinary-LoRA baseline")

    def create(self, rng):
        with _ordinary_lora_settings(rank=self.lora_rank, alpha=self.lora_alpha):
            model = KinovaPi05(self, rngs=nnx.Rngs(rng))
        # Native ordinary LoRA initializes A AND B randomly.  Explicit zero B
        # gives zero adapter contribution without introducing CL initialization.
        params = nnx_utils.state_map(nnx.state(model), nnx_utils.PathRegex(r".*lora_b$"),
                                     lambda value: value.replace(jnp.zeros_like(value.value)))
        nnx.update(model, params)
        return model


@dataclasses.dataclass(frozen=True)
class FrozenKinovaPi05Config(Pi0Config):
    """E0 base architecture, with the same masked validation loss method."""

    pi05: bool = True
    dtype: str = "bfloat16"
    action_horizon: int = 50
    action_dim: int = 32
    max_token_len: int = 200
    discrete_state_input: bool = True

    def create(self, rng):
        return KinovaPi05(self, rngs=nnx.Rngs(rng))


def adapter_initialization_summary(params):
    """Small summary after real initialization; does not restore any weights."""
    result = {"lora_b_tensors": 0, "all_lora_b_zero": True, "max_abs_lora_b": 0.0}
    for path, variable in params.flat_state().items():
        if str(path[-1]).endswith("lora_b"):
            value = np.asarray(variable.value)
            result["lora_b_tensors"] += 1
            maximum = float(np.max(np.abs(value)))
            result["max_abs_lora_b"] = max(result["max_abs_lora_b"], maximum)
            result["all_lora_b_zero"] &= maximum == 0.0
    if result["lora_b_tensors"] == 0:
        raise ValueError("No ordinary-LoRA B tensors found")
    return result
