#!/usr/bin/env bash
# Source from an activated OpenVLA environment; this does not install or run models.
_CL_VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENVLA_PROJECT_ROOT="$_CL_VLA_ROOT"
export REPO_ROOT="$_CL_VLA_ROOT"
export OPENVLA_BASE_PATH="${OPENVLA_BASE_PATH:-$_CL_VLA_ROOT/.local/models/openvla-7b}"
export VLA_PATH="${VLA_PATH:-$OPENVLA_BASE_PATH}"
export OPENVLA_DATA_ROOT="${OPENVLA_DATA_ROOT:-$_CL_VLA_ROOT/.local/data}"
export LOGS_ROOT="${LOGS_ROOT:-$_CL_VLA_ROOT/.local/runs/openvla}"
export HF_HOME="${HF_HOME:-$_CL_VLA_ROOT/.local/cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
case "${1:-libero}" in
  libero) _OPENVLA_CODE="$_CL_VLA_ROOT/openvla-oft";;
  robotwin) _OPENVLA_CODE="$_CL_VLA_ROOT/RoboTwin-main/policy/openvla-oft";;
  *) printf '%s\n' 'Usage: source scripts/env.sh {libero|robotwin}' >&2; return 2;;
esac
export OPENVLA_CODE_ROOT="$_OPENVLA_CODE"
export PYTHONPATH="$_OPENVLA_CODE:$_OPENVLA_CODE/vla-scripts${PYTHONPATH:+:$PYTHONPATH}"
unset _CL_VLA_ROOT _OPENVLA_CODE
