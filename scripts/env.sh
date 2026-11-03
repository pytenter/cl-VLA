#!/usr/bin/env bash
# Source from an installed Python3.11 environment. Use a fresh shell per profile.
_CL_VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CL_VLA_REPO_ROOT="$_CL_VLA_ROOT"
export PI05_DEPLOY_ROOT="${PI05_DEPLOY_ROOT:-$_CL_VLA_ROOT/.local/runtime}"
export PI05_BASE_PARAMS="${PI05_BASE_PARAMS:-$PI05_DEPLOY_ROOT/models/pi05_base/params}"
export PI05_PYTHON="${PI05_PYTHON:-$(command -v python3)}"
export PI05_LIBERO_FOURWAY_ROOT="${PI05_LIBERO_FOURWAY_ROOT:-$_CL_VLA_ROOT/.local/runs/libero_fourway}"
export PI05_LIBERO_REFERENCE_ROOT="${PI05_LIBERO_REFERENCE_ROOT:-$PI05_LIBERO_FOURWAY_ROOT}"
export PI05_LIBERO_PLAIN_ROOT="${PI05_LIBERO_PLAIN_ROOT:-$_CL_VLA_ROOT/.local/runs/libero_plain_prototype}"
export KINOVA_CANDIDATE_ROOT="${KINOVA_CANDIDATE_ROOT:-$_CL_VLA_ROOT/.local/data/kinova_candidate}"
export KINOVA_EXPERIMENT_ROOT="${KINOVA_EXPERIMENT_ROOT:-$_CL_VLA_ROOT/.local/runs/kinova}"
export CL_VLA_ARCHIVE_ROOT="${CL_VLA_ARCHIVE_ROOT:-$_CL_VLA_ROOT/.local/archive}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
case "${1:-libero-fourway}" in
  libero-fourway)
    export PI05_LIBERO_ROOT="$PI05_LIBERO_FOURWAY_ROOT"
    _CL_VLA_MODULES="$_CL_VLA_ROOT/experiments/libero_fourway"
    export HF_LEROBOT_HOME="$PI05_LIBERO_ROOT/datasets";;
  libero-plain-prototype)
    export PI05_LIBERO_ROOT="$PI05_LIBERO_PLAIN_ROOT"
    _CL_VLA_MODULES="$_CL_VLA_ROOT/experiments/libero_plain_prototype"
    export HF_LEROBOT_HOME="$PI05_LIBERO_ROOT/datasets";;
  kinova)
    _CL_VLA_MODULES="$_CL_VLA_ROOT/experiments/kinova"
    export HF_LEROBOT_HOME="$KINOVA_CANDIDATE_ROOT/lerobot"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 XLA_PYTHON_CLIENT_ALLOCATOR=bfc
    export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true'
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 JAX_DEFAULT_MATMUL_PRECISION=highest PYTHONHASHSEED=42
    unset XLA_CLIENT_MEM_FRACTION;;
  *) printf '%s\n' 'Usage: source scripts/env.sh {libero-fourway|libero-plain-prototype|kinova}' >&2; return 2;;
esac
export PYTHONPATH="$_CL_VLA_MODULES:$_CL_VLA_ROOT/pi05:$_CL_VLA_ROOT/pi05/src:$_CL_VLA_ROOT/pi05/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
unset _CL_VLA_ROOT _CL_VLA_MODULES
