"""Filesystem defaults for the published collaboration copy; no model changes."""
from __future__ import annotations
import os
from pathlib import Path
import sys
REPO_ROOT = Path(os.environ.get('CL_VLA_REPO_ROOT', Path(__file__).resolve().parents[4])).expanduser()

def project_path(kind: str, suffix: str = '') -> str:
    local = REPO_ROOT / '.local'
    deploy = Path(os.environ.get('PI05_DEPLOY_ROOT', local/'runtime')).expanduser()
    roots = {
        'deploy': deploy,
        'base_params': Path(os.environ.get('PI05_BASE_PARAMS', deploy/'models/pi05_base/params')).expanduser(),
        'libero_fourway': Path(os.environ.get('PI05_LIBERO_REFERENCE_ROOT', os.environ.get('PI05_LIBERO_FOURWAY_ROOT', local/'runs/libero_fourway'))).expanduser(),
        'libero_plain_prototype': Path(os.environ.get('PI05_LIBERO_PLAIN_ROOT', local/'runs/libero_plain_prototype')).expanduser(),
        'candidate': Path(os.environ.get('KINOVA_CANDIDATE_ROOT', local/'data/kinova_candidate')).expanduser(),
        'kinova': Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', local/'runs/kinova')).expanduser(),
        'archive': Path(os.environ.get('CL_VLA_ARCHIVE_ROOT', local/'archive')).expanduser(),
        'legacy': Path(os.environ.get('CL_VLA_LEGACY_DATA_ROOT', local/'legacy')).expanduser(),
    }
    if kind not in roots:
        raise ValueError(f'Unknown project path kind: {kind}')
    if kind == 'deploy':
        if suffix == '.venv/bin/python':
            return os.environ.get('PI05_PYTHON', sys.executable)
        if suffix == 'source/pi05' or suffix.startswith('source/pi05/'):
            return str(REPO_ROOT/'pi05'/suffix[len('source/pi05'):].lstrip('/'))
        if suffix == 'source/lerobot' or suffix.startswith('source/lerobot/'):
            return str(Path(os.environ.get('PI05_LEROBOT_SOURCE', deploy/'source/lerobot'))/suffix[len('source/lerobot'):].lstrip('/'))
    return str(roots[kind]/suffix)
