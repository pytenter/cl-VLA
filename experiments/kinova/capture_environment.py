"""Capture small, non-secret environment and source manifests without model loading."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess

ROOT = Path(os.environ.get('KINOVA_EXPERIMENT_ROOT', project_path('kinova', '')))
DEPLOY = Path(project_path('deploy', ''))
CANDIDATE = Path(project_path('candidate', ''))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from resource_guard import inspect_resources
    disk = shutil.disk_usage(ROOT)
    packages = {}
    for name in ['jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint', 'torch', 'lerobot', 'datasets', 'pyarrow', 'transformers', 'numpy']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = 'NOT_INSTALLED'
    result = {'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'host': socket.gethostname(),
              'user': subprocess.check_output(['id', '-un'], text=True).strip(), 'cwd': str(Path.cwd()),
              'python': platform.python_version(), 'executable': os.sys.executable, 'packages': packages,
              'candidate_exists': CANDIDATE.is_dir(), 'deployment_exists': DEPLOY.is_dir(),
              'disk_bytes': dict(zip(['total', 'used', 'free'], disk)), 'gpu_gate': inspect_resources(),
              'no_shared_environment_modifications': True, 'no_robot_interfaces_imported': True}
    (ROOT / 'reports/environment.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    source_paths = [
        DEPLOY / 'source/pi05/scripts/train.py', DEPLOY / 'source/pi05/src/openpi/models/pi0.py',
        DEPLOY / 'source/pi05/src/openpi/models/pi0_config.py', DEPLOY / 'source/pi05/src/openpi/models/gemma.py',
        DEPLOY / 'source/pi05/src/openpi/training/config.py', DEPLOY / 'source/pi05/src/openpi/training/data_loader.py',
        DEPLOY / 'source/pi05/src/openpi/training/checkpoints.py', DEPLOY / 'source/pi05/src/openpi/training/optimizer.py',
        DEPLOY / 'source/pi05/src/openpi/models/lora.py',
        CANDIDATE / 'candidate_config.json', CANDIDATE / 'assets/norm_stats.json',
        CANDIDATE / 'assets/statistics_provenance.json', CANDIDATE / 'scripts/kinova_adapter.py',
        DEPLOY / 'artifacts/model_manifest.json',
    ]
    manifest = [{'path': str(p), 'exists': p.is_file(), 'size': p.stat().st_size if p.is_file() else None,
                 'sha256': sha(p) if p.is_file() else None} for p in source_paths]
    (ROOT / 'manifests/referenced_source_files.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'environment': str(ROOT / 'reports/environment.json'), 'manifest_files': len(manifest),
                      'gpu_work_started': False, 'gate_eligible': result['gpu_gate']['eligible']}, indent=2))


if __name__ == '__main__':
    main()
