"""Download the official JAX checkpoint, pinned to GCS object generations."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(project_path('deploy', ''))
PREFIX = 'checkpoints/pi05_base/'


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.digest()


def fetch(obj):
    target = ROOT / 'models/pi05_base' / obj['name'][len(PREFIX):]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(obj['size'])
    expected_md5 = base64.b64decode(obj['md5Hash']) if obj.get('md5Hash') else None
    if not (target.exists() and target.stat().st_size == expected_size
            and (not expected_md5 or digest(target, 'md5') == expected_md5)):
        partial = target.with_name(target.name + '.part')
        for attempt in range(3):
            result = subprocess.run([
                'curl', '-L', '-f', '-C', '-', '--retry', '6', '--retry-delay', '5',
                '--connect-timeout', '10', '--speed-time', '90', '--speed-limit', '1024',
                '-sS', obj['mediaLink'], '-o', str(partial),
            ])
            if result.returncode == 0 and partial.stat().st_size == expected_size:
                if not expected_md5 or digest(partial, 'md5') == expected_md5:
                    partial.replace(target)
                    break
            if partial.exists() and partial.stat().st_size >= expected_size:
                partial.unlink()
            time.sleep(5)
        else:
            raise RuntimeError('Checkpoint object did not verify: ' + obj['name'])
    record = {'name': obj['name'], 'relative_path': str(target.relative_to(ROOT)),
              'generation': obj['generation'], 'bytes': expected_size,
              'gcs_md5': obj.get('md5Hash'), 'sha256': digest(target, 'sha256').hex()}
    print(json.dumps({'downloaded': record['relative_path'], 'bytes': expected_size}), flush=True)
    return record


def main():
    metadata_path = ROOT / 'artifacts/gcs_pi05_base_objects.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('nextPageToken'):
        raise RuntimeError('GCS object listing is incomplete')
    objects = [x for x in metadata['items'] if not x['name'].endswith('/')]
    status = ROOT / 'artifacts/model_download_status.json'
    status.write_text(json.dumps({'state': 'downloading', 'objects': len(objects),
                                 'bytes': sum(int(x['size']) for x in objects)}, indent=2))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            records = list(pool.map(fetch, objects))
        (ROOT / 'artifacts/model_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
        status.write_text(json.dumps({'state': 'complete', 'objects': len(records),
                                     'bytes': sum(x['bytes'] for x in records)}, indent=2) + '\n')
        print('OFFICIAL_PI05_BASE_DOWNLOAD_VERIFIED', flush=True)
    except Exception as exc:
        status.write_text(json.dumps({'state': 'failed', 'error': str(exc)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
