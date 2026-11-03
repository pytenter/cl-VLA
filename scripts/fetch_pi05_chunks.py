"""Resume official checkpoint objects using bounded, generation-pinned HTTP ranges."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import google_crc32c

ROOT = Path(project_path('deploy', ''))
CHUNK = 32 * 1024 * 1024
LOCK = threading.Lock()
PROGRESS = 0
TOTAL = 0


def checksum(path, algorithm):
    h = google_crc32c.Checksum() if algorithm == 'crc32c' else hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.digest()


def verify_object(path, obj):
    algorithm = 'md5' if obj.get('md5Hash') else 'crc32c'
    field = 'md5Hash' if algorithm == 'md5' else 'crc32c'
    assert checksum(path, algorithm) == base64.b64decode(obj[field]), str(path)


def status(state='downloading', **extra):
    p = ROOT / 'artifacts/model_download_status.json'
    temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps({'state': state, 'bytes': TOTAL,
                                    'downloaded_bytes': PROGRESS, **extra}, indent=2) + '\n')
    temporary.replace(p)


def fetch(job):
    global PROGRESS
    target = Path(job['chunk'])
    expected = job['end'] - job['start'] + 1
    if not target.exists() or target.stat().st_size != expected:
        temp = target.with_suffix('.part')
        existing = temp.stat().st_size if temp.exists() else 0
        assert existing <= expected
        with LOCK:
            PROGRESS += existing
            status()
        failures = 0
        while existing < expected:
            start = job['start'] + existing
            end = min(job['end'], start + 4 * 1024 * 1024 - 1)
            tail = target.with_suffix('.tail')
            headers = target.with_suffix('.headers')
            command = ['curl', '-fL', '--range', f'{start}-{end}',
                       '--connect-timeout', '10', '--max-time', '60',
                       '-sS', '-D', str(headers), job['url'], '-o', str(tail)]
            result = subprocess.run(command, capture_output=True, text=True)
            header = headers.read_text(errors='replace') if headers.exists() else ''
            ranges = re.findall(r'(?i)content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', header)
            received = tail.stat().st_size if tail.exists() else 0
            valid = (ranges and int(ranges[-1][0]) == start
                     and 0 < received <= end - start + 1
                     and result.returncode in (0, 18, 28, 56))
            if not valid:
                failures += 1
                print('RANGE_RETRY', target.name, result.returncode, result.stderr[:120], flush=True)
                if failures >= 6:
                    raise RuntimeError('Range download failed: ' + target.name)
                time.sleep(2 + failures)
                continue
            with temp.open('ab') as dest, tail.open('rb') as src:
                shutil.copyfileobj(src, dest)
            existing += received
            failures = 0
            tail.unlink()
            with LOCK:
                PROGRESS += received
                status()
        temp.replace(target)
        return
    with LOCK:
        PROGRESS += expected
        status()
        print(json.dumps({'downloaded_gib': round(PROGRESS / 2**30, 3),
                          'total_gib': round(TOTAL / 2**30, 3)}), flush=True)


def main():
    global PROGRESS, TOTAL
    meta = json.loads((ROOT / 'artifacts/gcs_pi05_base_objects.json').read_text())
    assert not meta.get('nextPageToken')
    objects = meta['items']
    TOTAL = sum(int(x['size']) for x in objects)
    plan_path = ROOT / 'artifacts/model_chunk_plan.json'
    chunk_dir = ROOT / 'cache/gcs_chunks'
    chunk_dir.mkdir(exist_ok=True)
    if plan_path.exists():
        plans = json.loads(plan_path.read_text())
    else:
        plans = []
        for obj in objects:
            target = ROOT / 'models/pi05_base' / obj['name'][len('checkpoints/pi05_base/'):]
            partial = target.with_name(target.name + '.part')
            prefix = partial.stat().st_size if partial.exists() else 0
            size = int(obj['size'])
            if prefix > size:
                raise RuntimeError('Oversized partial object: ' + str(partial))
            jobs = []
            for offset in range(prefix, size, CHUNK):
                end = min(size - 1, offset + CHUNK - 1)
                identifier = hashlib.sha256((obj['name'] + ':' + obj['generation']).encode()).hexdigest()[:20]
                jobs.append({'start': offset, 'end': end, 'url': obj['mediaLink'],
                             'chunk': str(chunk_dir / f'{identifier}_{offset}_{end}.chunk')})
            plans.append({'object': obj, 'target': str(target), 'prefix': prefix, 'jobs': jobs})
        plan_path.write_text(json.dumps(plans, indent=2) + '\n')
    pending = []
    records = []
    for plan in plans:
        obj = plan['object'];target = Path(plan['target'])
        if target.exists() and target.stat().st_size == int(obj['size']):
            PROGRESS += int(obj['size'])
        else:
            PROGRESS += plan['prefix']
            pending.extend(plan['jobs'])
    status(objects=len(objects), chunks=len(pending))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            for _ in pool.map(fetch, pending):
                pass
        for plan in plans:
            obj = plan['object'];target = Path(plan['target'])
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.part')
            if not target.exists():
                assembled = target.with_name(target.name + '.assembled')
                with assembled.open('wb') as dest:
                    remaining = plan['prefix']
                    if remaining:
                        with partial.open('rb') as src:
                            while remaining:
                                block = src.read(min(4 * 1024 * 1024, remaining))
                                if not block:
                                    raise RuntimeError('Saved partial prefix is incomplete')
                                dest.write(block);remaining -= len(block)
                    for job in plan['jobs']:
                        with Path(job['chunk']).open('rb') as src:
                            shutil.copyfileobj(src, dest, 4 * 1024 * 1024)
                assert assembled.stat().st_size == int(obj['size'])
                verify_object(assembled, obj)
                assembled.replace(target)
            verify_object(target, obj)
            records.append({'name': obj['name'], 'relative_path': str(target.relative_to(ROOT)),
                            'generation': obj['generation'], 'bytes': int(obj['size']),
                            'gcs_md5': obj.get('md5Hash'), 'gcs_crc32c': obj['crc32c'],
                            'verification_algorithm': 'md5' if obj.get('md5Hash') else 'crc32c',
                            'sha256': checksum(target, 'sha256').hex()})
            if partial.exists():
                partial.unlink()
            for job in plan['jobs']:
                p = Path(job['chunk'])
                if p.exists():
                    p.unlink()
            print('OBJECT_VERIFIED', obj['name'], flush=True)
        (ROOT / 'artifacts/model_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
        PROGRESS = TOTAL
        status('complete', objects=len(records))
        print('OFFICIAL_PI05_BASE_DOWNLOAD_VERIFIED', flush=True)
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
