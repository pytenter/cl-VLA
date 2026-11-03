
# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
from pathlib import Path
import base64
import hashlib
import json
import subprocess

root = Path(project_path('deploy', ''))
metadata_url = 'https://storage.googleapis.com/storage/v1/b/big_vision/o/paligemma_tokenizer.model'
result = subprocess.run(['curl', '-fsSL', '--connect-timeout', '8', '--max-time', '20', metadata_url],
                        check=True, capture_output=True, text=True)
metadata = json.loads(result.stdout)
path = root / 'cache/openpi/big_vision/paligemma_tokenizer.model'
path.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(['curl', '-fsSL', '--retry', '3', '--connect-timeout', '10', '--max-time', '90',
                metadata['mediaLink'], '-o', str(path)], check=True)
assert len(path.read_bytes()) == int(metadata['size'])
assert hashlib.md5(path.read_bytes()).digest() == base64.b64decode(metadata['md5Hash'])
metadata['local_path'] = str(path)
metadata['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'artifacts/tokenizer_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('OFFICIAL_TOKENIZER_VERIFIED', path.stat().st_size)
