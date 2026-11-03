"""Snapshot protected sources and compare their metadata after read-only validation."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
from pathlib import Path
import hashlib,json,sys
run=Path(sys.argv[1]);phase=sys.argv[2]
base=Path('/mnt/d/KinovaDatasets/review_20261031')
paths=[base/'raw_20261031.tar',*sorted((base/'candidate_rlds_20261031/kinova_dual_view/1.5.0').iterdir())]
raw=Path('/root/kinova_candidate_work/run_20261031/episodes')
paths.extend(p for p in raw.glob('*/metadata.json'));paths.extend(p for p in raw.glob('*/manifest.json'))
deploy=Path(project_path('deploy', ''))
paths.extend(p for p in (deploy/'models/pi05_base').rglob('*') if p.is_file())
paths.extend(p for p in (deploy/'source/pi05/src/openpi').rglob('*.py'))
snap={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths if p.is_file()}
dest=run/'reports'/f'protected_sources_{phase}.json';dest.write_text(json.dumps(snap,indent=2))
if phase=='after':
 before=json.loads((run/'reports/protected_sources_before.json').read_text());changed=[k for k,v in before.items() if snap.get(k)!=v]
 (run/'reports/source_preservation.json').write_text(json.dumps({'changed_paths':changed,'checked_files':len(before),'check':'size and modification timestamp snapshot during isolated conversion, before model validation, versus after; not a full content hash scan or all raw image files','all_preserved':not changed},indent=2));assert not changed,changed
print(phase,len(snap))
