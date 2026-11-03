"""Report missing model source prerequisites without importing ML dependencies."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
MODULES = ['action_heads', 'projectors', 'film_vit_wrapper',
           'backbones/llm/prompting', 'backbones/vision', 'vlms']
checks = []
for package in ['openvla-oft', 'RoboTwin-main/policy/openvla-oft']:
    models = ROOT/package/'prismatic/models'
    missing = [name for name in MODULES
               if not (models/(name+'.py')).is_file()
               and not (models/name/'__init__.py').is_file()]
    checks.append({'package': package, 'models_directory_present': models.is_dir(),
                   'missing_model_modules': missing})
complete = all(not row['missing_model_modules'] for row in checks)
print(json.dumps({'status': 'complete' if complete else 'incomplete_original_snapshot',
                  'packages': checks,
                  'note': 'Use a verified matching upstream source before training; no model or robot was started.'},
                 ensure_ascii=False, indent=2))
if not complete:
    raise SystemExit(2)
