"""Read-only compact progress, including the native trainer's buffered progress bars."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
from pathlib import Path
import os,json,re
r=Path(os.environ.get('PI05_LIBERO_ROOT',project_path('libero_fourway', '')))
s=json.loads((r/'status.json').read_text());j=s.get('current_job') or {};steps=[];tail=''
if j.get('log') and Path(j['log']).exists():
 with Path(j['log']).open('rb') as f:
  f.seek(max(0,Path(j['log']).stat().st_size-16000));tail=f.read().decode(errors='replace')
 steps=[int(v) for v in re.findall(r'\bStep\s+(\d+)',tail)]
 steps += [int(float(v)*(1000 if k else 1)) for v,k in re.findall(r'Progress on:\s*([\d.]+)(k?)it/',tail)]
out={'status':s['status'],'started_at':s.get('started_at'),'updated_at':s.get('updated_at'),'job':j.get('name'),
 'group':j.get('group'),'stage':j.get('stage'),'latest_logged_training_step':max(steps) if steps else j.get('last_logged_step'),
 'steps_per_stage':s.get('steps_per_stage'),'elapsed_seconds':j.get('elapsed_seconds'),'completed_evaluations':s.get('completed_evaluations',0),'total_evaluations':40,'error':s.get('error'),'log':j.get('log')}
print(json.dumps(out,ensure_ascii=False,indent=2))
