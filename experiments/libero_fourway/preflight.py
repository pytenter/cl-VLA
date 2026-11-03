"""Require completed real-data smoke evidence before the long four-way run."""
import json,os
from pathlib import Path
r=Path(os.environ['PI05_LIBERO_ROOT'])
def read(p):return json.loads((r/p).read_text())
d=read('reports/data_validation.json');assert sum(x['frames'] for x in d.values())==22680
assert [d[s]['episodes'] for s in 'ABCD']==[46,45,50,50]
for name in ['plain_A','prototype_A','prototype_B']:
 p=read(f'reports/pi05_libero_{name}_smoke_complete.json');assert p['steps']==3 and p['smoke']
e=read('eval/smoke_plain_A/evaluation.json');assert e['evaluation_valid'] and e['errors']==0 and e['episodes_completed']==2
p=read('reports/bank_preservation_smoke.json');assert p['status']=='passed' and p['protected_old_future_rows_changed']==0
for g in ['prototype','uniform']:
 v=read(f'smoke_buffers/{g}/task_A/validation.json');assert v['status']=='passed' and v['samples']==219
assert (r/'reports/keepalive.json').is_file()
out={'status':'passed','data_frames':22680,'episodes':191,'ordinary_updates_tested':3,'cl_updates_tested':3,'replay_updates_tested':3,'simulator_episodes_tested':2,'prototype_and_uniform_samples_each':219,'old_bank_protection_verified':True,'parameter_updates_are_only_smoke_so_far':True}
(r/'reports/preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
