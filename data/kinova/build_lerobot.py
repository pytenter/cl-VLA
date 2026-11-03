"""Build three split-isolated LeRobot 2.1 repositories with embedded lossless PNGs."""
import os
from pathlib import Path
import json,sys,time,numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.shared import normalize

run=Path(sys.argv[1]);started=time.monotonic()
records=json.loads((run/'reports/episode_audit.json').read_text());mappings=[]
names=[f'joint_position_deg_{i}' for i in range(7)]+[f'joint_velocity_deg_s_{i}' for i in range(7)]+[f'joint_torque_nm_{i}' for i in range(7)]+['tool_x_m','tool_y_m','tool_z_m','tool_theta_x_deg','tool_theta_y_deg','tool_theta_z_deg','gripper_raw']
features={'observation.images.primary':{'dtype':'image','shape':(640,480,3),'names':['height','width','channels']},
          'observation.images.wrist':{'dtype':'image','shape':(480,640,3),'names':['height','width','channels']},
          'observation.state':{'dtype':'float32','shape':(28,),'names':names},
          'action':{'dtype':'float32','shape':(7,),'names':['dx_m','dy_m','dz_m','dtheta_x_rad','dtheta_y_rad','dtheta_z_rad','gripper_open_fraction']},
          'source_time_ns':{'dtype':'int64','shape':(8,),'names':json.loads(next((run/'staging/train').glob('*/metadata.json')).read_text())['timestamp_fields']},
          'source_flags':{'dtype':'bool','shape':(3,),'names':['is_first','is_last','is_terminal']}}
for split in ['train','validation','test']:
 repo=f'kinova_candidate/{split}';out=run/'lerobot'/repo
 ds=LeRobotDataset.create(repo_id=repo,root=out,fps=10,features=features,robot_type='kinova_gen3_feedback_delta_candidate',use_videos=False)
 for row in [r for r in records if r['split']==split]:
  eid=row['original_episode_id'];src=run/'staging'/split/eid
  a=np.load(src/'numeric.npz');meta=json.loads((src/'metadata.json').read_text());n=row['steps'];index=ds.meta.total_episodes
  # Use the normal LeRobot save_episode path; images are already decoded and PNG encoded exactly once.
  b=ds.create_episode_buffer();b['size']=n;b['task']=[meta['language_instruction']]*n
  b['frame_index']=list(range(n));b['timestamp']=[i/10 for i in range(n)]
  b['observation.state']=list(a['state']);b['action']=list(a['action'])
  b['source_time_ns']=list(a['source_time_ns']);b['source_flags']=list(a['source_flags'])
  for name in ['primary','wrist']:b['observation.images.'+name]=[str(src/name/f'{i:06d}.png') for i in range(n)]
  ds.episode_buffer=b;ds.save_episode()
  mappings.append({'original_episode_id':eid,'candidate_episode_id':eid,'split':split,'lerobot_repo_id':repo,
    'lerobot_episode_index':index,'steps':n,'parquet_path':str(out/ds.meta.get_data_file_path(index)),
    'source_shard':row['source_shard'],'outcome':row['candidate_outcome'],'outcome_source':row['outcome_source']})
  (run/'reports/conversion_progress.json').write_text(json.dumps({'episodes':len(mappings),'last':eid}))
  print('converted',split,index,eid,n,flush=True)
 del ds
train=np.load(run/'assets/train_vectors.npz');norm={};distributions={}
for key in ['state','actions']:
 a=train[key].astype(np.float64);q01,q99=np.quantile(a,[.01,.99],axis=0)
 norm[key]=normalize.NormStats(mean=a.mean(0),std=a.std(0),q01=q01,q99=q99)
 distributions[key]={'count':len(a),'min':a.min(0).tolist(),'max':a.max(0).tolist(),'mean':a.mean(0).tolist(),
 'std':a.std(0).tolist(),'q01':q01.tolist(),'q99':q99.tolist(),'nonfinite_count':int((~np.isfinite(a)).sum()),
 'below_q01_fraction':(a<q01).mean(0).tolist(),'above_q99_fraction':(a>q99).mean(0).tolist(),
 'near_constant_dimensions_std_below_1e_6':np.where(a.std(0)<1e-6)[0].tolist()}
normalize.save(run/'assets',norm)
(run/'assets/statistics_provenance.json').write_text(json.dumps({'split':'train','episodes':60,'steps':len(train['state']),
 'episode_ids':[m['original_episode_id'] for m in mappings if m['split']=='train'],
 'calculation':'each original training step counted once; float64 population mean/std; numpy linear exact q01/q99; no validation/test; no action-chunk padding counted',
 'state_order':names,'distributions':distributions},ensure_ascii=False,indent=2))
(run/'reports/trajectory_mapping.json').write_text(json.dumps(mappings,ensure_ascii=False,indent=2))
import csv
with (run/'reports/trajectory_mapping.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(mappings[0]));w.writeheader();w.writerows(mappings)
(run/'reports/conversion_complete.json').write_text(json.dumps({'episodes':len(mappings),'steps':sum(m['steps'] for m in mappings),'seconds':time.monotonic()-started,'image_storage':'embedded PNG bytes','time_index':'nominal 10Hz, original nanosecond timestamps retained separately','parameter_updates':0},indent=2))
