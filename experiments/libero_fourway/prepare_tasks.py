"""Build independent, single-task LeRobot repositories and per-task training statistics."""
from pathlib import Path
import json,os,hashlib,numpy as np,pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.shared import normalize
root=Path(os.environ['PI05_LIBERO_ROOT']);source=root/'data/physical-intelligence-libero'
manifest=json.loads((root/'reports/source_download_manifest.json').read_text())
info=json.loads((source/'info.json').read_text());features={k:v for k,v in info['features'].items() if k in ['image','wrist_image','state','actions']}
for v in features.values():v['shape']=tuple(v['shape'])
mapping=[];summary={}
for stage in 'ABCD':
 repo=f'libero_fourway/task_{stage}';out=root/'datasets'/repo
 ds=LeRobotDataset.create(repo,root=out,fps=10,features=features,robot_type='panda',use_videos=False)
 states=[];actions=[]
 for e in sorted([x for x in manifest if x['stage']==stage],key=lambda x:x['episode_index']):
  table=pq.read_table(e['path']);n=table.num_rows;assert n==e['length'];idx=ds.meta.total_episodes
  state=np.stack(table['state'].to_pylist()).astype(np.float32);act=np.stack(table['actions'].to_pylist()).astype(np.float32)
  assert state.shape==(n,8) and act.shape==(n,7) and np.isfinite(state).all() and np.isfinite(act).all()
  image_dir=root/'data/temp_images'/stage/str(idx);image_dir.mkdir(parents=True,exist_ok=False)
  b=ds.create_episode_buffer();b.update(size=n,task=[e['tasks'][0]]*n,frame_index=list(range(n)),timestamp=[i/10 for i in range(n)])
  b['state']=list(state);b['actions']=list(act)
  for name in ['image','wrist_image']:
   paths=[]
   for i,image in enumerate(table[name].to_pylist()):
    data=image['bytes'];assert data
    ext='png' if data.startswith(b'\x89PNG') else 'jpg';p=image_dir/f'{name}_{i:06d}.{ext}';p.write_bytes(data);paths.append(str(p))
   b[name]=paths
  ds.episode_buffer=b;ds.save_episode()
  mapping.append({'stage':stage,'repo_id':repo,'episode_index':idx,'source_episode_index':e['episode_index'],'frames':n,'source_sha256':e['sha256'],'source_revision':e['revision'],'source_relative_path':e['relative_path']})
  states.append(state);actions.append(act)
  import shutil
  shutil.rmtree(image_dir)
  print('converted',stage,idx,e['episode_index'],n,flush=True)
 norm={};stats={}
 for key,x in [('state',np.concatenate(states)),('actions',np.concatenate(actions))]:
  x=x.astype(np.float64);q01,q99=np.quantile(x,[.01,.99],axis=0);norm[key]=normalize.NormStats(mean=x.mean(0),std=x.std(0),q01=q01,q99=q99)
  stats[key]={'min':x.min(0).tolist(),'max':x.max(0).tolist(),'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'q01':q01.tolist(),'q99':q99.tolist()}
 normalize.save(root/'assets'/repo,norm)
 provenance={'repo_id':repo,'task':e['tasks'][0],'episodes':len(states),'frames':sum(len(s) for s in states),'statistics':stats,'source_episode_indices':[m['source_episode_index'] for m in mapping if m['stage']==stage],'calculation':'current task original steps only, each once; exact numpy q01/q99, population std; no future tasks or replay/chunk padding in stats'}
 (root/'assets'/repo/'provenance.json').write_text(json.dumps(provenance,indent=2));summary[stage]={k:provenance[k] for k in ['repo_id','task','episodes','frames']}
 del ds
for stage in 'ABCD':
 p=root/'datasets/libero_fourway'/f'task_{stage}'/'meta/episodes.jsonl'
 episode_rows=[json.loads(line) for line in p.read_text().splitlines()]
 by_index={m['episode_index']:m for m in mapping if m['stage']==stage}
 for row in episode_rows:
  m=by_index[row['episode_index']]
  row.update(source_episode_index=m['source_episode_index'],source_revision=m['source_revision'],source_relative_path=m['source_relative_path'])
 p.write_text('\n'.join(json.dumps(row) for row in episode_rows)+'\n')
(root/'reports/episode_mapping.json').write_text(json.dumps(mapping,indent=2))
(root/'reports/tasks_prepared.json').write_text(json.dumps(summary,indent=2))
