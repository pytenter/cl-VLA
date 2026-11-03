"""Download only the four tasks from a pinned public LeRobot dataset, using curl."""
from pathlib import Path
import json,os,subprocess,hashlib,concurrent.futures,threading,time
root=Path(os.environ['PI05_LIBERO_ROOT']);base=root/'data/physical-intelligence-libero';base.mkdir(exist_ok=True)
revision='a4336d589d589045d1c56423ffdf3b88a0e19b1f'
tasks=[json.loads(s) for s in (base/'tasks.jsonl').read_text().splitlines()]
eps=[json.loads(s) for s in (base/'episodes.jsonl').read_text().splitlines()]
task_ids=[30,34,29,16];manifest=[]
for letter,taskid in zip('ABCD',task_ids):
 for e in eps:
  if tasks[taskid]['task'] in e['tasks']:
   idx=e['episode_index'];manifest.append({**e,'stage':letter,'source_task_index':taskid,'relative_path':f'data/chunk-{idx//1000:03d}/episode_{idx:06d}.parquet'})
(root/'reports/source_episodes.json').write_text(json.dumps(manifest,indent=2))
lock=threading.Lock();results=[]
def fetch(e):
 rel=e['relative_path'];p=base/rel;p.parent.mkdir(parents=True,exist_ok=True);partial=p.with_suffix('.part');headers=p.with_suffix('.headers')
 if not p.exists():
  url=f'https://hf-mirror.com/datasets/physical-intelligence/libero/resolve/{revision}/{rel}'
  subprocess.run(['curl','-fSL','--connect-timeout','15','--max-time','600','--retry','5','--retry-delay','2','--retry-all-errors','-C','-','-D',str(headers),'-o',str(partial),url],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
  partial.rename(p)
 digest=hashlib.sha256(p.read_bytes()).hexdigest();expected=None
 if headers.exists():
  for line in headers.read_text(errors='replace').splitlines():
   if line.lower().startswith('x-linked-etag:'):
    value=line.split(':',1)[1].strip().strip('"')
    if len(value)==64:expected=value
 if expected and expected!=digest:raise ValueError('SHA256 mismatch '+rel)
 assert p.read_bytes()[:4]==b'PAR1'
 item={**e,'path':str(p),'bytes':p.stat().st_size,'sha256':digest,'expected_lfs_sha256':expected,'revision':revision}
 with lock:
  results.append(item);(root/'reports/download_progress.json').write_text(json.dumps({'complete':len(results),'total':len(manifest),'bytes':sum(x['bytes'] for x in results),'last':rel}))
 print('downloaded',e['stage'],e['episode_index'],p.stat().st_size,flush=True)
 return item
with concurrent.futures.ThreadPoolExecutor(8) as pool:completed=list(pool.map(fetch,manifest))
(root/'reports/source_download_manifest.json').write_text(json.dumps(completed,indent=2))
(root/'reports/download_complete.json').write_text(json.dumps({'episodes':len(completed),'bytes':sum(e['bytes'] for e in completed),'revision':revision},indent=2))
