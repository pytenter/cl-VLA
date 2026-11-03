"""Download pinned LIBERO simulation assets with curl; no training data or model weights."""
from pathlib import Path,PurePosixPath
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,subprocess,sys,time

root=Path(sys.argv[1]);metadata=json.loads((root/'reports/libero_assets_meta.json').read_text())
revision='0b3ea86be5fe169d0fd036ae63d1070ec09e90f6';assert metadata['sha']==revision
out=root/'simulator_assets';out.mkdir(parents=True,exist_ok=True);reports=root/'reports/simulator';reports.mkdir(exist_ok=True,parents=True)
names=[r['rfilename'] for r in metadata['siblings'] if r['rfilename']!='.gitattributes']
started=time.monotonic();done=[];failed=[]
def download(name):
 rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
 dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_name(dest.name+'.part')
 if not dest.is_file():
  url=f'https://hf-mirror.com/datasets/lerobot/libero-assets/resolve/{revision}/{quote(name,safe="/")}?download=true'
  command=['curl','--fail','--location','--retry','4','--retry-delay','2','--connect-timeout','20','--max-time','240','--silent','--show-error','--continue-at','-','--output',str(part),url]
  result=subprocess.run(command,capture_output=True,text=True,timeout=1100)
  if result.returncode:return {'path':name,'error':result.stderr[-1200:],'returncode':result.returncode}
  part.replace(dest)
 return {'path':name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
with ThreadPoolExecutor(max_workers=6) as pool:
 futures={pool.submit(download,name):name for name in names}
 for future in as_completed(futures):
  try:row=future.result()
  except Exception as exc:row={'path':futures[future],'error':repr(exc)}
  (failed if 'error' in row else done).append(row)
  if (len(done)+len(failed))%20==0:print('assets',len(done),'/',len(names),'failed',len(failed),flush=True)
  p=reports/'asset_download_progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'complete':len(done),'total':len(names),'failed':len(failed),'last':row['path']}));tmp.replace(p)
manifest={'repo':'lerobot/libero-assets','revision':revision,'files':sorted(done,key=lambda r:r['path']),'failures':failed,'seconds':time.monotonic()-started}
(reports/'asset_manifest.json').write_text(json.dumps(manifest,indent=2));print('ASSETS_COMPLETE',len(done),len(failed),manifest['seconds'])
if failed:raise SystemExit(1)
