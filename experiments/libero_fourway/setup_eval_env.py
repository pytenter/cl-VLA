"""Create a simulator-only environment, reusing deployed libraries through a read-only .pth."""

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
from pathlib import Path
import json,subprocess,sys

root=Path(sys.argv[1]);deploy=Path(project_path('deploy', ''))
environment=root/'.venv_eval';python=environment/'bin/python';report=root/'reports/simulator';report.mkdir(parents=True,exist_ok=True)
if not python.exists():
 subprocess.run([str(deploy/'.venv/bin/python'),'-m','venv','--without-pip',str(environment)],check=True)
site=environment/'lib/python3.11/site-packages'
site.mkdir(parents=True,exist_ok=True)
(site/'shared_deployed_runtime.pth').write_text(str(deploy/'.venv/lib/python3.11/site-packages')+'\n')
packages=['hf-libero==0.1.4','robosuite==1.4.1','mujoco==3.2.3','bddl==1.0.1','easydict==1.13','gym==0.26.2','gym-notices==0.0.8','future==1.0.0']
command=[str(deploy/'.bootstrap/bin/uv'),'pip','install','--python',str(python),'--no-deps','--index-url','https://mirrors.bfsu.edu.cn/pypi/web/simple',*packages]
result=subprocess.run(command,capture_output=True,text=True)
(report/'install.log').write_text(result.stdout+result.stderr)
print(result.stdout+result.stderr)
if result.returncode:raise SystemExit(result.returncode)
code='import importlib.metadata as m,json; names='+repr(['hf-libero','robosuite','mujoco','bddl','easydict','gym','numba','numpy','torch','PyOpenGL'])+';print(json.dumps({n:{"version":m.version(n),"requires":m.requires(n)} for n in names},indent=2))'
result=subprocess.run([str(python),'-c',code],capture_output=True,text=True,check=True)
(report/'environment_metadata.json').write_text(result.stdout)
print('EVAL_ENV_READY',python)
