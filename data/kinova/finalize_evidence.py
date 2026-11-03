
# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path
from pathlib import Path
import json,hashlib,sys,shutil
run=Path(sys.argv[1]);R=run/'reports'
pipeline=json.loads((R/'pipeline_validation.json').read_text())
assert pipeline['parameter_updates']==0 and all(v['status']=='passed' for v in pipeline['inference']['cases'].values())
assert pipeline['language']['truncated_count']==0 and pipeline['state_control']['all_28_dimensions_change_tokens']
deploy=Path(project_path('deploy', ''))
for name in ['source/pi05/scripts/train.py','source/pi05/src/openpi/training/data_loader.py','source/pi05/src/openpi/training/config.py','source/pi05/src/openpi/transforms.py','source/pi05/src/openpi/models/tokenizer.py','source/pi05/src/openpi/models/pi0.py','source/pi05/src/openpi/models/pi0_config.py','source/lerobot/lerobot/common/datasets/lerobot_dataset.py']:
 shutil.copyfile(deploy/name,run/'source_evidence'/name.replace('/','__'))
visual=json.loads((R/'visualization_complete.json').read_text())
manual={'scope':'Agent visually inspected six training episode overview sheets and the first episode timestamp-labelled uncompensated/compensated/wrist adjacent-event sheet; selected frames only, not continuous full videos',
 'episode_ids':visual['detailed_alignment_train_episodes'],
 'observations':['Selected scenes show object near gripper during closing, object later visible inside target box, and arm returning away from the object.',
 'No obvious 90-degree image rotation or gross RGB/BGR swap is apparent in reviewed sheets; no color-card verification.',
 'First reviewed episode closing steps 75-77 reuse the external image while wrist/feedback change; fine event timing cannot be resolved from those external frames.',
 'Global maximum tool z often occurs on return; dedicated closure-to-reopening motion sheets supplied instead of calling that maximum a verified lift event.',
 'First old episode final gripper raw is about 25.44, so full-open endpoint is not required to observe object already in box; precise release/stability and full task success remain unverified.'],
 'exposure_synchronization_confirmed':False,'all_107_successes_revalidated':False}
(R/'manual_visual_review.json').write_text(json.dumps(manual,ensure_ascii=False,indent=2))
repeat=json.loads((R/'event_repeat_diagnostic.json').read_text())
report=run/'REPORT_zh.md';text=report.read_text();addition='''
### 补充：已实际查看的画面与关键事件重复

本次实际查看了所选 6 条训练轨迹的双视角概览，以及首条训练轨迹的无补偿/继承补偿/腕部相邻事件对照。抽帧可见物体在夹爪附近，后续画面在目标盒内可见物体，机械臂随后移开；这支持粗略阶段顺序，但未连续观看全部 107 条视频，也未独立重标成功。首条旧轨迹闭合附近 75～77 步外部图相同，腕部画面和反馈仍变化，说明重复帧确实限制关键时刻的外部视觉分辨率。

在最大闭合变化/最大张开变化的前后各 3 步窗口内，训练集主视角相邻像素完全相同的比例分别为 **52.78%（190/360）和 48.61%（175/360）**，60 条训练轨迹在这两个窗口内均至少出现一对重复。此固定窗口仅作诊断，未据验证或测试数据制定重采样规则。逐 split 数值在 `reports/event_repeat_diagnostic.json`，视觉观察范围在 `reports/manual_visual_review.json`。

保护检查在独立转换期间、模型验证前建立快照，随后核对 364 个原始 RLDS/归档、候选 metadata/manifest、权重及工程 Python 源文件的大小与修改时间，未发现变化。它不是对全部原始图像的逐文件内容哈希复核。候选文件与源 RLDS 的 SHA-256 清单另见 `reports/version_hashes.json`。
'''
text=text.replace('## C. 需要补充证据或重新采集',addition+'\n## C. 需要补充证据或重新采集');report.write_text(text)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
src=Path('/mnt/d/KinovaDatasets/review_20261031/candidate_rlds_20261031/kinova_dual_view/1.5.0')
hashes={'source_rlds':[digest(p) for p in sorted(src.iterdir()) if p.is_file()],
        'candidate_parquet':[digest(p) for p in sorted((run/'lerobot').rglob('*.parquet'))],
        'candidate_metadata':[digest(p) for p in sorted((run/'lerobot').rglob('meta/*.json*'))]}
(R/'version_hashes.json').write_text(json.dumps(hashes,indent=2))
print('EVIDENCE_FINALIZED',len(hashes['candidate_parquet']))
