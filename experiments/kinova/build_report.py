"""Refresh Chinese Kinova experiment reports from saved evidence (stdlib only).

This script does not import models, inspect CUDA, change workflow, or launch jobs.
Only REPORT_zh.md, NEXT_STEPS_zh.md and reports/report_generation.json are written.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path


CPU_TESTS = [
    ("committed32源日志/提交元数据只读审计", ["smoke_recovery_source_audit.json"]),
    ("恢复脚本CPU控制检查（不计实际GPU恢复）", ["recover_smoke_cpu.json"]),
    ("恢复流程历史CPU检查", ["resume_commit32_workflow_cpu.json"]),
    ("当前三段续训路由CPU检查", ["segmented_workflow_cpu.json"]),
    ("当前stop-at/总LR预算/loader重放CPU检查", ["segmented_training_cpu.json"]),
    ("确定性flags/最高矩阵精度/71参数hash门禁CPU检查", ["deterministic_training_entry_cpu.json"]),
    ("E0必须记录71 BF16参数hash的评估接口CPU检查", ["deterministic_evaluation_cpu.json"]),
    ("两次smoke相同参数/输入却输出不同的CPU证据复核", ["same_graph_cross_process_nondeterminism_cpu.json"]),
    ("同图E0构造/评估接口CPU模拟检查", ["same_graph_evaluation_cpu.json"]),
    ("同图E0工作流门禁与训练数学AST检查", ["same_graph_training_entry_cpu.json"]),
    ("动作loss/维度与时间mask、小张量反向传播", ["masked_loss_cpu.json"]),
    ("真实native数据读取/episode边界/腕部mask", ["native_loader_validation.json"]),
    ("有效元素加权梯度累积与跳过无效更新", ["accumulation_unit_tests.json"]),
    ("离线误差与checkpoint选择逻辑", ["offline_metric_cpu_test_v2.json", "offline_metric_cpu_test.json", "evaluation_metrics_cpu_test.json"]),
    ("预测比较器与固定输入摘要逻辑", ["restore_comparison_cpu_test_v3.json", "restore_comparison_cpu_test_v2.json", "restore_comparison_cpu_test.json", "checkpoint_comparison_cpu_test.json"]),
    ("验证数据接口CPU检查", ["evaluation_data_interface_cpu_test.json"]),
    ("评估运行流程CPU微型模型测试", ["evaluation_runtime_cpu_toy_test.json"]),
    ("评估GPU资源门禁CPU检查", ["evaluation_resource_gate_cpu_test.json"]),
    ("本次CPU原生恢复/目标dtype/单leaf串行数值检查", ["bounded_native_weights_cpu.json"]),
    ("本次CPU恢复后/设备传输前clean cache维护检查", ["memory_maintenance_cpu.json"]),
    ("本次native维护hook时序/无实际OS写入CPU检查", ["cacheguard_native_cpu.json"]),
    ("本次评估加载接口AST/CPUstub检查（不是实际模型验证）", ["bounded_inference_cpu.json"]),
    ("旧完整NumPy原生加载CPU检查（历史，不认证本次改动）", ["native_weights_cpu.json"]),
    ("本次分阶段内存采样CPU检查（不计模型/GPU实测）", ["phase_memory_cpu.json"]),
    ("旧存储chunk恢复CPU检查（历史，不认证本次原生路径）", ["chunked_restore_cpu.json"]),
    ("旧逐leaf恢复CPU检查（历史实现，不认证本次chunk路径）", ["streaming_restore_current_cpu.json", "streaming_restore_cpu.json"]),
    ("旧I/O CPU检查（历史；不证明错误restore构造器限流生效）", ["bounded_io_cpu.json"]),
    ("空闲clean pagecache回收CPU模拟检查（未实际回收/GPU）", ["idle_reclaim_cpu.json"]),
    ("Windows缓存/低余量处理CPU检查", ["windows_memory_cpu_test.json"]),
    ("Windows实际采样检查（无模型/GPU）", ["windows_memory_live_test.json"]),
    ("Windows监控存活与清理检查", ["monitor_health_cpu_live.json", "windows_monitor_no_orphan_test.json"]),
    ("Windows稳定观察窗口CPU检查", ["window_guard_cpu.json"]),
    ("子进程资源拒绝发生于JAX初始化之前", ["resource_defer_cpu.json"]),
    ("启动前资源变化返回队列且保留日志", ["resource_race_workflow_cpu.json"]),
    ("本次独占GPU资源策略CPU检查", ["exclusive_resource_cpu.json", "new_exclusive_resource_cpu.json"]),
]
REAL_CHECKS = [
    ("同图零更新E0与smoke初始预测严格等价", "base_zero_lora_equivalence.json"),
    ("独立两次checkpoint恢复参数/Adam/step一致", "full_state_restore.json"),
    ("独立两次checkpoint恢复预测严格一致", "checkpoint_prediction_restore.json"),
]


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def cell(value):
    if value is None:
        return "未实测/UNKNOWN"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.8g}" if math.isfinite(value) else "非有限值"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def saved_timestamp(value):
    """Sort saved snapshots within a provenance tier, never across tiers."""
    if not isinstance(value, dict):
        return float("-inf")
    text = value.get("checked_at") or value.get("updated_at")
    if isinstance(text, (int, float)):
        return float(text)
    try:
        parsed = dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=dt.timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
    except (ValueError, TypeError):
        return float("-inf")


def choose_resource_gate(workflow, saved_gate, environment):
    """Current-run workflow wins even if a copied legacy file has a later date."""
    current = [("workflow.json#" + key, workflow.get(key)) for key in ("resource_gate", "gpu_gate")
               if isinstance(workflow.get(key), dict) and workflow.get(key)]
    if current:
        source, gate = max(current, key=lambda item: saved_timestamp(item[1]))
        return source, gate, True
    if isinstance(saved_gate, dict) and saved_gate:
        return "reports/last_gpu_gate.json（回退快照）", saved_gate, False
    gate = environment.get("gpu_gate")
    return ("reports/environment.json#gpu_gate（历史回退）", gate, False) if isinstance(gate, dict) and gate else (None, {}, False)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def gib(value):
    value = number(value)
    return value / 2**30 if value is not None else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_action_continuity(root):
    """Optional CPU-only source_time column reread; never reads image columns."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    candidate = Path(project_path('candidate', ''))
    rows, totals = [], {"pairs": 0, "action_target_matches_next_robot": 0,
                        "target_robot_matches_next_robot": 0, "next_current_minus_target_ns_min": None,
                        "next_current_minus_target_ns_max": None}
    for split in ("train", "validation", "test"):
        repo = candidate / "lerobot/kinova_candidate" / split
        info = json.loads((repo / "meta/info.json").read_text())
        names = info["features"]["source_time_ns"]["names"]
        target = names.index("action_target_host_monotonic_ns")
        target_robot = names.index("target_robot_host_monotonic_ns")
        current_robot = names.index("robot_host_monotonic_ns")
        for line in (repo / "meta/episodes.jsonl").read_text().splitlines():
            episode = json.loads(line)
            idx = episode["episode_index"]
            path = repo / info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
            values = np.asarray(pq.read_table(path, columns=["source_time_ns"], use_threads=False)["source_time_ns"].to_pylist(), dtype=np.int64)
            d = values[1:, current_robot] - values[:-1, target]
            d_robot = values[1:, current_robot] - values[:-1, target_robot]
            entry = {"split": split, "episode_index": idx, "parquet_path": str(path), "steps": len(values),
                     "pairs": len(d), "action_target_matches_next_robot": int(np.sum(d == 0)),
                     "target_robot_matches_next_robot": int(np.sum(d_robot == 0)),
                     "next_current_minus_action_target_ns_min": int(d.min()) if len(d) else None,
                     "next_current_minus_action_target_ns_max": int(d.max()) if len(d) else None,
                     "mismatch_examples": [{"current_frame": int(i), "target_ns": int(values[i, target]),
                                            "next_robot_ns": int(values[i + 1, current_robot])}
                                           for i in np.where(d != 0)[0][:5]],
                     "source_time_int64_sha256": hashlib.sha256(values.tobytes()).hexdigest()}
            rows.append(entry)
            totals["pairs"] += len(d)
            totals["action_target_matches_next_robot"] += entry["action_target_matches_next_robot"]
            totals["target_robot_matches_next_robot"] += entry["target_robot_matches_next_robot"]
    ds_min = [r["next_current_minus_action_target_ns_min"] for r in rows if r["pairs"]]
    ds_max = [r["next_current_minus_action_target_ns_max"] for r in rows if r["pairs"]]
    totals["next_current_minus_target_ns_min"], totals["next_current_minus_target_ns_max"] = min(ds_min), max(ds_max)
    output = {"status": "complete", "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "actual Parquet source_time_ns only, each episode separately; no images/model/GPU",
              "candidate_root": str(candidate), "episodes": rows, "totals": totals,
              "all_pairs_continue_exactly": totals["pairs"] == totals["action_target_matches_next_robot"] == totals["target_robot_matches_next_robot"],
              "meaning": "Each action is its own start-feedback to own target-feedback increment. Exact next-row continuation is a measured property of this D0, not assumed from a delta label.",
              "test_usage": "timestamp integrity only; no model score or checkpoint selection"}
    atomic_text(root / "reports/data_action_continuity.json", pretty(output) + "\n")
    print(pretty({k: output[k] for k in ("status", "totals", "all_pairs_continue_exactly")}))


class Evidence:
    def __init__(self, root):
        self.root = root
        self.reads, self.absent, self.errors = {}, [], []

    def read(self, relative, default=None):
        path = self.root / relative
        if not path.exists():
            if str(relative) not in self.absent:
                self.absent.append(str(relative))
            return default
        try:
            data = path.read_bytes()
            value = json.loads(data)
            self.reads[str(relative)] = {"sha256": hashlib.sha256(data).hexdigest(),
                                         "mtime_ns": path.stat().st_mtime_ns, "bytes": len(data)}
            return value
        except (OSError, ValueError) as exc:
            self.errors.append({"path": str(relative), "error": repr(exc)})
            return default

    def first_report(self, names):
        for name in names:
            if (self.root / "reports" / name).exists():
                return "reports/" + name, self.read("reports/" + name, {})
        for name in names:
            self.read("reports/" + name)
        return None, None


def phase(ev, name, target):
    evidence = ev.read(f"reports/{name}_training_evidence.json", {})
    status = ev.read(f"reports/{name}_training_status.json", {})
    complete = ev.read(f"reports/{name}_complete.json", {})
    current = max([int(v.get("optimizer_step", 0) or 0) for v in (evidence, status, complete)] + [0])
    trained = (evidence.get("status") == "trained" or evidence.get('status') == 'complete'
               and evidence.get('independent_full_restore_verified') is True
               and evidence.get('independent_prediction_restore_verified') is True) and current >= target
    checkpoint = evidence.get("checkpoint") or complete.get("checkpoint") or complete.get('source_checkpoint')
    return {"name": name, "target": target, "optimizer_step": current, "trained": trained,
            "evidence": evidence, "status": status, "complete": complete,
            "checkpoint": checkpoint, "checkpoint_exists": Path(checkpoint).is_dir() if checkpoint else False}


def check_status(value):
    if value is None or not value:
        return "未执行（证据文件不存在或无法读取）"
    return value.get("status", "文件存在；未声明通过状态")


def make_report(root):
    ev = Evidence(root)
    data = ev.read("reports/data_summary.json", {})
    episodes = ev.read("reports/data_episode_audit.json", [])
    quality = ev.read("reports/data_quality_interpretation.json", {})
    continuity = ev.read("reports/data_action_continuity.json", {})
    continuation = ev.read("reports/continuation_data_check.json", {})
    continuation_manifest = ev.read("manifests/continuation_provenance.json", {})
    if not continuation_manifest:
        continuation_manifest = ev.read("manifests/continuation.json", {})
    inherited_e0 = continuation_manifest.get('inherited_E0', {})
    legacy_e0 = continuation_manifest.get('legacy_E0', {})
    environment = ev.read("reports/environment.json", {})
    layout = ev.read("reports/parameter_layout.json", {})
    workflow = ev.read("workflow.json", {})
    resource_policy = ev.read("configs/resource_policy.json", {})
    shared_policy = (any("shar" in str(resource_policy.get(k, "")).lower()
                         or "concurrent" in str(resource_policy.get(k, "")).lower()
                         for k in ("mode", "strategy", "policy", "name"))
                     or resource_policy.get("allow_concurrent_libero") is True
                     or resource_policy.get("allow_libero_sharing") is True)
    shared_policy = shared_policy and resource_policy.get("user_authorized") is True
    exclusive_policy = str(resource_policy.get("mode", "")).startswith("exclusive")
    copied_gate = ev.read("reports/last_gpu_gate.json", {})
    gate_source, last_gate, gate_from_workflow = choose_resource_gate(workflow, copied_gate, environment)
    gate_shared = "shar" in pretty(last_gate.get("resource_policy", last_gate.get("policy", last_gate.get("mode", "")))).lower()
    historical_gate = bool(last_gate and not gate_from_workflow and (continuation_manifest or (resource_policy and shared_policy and not gate_shared)))
    windows_current = ev.read("reports/windows_current.json", {})
    windows_source = "reports/windows_current.json"
    if not windows_current:
        monitor_windows = workflow.get("shared_resource_monitor", {}).get("windows_memory", {})
        gate_windows = last_gate.get("windows_memory", {})
        windows_current = monitor_windows if isinstance(monitor_windows, dict) and monitor_windows else gate_windows
        windows_source = "workflow.json#shared_resource_monitor.windows_memory（回退）" if monitor_windows else str(gate_source) + ".windows_memory（回退）"
    windows_monitor_status = ev.read("reports/windows_monitor_status.json", {})
    history = sorted(p.relative_to(root).as_posix() for p in (root / "manifests/revisions").glob("share_gpu_*") if p.is_dir())
    smoke, pilot = phase(ev, "smoke", 32), phase(ev, "pilot", 2000)
    recovery_source = continuation_manifest.get('recovery', {})
    if recovery_source.get('source_optimizer_updates') == 32 and smoke['optimizer_step'] < 32:
        smoke['optimizer_step'] = 32
        smoke['checkpoint'] = recovery_source.get('source_checkpoint')
        smoke['checkpoint_exists'] = Path(smoke['checkpoint']).is_dir() if smoke['checkpoint'] else False
    segmented = continuation_manifest.get('segmented_pilot_resume', {})
    if segmented and pilot['optimizer_step'] < 500:
        pilot['optimizer_step'] = 500
        pilot['checkpoint'] = segmented.get('copied_checkpoint')
        pilot['checkpoint_exists'] = Path(pilot['checkpoint']).is_dir() if pilot['checkpoint'] else False
    smoke_config = ev.read("configs/smoke_effective.json", {})
    pilot_config = ev.read("configs/pilot_effective.json", {})
    plan = ev.read("configs/validation_plan.json", {})
    evaluation_config = ev.read("configs/evaluation_config.json", {})
    selection = ev.read("configs/checkpoint_selection.json", {})
    cpu = [(label, *ev.first_report(names)) for label, names in CPU_TESTS]
    real = [(label, name, ev.read("reports/" + name)) for label, name in REAL_CHECKS]
    evals = []
    for label, step, path in ([("E0_wrist_frozen", 0, "evaluations/E0_wrist_frozen/summary.json")]
        + [("E1_wrist_pilot", step, f"evaluations/E1_wrist_pilot/step_{step}/summary.json") for step in (500, 1000, 1500, 2000)]):
        value = ev.read(path)
        evals.append((label, step, path, value))
    now = dt.datetime.now(dt.timezone.utc)
    updated = now.astimezone(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S 北京时间")
    lines = ["# Kinova Gen3 + π0.5 单任务离线实验报告", "", f"更新时间：{updated}。实验根目录：`{root}`。", ""]

    def add(value):
        lines.extend([value, ""])

    total = smoke["optimizer_step"] + pilot["optimizer_step"]
    if segmented:
        add(f"**本轮从完整保存的pilot500继续，当前轨迹步数{pilot['optimizer_step']}；不把源实验未保存的501–1000当作已恢复。** E0和smoke32及其独立恢复验收已验证复用，不再做GPU重复检查。三个进程分别恢复/保存到1000、1500、2000，每段后退出释放内存池，总学习率预算始终2000。本轮计划新增1500更新，随后执行原四个checkpoint验证。")
    elif recovery_source:
        add(f"**源实验已完成32次真实optimizer更新并提交checkpoint32；本轮不重复训练这32步。恢复验收{'已通过' if smoke['trained'] else '尚未完成'}，本轮新增smoke更新0步，pilot累计{pilot['optimizer_step']}步。** 原最后RAM快照未留下，首次checkpoint恢复是新的对照参考；两次恢复一致不能补证原RAM到保存等价。")
    elif not total:
        add("**CPU数据与工程检查已有证据；本次可核实的真实π0.5 optimizer更新为0步，32步smoke和最多2000步pilot仍是准备方案。训练峰值、checkpoint与模型离线验证分别以对应实测证据为准，不能由准备配置推断完成。**")
    else:
        add(f"**本次已有真实π0.5更新证据：smoke累计{smoke['optimizer_step']}步，pilot累计{pilot['optimizer_step']}步。pilot是否完成2000步：{'已完成训练更新' if pilot['trained'] else '尚未完成'}；完整恢复验收及验证结果分别见下表。**")
    add("本任务是一个真实抓取放盒任务的腕部单视角D0探索性基线。全部107条人工复核尚待完成，两路曝光/状态同步仍UNVERIFIED；离线loss或动作误差不是真机成功率。没有开展多任务CL或遗忘率实验。")

    if continuation_manifest:
        previous = continuation_manifest.get("previous_failed_experiment", continuation.get("previous_failed_run", "UNKNOWN"))
        previous_initial = continuation_manifest.get("previous_initial_failed_experiment")
        add("## 续作来源与证据边界")
        add(f"本目录从`{previous}`已提交的32步只读恢复；复用其已验证60块/71BF16hash的确定性同图E0，逐文件SHA与来源明确记录。collect建立首次恢复快照，verify在独立进程比较完整参数/Adam/step/固定预测，全部通过后才从原始BASE另起pilot2000。E0和32步训练属于来源，不冒充本轮新执行。旧日志/源码/完整checkpoint保留，本轮不向旧run写入。")
        if previous_initial:
            add(f"更早的首次尝试`{previous_initial}`同样保留。新授权、此前失败现场和两套已完成LIBERO结果的保护SHA记录在[continuation_provenance.json](manifests/continuation_provenance.json)。没有将旧smoke/pilot更新或旧检查点复制为本次完成证据。")
        if inherited_e0:
            add(f"**E0为明确复用，非本轮重新执行**：来源`{inherited_e0.get('source_root')}`，固定20条validation/60个block-seed和完整71 BF16参数hash，summary/预测/参考共{len(inherited_e0.get('files', []))}份按SHA复制。选择分数`{inherited_e0.get('selection_score')}`，plan `{inherited_e0.get('plan_sha256')}`，norm `{inherited_e0.get('norm_stats_sha256')}`，输入/噪声 `{inherited_e0.get('validation_input_and_noise_sha256')}`。新workflow只登记复用，不重新加载E0模型；来源证明见lineage及cpu_preflight。")
        if continuation:
            add(f"新run实际快速复核：`{continuation.get('checked_at', 'UNKNOWN')}`，状态`{continuation.get('status', 'UNKNOWN')}`；重读三个repo meta与全部107份Parquet的state/action/source_time数值列，数值payload与旧审计一致：{cell(continuation.get('all107_numerical_payloads_match_prior_audit'))}。实际划分`{continuation.get('splits', {})}`；{continuation.get('referenced_sources_checked', 'UNKNOWN')}个小型reference文件SHA全部未变：{cell(continuation.get('all_referenced_sources_unchanged'))}。norm SHA256 `{continuation.get('norm_stats_sha256', 'UNKNOWN')}`。")
            add("本次快速复核没有重解码图像、重跑746个原始反馈抽查或321个native loader位置，也没有加载完整基座数组。只读核对基座identity manifest和checkpoint小metadata，不宣称重新哈希全部大型权重。逐文件新旧来源与SHA见[continuation_data_check.json](reports/continuation_data_check.json)及[continuation_data_evidence.json](manifests/continuation_data_evidence.json)。")
        else:
            add("新run快速复核记录尚未出现，不能仅凭复制的旧报告声称候选来源已重新检查。")

    add("## 当前运行状态与资源")
    if continuation_manifest:
        previous_root = continuation_manifest.get("previous_failed_experiment")
        old_workflow = ev.read(Path(previous_root) / "workflow.json", {}) if previous_root else {}
        add(f"**源停止历史，不是本轮失败**：`{previous_root}`旧workflow状态`{old_workflow.get('status', 'UNKNOWN')}`，error=`{cell(old_workflow.get('error'))}`。205332通过60块E0、初始71BF16身份与严格预测检查，完成32更新并提交32；保存收尾被资源保护停止，最后RAM参数/Adam/预测快照未保留。本轮补做只读恢复验收，不重训32；更早输出差异和内存失败分别保留。")
    execution = ev.read('configs/experiment.json', {}).get('deterministic_execution', {})
    add('## 本轮数值执行与参数身份验收')
    add('固定XLA_FLAGS=--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true、CUBLAS_WORKSPACE_CONFIG=:4096:8、JAX_DEFAULT_MATMUL_PRECISION=highest。模型/损失/优化超参数保持，数值执行策略明确改变，本轮执行策略与205332一致，明确复用其已验证E0并核对SHA。CPU解析/配置通过不证明完整GPU确定性已实现。')
    add('E0 factory显式hash_parameters=True，记录51个base+20个adapter的完整BF16推理视图hash；smoke capture前也记录71个BF16 hash，必须逐项完全一致，才继续原atol=rtol=1e-5预测/FM/count比较。参数hash相等不能替代严格输出或完整checkpoint/Adam/step恢复验收；不放宽或跳过任何原验收。')
    if execution:
        add('保存的执行配置：\n\n```json\n' + pretty(execution) + '\n```')
    discrepancy = ev.read('reports/same_graph_cross_process_nondeterminism_cpu.json', {})
    add(f"两次旧smoke保存证据的CPU审计状态`{discrepancy.get('status', '未确认')}`，范围`{discrepancy.get('scope', 'UNKNOWN')}`；这表示差异被核实，不是模型通过。202704旧E0未记录71 BF16 hash，不能从旧mixed smoke hash反推E0身份。本轮补齐此缺口。")
    add('## 新E0与旧纯Frozen基线的边界')
    add('本轮复用205332已完成的同图零更新E0：native seed42、BF16 view、71hash、20episode/60块及314159参考，来源与SHA明确。源32步训练前已通过71hash身份与严格1e-5初始输出，均不冒充新执行。旧纯Frozen E0仍只作legacy。')
    if legacy_e0:
        add(f"旧纯Frozen E0仅保留为legacy：来源`{legacy_e0.get('source_root')}`，{len(legacy_e0.get('files', []))}个文件按SHA隔离复制至legacy_evaluations/legacy_reports，旧选择分数`{legacy_e0.get('selection_score')}`。它不是本轮E0完成证据，**不用于E1改善判断**。")
    diag_gate = ev.read('reports/same_graph_diagnostic_gate.json', {})
    add(f"共享权重GPU诊断的分项记录见reports/shared_graph_diagnostic及CPU只读门禁same_graph_diagnostic_gate.json（状态`{diag_gate.get('status', '未确认')}`）：51个基座数组共享身份、10个B全零；LoRA对上次LoRA的prediction/FM/count三项差异均0；共享Frozen对旧纯Frozen预测差0，但FM numerator差约0.280685，故全局references_reproduced=false，不能称全部指标复现。共享参数两种图的预测差约0.0673447、FM numerator差1.0546875。CPU微型Einsum/FFN六例差异0，也不能单独解释完整GPU差异。新基线变更是公开的实验定义变更，不是放宽容差。")
    if workflow:
        add(f"工作流状态：`{workflow.get('status', 'UNKNOWN')}`；阶段：`{workflow.get('phase', workflow.get('current_phase', workflow.get('current_job', 'UNKNOWN')))}`。更新：`{workflow.get('updated_at', 'UNKNOWN')}`。完整记录：[workflow.json](workflow.json)。")
    else:
        add("工作流状态文件尚不存在，不能据此声称后台worker已经启动或已完成。")
    if resource_policy:
        add("当前生效资源策略来源：[configs/resource_policy.json](configs/resource_policy.json)。报告仅展示已保存配置，不更改门禁。\n\n```json\n" + pretty(resource_policy) + "\n```")
        policy_info = ev.reads.get("configs/resource_policy.json", {})
        if policy_info.get("mtime_ns"):
            policy_mtime = dt.datetime.fromtimestamp(policy_info["mtime_ns"] / 1e9, dt.timezone.utc).isoformat()
            add(f"当前策略文件修改时间UTC：`{policy_mtime}`；SHA256 `{policy_info.get('sha256')}`。这是配置文件标识，不代表实际恢复或训练已通过。")
        if exclusive_policy:
            add("**本次采用独占GPU策略。** 两套LIBERO实验已结束，本次仍要核查没有外来CUDA作业及GPU、Linux内存和Windows提交余量。当前配置采用JAX fraction0.85、关闭预分配，Windows启动提交余量至少16GiB并观察稳定30秒；具体阈值以本页JSON和当前workflow门禁为准。旧共卡.40分配器探测不能作为本轮.85或完整模型训练已成功的证据。")
            add("运行保护只停止本次Kinova自身，保留所有旧实验。JAX fraction是分配器策略，不是MIG硬分区或整进程显存保证；资源余量达标也不等于32步smoke、checkpoint恢复或2000步pilot已经完成。")
            if resource_policy.get('drop_clean_cache_after_native_cpu_restore'):
                add("本次新增CPU恢复后、设备传输前的clean pagecache维护：仅在己方受保护的独占模型流程调用；drop_caches回收的是Linux系统范围clean pagecache，并非只清某个进程，CPU参数树恢复完成并释放读取临时对象后，维护通过才继续merge/GPU init或eval device_put。它可能执行sync/drop_caches=1，属于暂态清洁缓存维护，不改pagefile/.wslconfig/swap、不重启、不降低任何启动/运行门槛，也不保证后续内存/训练一定成功。CPU测试和hook先后顺序不算实际回收或GPU结果，实际事件以新root记录为准。")
            if resource_policy.get("idle_clean_pagecache_reclaim"):
                add("本次允许每个execute阶段最多一次受条件保护的空闲clean pagecache回收：仅在无GPU/模型作业、Windows提交余量低于16GiB且Linux cache/anon达到策略条件时尝试，sync后再次检查才写drop_caches=1。不降低16GiB/30秒准入及运行保护阈值，不杀其他作业、不改数据文件；CPU模拟通过不能证明实际回收发生或保证改善。实际事件见reports/idle_reclaim.jsonl。")
        else:
            add("当前策略文件未声明exclusive模式，与本次独占方案不一致，须在启动前核对；本报告不会自行修改策略或启动GPU。")
    else:
        add("本次资源策略文件尚不存在，不能据旧共卡或空闲门禁快照认定当前独占资源条件已通过。")
    if last_gate:
        add(f"{'历史回退' if historical_gate else '当前workflow保存的' if gate_from_workflow else '回退文件保存的'}GPU门禁检查，来源`{gate_source}`：`{last_gate.get('checked_at', 'UNKNOWN')}`，eligible=`{last_gate.get('eligible', 'UNKNOWN')}`。生成报告时不查询GPU。" + (" 当前workflow门禁不存在；此历史回退不能证明本次已满足资源条件。" if historical_gate else " 当前workflow记录优先于复制的last_gpu_gate文件，与文件时间先后无关。" if gate_from_workflow else " 这是回退快照，不作为当前workflow已经通过的证明。"))
        if last_gate.get("blocking_reasons"):
            add(("历史回退阻塞原因（不作为当前独占规则）：" if historical_gate else "该次检查的阻塞原因：") + "\n\n" + "\n".join("- " + str(v) for v in last_gate["blocking_reasons"]))
        if last_gate.get("gpus"):
            add(table(["GPU", "总显存MiB", "已用MiB", "空闲MiB", "利用率%"],
                      [(v.get("name"), v.get("total_mib"), v.get("used_mib"), v.get("free_mib"), v.get("utilization_percent")) for v in last_gate["gpus"]]))
    maintenance_records = []
    for path in sorted((root/'reports').glob('device_transfer_reclaim_*.json')):
        value = ev.read(str(path.relative_to(root)), {})
        if value:
            maintenance_records.append((str(path.relative_to(root)), value))
    if maintenance_records:
        add('### 本轮CPU恢复后/传输前维护记录')
        for path, value in maintenance_records:
            add(f'来源`{path}`，原字段如下；仅有started记录或缺少完成值不能算回收成功。\n\n```json\n' + pretty({k: value.get(k) for k in ('status', 'performed', 'pid', 'started_at', 'finished_at', 'before', 'after', 'error')}) + '\n```')
        add('逐事件时序见reports/device_transfer_reclaim.jsonl。维护完成不代替模型训练、checkpoint恢复或验证验收。')
    else:
        add('本轮CPU恢复后/传输前实际维护尚无保存证据；memory_maintenance_cpu.json的模拟sync/临时fixture写入不算生产环境已回收缓存。')
    if windows_current:
        sampled_unix = number(windows_current.get("sampled_at_unix"))
        received_unix = number(windows_current.get("received_at_unix"))
        sampled_age = now.timestamp() - sampled_unix if sampled_unix is not None else None
        received_age = now.timestamp() - received_unix if received_unix is not None else None
        max_age = number(resource_policy.get("windows_sample_max_age_seconds"))
        ages = [v for v in (sampled_age, received_age) if v is not None]
        age_note = ("按墙钟估算已超出缓存年龄上限" if ages and max_age is not None and max(ages) > max_age
                    else "存在未来时间/时钟差，不能据此认证新鲜度" if ages and min(ages) < 0
                    else "仅展示缓存年龄，不重新认证门禁新鲜度")
        add("### Windows当前保存的提交内存缓存")
        add(f"来源`{windows_source}`；Windows采样时间`{windows_current.get('sampled_at', 'UNKNOWN')}`，sampled_at_unix=`{windows_current.get('sampled_at_unix', 'UNKNOWN')}`。模式`{windows_current.get('sampling_mode', 'UNKNOWN')}`，sequence=`{windows_current.get('sequence', 'UNKNOWN')}`。这是已保存缓存，报告不会调用Windows采样器或刷新数据。")
        add(table(["缓存指标", "值"], [
            ("已提交内存GiB", gib(windows_current.get("committed_bytes"))),
            ("提交上限GiB", gib(windows_current.get("commit_limit_bytes"))),
            ("当前剩余提交额度GiB", gib(windows_current.get("commit_free_bytes"))),
            ("Windows物理可用内存GiB", gib(windows_current.get("physical_available_bytes"))),
            ("距Windows采样秒（按报告生成墙钟估算）", sampled_age),
            ("距WSL接收秒（按报告生成墙钟估算）", received_age),
            ("策略缓存年龄上限秒", max_age),
            ("滚动观察窗口实际秒", windows_current.get("rolling_commit_window_seconds")),
            ("策略要求稳定窗口秒", resource_policy.get("windows_stable_window_seconds")),
            ("滚动窗口最低剩余提交额度GiB", gib(windows_current.get("rolling_min_commit_free_bytes"))),
            ("滚动窗口最大样本间隔秒", windows_current.get("rolling_max_sample_gap_seconds")),
            ("策略启动最低提交余量GiB", resource_policy.get("min_windows_commit_free_gib_before_launch")),
            ("策略运行停止余量GiB", resource_policy.get("abort_windows_commit_free_gib")),
        ]))
        add(age_note + "。实际门禁还检查boot标识、单调时钟和稳定观察窗口；本报告只读缓存，不重做这些检查。未记录的滚动值为UNKNOWN，不能用单次高余量代替完整稳定窗口。提交额度、WSL可用内存和GPU空闲显存分别计量。")
    else:
        add("Windows当前缓存尚无可读证据：未找到reports/windows_current.json或可用workflow回退，不能用旧失败后快照代替当前提交余量。")
    if windows_monitor_status:
        add("Windows监控状态文件（仅缓存状态，不复查进程存活）：\n\n```json\n" + pretty({k: windows_monitor_status.get(k) for k in ("status", "updated_at_unix", "last_sample_unix", "windows_pid", "python_pid", "error")}) + "\n```")
    add("旧waiting_resources、资源中止和此前smoke加载失败分别属于历史尝试；日志、checkpoint、配置、源码及被保护manifest/results保留。本次不改Windows pagefile、.wslconfig、swap或重启；已授权的受条件保护空闲clean pagecache维护不改变这些持久配置。门禁等待、权重加载和真实optimizer更新分别记录。")
    disk = environment.get("disk_bytes", {})
    if continuation_manifest and environment.get("cwd") != str(root):
        add("下方environment.json是复用的旧run历史环境快照，不是新run即时资源检查；时间、cwd和磁盘余量均保留原值。旧workflow和旧frozen code manifest没有复制为新run活动状态，历史归档仍位于上一轮目录。")
    add(f"已查主机/用户：`{environment.get('host', 'UNKNOWN')}` / `{environment.get('user', 'UNKNOWN')}`；Python `{environment.get('python', 'UNKNOWN')}`，解释器 `{environment.get('executable', 'UNKNOWN')}`。环境检查时间 `{environment.get('checked_at', 'UNKNOWN')}`；当时可用磁盘 {disk.get('free', 0) / 2**30:.1f}GiB。共享环境是否改动：{'未改动' if environment.get('no_shared_environment_modifications') is True else '证据未确认'}。")

    add("## 数据：续作快查与历史审计" if continuation_manifest else "## 已重新核实的数据")
    add(table(["划分", "episode", "原始步数"], [(s, data.get("splits", {}).get(s, {}).get("episodes"), data.get("splits", {}).get(s, {}).get("steps")) for s in ("train", "validation", "test")]))
    add(f"原始tar metadata记录{data.get('raw_record_count_from_archive', 'UNKNOWN')}条，状态分布`{data.get('raw_archive_status_counts', {})}`；部分展开目录只有{data.get('raw_expanded_metadata_count_current', data.get('raw_inventory_count_current', 'UNKNOWN'))}份metadata，不能当原始总量。候选{data.get('candidate_episodes', 'UNKNOWN')}条/{data.get('candidate_steps', 'UNKNOWN')}步。与用户报告一致：{cell(data.get('user_report_matches'))}；split交集：`{data.get('split_episode_id_intersections', {})}`。")
    add(("以下为旧run全审计结果，新run只验证数值payload/metadata/reference身份延续：" if continuation_manifest else "") + f"逐episode读取实际Parquet数值并独立抽查{data.get('independent_action_check_samples', 'UNKNOWN')}个原始反馈转移，最大动作绝对误差{cell(data.get('max_action_abs_error'))}，状态误差{cell(data.get('max_state_abs_error'))}。数值审计通过：{cell(data.get('numerical_checks_pass'))}。train-only mean/std/q01/q99来自60条/8011原始步，每步只计一次；与实际训练Parquet重算逐项吻合，详见[data_normalization_recheck.json](reports/data_normalization_recheck.json)。")
    add("28维状态顺序为关节位置7（度）、关节速度7（度/秒）、关节力矩7（N·m）、末端xyz3（米）、Kortex theta_xyz3（度）、实际夹爪位置1（原始反馈）。7维动作不是关节命令：该行自身起点反馈到自身目标反馈的xyz差3（米）、绕回[-180,180)后的欧拉分量差转弧度3、目标时刻实际夹爪反馈映射1。每个50步块逐行保留各行增量，不相对块起点。没有再次DeltaActions/AbsoluteActions。")
    if continuity:
        v = continuity.get("totals", {})
        add(("复用旧run连续性实查（新run数值SHA一致）：" if continuation_manifest else "额外重读实际source_time列：") + f"action_target与下一行robot_time匹配{v.get('action_target_matches_next_robot')}/{v.get('pairs')}对；target_robot匹配{v.get('target_robot_matches_next_robot')}/{v.get('pairs')}对，下一行起点减当前目标的时间差范围[{v.get('next_current_minus_target_ns_min')},{v.get('next_current_minus_target_ns_max')}]ns。这是该D0实测衔接结果，不是仅凭delta标签作出的假设，见[data_action_continuity.json](reports/data_action_continuity.json)。")
    else:
        add("当前行target与下一行state是否严格同一时刻：额外连续性检查尚无证据；每行自身起点→自身target的增量语义不能自动推出严格衔接。")
    add("夹爪使用open raw=0.4386、closed raw=99.1228映射并裁剪到[0,1]，1张开/0闭合；实际反馈可能受物体阻挡，不等于操作者目标位置。107条commands.jsonl为空。控制参考系、执行姿态顺序/TCP、周期与夹爪API对应关系仍UNKNOWN。")
    images = quality.get("images", {})
    if images:
        add(table(["视角", "旧审计PNG容器检查条目" if continuation_manifest else "PNG容器检查条目", "旧审计RGB解码张数" if continuation_manifest else "本轮RGB解码张数", "相邻完全同PNG", "比例"],
                  [(v, images[v].get("entries"), images[v].get("decoded_rgb_samples"), f"{images[v].get('identical_encoded_pairs')}/{images[v].get('adjacent_pairs')}", f"{images[v].get('repeat_ratio', 0):.2%}") for v in images]))
    add("源主视角HWC 640×480×3、腕部480×640×3。候选嵌入PNG；" + ("全部容器结构检查及首中末RGB解码来自旧run，本次没有重新解码图像。" if continuation_manifest else "本轮全部容器结构检查，首中末实际RGB解码，未声称本轮全量逐像素重解码。") + "旧图像转换/像素保真证据与当前检查范围分开保留。")
    if episodes:
        dt_min = min(r["actual_action_interval_ms"]["min"] for r in episodes)
        dt_max = max(r["actual_action_interval_ms"]["max"] for r in episodes)
        add(f"真实动作间隔范围{dt_min:.5f}～{dt_max:.5f}ms。名义timestamp采用10Hz索引，不是根据它插值动作。原始8列纳秒时间保留；候选时间列无倒序，外部原日志左右流交错行的顺序与单路时钟倒退必须区分。")
    tail = quality.get("tail", {})
    if tail:
        trim = tail.get("tail_trim_seconds", {})
        add(f"候选最后动作目标至原始结束间隔{cell(trim.get('min'))}～{cell(trim.get('max'))}秒；{tail.get('episodes_with_later_raw_feedback', 'UNKNOWN')}条仍有更晚原始反馈，{tail.get('episodes_with_later_raw_wrist_records', 'UNKNOWN')}条仍有更晚腕部记录。这支持旧导出截断后原始记录仍被保留，不能据此直接证明释放/成功，也不必要求全部重采。")
    add("**全部107条仍为manual_review_pending**。既有success标注及manifest绑定保留为来源信息，不自动等同重新确认成功。全轨迹首尾、各时间统计、原始后续记录和独立数值样本见[逐轨迹CSV](reports/data_episode_audit.csv)、[逐轨迹JSON](reports/data_episode_audit.json)、[动作核对样本](reports/data_action_reconstruction_samples.json)。[review入口](review/index.html)提供全部107条逐帧/名义10Hz播放及原始尾帧，只读引用源图像，下载文本包不包含全部图像。")
    add("外部2.08秒补偿继承原经验估计，原挥手校准记录不在历史包内。外部估计内容年龄不是真实残余同步误差。external_sync_status=UNVERIFIED，wrist_sync_status=UNVERIFIED；没有共同曝光/状态事件，不能仅凭时间单调断言无漂移。")

    add("## 专用模型、输入与训练方案")
    model = layout.get("model", {})
    add(f"π0.5普通单任务LoRA，PaliGemma和动作专家rank={model.get('lora_rank', 'UNKNOWN')}、alpha={model.get('lora_alpha', 'UNKNOWN')}、dropout={model.get('lora_dropout', 'UNKNOWN')}；CL bank关闭。基座只读路径为`/root/deployments/pi05_cl_lora_20261031/models/pi05_base/params`，实际运行路径以effective配置和训练证据为准。基座未重新下载。")
    add("训练与验证只激活left_wrist_0_rgb；base_0_rgb和right_wrist_0_rgb为零图且mask=false，不复制腕图伪造双视角。保持比例resize+pad到224×224。28维状态按train q01/q99仿射归一化，经离散状态文本token进入π0.5，再补4零到32维；max_token_len200。动作归一化50×7后补25维到50×32，推理按同一统计反归一化取前7维。")
    add(table(["配置项", "准备值（未运行时不代表已生效）", "真实运行证据"], [
        ("实际batch / 梯度累积 / 有效batch", "1 / 4 / 4", "configs/smoke_effective.json或pilot_effective.json"),
        ("优化器 / 梯度裁剪 / weight decay", "分组AdamW / 全局1.0 / 0", "effective配置与训练证据"),
        ("LoRA peak LR / 动作轻量头peak LR", "1e-4 / 1e-5", "同上"),
        ("warmup / cosine decay / seed", "100 / 2000 / 42", "同上"),
        ("EMA / KD / replay", "全部关闭", "单任务基线"),
        ("smoke / pilot上限", "32 / 2000真实optimizer steps", "以真实step证据为准"),
        ("pilot保存步", "500、1000、1500、2000", "未出现文件即未确认保存"),
    ]))
    add("smoke与pilot使用独立目录；pilot重新从原始基座初始化，不继承smoke更新。仅加载params与恢复完整optimizer/step不是同一操作。冻结参数转BF16，可训练参数和Adam状态不应概括为全部BF16；显存峰值必须实际测量。")
    add("本次保持原生Orbax CPU数据恢复与native merge/init_train_state，在NumPy读取前按每leaf RestoreArgs.dtype直接恢复目标精度：冻结参数BF16、计划可训练参数FP32。使用ocp.Checkpointer(ocp.PyTreeCheckpointHandler(restore_concurrent_gb=6,type_handler_registry=private_registry))；private SequentialNumpyHandler通过锁和单leaf调用委托native NumpyHandler.deserialize。全部CPU leaf恢复后进入native.maintenance.before/after，再返回merge/GPU init；eval同样在device_put之前维护。数据不走自定义TensorStore reader或GPU优先/DUS；完整目标dtype host树仍可能保留。")
    add("**6GB restore_concurrent_gb声明本身不约束固定版本NumPy handler**，因为其未使用byte_limiter；真正串行来自锁和单leaf native调用，不能把6GB称为进程/Windows提交峰值上限。共享TensorStore context的16MiB cache/coalescing和I/O/copy各1也不是整进程上限。新reports/bounded_native_weights_cpu.json证明tiny CPU值/shape/混合目标dtype及单leaf并发；bounded_inference_cpu.json只证明AST/CPUstub接口，不是实际模型推理。旧原生/逐leaf/chunk报告均是历史，不能认证新实现。CheckpointManager保存接口另行保留。")
    cache_reuse = ev.read('manifests/cache_reuse.json', {})
    add(f"已完成Arrow缓存来源[cache_reuse.json](manifests/cache_reuse.json)：状态`{cache_reuse.get('status', '未确认')}`，来源`{cache_reuse.get('source_root', 'UNKNOWN')}`，目标HF_DATASETS_CACHE=`{cache_reuse.get('destination_cache', 'UNKNOWN')}`。独立复制17个完成Arrow/metadata文件共7603798421B，train8011行（12文件/5659005114B）和validation2754行（5文件/1944793307B），排除lock/incomplete；validation数据缓存不等于本轮模型评分。预检核metadata/大小/目录身份，不宣称重哈希全payload或更改样本；实际缓存命中以新日志为准。")
    phase_evidence = ev.read("reports/phase_memory_cpu.json", {})
    add(f"分阶段采样CPU证据状态`{phase_evidence.get('status', '未执行')}`。正式采样写入logs/phase_memory.jsonl，只用stdlib读取/proc RSS/PSS及已有Windows提交缓存；虚拟机私有计数是否实际记录，以缓存字段为准，不凭计划推断。采样器不启动额外GPU模型或修改OS设置。各内存域分别展示；样本max/min是观测值，不是理论峰值保证。尚无本轮模型阶段记录时，不把CPU tiny process的RSS/PSS当成π0.5加载/训练峰值。")
    native_identity = ev.read("reports/native_protocol_identity.json", {})
    add(f"本轮数值协议源码快查状态`{native_identity.get('status', '未执行')}`；范围`{native_identity.get('scope', 'UNKNOWN')}`。它检查未更改文件和指定数值函数AST，不能认证明确更改的load_model或整套GPU模型行为。")
    memory_runs = []
    for path in sorted((root / 'reports').glob('phase_memory_*.json')):
        if path.name == 'phase_memory_cpu.json' or path.name.startswith(('phase_memory_current_', 'phase_memory_phase_')):
            continue
        value = ev.read(str(path.relative_to(root)), {})
        if value.get('schema') == 'kinova-phase-memory-v1' and 'sampled_maxima_bytes' in value:
            memory_runs.append((str(path.relative_to(root)), value))
    if memory_runs:
        add("### 本轮进程内存采样汇总（各域独立，不相加）")
        add(table(["label / PID", "采样数 / 降级数", "Linux进程RSS最大GiB", "Linux进程PSS最大GiB", "Windows提交最大GiB", "Windows剩余提交最小GiB", "证据"], [
            (f"{v.get('label')} / {v.get('pid')}", f"{v.get('samples')}/{v.get('degraded_samples')}",
             gib(v.get('sampled_maxima_bytes', {}).get('linux.process.VmRSS_bytes')),
             gib(v.get('sampled_maxima_bytes', {}).get('linux.smaps_rollup.Pss_bytes')),
             gib(v.get('sampled_maxima_bytes', {}).get('windows.committed_bytes')),
             gib(v.get('sampled_minima_bytes', {}).get('windows.commit_free_bytes')), path) for path, v in memory_runs]))
        add("阶段边界及每秒样本见logs/phase_memory.jsonl；此表为进程已记录样本汇总，不将CPU测试、未观测字段或其他进程占用冒充本模型峰值。")
    else:
        add("本轮模型进程内存汇总尚无证据；加载/初始化/训练各阶段RSS、PSS及Windows提交峰值均未实测。")
    add(f"参数清单来自CPU abstract eval_shape：总数{cell(layout.get('total_parameters'))}，可训练{cell(layout.get('trainable_parameters'))}，共{cell(layout.get('trainable_tensors'))}个张量；分组`{layout.get('group_counts', {})}`。它证明形状与冻结计划，尚不等同完整权重已经加载、存在真实梯度或参数已经更新。")
    parameters = layout.get("parameters", [])
    if parameters:
        add("### 计划可训练张量完整名称")
        add(table(["参数路径", "形状", "数量", "组"], [(p["path"], p["shape"], p["count"], p.get("optimizer_group")) for p in parameters if p.get("trainable")]))
        add("其余参数冻结，包括视觉编码器和两专家非LoRA主体。完整逐项冻结/可训练名称、形状和数量见[parameter_layout.json](reports/parameter_layout.json)。")
    add("## 损失mask及CPU工程验证")
    if continuation_manifest:
        add("以下通过状态主要复用旧run CPU证据，保留其原文件SHA和时间；本次快速复核不重执行这些测试。新run如重新执行某项，必须以新日志另行记录，不能仅因复制文件就称为新执行。")
    add("保持flow matching：先将无效GT维度/时间规范为零，target=noise-canonical_actions；仅前7维且真实episode有效时间位置参与平方误差，分子除以有效元素总数。不同有效长度microbatch先累积分子梯度、最后除总有效元素，不能平均各自均值。全无效计数0，训练器跳过或拒绝更新；不产生除零。原生全维噪声和内部attention保留，不能将此描述成普通动作回归。")
    add(table(["检查", "结果", "证据", "范围"], [(label, check_status(value), path or "不存在", (value or {}).get("scope", "CPU数值/格式检查；不构成完整π0.5训练")) for label, path, value in cpu]))
    add("**CPU小张量、标量模型、微型原生LoRA的梯度/参数更新，不计入真实π0.5 optimizer steps。预测比较器的合成数组测试也不等于实际checkpoint恢复。** native loader确实检查真实数据首中末321位置并保留mask/来源，但没有据此宣称完整模型训练通过。")
    add("继承的evaluation_resource_gate_cpu_test.json与旧shared_allocator_probe等属于旧策略历史，不作为当前独占门禁或完整GPU训练通过的证明。当前独占策略CPU检查、实时门禁和完整模型运行分别提供证据。")

    add("## 真实训练、显存和恢复状态")
    add("本节仅列本新run实测：nvidia-smi为整卡采样，JAX allocator为本进程分配器统计，二者不能混称；旧共卡耗时或旧scalar探测不充当本次独占训练峰值。未测项保持UNKNOWN。")
    add(table(["实验", "累计真实optimizer step", "目标", "训练/恢复阶段通过", "checkpoint"], [(s["name"], s["optimizer_step"], s["target"], s["trained"], s["checkpoint"] or "未保存/未确认") for s in (smoke, pilot)]))
    for s in (smoke, pilot):
        evidence = s["evidence"]
        add(f"`{s['name']}`：训练证据状态`{evidence.get('status', s['status'].get('status', '未执行'))}`；报告的更新数{ s['optimizer_step'] }。checkpoint目录当前存在：{cell(s['checkpoint_exists'])}。")
        measured = [("单步平均耗时秒", evidence.get("mean_optimizer_seconds")),
                    ("暖机后平均耗时秒", evidence.get("mean_warm_optimizer_seconds")),
                    ("首步整卡采样峰值MiB", evidence.get("first_update_sampled_whole_card_peak_mib")),
                    ("训练期整卡采样峰值MiB", evidence.get("sampled_whole_card_peak_mib", s["status"].get("sampled_whole_card_peak_mib"))),
                    ("冻结参数未改变", evidence.get("frozen_parameters_unchanged")),
                    ("改变的LoRA张量", evidence.get("changed_lora_tensors")),
                    ("改变的轻量头张量", evidence.get("changed_head_tensors"))]
        add(table(["实测项", "结果"], measured))
        if evidence.get("jax_allocator_after_first_update") or evidence.get("jax_allocator_final"):
            add("JAX allocator测量与整卡nvidia-smi采样不同；整卡0.5秒采样可能漏掉短峰值。allocator原始证据：\n\n```json\n" + pretty({k: evidence.get(k) for k in ("jax_allocator_after_first_update", "jax_allocator_final")}) + "\n```")
        if evidence.get("error"):
            add("已记录训练异常：`" + str(evidence["error"]) + "`。保留日志/有效checkpoint，不把失败写成通过。")
    add(table(["完整模型验收项", "状态", "证据"], [(label, check_status(value), "reports/" + name) for label, name, value in real]))
    all_real_pass = all(value and value.get("status") == "passed" for _, _, value in real)
    add(f"提交32恢复验收：{'通过' if smoke['trained'] and all_real_pass else '尚未全部通过/尚未执行'}。参考来自首次checkpoint恢复；原训练最后RAM快照缺失，不能补证原RAM到保存等价。真实32更新来自源日志与提交marker，本轮恢复更新0。")

    add("## 固定离线验证与checkpoint选择")
    add(f"固定计划文件[validation_plan.json](configs/validation_plan.json)：validation {plan.get('episode_count', 'UNKNOWN')}条，horizon{plan.get('action_horizon', 'UNKNOWN')}；每episode first=0、mid=floor((length-1)/2)、late=max(0,length-10)，固定seed `{plan.get('seeds', 'UNKNOWN')}`、去噪{plan.get('denoise_steps', 'UNKNOWN')}步。E0/E1使用相同腕部、状态、train统计、有效mask及输入/初始噪声，不能从多次预测挑最接近答案一次。")
    add("预先固定选择指标为首10个有效时间步的归一化组均方误差：xyz三维均值、欧拉三维均值、夹爪一维误差三组各1/3；先块/seed等权，再episode等权。只允许500/1000/1500/2000步E1参加，分数相同取较早步；E0仅比较，不作为E1候选。米、弧度和夹爪原单位误差分别报告，未经归一化不混加。抓取/放置阶段真值UNKNOWN。")
    def valid_current_evaluation(label, result):
        if not result or result.get('evaluation_valid') is not True or result.get('status') != 'complete':
            return False
        if label != 'E0_wrist_frozen':
            return True
        baseline = result.get('config_metadata', {})
        return (baseline.get('baseline_kind') == 'E0_same_graph_zero_update'
                and baseline.get('graph_kind') == 'zero_lora_same_as_training'
                and baseline.get('optimizer_step') == 0
                and baseline.get('parameter_hashes_recorded') is True
                and len(baseline.get('parameter_hashes', {}).get('base', {})) == 51
                and len(baseline.get('parameter_hashes', {}).get('adapter', {})) == 20)
    eval_table = []
    for label, step, path, result in evals:
        metric = (result or {}).get("dataset_metrics") or {}
        valid = valid_current_evaluation(label, result)
        eval_state = ("复用旧run有效E0（本轮未重跑）" if label == "E0_wrist_frozen" and inherited_e0 else "本轮有效完成") if valid else ("未执行" if result is None else "未完成/无效")
        eval_table.append((label, step, eval_state,
                           metric.get("flow_matching_loss") if valid else None, metric.get("selection_score") if valid else None, path))
        if result is not None:
            add(f"`{label}/step{step}`记录：errors={result.get('errors', 'UNKNOWN')}，完成block/seed={result.get('records_completed', 'UNKNOWN')}/{result.get('expected_block_seed_count', 'UNKNOWN')}；缺失={result.get('missing_records', 'UNKNOWN')}。")
    add(table(["模型", "step", "真实评估状态", "有效FM验证loss", "选择分数", "证据"], eval_table))
    for label, step, path, result in evals:
        if valid_current_evaluation(label, result):
            metric = result.get("dataset_metrics") or {}
            add(f"### {label} step{step} {'旧run实测复用误差' if label == 'E0_wrist_frozen' and inherited_e0 else '本轮实测误差'}")
            add("```json\n" + pretty({"dataset_metrics": metric, "point_horizon_coverage": result.get("point_horizon_coverage"),
                                      "task_phase": result.get("task_phase", "UNKNOWN"), "summary": path}) + "\n```")
    if selection:
        add("checkpoint选择实际结果（文件存在不代替有效评估检查）：\n\n```json\n" + pretty({k: selection.get(k) for k in ("status", "selected", "E0", "E1_improved_over_E0", "all_declared_steps_evaluated", "test_split_used")}) + "\n```")
    else:
        add("尚无configs/checkpoint_selection.json，没有已选checkpoint，也没有E0/E1改善结论。")
    model_test_used = any((v or {}).get("test_split_used") is True for _, _, _, v in evals) or selection.get("test_split_used") is True
    add("测试集使用：" + ("发现模型评估/选择证据声明使用test，须逐项复核用途。" if model_test_used else "仅做数据完整性/边界检查和准备复核材料；未发现模型评分、调参或checkpoint选择使用test的证据。本轮保留test，不将格式检查称为留出模型评估。"))

    add("## 交付与继续运行")
    add("- 数据与复核：[data_report_zh.md](reports/data_report_zh.md)、[review/index.html](review/index.html)、[manual_review.csv](review/manual_review.csv)。\n- 模型/参数：[parameter_layout.json](reports/parameter_layout.json)、`scripts/kinova_config.py`、`scripts/kinova_model.py`。\n- 训练入口：`scripts/train_kinova.py --help`；实际GPU工作受resource_guard保护。\n- 验证入口：`scripts/evaluate_offline.py --help`；固定计划在configs/。\n- 本报告：`scripts/build_report.py --root " + str(root) + "`只读取已有证据并刷新文档，不触发训练。\n- 下一步：[NEXT_STEPS_zh.md](NEXT_STEPS_zh.md)。")
    add("当前数据允许在工程与资源门禁通过后开展明确标记的D0腕部探索性离线训练；人工复核/同步未完成本身不被偷换成要求再次授权。现有资源阻塞不会通过终止他人任务解决。正式双视角结论和自主真机能力仍无证据。")
    add("## 报告证据边界")
    add("本报告由保存的JSON证据动态生成，不加载模型、不查询GPU、不执行optimizer。缺失文件只能记为未执行/未确认；不能由准备配置推断实际完成。所读取文件及SHA、未找到路径和读取错误见[report_generation.json](reports/report_generation.json)。")
    if ev.errors:
        add("读取错误：\n\n```json\n" + pretty(ev.errors) + "\n```")
    report = "\n".join(lines)

    next_lines = ["# 下一步", "", f"更新时间：{updated}。", "",
        "1. **从完整pilot500三段续训**：复用已验证E0和smoke32恢复；完整native500与首500条日志已独立复制。三个进程resume到1000/1500/2000，重放start_step×4个microbatch后续跑，每段保存退出，总LR预算仍2000。最后原四checkpoint评估/选择不变。旧未保存501–1000保留原处，不算当前恢复进度。", "",
        "2. **完成全部107条人工示范复核**：使用review/index.html和manual_review.csv逐条连续查看，接近/夹取/抬起/运输/放置/释放/结束均保留审阅者、时间和证据。当前全部manual_review_pending；无法判断填UNKNOWN。先查看原始保留尾部，再决定独立新版导出或局部重采，不覆盖D0。", "",
        "3. **完成独立同步校准设计并在新记录验证**：外部与腕部同步均UNVERIFIED。使用共同可见、唯一编码的时间事件，同一网络/解码/缓存配置，多次独立启动并覆盖开头/中间/结束和漂移；记录曝光或设备时钟、PTS、接收、解码、反馈请求起止及不确定性。新结果不能自动证明历史数据同步。", "",
        "4. **按固定validation规则比较E0与E1**：同一20episode、固定起点/噪声/mask/统计，分别报告有效FM loss和1/5/10/20/50步实际存在的物理分量误差，按预定首10步归一化三组等权误差选择E1；没有改善就如实记录。test继续保留，不能用离线误差声称真机成功率。", "",
        "5. **真机前补齐UNKNOWN控制接口**：验证参考系、TCP偏置、姿态表示/顺序、控制模式/周期和夹爪命令映射。反馈重建动作不是已验证下发命令；本轮不操作机器人。同步方案、数据版本和预算固定后再考虑腕部/双视角对照。", ""]
    if continuation_manifest:
        next_lines.insert(4, "本轮明确复用205332有效E0和真实提交32，不重复训练。恢复验收前不称完整smoke通过；两次恢复一致也不能补证缺失的原RAM保存快照。\n")
    draft = root / "reports/data_sync_next_steps_zh.md"
    if draft.exists():
        next_lines.extend(["## 同步与示范复核的详细实施建议", "", draft.read_text(), ""])
    atomic_text(root / "REPORT_zh.md", report)
    atomic_text(root / "NEXT_STEPS_zh.md", "\n".join(next_lines))
    record = {"generated_at": now.isoformat(), "root": str(root), "script_sha256": sha(Path(__file__)),
              "actual_optimizer_steps": {"smoke": smoke["optimizer_step"], "pilot": pilot["optimizer_step"]},
              "smoke_updates_executed_this_run": 0, "source_smoke_updates": 32,
              "recovery_reference_is_original_live_RAM": False,
              "cpu_toy_steps_counted_as_pi05": False, "source_json": ev.reads,
              "missing_json": ev.absent, "read_errors": ev.errors,
              "model_or_gpu_loaded_by_report": False, "written": ["REPORT_zh.md", "NEXT_STEPS_zh.md"]}
    atomic_text(root / "reports/report_generation.json", pretty(record) + "\n")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(os.environ.get("KINOVA_EXPERIMENT_ROOT", project_path('archive', 'run_20261103_094715_segmented'))))
    p.add_argument("--audit-action-continuity", action="store_true", help="First reread only actual source_time columns with CPU numpy/pyarrow, then generate reports")
    args = p.parse_args()
    if args.audit_action_continuity:
        audit_action_continuity(args.root.resolve())
    result = make_report(args.root.resolve())
    print(pretty({"written": result["written"], "actual_optimizer_steps": result["actual_optimizer_steps"],
                  "missing_evidence_files": len(result["missing_json"]), "read_errors": result["read_errors"]}))


if __name__ == "__main__":
    main()
