"""Serial, resumable four-way π0.5 LIBERO experiment supervisor.

Only the explicit main entry launches work. Importing this module or using
--dry-run never launches training, simulation, or model serving. The supervisor
uses the standard library so its resident process does not initialize JAX/CUDA.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import ast
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid


GROUPS = ("prototype", "uniform", "noreplay", "plain")
STAGES = "ABCD"
PROTOCOL = "pi05-libero-fourway-supervisor-v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def task_specs(script_dir: Path) -> dict:
    # Read the literal task registry without importing model/config dependencies.
    tree = ast.parse((script_dir / "experiment_config.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in node.targets):
            tasks = ast.literal_eval(node.value)
            if set(tasks) != set(STAGES):
                raise ValueError("Expected exactly four task specifications A..D")
            return tasks
    raise ValueError("experiment_config.py has no literal TASKS registry")


def log_tail(path: Path, limit: int = 12000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - limit))
        return file.read().decode("utf-8", errors="replace").replace("\r", "\n")


def terminate_group(process: subprocess.Popen | None, grace: float = 20.0) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def alive_owned_job(pid, root: Path) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return str(root / "scripts") in command and any(
        name in command for name in ("train_stage.py", "build_memory.py", "serve_libero.py", "evaluate_libero.py")
    )


def check_policy_port(port: int) -> None:
    # Match the asyncio server's POSIX reuse behavior: a recently closed server
    # can leave TIME_WAIT sockets without any listener owning this port.
    # SO_REUSEADDR allows that case; an active listener still rejects bind.
    with socket.socket() as port_check:
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            port_check.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"Policy port {port} is unavailable; refusing to kill/reuse an unknown server"
            ) from error


class Supervisor:
    def __init__(self, args, root: Path, scripts: Path, tasks: dict):
        self.args, self.root, self.scripts, self.tasks = args, root, scripts, tasks
        self.python = root / ".venv/bin/python"
        self.eval_python = root / ".venv_eval/bin/python"
        self.started = time.monotonic()
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        old = read_json(root / "status.json") if (root / "status.json").exists() else {}
        if old.get("protocol") == PROTOCOL:
            for pid in [old.get("pid"), old.get("server_pid")]:
                if alive_owned_job(pid, root):
                    raise RuntimeError(f"Previous experiment job PID {pid} is still alive; refusing a concurrent duplicate")
            if not args.resume:
                raise FileExistsError("Experiment status already exists; explicitly use --resume")
        self.previous_elapsed = float(old.get("elapsed_seconds", 0.0)) if old.get("protocol") == PROTOCOL else 0.0
        self.state = {"protocol": PROTOCOL, "status": "running", "run_id": self.run_id,
                      "orchestrator_pid": os.getpid(), "pid": None, "server_pid": None,
                      "started_at": utc_now(), "updated_at": utc_now(), "current_job": None,
                      "elapsed_seconds": self.previous_elapsed, "steps_per_stage": args.steps,
                      "batch_size": args.batch_size, "groups": list(GROUPS), "training_seed": 42,
                      "evaluation_seed": 7, "policy_seed": 0, "evaluation_trials": 50,
                      "completed_jobs": old.get("completed_jobs", []) if args.resume else [],
                      "events": old.get("events", [])[-100:] if args.resume else []}
        self.active = None
        self.server = None
        self.attempt_logs = root / "logs" / self.run_id
        self.attempt_logs.mkdir(parents=True, exist_ok=False)
        self.train_env = self.training_environment()
        self.update()

    def training_environment(self):
        env = os.environ.copy()
        deploy = Path(env.get("PI05_DEPLOY_ROOT", project_path('deploy', '')))
        env.update(PI05_LIBERO_ROOT=str(self.root), HF_LEROBOT_HOME=str(self.root / "datasets"),
                   HF_DATASETS_CACHE=str(self.root / "cache/datasets"), WANDB_MODE="disabled",
                   HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env.setdefault("OPENPI_DATA_HOME", str(deploy / "cache/openpi"))
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("OPENBLAS_NUM_THREADS", "4")
        paths = [self.scripts, self.root / "source/pi05", self.root / "source/pi05/src",
                 self.root / "source/pi05/packages/openpi-client/src", deploy / "source/lerobot"]
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths)
        env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")
        return env

    def update(self, **values):
        self.state.update(values)
        self.state["updated_at"] = utc_now()
        self.state["elapsed_seconds"] = self.previous_elapsed + time.monotonic() - self.started
        write_atomic(self.root / "status.json", self.state)

    def event(self, text: str, **values):
        self.state["events"].append({"at": utc_now(), "message": text, **values})
        self.state["events"] = self.state["events"][-200:]
        self.update()
        print(f"{utc_now()} {text}", flush=True)

    def latest_params(self, group: str, stage: str) -> Path:
        # Same completion rule as experiment_config.latest_params, without JAX import.
        folder = self.root / "checkpoints" / f"pi05_libero_{group}_{stage}" / "seed42"
        if folder.resolve() != folder:
            raise ValueError(f"Refusing a symlinked cross-group checkpoint root: {folder}")
        options = sorted((int(p.name), p / "params") for p in folder.iterdir()
                         if p.is_dir() and p.name.isdigit() and (p / "params/_METADATA").is_file()) if folder.is_dir() else []
        if not options:
            raise FileNotFoundError(f"No completed checkpoint under {folder}")
        params = options[-1][1]
        if params.resolve() != params:
            raise ValueError(f"Refusing symlinked checkpoint parameters: {params}")
        return params

    def _heartbeat(self, path: Path, start: float):
        tail = log_tail(path)
        job = dict(self.state["current_job"] or {})
        matches = re.findall(r"\b(?:Step\s+|step\s*[=:]\s*)(\d+)", tail, flags=re.IGNORECASE)
        if matches:
            job["last_logged_step"] = int(matches[-1])
        job.update(elapsed_seconds=time.monotonic() - start, log_tail=tail[-4000:])
        if job.get("kind") == "evaluate" and job.get("output_dir"):
            summary_path = Path(job["output_dir"]) / "evaluation.json"
            if summary_path.is_file():
                try:
                    summary = read_json(summary_path)
                    job.update(episodes_completed=summary.get("episodes_completed", 0),
                               trials=summary.get("trials", 50), successes=summary.get("successes", 0),
                               errors=summary.get("errors", 0))
                except json.JSONDecodeError:
                    pass  # A concurrently written summary can be incomplete; the final read must validate.
        self.update(current_job=job)
        if job.get("kind") == "train" and re.search(
            r"\b(?:loss(?:_\w+)?|grad_norm)\s*[=:]\s*(?:nan|[+-]?inf)(?=[,\s}]|$)", tail, re.IGNORECASE
        ):
            raise RuntimeError("Training reported nonfinite loss/gradient; preserving logs and stopping")

    def run_job(self, name: str, command: list[str], timeout: float, *, kind: str, **details):
        path = self.attempt_logs / (name + ".log")
        started = time.monotonic()
        job = {"name": name, "kind": kind, "command": command, "log": str(path),
               "started_at": utc_now(), "timeout_seconds": timeout, **details}
        self.update(current_job=job, pid=None)
        self.event(f"Starting {name}")
        with path.open("w") as log:
            process = subprocess.Popen(command, cwd=self.root, env=self.train_env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.active = process
            self.update(pid=process.pid)
            try:
                while process.poll() is None:
                    self._heartbeat(path, started)
                    if self.server is not None and self.server.poll() is not None:
                        raise RuntimeError(f"Model server exited during {name}, code {self.server.returncode}")
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(f"{name} exceeded {timeout:.0f}s; log: {path}")
                    time.sleep(self.args.poll_seconds)
                self._heartbeat(path, started)
                if process.returncode != 0:
                    raise RuntimeError(f"{name} failed with exit code {process.returncode}; log: {path}\n{log_tail(path)[-3500:]}")
            except BaseException:
                terminate_group(process)
                raise
            finally:
                self.active = None
        completed = {**job, "finished_at": utc_now(), "seconds": time.monotonic() - started, "exit_code": 0}
        self.state["completed_jobs"].append(completed)
        self.update(pid=None, current_job=None)
        self.event(f"Completed {name}")

    def train(self, group: str, stage: str) -> Path:
        marker = self.root / "reports" / f"pi05_libero_{group}_{stage}_complete.json"
        if self.args.resume and marker.is_file():
            saved = read_json(marker)
            if (saved.get("group"), saved.get("stage"), saved.get("steps"), saved.get("batch_size"), saved.get("smoke")) != (
                group, stage, self.args.steps, self.args.batch_size, False
            ):
                raise ValueError(f"Existing stage marker has incompatible protocol: {marker}")
            params = self.latest_params(group, stage)
            self.verify_training_source(group, stage)
            self.event(f"Resume: training already complete for {group}/{stage}", checkpoint=str(params))
            return params
        command = [str(self.python), str(self.scripts / "train_stage.py"), "--group", group, "--stage", stage,
                   "--steps", str(self.args.steps), "--batch-size", str(self.args.batch_size)]
        if self.args.resume:
            command.append("--resume")
        self.run_job(f"train_{group}_{stage}", command, self.args.training_timeout, kind="train", group=group, stage=stage)
        if not marker.is_file():
            raise RuntimeError(f"Training exited without its explicit completion marker: {marker}")
        saved = read_json(marker)
        if saved.get("steps") != self.args.steps or saved.get("smoke") or saved.get("group") != group or saved.get("stage") != stage:
            raise ValueError(f"Invalid training completion marker: {marker}")
        params = self.latest_params(group, stage)
        self.verify_training_source(group, stage)
        return params

    def verify_training_source(self, group: str, stage: str):
        configuration = read_json(self.root / "reports/configurations" / f"pi05_libero_{group}_{stage}.json")
        source = Path(configuration["weight_loader"]["params_path"]).resolve()
        expected = (Path(project_path('base_params', ''))
                    if stage == "A" else self.latest_params(group, STAGES[STAGES.index(stage) - 1]))
        if source != expected:
            raise ValueError(f"Stage {group}/{stage} used a different group's or stage's checkpoint: {source}")

    def _evaluation_valid(self, result: dict, group: str, trained_stage: str, task: str, params: Path) -> bool:
        suite, task_id, language = self.tasks[task]
        metadata = result.get("server_metadata", {})
        checks = [result.get("status") == "complete", result.get("evaluation_valid") is True,
                  result.get("errors") == 0, result.get("episodes_completed") == 50, result.get("trials") == 50,
                  result.get("seed") == 7, result.get("num_steps_wait") == 10, result.get("replan_steps") == 5,
                  result.get("suite") == suite, result.get("task_id") == task_id, result.get("language") == language,
                  metadata.get("config_name") == f"pi05_libero_{group}_{trained_stage}",
                  metadata.get("checkpoint") == str(params.parent), metadata.get("task_bank_id") == STAGES.index(task),
                  metadata.get("expected_language") == language,
                  metadata.get("norm_stats_file") == str(self.root / "assets/libero_fourway" / f"task_{task}/norm_stats.json")]
        successes = result.get("successes")
        if not isinstance(successes, int) or not 0 <= successes <= 50:
            return False
        return all(checks) and abs(float(result.get("success_rate", -1)) - successes / 50) < 1e-12

    def evaluate(self, group: str, trained_stage: str, task: str, params: Path):
        folder = self.root / "evaluations" / group / f"stage_{trained_stage}" / f"task_{task}"
        pointer = folder / "complete.json"
        if self.args.resume and pointer.is_file():
            existing = read_json(pointer)
            summary_path = Path(existing["evaluation_json"])
            if summary_path.is_file() and self._evaluation_valid(read_json(summary_path), group, trained_stage, task, params):
                self.event(f"Resume: valid evaluation already complete for {group}/{trained_stage}/{task}")
                return
            raise ValueError(f"Existing success marker is invalid or belongs to another checkpoint: {pointer}")
        folder.mkdir(parents=True, exist_ok=True)
        attempt = folder / ("attempt_" + self.run_id)
        attempt.mkdir(exist_ok=False)
        metadata_path = attempt / "server_metadata.json"
        server_log = attempt / "server.log"
        name = f"eval_{group}_{trained_stage}_{task}"
        bank = STAGES.index(task)
        check_policy_port(self.args.port)
        command = [str(self.python), str(self.scripts / "serve_libero.py"),
            "--config-name", f"pi05_libero_{group}_{trained_stage}",
            "--config-factory", "experiment_config:config_from_name", "--checkpoint", str(params.parent),
            "--norm-stats-dir", str(self.root / "assets/libero_fourway" / f"task_{task}"),
            "--bank-id", str(bank), "--expected-language", self.tasks[task][2],
            "--metadata-out", str(metadata_path), "--port", str(self.args.port), "--seed", "0", "--denoise-steps", "10"]
        started = time.monotonic()
        log = server_log.open("w")
        try:
            self.server = subprocess.Popen(command, cwd=self.root, env=self.train_env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.update(server_pid=self.server.pid, current_job={"name": name + "_server_start", "kind": "server_start",
                        "group": group, "stage": trained_stage, "task": task, "log": str(server_log), "started_at": utc_now()})
            while True:
                if self.server.poll() is not None:
                    raise RuntimeError(f"Policy server exited before readiness: {log_tail(server_log)[-3500:]}")
                if time.monotonic() - started > self.args.server_ready_timeout:
                    raise TimeoutError(f"Model server did not become ready within {self.args.server_ready_timeout}s")
                self._heartbeat(server_log, started)
                metadata = None
                if metadata_path.is_file():
                    try:
                        metadata = read_json(metadata_path)
                    except json.JSONDecodeError:
                        pass
                if metadata is not None:
                    if (metadata.get("config_name"), metadata.get("checkpoint"), metadata.get("task_bank_id"), metadata.get("expected_language")) != (
                        f"pi05_libero_{group}_{trained_stage}", str(params.parent), bank, self.tasks[task][2]
                    ):
                        raise ValueError(f"Ready server metadata mismatch: {metadata_path}")
                    try:
                        with socket.create_connection(("127.0.0.1", self.args.port), timeout=1.0):
                            break
                    except OSError:
                        pass
                time.sleep(min(self.args.poll_seconds, 1.0))
            evaluation_command = ["/bin/bash", "-c", 'source "$1"; shift; exec "$@"', "libero-eval",
                str(self.root / "eval_env.sh"), str(self.eval_python), str(self.scripts / "evaluate_libero.py"),
                "--task-label", task, "--trials", "50", "--seed", "7", "--num-steps-wait", "10",
                "--replan-steps", "5", "--expected-bank-id", str(bank), "--expected-language", self.tasks[task][2],
                "--policy-port", str(self.args.port), "--output-dir", str(attempt), "--video-mode", "first"]
            self.run_job(name, evaluation_command, self.args.evaluation_timeout, kind="evaluate",
                         group=group, stage=trained_stage, task=task, server_log=str(server_log), output_dir=str(attempt))
            summary_path = attempt / "evaluation.json"
            if not summary_path.is_file() or not self._evaluation_valid(read_json(summary_path), group, trained_stage, task, params):
                raise RuntimeError(f"Evaluation exited but failed its completion/protocol validation: {summary_path}")
            write_atomic(pointer, {"group": group, "trained_stage": trained_stage, "task": task,
                                  "checkpoint": str(params), "evaluation_json": str(summary_path), "completed_at": utc_now()})
            self.results()
        finally:
            terminate_group(self.server)
            self.server = None
            log.close()
            self.update(server_pid=None)

    def memory(self, group: str, stage: str, params: Path):
        output = self.root / "buffers" / group / f"task_{stage}"
        if output.exists():
            valid = False
            if (output / "meta.json").is_file() and (output / "validation.json").is_file() and (output / "manifest.jsonl").is_file():
                meta, validation = read_json(output / "meta.json"), read_json(output / "validation.json")
                expected_method = "multimodal_prototype_v2_libero_rotvec" if group == "prototype" else "uniform_time_matched_per_episode"
                valid = (validation.get("status") == "passed" and validation.get("samples", 0) > 0
                    and meta.get("method") == expected_method and meta.get("cl_task_id") == STAGES.index(stage)
                    and meta.get("repo_id") == f"libero_fourway/task_{stage}" and meta.get("source_checkpoint") == str(params))
            if not self.args.resume:
                raise FileExistsError(f"Memory already exists; use --resume: {output}")
            if valid:
                self.event(f"Resume: valid memory already complete for {group}/{stage}")
                return
            archived = output.with_name(output.name + ".incomplete_" + self.run_id)
            output.rename(archived)
            self.event("Preserved incomplete memory before rebuilding", previous=str(archived))
        command = [str(self.python), str(self.scripts / "build_memory.py"), "--group", group, "--stage", stage,
                   "--checkpoint", str(params), "--output-dir", str(output)]
        if group == "uniform":
            command += ["--reference-buffer", str(self.root / "buffers/prototype" / f"task_{stage}")]
        self.run_job(f"memory_{group}_{stage}", command, self.args.memory_timeout, kind="memory", group=group, stage=stage)
        if not (output / "validation.json").is_file() or read_json(output / "validation.json").get("status") != "passed":
            raise RuntimeError(f"Memory builder exited without valid output: {output}")

    def results(self, complete: bool = False):
        matrices, records, invalid = {}, {}, []
        for group in GROUPS:
            matrix = {stage: {task: None for task in STAGES} for stage in STAGES}
            group_records = {}
            for stage in STAGES:
                for task in STAGES[:STAGES.index(stage) + 1]:
                    marker = self.root / "evaluations" / group / f"stage_{stage}" / f"task_{task}/complete.json"
                    if not marker.is_file():
                        continue
                    pointer = read_json(marker)
                    summary = read_json(Path(pointer["evaluation_json"]))
                    params = Path(pointer["checkpoint"])
                    if not self._evaluation_valid(summary, group, stage, task, params):
                        invalid.append(str(marker))
                        continue
                    matrix[stage][task] = summary["success_rate"]
                    group_records[f"{stage}/{task}"] = {"successes": summary["successes"], "episodes": 50,
                        "success_rate": summary["success_rate"], "evaluation_json": pointer["evaluation_json"],
                        "checkpoint": str(params)}
            final = matrix["D"]
            final_average = sum(final.values()) / 4 if all(v is not None for v in final.values()) else None
            old_average = sum(final[t] for t in "ABC") / 3 if all(final[t] is not None for t in "ABC") else None
            bwt = sum(final[t] - matrix[t][t] for t in "ABC") / 3 if old_average is not None and all(matrix[t][t] is not None for t in "ABC") else None
            matrices[group] = {"success_rate_matrix": matrix, "final_average": final_average,
                               "final_old_task_average": old_average, "backward_transfer": bwt}
            records[group] = group_records
        count = sum(len(v) for v in records.values())
        result = {"protocol": PROTOCOL, "status": "complete" if complete else "running", "updated_at": utc_now(),
                  "groups": matrices, "evaluations": records, "completed_evaluations": count,
                  "expected_evaluations": 40, "invalid_results": invalid, "training_seed": 42,
                  "evaluation_seed": 7, "policy_seed": 0, "trials_per_task": 50, "tasks": self.tasks,
                  "scope": "single training seed; fixed 50 simulator initial states per task, no confidence claim across training seeds"}
        if complete and (count != 40 or invalid):
            raise RuntimeError(f"Cannot complete experiment: {count}/40 valid evaluations; invalid={invalid}")
        write_atomic(self.root / "RESULTS.json", result)
        self.update(completed_evaluations=count, expected_evaluations=40)

    def run(self):
        try:
            for group in GROUPS:
                for stage in STAGES:
                    params = self.train(group, stage)
                    for task in STAGES[:STAGES.index(stage) + 1]:
                        self.evaluate(group, stage, task, params)
                    if group in ("prototype", "uniform") and stage != "D":
                        self.memory(group, stage, params)
                    self.results()
            self.results(complete=True)
            self.update(status="complete", current_job=None, pid=None, server_pid=None, finished_at=utc_now())
            self.event("All four groups, 16 stages, and 40 evaluation sessions completed")
        except BaseException as error:
            failed_job = self.state.get("current_job") or (self.state["completed_jobs"][-1] if self.state["completed_jobs"] else None)
            terminate_group(self.active)
            terminate_group(self.server)
            self.update(status="failure", failed_job=failed_job, pid=None, server_pid=None,
                        finished_at=utc_now(), error=str(error), traceback=traceback.format_exc())
            results_path = self.root / "RESULTS.json"
            if results_path.is_file():
                results = read_json(results_path)
                results.update(status="failure", updated_at=utc_now(), error=str(error), failed_job=failed_job)
                write_atomic(results_path, results)
            raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the fixed job order; launch nothing and write nothing")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--training-timeout", type=float, default=86400.0)
    parser.add_argument("--evaluation-timeout", type=float, default=3600.0)
    parser.add_argument("--server-ready-timeout", type=float, default=300.0)
    parser.add_argument("--memory-timeout", type=float, default=7200.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.steps, args.batch_size, args.poll_seconds, args.training_timeout, args.evaluation_timeout,
           args.server_ready_timeout, args.memory_timeout) <= 0 or not 1 <= args.port <= 65535:
        raise ValueError("Steps, batch size, polling, timeouts and port must be positive and valid")
    root = Path(os.environ.get("PI05_LIBERO_ROOT", project_path('libero_fourway', ''))).expanduser().resolve()
    scripts = root / "scripts"
    if args.dry_run and not scripts.is_dir():
        scripts = Path(__file__).resolve().parent
    tasks = task_specs(scripts)
    plan = [{"group": group, "stage": stage, "steps": args.steps,
             "evaluate_tasks": list(STAGES[:STAGES.index(stage) + 1]),
             "memory": group if group in ("prototype", "uniform") and stage != "D" else None}
            for group in GROUPS for stage in STAGES]
    if args.dry_run:
        print(json.dumps({"root": str(root), "jobs": plan, "evaluations": 40, "tasks": tasks,
                          "training_started": False, "parameter_updates": 0}, ensure_ascii=False, indent=2))
        return
    required = [root / ".venv/bin/python", root / ".venv_eval/bin/python", root / "eval_env.sh",
                *(scripts / name for name in ["train_stage.py", "build_memory.py", "serve_libero.py", "evaluate_libero.py"])]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"Missing runtime files: {missing}")
    with (root / ".run_experiment.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another experiment supervisor holds the run lock") from error
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Supervisor received signal {signum}; terminating its own child jobs")
        signal.signal(signal.SIGTERM, interrupted)
        supervisor = Supervisor(args, root, scripts, tasks)
        write_atomic(root / "run_plan.json", {"protocol": PROTOCOL, "jobs": plan, "tasks": tasks,
                                             "steps_per_stage": args.steps, "batch_size": args.batch_size})
        supervisor.run()


if __name__ == "__main__":
    main()
