"""Read-only progress for the isolated ordinary-LoRA + fixed-prototype run.

Counts come from validated completion markers, never from the planned totals.
No imports initialize JAX, model serving, simulation, or training.
"""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from run_experiment import (EXPECTED_EVALUATIONS, EXPECTED_TRAINING_STAGES, GROUPS,
                            PROTOCOL, STAGES, Supervisor, read_json, task_specs)

DEFAULT_ROOT = project_path('libero_plain_prototype', '')


def collect_progress(root):
    root = Path(root).expanduser().resolve()
    output = {"protocol": PROTOCOL, "root": str(root), "status": "not_started", "job": None,
              "completed_training_stages": 0, "total_training_stages": EXPECTED_TRAINING_STAGES,
              "completed_evaluations": 0, "total_evaluations": EXPECTED_EVALUATIONS,
              "expected_evaluations": EXPECTED_EVALUATIONS, "groups": list(GROUPS),
              "latest_logged_training_step": None, "errors": []}
    status_path = root / "status.json"
    if not status_path.is_file():
        return output
    state = read_json(status_path)
    if state.get("protocol") != PROTOCOL:
        output.update(status="unrecognized_protocol", error="Refusing to report another experiment as this run")
        return output
    job = state.get("current_job") or {}
    output.update(status=state.get("status", "UNKNOWN"), started_at=state.get("started_at"),
                  updated_at=state.get("updated_at"), job=job.get("name"), kind=job.get("kind"),
                  group=job.get("group"), stage=job.get("stage"), task=job.get("task"),
                  steps_per_stage=state.get("steps_per_stage"), elapsed_seconds=job.get("elapsed_seconds"),
                  total_elapsed_seconds=state.get("elapsed_seconds"), error=state.get("error"), log=job.get("log"),
                  pid=state.get("pid"), server_pid=state.get("server_pid"))
    if job.get("kind") == "train":
        steps = []
        path = Path(job["log"]) if job.get("log") else None
        if path and path.is_file():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16000))
                tail = stream.read().decode(errors="replace")
            steps.extend(int(value) for value in re.findall(r"\bStep\s+(\d+)", tail))
            steps.extend(int(float(value)*(1000 if suffix else 1))
                         for value, suffix in re.findall(r"Progress on:\s*([\d.]+)(k?)it/", tail))
        output["latest_logged_training_step"] = max(steps) if steps else job.get("last_logged_step")
    # Construct only a read-only validation view: __init__ writes status/logs and
    # must never run from a progress query.
    checker = object.__new__(Supervisor)
    checker.root = root
    checker.args = SimpleNamespace(steps=state.get("steps_per_stage", 10000), batch_size=state.get("batch_size", 2))
    try:
        checker.tasks = task_specs(root / "scripts")
        training, invalid_training = checker.training_records()
        output["completed_training_stages"] = len(training)
        output["errors"].extend(invalid_training)
        for group in GROUPS:
            for stage in STAGES:
                for task in STAGES[:STAGES.index(stage)+1]:
                    marker = root / "evaluations" / group / f"stage_{stage}" / f"task_{task}" / "complete.json"
                    if not marker.is_file():
                        continue
                    try:
                        pointer = read_json(marker)
                        result = read_json(Path(pointer["evaluation_json"]))
                        if not checker._evaluation_valid(result, group, stage, task, Path(pointer["checkpoint"])):
                            raise ValueError("Evaluation protocol/checkpoint validation failed")
                        output["completed_evaluations"] += 1
                    except (OSError, ValueError, KeyError) as error:
                        output["errors"].append({"marker": str(marker), "error": str(error)})
    except (OSError, ValueError, KeyError) as error:
        output["errors"].append({"error": str(error)})
    output["completion_verified"] = (output["completed_training_stages"] == EXPECTED_TRAINING_STAGES
                                       and output["completed_evaluations"] == EXPECTED_EVALUATIONS
                                       and not output["errors"])
    if output["status"] == "complete" and not output["completion_verified"]:
        output["status"] = "invalid_completion_evidence"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PI05_LIBERO_ROOT", DEFAULT_ROOT)))
    args = parser.parse_args()
    print(json.dumps(collect_progress(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
