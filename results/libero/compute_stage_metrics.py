"""Compute documented stage-end metrics from completed LIBERO evaluation counts.

Read-only inputs; no server connection, model import, training, or evaluation.
The stage-end definitions are from arXiv:2602.10503v1, Sec. V-C.1.
They differ from the within-task learning-curve FWT in original LIBERO.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASKS = ("A", "B", "C", "D")
SOURCES = (
    ROOT / "fourway.json",
    ROOT / "plain_prototype.json",
)
CONFIGS = (
    (0, "plain", "普通 LoRA，无回放"),
    (0, "noreplay", "CL-LoRA，无回放"),
    (0, "prototype", "CL-LoRA＋原型回放"),
    (0, "uniform", "CL-LoRA＋均匀回放（ER）"),
    (1, "plain_prototype", "普通 LoRA＋原型回放"),
)


def record(value: Fraction) -> dict:
    """Percentage-scale values; NBT is a difference in percentage points."""
    return {
        "value": float(value * 100),
        "display": f"{float(value * 100):.2f}",
        "exact_percentage_fraction": str(value * 100),
    }


def main():
    payloads = [p.read_bytes() for p in SOURCES]
    documents = [json.loads(payload) for payload in payloads]
    assert all(d["status"] == "complete" for d in documents)
    groups = []
    for source_index, group, title in CONFIGS:
        source = documents[source_index]
        reported = source["groups"][group]["success_rate_matrix"]
        matrix = [[None] * 4 for _ in TASKS]
        counts = []
        for i, stage in enumerate(TASKS):
            for j, task in enumerate(TASKS):
                if j > i:
                    assert reported[stage][task] is None
                    continue
                result = source["evaluations"][group][f"{stage}/{task}"]
                assert type(result["successes"]) is int
                assert result["episodes"] == 50
                assert 0 <= result["successes"] <= result["episodes"]
                rate = Fraction(result["successes"], result["episodes"])
                assert math.isclose(float(rate), reported[stage][task], abs_tol=1e-12)
                assert math.isclose(float(rate), result["success_rate"], abs_tol=1e-12)
                matrix[i][j] = rate
                counts.append({"stage": stage, "task": task,
                               "successes": result["successes"], "episodes": result["episodes"]})
        diagonal = [matrix[j][j] for j in range(4)]
        auc_per_task = [sum(matrix[i][j] for i in range(j, 4)) / (4-j) for j in range(4)]
        nbt_per_task = [sum(diagonal[j] - matrix[i][j] for i in range(j+1, 4)) / (3-j)
                        if j < 3 else Fraction(0) for j in range(4)]
        fwt = sum(diagonal) / 4
        auc = sum(auc_per_task) / 4
        nbt = sum(nbt_per_task) / 4
        final = sum(matrix[3]) / 4
        assert math.isclose(float(final), source["groups"][group]["final_average"], abs_tol=1e-12)
        groups.append({
            "configuration": group, "name_zh": title,
            "source": SOURCES[source_index].name,
            "measured_cells": counts,
            "success_rate_matrix_percent": [[float(x*100) if x is not None else None for x in row] for row in matrix],
            "AUC_percent": record(auc), "FWT_percent": record(fwt),
            "NBT_percentage_points": record(nbt), "final_average_percent": record(final),
            "per_task": {task: {"initial_success_percent": record(diagonal[j]),
                                "AUC_percent": record(auc_per_task[j]),
                                "NBT_percentage_points": record(nbt_per_task[j])}
                         for j, task in enumerate(TASKS)},
            "alternative_NBT_prior_tasks_only_percentage_points": record(sum(nbt_per_task[:3])/3),
        })
    assert all(p.read_bytes() == original for p, original in zip(SOURCES, payloads))
    output = {
        "schema": "pi05-libero-stage-end-metrics-v1", "status": "calculated",
        "task_order": TASKS, "training_seed": 42, "single_training_seed": True,
        "metrics_reference": "https://arxiv.org/html/2602.10503v1#S5.SS3.SSS1",
        "reference_section": "V-C.1, Evaluation Protocols (stage-end matrix definitions)",
        "screenshot_formula_verified": False,
        "formulas": {
            "S_i_j": "success rate on task j after completing stage i; indices1..K; K=4",
            "FWT": "(1/K) sum_j S_j_j",
            "AUC": "(1/K) sum_j [(1/(K-j+1)) sum_{i=j..K} S_i_j]",
            "NBT": "(1/K) sum_{j=1..K-1} [(1/(K-j)) sum_{i=j+1..K} (S_j_j-S_i_j)]",
            "NBT_last_task_convention": "0; final task has no later stage. Included in K-task outer mean.",
            "NBT_direction": "lower is better; negative denotes improvement on previously learned tasks",
        },
        "limitations": [
            "Screenshot names metrics but provides no formula; this calculation states its adopted definitions explicitly.",
            "These are four-task stage-end metrics, not the original LIBERO within-task learning-curve FWT/AUC.",
            "No unlearned upper-triangular entries are filled with zero.",
            "No multi-seed mean, standard deviation, or statistical significance is inferred.",
            "Plain-LoRA prototype replay uses fixed prototype samples selected by the CL-LoRA prototype run.",
        ],
        "sources": [{"path": p.name, "sha256": hashlib.sha256(payload).hexdigest()}
                    for p, payload in zip(SOURCES, payloads)],
        "groups": groups, "source_files_unchanged": True,
        "training_or_robot_execution": False,
    }
    destination = Path(__file__).with_name("RESULTS_stage_end.json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for row in groups:
        print(f"{row['name_zh']}: AUC={row['AUC_percent']['display']}, "
              f"FWT={row['FWT_percent']['display']}, NBT={row['NBT_percentage_points']['display']}, "
              f"final={row['final_average_percent']['display']}")
    print(destination)


if __name__ == "__main__":
    main()
