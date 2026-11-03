"""Sequential CPU checkpoint comparison for actual CL task-bank preservation.

This reads one complete native params checkpoint at a time with restore_params,
retains only hashes/small statistics of the requested tensors, releases it, then
loads the next checkpoint.  It does not compare the base backbone or execute a
model forward, gradient, optimizer update, or robot command.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import flax.traverse_util
import jax
import numpy as np

from openpi.models import gemma
from openpi.models import model as model_lib
from stable_training import bank_axis


def _fingerprint(value: np.ndarray, chunk_elements: int = 1_000_000) -> dict:
    """Digest exact bytes; compute full small statistics with bounded temporaries."""
    array = np.ascontiguousarray(value)
    flat = array.reshape(-1)
    header = {"shape": list(array.shape), "dtype": str(array.dtype)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode("utf-8") + b"\0")
    finite_count = 0
    total_sum = 0.0
    total_squared = 0.0
    minimum, maximum = math.inf, -math.inf
    for start in range(0, flat.size, chunk_elements):
        chunk = flat[start : start + chunk_elements]
        digest.update(chunk.view(np.uint8).tobytes(order="C"))
        numeric = chunk.astype(np.float64)
        finite = np.isfinite(numeric)
        finite_count += int(finite.sum())
        if finite.all():
            total_sum += float(numeric.sum(dtype=np.float64))
            total_squared += float(np.dot(numeric, numeric))
            if numeric.size:
                minimum = min(minimum, float(numeric.min()))
                maximum = max(maximum, float(numeric.max()))
        else:
            # Invalid numbers remain part of the exact byte digest.  Metrics
            # explicitly summarize only their finite subset.
            finite_numeric = numeric[finite]
            total_sum += float(finite_numeric.sum(dtype=np.float64))
            total_squared += float(np.dot(finite_numeric, finite_numeric))
            if finite_numeric.size:
                minimum = min(minimum, float(finite_numeric.min()))
                maximum = max(maximum, float(finite_numeric.max()))
    return {
        **header,
        "sha256": digest.hexdigest(),
        "elements": int(flat.size),
        "nonfinite_elements": int(flat.size) - finite_count,
        "min": minimum if finite_count else None,
        "max": maximum if finite_count else None,
        "mean": total_sum / finite_count if finite_count else None,
        "rms": math.sqrt(total_squared / finite_count) if finite_count else None,
    }


def _release_memory() -> None:
    gc.collect()
    # NumPy/Orbax host allocations can remain in libc arenas after gc.  This
    # optional Linux trim improves separation of the two checkpoint reads.
    try:
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except (AttributeError, OSError):
        pass


def checkpoint_snapshot(path: str | Path, num_tasks: int) -> dict:
    started = time.monotonic()
    params = model_lib.restore_params(str(path), restore_type=np.ndarray, dtype=None)
    flattened = flax.traverse_util.flatten_dict(params)
    summary = {"path": str(Path(path).resolve()), "bank_tensors": {}, "shared_lora_b": {}}
    try:
        for key in sorted(flattened):
            name = "/".join(map(str, key))
            is_bank = "lora_b_bank" in name or "cl_block_weights_" in name
            is_regular_b = str(key[-1]).endswith("lora_b") and "layers" in key
            if not is_bank and not is_regular_b:
                continue
            value = np.asarray(flattened[key])
            if is_bank:
                axis = bank_axis(key, tuple(value.shape), num_tasks)
                if axis is None:
                    continue
                rows = {}
                for task in range(num_tasks):
                    selected = np.take(value, task, axis=axis)
                    rows[str(task)] = _fingerprint(selected)
                    del selected
                summary["bank_tensors"][name] = {
                    "shape": list(value.shape), "dtype": str(value.dtype), "task_axis": axis, "rows": rows,
                }
            else:
                # In this engine's scanned Gemma tensors, regular LoRA-B is
                # used by shared layers.  Specific layers select the bank.
                expert = 1 if any(str(part).endswith("_1") for part in key) else 0
                variant = "gemma_300m_cl_lora" if expert else "gemma_2b_cl_lora"
                layer_ids = list(gemma.get_config(variant).shared_pos)
                if not layer_ids or max(layer_ids) >= value.shape[0]:
                    raise ValueError(f"Shared-layer configuration does not fit {name}: {value.shape}, {layer_ids}")
                selected = np.take(value, layer_ids, axis=0)
                summary["shared_lora_b"][name] = {
                    "full_tensor_shape": list(value.shape), "layer_axis": 0,
                    "shared_layer_indices": layer_ids, "fingerprint": _fingerprint(selected),
                }
                del selected
            del value
        if len(summary["bank_tensors"]) != 12:
            raise ValueError(f"Expected the configured 12 bank tensors; found {len(summary['bank_tensors'])} in {path}")
        if not summary["shared_lora_b"]:
            raise ValueError(f"No shared-layer regular LoRA-B tensors found in {path}")
        summary["seconds"] = time.monotonic() - started
        return summary
    finally:
        del flattened
        del params


def compare_snapshots(before: dict, after: dict, task_id: int, num_tasks: int) -> dict:
    issues = []
    bank_results = []
    shared_results = []
    before_banks, after_banks = before["bank_tensors"], after["bank_tensors"]
    if set(before_banks) != set(after_banks):
        issues.append({"kind": "bank_path_mismatch", "only_before": sorted(set(before_banks)-set(after_banks)),
                       "only_after": sorted(set(after_banks)-set(before_banks))})
    protected_rows = 0
    protected_changes = 0
    current_changes = 0
    for name in sorted(set(before_banks) & set(after_banks)):
        a, b = before_banks[name], after_banks[name]
        if (a["shape"], a["dtype"], a["task_axis"]) != (b["shape"], b["dtype"], b["task_axis"]):
            issues.append({"kind": "bank_layout_changed", "path": name})
        for task in range(num_tasks):
            fa, fb = a["rows"][str(task)], b["rows"][str(task)]
            changed = fa["sha256"] != fb["sha256"]
            role = "current" if task == task_id else "old" if task < task_id else "future"
            bank_results.append({"path": name, "task_id": task, "role": role, "changed": changed,
                                 "before": fa, "after": fb})
            if task != task_id:
                protected_rows += 1
                protected_changes += int(changed)
            else:
                current_changes += int(changed)
            if fa["nonfinite_elements"] or fb["nonfinite_elements"]:
                issues.append({"kind": "nonfinite_bank", "path": name, "task_id": task})
    ba, aa = before["shared_lora_b"], after["shared_lora_b"]
    if set(ba) != set(aa):
        issues.append({"kind": "shared_lora_b_path_mismatch"})
    shared_changes = 0
    for name in sorted(set(ba) & set(aa)):
        a, b = ba[name], aa[name]
        if a["shared_layer_indices"] != b["shared_layer_indices"]:
            issues.append({"kind": "shared_layer_indices_changed", "path": name})
        fa, fb = a["fingerprint"], b["fingerprint"]
        changed = fa["sha256"] != fb["sha256"]
        shared_changes += int(changed)
        shared_results.append({"path": name, "shared_layer_indices": a["shared_layer_indices"],
                               "changed": changed, "before": fa, "after": fb})
        if fa["nonfinite_elements"] or fb["nonfinite_elements"]:
            issues.append({"kind": "nonfinite_shared_lora_b", "path": name})
    if protected_changes:
        issues.append({"kind": "protected_bank_rows_changed", "count": protected_changes})
    if current_changes == 0:
        issues.append({"kind": "no_current_bank_tensor_changed"})
    if shared_changes == 0:
        issues.append({"kind": "no_shared_lora_b_tensor_changed"})
    return {
        "status": "passed" if not issues else "failed",
        "task_id": task_id, "num_tasks": num_tasks,
        "bank_tensor_count": len(bank_results) // num_tasks,
        "protected_old_future_rows_checked": protected_rows,
        "protected_old_future_rows_changed": protected_changes,
        "current_bank_tensors_changed": current_changes,
        "shared_lora_b_tensors_checked": len(shared_results),
        "shared_lora_b_tensors_changed": shared_changes,
        "bank_row_comparisons": bank_results, "shared_lora_b_comparisons": shared_results,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="Native complete params checkpoint before the stage")
    parser.add_argument("--after", required=True, help="Native complete params checkpoint after the stage")
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.task_id < args.num_tasks:
        parser.error("task-id must be within [0,num-tasks)")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        devices = jax.devices()
        if any(device.platform != "cpu" for device in devices):
            raise RuntimeError(f"This comparison must run on CPU only: {devices}")
        before = checkpoint_snapshot(args.before, args.num_tasks)
        _release_memory()
        print("BEFORE_DIGESTED_AND_FULL_TREE_RELEASED", flush=True)
        after = checkpoint_snapshot(args.after, args.num_tasks)
        _release_memory()
        print("AFTER_DIGESTED_AND_FULL_TREE_RELEASED", flush=True)
        result = compare_snapshots(before, after, args.task_id, args.num_tasks)
        result.update({
            "before": before["path"], "after": after["path"],
            "before_restore_and_digest_seconds": before["seconds"],
            "after_restore_and_digest_seconds": after["seconds"],
            "seconds": time.monotonic() - started,
            "devices": [str(device) for device in devices],
            "hash_definition": "SHA-256(dtype/shape header + exact C-order raw bytes), not approximate numerical comparison",
            "comparison_scope": "all 12 configured LoRA-B/block-gate bank tensors, every bank row; shared-layer slices of regular LoRA-B only",
            "not_compared": "base backbone, LoRA-A, action/time heads, optimizer states, task success, and other parameters",
            "memory_scope": "native restore_params reads a complete checkpoint, one checkpoint at a time; only hashes/scalar metrics retained between reads",
            "model_forward_calls": 0, "parameter_updates": 0,
        })
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({k: result[k] for k in ["status", "protected_old_future_rows_changed", "current_bank_tensors_changed", "shared_lora_b_tensors_changed", "seconds"]}), flush=True)
        return 0 if result["status"] == "passed" else 1
    except Exception as exc:
        _release_memory()
        output.write_text(json.dumps({"status": "error", "error": repr(exc), "seconds": time.monotonic()-started,
                                      "scope": "comparison did not complete; no preservation claim"}, indent=2))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
