"""Small CPU-only continuation provenance check; no images or model arrays read."""
from __future__ import annotations

# Publication: resolve machine-specific filesystem defaults from environment.
from openpi.shared.project_paths import project_path

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def check(root, previous):
    start = time.monotonic()
    candidate = Path(project_path('candidate', ''))
    base = Path(project_path('base_params', ''))
    old = read(previous / "manifests/data_sources.json")
    ref = read(previous / "manifests/referenced_source_files.json")
    by_index = {(r["split"], r["episode_index"]): r for r in old["episodes"]}
    reference_checks, issues, datasets, ids = [], [], {}, {}
    for item in ref:
        p = Path(item["path"])
        actual = sha(p) if p.is_file() else None
        same = actual == item.get("sha256") and p.stat().st_size == item.get("size") if p.is_file() else False
        reference_checks.append({"path": str(p), "exists": p.is_file(), "expected_sha256": item.get("sha256"),
                                 "actual_sha256": actual, "unchanged": same})
        if not same:
            issues.append("referenced source changed or missing: " + str(p))
    for split in ("train", "validation", "test"):
        repo = candidate / "lerobot/kinova_candidate" / split
        info = read(repo / "meta/info.json")
        episodes = [json.loads(line) for line in (repo / "meta/episodes.jsonl").read_text().splitlines() if line]
        rows = []
        ids[split] = []
        for ep in episodes:
            idx, n = int(ep["episode_index"]), int(ep["length"])
            source = by_index[(split, idx)]
            eid = source["episode_id"]
            ids[split].append(eid)
            path = repo / info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
            pf = pq.ParquetFile(path)
            data = pf.read(columns=["observation.state", "action", "source_time_ns"], use_threads=False)
            state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float64)
            action = np.asarray(data["action"].to_pylist(), dtype=np.float64)
            times = np.asarray(data["source_time_ns"].to_pylist(), dtype=np.int64)
            value_sha = hashlib.sha256(state.tobytes() + action.tobytes() + times.tobytes()).hexdigest()
            valid = (pf.metadata.num_rows == n == source["steps"] and state.shape == (n, 28)
                     and action.shape == (n, 7) and times.shape == (n, 8)
                     and np.isfinite(state).all() and np.isfinite(action).all())
            unchanged = value_sha == source["numeric_sha256"]
            if not valid or not unchanged:
                issues.append("candidate numerical/source length changed: " + eid)
            rows.append({"episode_index": idx, "episode_id": eid, "steps": n, "parquet_path": str(path),
                         "row_count": pf.metadata.num_rows, "numeric_finite_shapes_valid": bool(valid),
                         "numeric_sha256": value_sha, "previous_numeric_sha256": source["numeric_sha256"],
                         "numerical_payload_unchanged": unchanged,
                         "parquet_bytes_unchanged": path.stat().st_size == source["parquet_bytes"],
                         "parquet_mtime_unchanged": path.stat().st_mtime_ns == source["parquet_mtime_ns"]})
        metadata_same = info == old["candidate_info"][split]
        if not metadata_same:
            issues.append("candidate repo info differs from prior actual snapshot: " + split)
        datasets[split] = {"episodes": len(rows), "steps": sum(r["steps"] for r in rows),
                           "actual_info_total_episodes": info["total_episodes"],
                           "actual_info_total_frames": info["total_frames"], "repo_info_unchanged": metadata_same,
                           "metadata_sha256": {n: sha(repo / "meta" / n) for n in ("info.json", "episodes.jsonl", "tasks.jsonl")},
                           "episode_checks": rows}
    overlaps = {a + "/" + b: sorted(set(ids[a]) & set(ids[b]))
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(overlaps.values()):
        issues.append("candidate split ID intersections are nonempty")
    copies = []
    # These are inherited CPU/metadata artifacts, not new execution evidence.
    for p in sorted((root / "reports").glob("*.json")):
        source = previous / "reports" / p.name
        if source.is_file() and not p.name.startswith("continuation_"):
            same = sha(p) == sha(source)
            copies.append({"destination": str(p), "source": str(source), "sha256": sha(p),
                           "copied_verbatim_at_check": same,
                           "execution_claim": "old-run evidence reused; not rerun by this continuation check" if same
                           else "content differs; requires its own new-run execution/provenance evidence"})
    base_meta = [{"path": str(base / n), "exists": (base / n).is_file(),
                  "sha256": sha(base / n) if (base / n).is_file() else None,
                  "bytes": (base / n).stat().st_size if (base / n).is_file() else None}
                 for n in ("_METADATA", "_CHECKPOINT_METADATA", "manifest.ocdbt", "_sharding")]
    base_tree = read(base / "_METADATA") if (base / "_METADATA").is_file() else {}
    review = root / "review"
    review.mkdir(exist_ok=True)
    links = {}
    for name in ("index.html", "episodes", "candidate_staging", "raw_episodes", "legacy_previews"):
        target, dest = previous / "review" / name, review / name
        if dest.is_symlink():
            if dest.resolve() != target.resolve():
                raise FileExistsError(dest)
        elif dest.exists():
            raise FileExistsError(dest)
        else:
            dest.symlink_to(target, target_is_directory=target.is_dir())
        links[name] = {"link": str(dest), "target": str(target), "exists": dest.exists()}
    # A fresh small worksheet may be edited without writing through to the old run.
    worksheet = review / "manual_review.csv"
    if not worksheet.exists():
        shutil.copyfile(previous / "review/manual_review.csv", worksheet)
    counts = {s: {k: v for k, v in values.items() if k in ("episodes", "steps")} for s, values in datasets.items()}
    result = {"status": "passed" if not issues else "failed", "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "new_run_root": str(root), "previous_failed_run": str(previous),
              "scope": "fresh actual metadata and all107 Parquet numerical columns only;14 small referenced files; base metadata identity. No image decode, no model tensor load, no GPU or optimizer.",
              "candidate_root": str(candidate), "candidate_episodes": sum(v["episodes"] for v in counts.values()),
              "candidate_steps": sum(v["steps"] for v in counts.values()), "splits": counts,
              "expected_counts_match": counts == {"train": {"episodes": 60, "steps": 8011}, "validation": {"episodes": 20, "steps": 2754}, "test": {"episodes": 27, "steps": 3758}},
              "split_episode_id_intersections": overlaps, "datasets": datasets,
              "norm_stats_sha256": sha(candidate / "assets/norm_stats.json"),
              "referenced_sources_checked": len(reference_checks),
              "all_referenced_sources_unchanged": all(v["unchanged"] for v in reference_checks),
              "referenced_source_checks": reference_checks,
              "all107_numerical_payloads_match_prior_audit": all(r["numerical_payload_unchanged"] for v in datasets.values() for r in v["episode_checks"]),
              "base_weights": {"params_dir": str(base), "exists": base.is_dir(), "metadata": base_meta,
                               "tree_entry_count": len(base_tree.get("tree_metadata", {})),
                               "identity_manifest": project_path('deploy', 'artifacts/model_manifest.json'),
                               "full_large_weight_arrays_rehashed": False,
                               "qualification": "small identity manifest and checkpoint metadata checked; not a proof of full array-byte identity"},
              "inherited_report_evidence": copies,
              "prior_full_audit_scope": "previous run performed full image-container checks,642 RGB samples,746 raw-feedback reconstructions and321 native loader samples; these were NOT rerun by this quick continuation check",
              "manual_review_status": "all107 manual_review_pending; old preview material reused, no new success judgments",
              "external_sync_status": "UNVERIFIED", "wrist_sync_status": "UNVERIFIED",
              "review_links": links, "new_manual_review_worksheet": str(worksheet),
              "new_run_gpu_evidence_rule": "Only newly written training/evaluation/restore evidence under this new root counts as this continuation's GPU results; copied reports do not count as new execution",
              "optimizer_steps_executed": 0, "gpu_used": False, "robot_commands_sent": 0,
              "issues": issues, "elapsed_seconds": time.monotonic() - start}
    write(root / "reports/continuation_data_check.json", result)
    write(root / "manifests/continuation_data_evidence.json", {"script_sha256": sha(Path(__file__)),
          "checked_at": result["checked_at"], "previous_data_manifest": str(previous / "manifests/data_sources.json"),
          "previous_data_manifest_sha256": sha(previous / "manifests/data_sources.json"),
          "previous_reference_manifest_sha256": sha(previous / "manifests/referenced_source_files.json"),
          "new_run": str(root), "full_audit_reexecuted": False})
    print(json.dumps({k: result[k] for k in ("status", "splits", "all107_numerical_payloads_match_prior_audit", "referenced_sources_checked", "all_referenced_sources_unchanged", "issues", "elapsed_seconds")}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(project_path('kinova', '')))
    p.add_argument("--previous-root", type=Path, default=Path(project_path('archive', 'run_20261101_113646')))
    a = p.parse_args()
    if a.root.resolve() == a.previous_root.resolve():
        p.error("continuation output root must differ from protected previous experiment")
    check(a.root.resolve(), a.previous_root.resolve())
