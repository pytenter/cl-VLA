"""Stdlib CPU child-process tests for the workflow prelaunch deferral loop."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import run_workflow as workflow


def make_execute(root):
    tree = ast.parse(Path(workflow.__file__).read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    execute = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    wrapper = ast.FunctionDef(name="make_execute", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                             kw_defaults=[], defaults=[]), decorator_list=[], body=[
        ast.Assign(targets=[ast.Name(id="active", ctx=ast.Store())], value=ast.Constant(value=None)),
        copy.deepcopy(execute), ast.Return(value=ast.Name(id="execute", ctx=ast.Load()))])
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    state = {"completed_jobs": [], "resource_deferrals": []}
    waits = []
    scope = dict(workflow.__dict__)
    scope.update(ROOT=root, state=state, recorded={"test": "frozen"}, manifest=lambda: {"test": "frozen"},
                 monitor=SimpleNamespace(assert_healthy=lambda: None),
                 runtime_resource_snapshot=lambda monitor: {'eligible': True, 'test_only': True},
                 args=SimpleNamespace(max_wait_hours=1, wait=True),
                 wait_idle=lambda **kwargs: waits.append(kwargs), rebuild_report=lambda: None,
                 update=lambda **fields: state.update(fields),
                 time=SimpleNamespace(monotonic=time.monotonic, sleep=lambda seconds: time.sleep(.01)))
    exec(compile(module, str(workflow.__file__), "exec"), scope)
    return scope["make_execute"](), state, waits


def fresh(root, name):
    path = root / name
    (path / "logs").mkdir(parents=True)
    (path / "configs").mkdir()
    (path / "reports").mkdir()
    (path / "configs/resource_policy.json").write_text(json.dumps({"mode": "cpu_test_no_gpu"}))
    return path


def run_tests():
    checks = {}
    with tempfile.TemporaryDirectory(prefix="kinova_resource_race_cpu_") as temporary:
        root = Path(temporary)
        queued = fresh(root, "defer_then_success")
        counter = queued / "child_count"
        command = [sys.executable, "-c", (
            "from pathlib import Path; import json,sys; p=Path(sys.argv[1]); "
            "n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
            "print('KINOVA_RESOURCE_DEFERRED '+json.dumps({'eligible':False,'blocking_reasons':['mock commit dip'],'attempt':n}) "
            "if n<=2 else 'CPU_ACTUAL_JOB_COMPLETED',flush=True); sys.exit(75 if n<=2 else 0)"), str(counter)]
        execute, state, waits = make_execute(queued)
        execute("job", command)
        assert int(counter.read_text()) == 3 and len(waits) == 3
        assert len({id(row['reclaimer']) for row in waits}) == 1
        assert len(state["resource_deferrals"]) == 2 and len(state["completed_jobs"]) == 1
        archived = [Path(row["log"]) for row in state["resource_deferrals"]]
        assert len(set(archived)) == 2 and all(path.exists() for path in archived)
        assert all(row["gate"]["eligible"] is False and row["exit_code"] == 75 for row in state["resource_deferrals"])
        assert all("KINOVA_RESOURCE_DEFERRED " in path.read_text() for path in archived)
        assert (queued / "logs/job.log").read_text().strip() == "CPU_ACTUAL_JOB_COMPLETED"
        checks["exit75_marker_and_blocked_gate_requeues_iteratively"] = True
        checks["each_prelaunch_deferral_has_distinct_preserved_log_and_gate_evidence"] = True
        checks["guard_wait_repeated_before_each_launch"] = True
        checks["idle_reclaim_budget_survives_all_prelaunch_deferrals"] = True

        for case, code, output in [
            ("non75_with_marker", 1, 'KINOVA_RESOURCE_DEFERRED {"eligible":false}'),
            ("75_without_marker", 75, "ordinary error without prelaunch contract"),
            ("75_malformed_gate", 75, "KINOVA_RESOURCE_DEFERRED notJSON"),
            ("75_eligible_gate", 75, 'KINOVA_RESOURCE_DEFERRED {"eligible":true}'),
        ]:
            directory = fresh(root, case)
            execute, state, waits = make_execute(directory)
            command = [sys.executable, "-c", "import sys;print(sys.argv[1],flush=True);sys.exit(int(sys.argv[2]))", output, str(code)]
            try:
                execute("job", command)
            except RuntimeError as error:
                assert f"code {code}" in str(error)
            else:
                raise AssertionError(f"{case} must stop the branch")
            assert len(waits) == 1 and not state["resource_deferrals"] and not state["completed_jobs"]
            assert (directory / "logs/job.log").read_text().strip() == output
        checks["non75_or_missing_invalid_marker_never_retried"] = True

        protected = fresh(root, "prior_actual_attempt")
        actual_log = protected / "logs/job.log"
        actual_log.write_text("prior actual model error; preserve exactly\n")
        execute, state, waits = make_execute(protected)
        sentinel = protected / "should_not_exist"
        try:
            execute("job", [sys.executable, "-c", "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('ran')", str(sentinel)])
        except FileExistsError:
            pass
        else:
            raise AssertionError("An existing actual attempt must block launch")
        assert not sentinel.exists() and actual_log.read_text() == "prior actual model error; preserve exactly\n"
        checks["existing_actual_log_prevents_launch_and_is_not_overwritten"] = True

        source = protected / "logs/new.log"
        source.write_text("new prelaunch deferral\n")
        archive = protected / "logs/new.resource_deferred_collision.log"
        archive.write_text("prior archived evidence\n")
        with mock.patch.object(workflow.uuid, "uuid4", return_value=SimpleNamespace(hex="collision")):
            try:
                workflow.archive_resource_deferral(source)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Archive collisions must never overwrite evidence")
        assert source.read_text() == "new prelaunch deferral\n" and archive.read_text() == "prior archived evidence\n"
        checks["archive_collision_preserves_both_existing_files"] = True
    return {"status": "passed", "scope": "Stdlib CPU subprocesses exercise the current execute AST and actual log helpers; no workflow/GPU/model launch",
            "checks": checks, "gpu_used": False, "jax_imported": "jax" in sys.modules,
            "model_optimizer_steps": 0, "contract": "exit75 plus exactly one marker line containing JSON object eligible=false"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_tests()
    assert not result["jax_imported"]
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
