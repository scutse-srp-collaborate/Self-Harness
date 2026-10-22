#!/usr/bin/env python3
"""Run a bounded SWE-bench Verified train-only pilot.

This entry point enforces the pilot contract:

* only frozen ``split=train`` task ids (at most five, unique, never heldout);
* one repeat count from config (default 2) and case concurrency from config;
* no proposer, no candidates, no workflow/acceptance iteration;
* the unified result is marked ``pilot`` and a pilot summary is written next to it;
* missing API key or missing data/repositories stops the pilot before any model call.

The main 100-task manifest is never modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("eval/src", "eval/scripts"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from self_harness_eval.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

from self_harness_eval.config import ConfigError, load_eval_config  # noqa: E402
from self_harness_eval.manifest import load_swebench_manifest  # noqa: E402
from self_harness_eval.pilot import (  # noqa: E402
    PILOT_MAX_TASKS,
    PilotError,
    docker_runtime_blockers,
    pilot_manifest_payload,
    validate_pilot_task_ids,
    write_pilot_outputs,
)
from self_harness_eval.runner_common import RunnerError  # noqa: E402
from self_harness_eval.swebench import run_swebench_eval  # noqa: E402
from prepare_swebench_data import data_root_from_environment, dataset_paths  # noqa: E402

API_KEY_ENV = "SELF_HARNESS_MODEL_API_KEY"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded SWE-bench train pilot.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--result-id")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-tasks", type=int, default=PILOT_MAX_TASKS)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_eval_config(args.config)
    except ConfigError as exc:
        print(f"error: cannot load config: {exc}", file=sys.stderr)
        return 2
    if config.eval.benchmark != "swe_bench_verified":
        print("error: pilot requires a swe_bench_verified config", file=sys.stderr)
        return 2
    manifest = load_swebench_manifest(config.dataset.manifest_path)
    requested = tuple(args.task_id) if args.task_id else tuple(config.eval.task_ids or ())
    try:
        tasks = validate_pilot_task_ids(manifest, requested, max_tasks=args.max_tasks)
    except PilotError as exc:
        print(f"error: invalid pilot task selection: {exc}", file=sys.stderr)
        return 2
    repeats = int(args.repeats or config.eval.repeats)
    if repeats < 1:
        print("error: --repeats must be >= 1", file=sys.stderr)
        return 2

    data_root = data_root_from_environment()
    paths = dataset_paths(data_root)
    run_id = args.run_id or f"swebench-pilot5-{time.strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (paths["runs"] / run_id).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_payload = pilot_manifest_payload(manifest, tasks, repeats=repeats)

    blockers: list[str] = []
    if not os.environ.get(API_KEY_ENV, "").strip():
        blockers.append(
            f"{API_KEY_ENV} is not set in the runtime environment; refusing to make real model "
            "calls. Set the variable in the environment and rerun; do not paste the key into chat."
        )
    if not paths["json"].exists() or not paths["meta"].exists():
        blockers.append(f"pinned dataset export is missing under {paths['datasets']}")
    repos_dir = Path(config.dataset.repos_dir) if config.dataset.repos_dir else paths["repos"]
    for task in tasks:
        repo_path = repos_dir / str(task["repository"])
        if not (repo_path / ".git").is_dir():
            blockers.append(f"repository clone is missing: {repo_path}")
    blockers.extend(
        docker_runtime_blockers(
            config.constraints.docker_image
            or os.environ.get("SELF_HARNESS_AGENT_IMAGE", "python:3.12")
        )
    )
    if blockers:
        blocked = {
            "format": "self_harness.swebench_pilot_blocked.v1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "output_dir": str(output_dir),
            "blockers": blockers,
            "pilot": pilot_payload,
            "model_calls_made": False,
            "result_written": False,
            "note": (
                "The pilot was stopped before any model call. No fake evaluator or synthetic "
                "resolved label was substituted."
            ),
        }
        blocked_path = output_dir / "pilot_blocked.json"
        blocked_path.write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        message = "pilot blocked: " + "; ".join(blockers)
        print(message, file=sys.stderr)
        print(f"wrote {blocked_path}", file=sys.stderr)
        return 2

    config = dataclasses.replace(
        config,
        eval=dataclasses.replace(
            config.eval,
            pilot=True,
            pilot_max_tasks=args.max_tasks,
            repeats=repeats,
            splits=("train",),
            task_ids=tuple(task["task_id"] for task in tasks),
            case_concurrency=max(1, config.eval.case_concurrency),
        ),
    )
    run_id = run_id
    result_id = args.result_id or run_id
    try:
        result = run_swebench_eval(
            config=config,
            output_dir=output_dir,
            run_id=run_id,
            result_id=result_id,
            harness_version_id=f"{run_id}-baseline",
            harness_kind="baseline",
            splits=("train",),
            task_ids=tuple(task["task_id"] for task in tasks),
            repeats=repeats,
            reuse_existing=args.reuse_existing,
            pilot=pilot_payload,
        )
    except RunnerError as exc:
        print(f"error: pilot evaluation failed: {exc}", file=sys.stderr)
        return 3
    summary_path, markdown_path = write_pilot_outputs(
        result=result, pilot=pilot_payload, output_dir=output_dir
    )
    summary = {
        "run_id": run_id,
        "result_id": result_id,
        "output_dir": str(output_dir),
        "result_path": str(output_dir / "result.json"),
        "pilot_summary_json": str(summary_path),
        "pilot_summary_md": str(markdown_path),
        "status": result.get("status"),
        "scope": "pilot_train_only",
        "task_ids": [task["task_id"] for task in tasks],
        "repeats": repeats,
        "heldout_run": False,
        "candidate_iteration": False,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
