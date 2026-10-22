#!/usr/bin/env python3
"""Deterministic fake SWE-bench *command* evaluator for smoke tests.

Writes ``evaluation.json`` with a boolean ``resolved`` per instance.  The
resolved label comes from the fixture agent's patch marker (or an explicit
comma-separated task list), never from the harness being evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolved_marker(patch: str) -> bool:
    return "# SOLVED" in patch and "# NOT_RESOLVED" not in patch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolved-task-ids", default="")
    args = parser.parse_args(argv)

    resolved_env = os.environ.get("SELF_HARNESS_FAKE_RESOLVED_TASKS", "")
    resolved_ids = {
        item.strip()
        for item in (args.resolved_task_ids + "," + resolved_env).split(",")
        if item.strip()
    }
    predictions = []
    if args.predictions.exists():
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                predictions.append(json.loads(line))
    if not predictions:
        print("no predictions supplied", file=sys.stderr)
        return 3
    report = {}
    for prediction in predictions:
        task_id = str(prediction.get("instance_id"))
        patch = str(prediction.get("model_patch") or "")
        resolved = bool(task_id in resolved_ids) if resolved_ids else _resolved_marker(patch)
        report[task_id] = {
            "resolved": resolved,
            "error": None if resolved else "fixture evaluator: patch did not apply cleanly",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fake evaluator resolved {sum(1 for v in report.values() if v['resolved'])}/{len(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
