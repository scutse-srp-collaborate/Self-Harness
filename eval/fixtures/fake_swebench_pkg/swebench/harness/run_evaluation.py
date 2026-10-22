#!/usr/bin/env python3
"""Fake official SWE-bench evaluator entry point.

Mimics the official CLI contract: reads a predictions JSONL, writes a report
JSON keyed by instance id with a boolean ``resolved``.  The fixture marks a
prediction resolved when its patch contains the ``SOLVED`` marker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolved_marker(patch: str) -> bool:
    return "# SOLVED" in patch and "# NOT_RESOLVED" not in patch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--max_workers", default="1")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--instance_ids", nargs="*", default=[])
    parser.add_argument("--report_dir", default=".")
    parser.add_argument("--split", default="test")
    args, _unknown = parser.parse_known_args(argv)

    predictions = []
    for line in Path(args.predictions_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            predictions.append(json.loads(line))
    report = {}
    for prediction in predictions:
        task_id = str(prediction.get("instance_id"))
        patch = str(prediction.get("model_patch") or "")
        report[task_id] = {
            "resolved": _resolved_marker(patch),
            "error": None if _resolved_marker(patch) else "fixture: no SOLVED marker",
            "tests_status": {},
        }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(predictions[0].get("model_name_or_path", "model")) if predictions else "model"
    safe_model = "".join(char if char.isalnum() or char in "._-" else "_" for char in model_name)
    target = report_dir / f"{safe_model}.{args.run_id}.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fixture official evaluator wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
