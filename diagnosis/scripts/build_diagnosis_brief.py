#!/usr/bin/env python3
"""Build a deterministic train-only diagnosis brief from a unified eval result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_harness_diagnosis.evidence import (  # noqa: E402
    HeldoutLeakError,
    write_diagnosis_brief,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a train-only Self-Harness diagnosis brief.")
    parser.add_argument("--result", required=True, type=Path, help="Unified result.json.")
    parser.add_argument("--output", required=True, type=Path, help="Markdown brief output path.")
    parser.add_argument("--evidence-output", required=True, type=Path, help="JSON evidence output path.")
    parser.add_argument("--heldout-ids-file", required=True, type=Path)
    parser.add_argument("--train-ids-file", type=Path)
    parser.add_argument("--run-root", type=Path, help="Run root for resolving relative artifact paths.")
    args = parser.parse_args(argv)

    result_path = args.result.expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    heldout_ids = _read_ids(args.heldout_ids_file)
    train_ids = _read_ids(args.train_ids_file) if args.train_ids_file else None
    run_root = args.run_root.expanduser().resolve() if args.run_root else result_path.parent
    try:
        bundle = write_diagnosis_brief(
            result=result,
            result_path=result_path,
            output_path=args.output.expanduser().resolve(),
            evidence_path=args.evidence_output.expanduser().resolve(),
            heldout_task_ids=heldout_ids,
            train_task_ids=train_ids,
            run_root=run_root,
        )
    except HeldoutLeakError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(
        f"wrote {args.output} "
        f"(clusters={len(bundle.get('clusters') or [])}, "
        f"failed_cases={len(bundle.get('case_evidence') or [])})"
    )
    return 0


def _read_ids(path: Path) -> list[str]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("task_ids", "heldout_task_ids", "held_out_task_ids", "train_task_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
    raise SystemExit(f"cannot read task ids from {path}")


if __name__ == "__main__":
    raise SystemExit(main())
