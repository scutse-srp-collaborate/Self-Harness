#!/usr/bin/env python3
"""Apply the Self-Harness non-regression Pareto acceptance gate.

The gate compares a candidate evaluation result against its parent/baseline
result.  It writes ``acceptance.json`` and never treats non-comparable or
unscoreable evaluations as ordinary rejects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("eval/src", "acceptance/src"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from self_harness_acceptance import (  # noqa: E402
    AcceptanceError,
    DEFAULT_SPLITS,
    DEFAULT_EXPECTED_REPEATS,
    load_result,
    run_acceptance_gate,
    write_acceptance,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the Self-Harness acceptance gate.")
    parser.add_argument("--baseline-result", required=True, type=Path)
    parser.add_argument("--candidate-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", action="append", help="Split to gate. Defaults to train and heldout.")
    parser.add_argument("--expected-repeats", type=int, default=DEFAULT_EXPECTED_REPEATS)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing acceptance.json if it matches the same result ids.",
    )
    args = parser.parse_args(argv)

    baseline_path = args.baseline_result.expanduser().resolve()
    candidate_path = args.candidate_result.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    splits = tuple(args.split or DEFAULT_SPLITS)
    try:
        baseline = load_result(baseline_path)
        candidate = load_result(candidate_path)
        if args.reuse_existing and output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                existing.get("baseline_result_id") == baseline.get("result_id")
                and existing.get("candidate_result_id") == candidate.get("result_id")
                and existing.get("rule_version")
            ):
                print(f"{existing.get('decision')}: reused {output_path}")
                return 0
        decision = run_acceptance_gate(
            baseline_result=baseline,
            candidate_result=candidate,
            baseline_result_path=baseline_path,
            candidate_result_path=candidate_path,
            splits=splits,
            expected_repeats=args.expected_repeats,
        )
        write_acceptance(output_path, decision)
    except AcceptanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{decision['decision']}: {decision['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
