#!/usr/bin/env python3
"""Deterministic fake proposer for workflow fixture/smoke tests.

Emits exactly ``--width`` mechanism-diverse proposals that edit different
prompt hooks with the ``FAKE_HARNESS_GOOD`` marker.  Modes via
SELF_HARNESS_FAKE_PROPOSER_MODE: improve (default), decline, duplicate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MARKER = "FAKE_HARNESS_GOOD"
HOOKS = (
    ("build_system_prompt", "prompt_instruction", "system_prompt_guidance"),
    ("build_verification_instruction", "prompt_instruction", "verification_guidance"),
    ("build_failure_recovery_instruction", "prompt_instruction", "recovery_guidance"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--train-ids-file", type=Path)
    args = parser.parse_args(argv)

    train_ids: list[str] = []
    if args.train_ids_file and args.train_ids_file.exists():
        payload = json.loads(args.train_ids_file.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            train_ids = [str(item) for item in payload]
    mode = os.environ.get("SELF_HARNESS_FAKE_PROPOSER_MODE", "improve").strip().lower()

    proposals = []
    for slot in range(1, max(1, args.width) + 1):
        hook, family, cluster = HOOKS[(slot - 1) % len(HOOKS)]
        if mode == "decline":
            proposals.append(
                {
                    "proposal_id": f"decline-{slot:02d}",
                    "title": "No safe mechanism",
                    "selection_decision": "decline",
                    "selected_cluster": "none",
                    "selected_surface": "none",
                    "mechanism": "none",
                    "mechanism_family": "prompt_instruction",
                    "exact_hook": "none",
                    "mechanism_to_hook_rationale": "declined",
                    "why_not_noop": "",
                    "decline_reason": "fixture declines to propose",
                    "skipped_clusters": ["fixture"],
                    "expected_affected_cases": [],
                    "protected_passing_cases": [],
                    "why_distinct": f"decline {slot}",
                    "net_gain_hypothesis": "",
                    "regression_guard": "",
                    "summary": "Fixture decline proposal.",
                    "final_message": "",
                    "candidate_values": {},
                }
            )
            continue
        value = (
            f"You are a fixture harness. {MARKER}. Guidance family: {cluster}. "
            f"Slots marker: {slot}."
        )
        proposal_id = "fixture-" + hook.replace("build_", "").replace("_instruction", "")
        if mode == "duplicate":
            proposal_id = "fixture-duplicate"
        proposals.append(
            {
                "proposal_id": proposal_id,
                "title": f"Fixture {hook} edit",
                "selection_decision": "eval_candidate",
                "selected_cluster": cluster,
                "selected_surface": hook,
                "mechanism": f"fixture {hook} guidance",
                "mechanism_family": family,
                "exact_hook": hook,
                "mechanism_to_hook_rationale": "fixture maps prompt guidance directly to the hook",
                "why_not_noop": "fixture evidence requires a bounded prompt mechanism",
                "decline_reason": "",
                "skipped_clusters": [],
                "expected_affected_cases": train_ids[:2],
                "protected_passing_cases": [],
                "why_distinct": f"fixture hook {hook}",
                "net_gain_hypothesis": "fixture marker causes the fixture agent to emit SOLVED",
                "regression_guard": "other prompt hooks stay unchanged",
                "summary": f"Fixture proposal editing {hook}.",
                "final_message": f"Edit {hook}",
                "candidate_values": {hook: value},
            }
        )
    args.response.parent.mkdir(parents=True, exist_ok=True)
    args.response.write_text(
        json.dumps({"proposals": proposals}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"fake proposer wrote {len(proposals)} proposals ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
