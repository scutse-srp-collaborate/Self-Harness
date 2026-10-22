#!/usr/bin/env python3
"""Self-Harness experiment CLI.

Commands: init, baseline, round, run, resume, report, eval.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in (
    "eval/src",
    "diagnosis/src",
    "proposer/src",
    "acceptance/src",
    "workflow/src",
):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from self_harness_eval.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

from self_harness_eval.config import load_eval_config  # noqa: E402
from self_harness_workflow import Orchestrator, ResumeMismatch, WorkflowError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Self-Harness experiment workflow.")
    parser.add_argument(
        "command",
        choices=("init", "baseline", "round", "run", "resume", "report", "eval"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, help="Explicit run directory (defaults to output_root/run_id).")
    parser.add_argument("--run-id", help="Run id shared by baseline and candidates.")
    parser.add_argument("--version-id", help="Version id for the 'eval' command.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse validated snapshots/results.")
    parser.add_argument("--max-rounds", type=int, help="Override [run].max_rounds for this invocation.")
    parser.add_argument("--proposal-width", type=int, help="Override [run].proposal_width.")
    parser.add_argument(
        "--max-candidate-evaluations",
        type=int,
        help="Override [run].max_candidate_evaluations.",
    )
    parser.add_argument("--proposer-command", help="External proposer command template.")
    parser.add_argument("--proposer-response", type=Path, help="Use a ready-made proposer response JSON.")
    parser.add_argument("--json", action="store_true", help="Print the command summary as JSON.")
    args = parser.parse_args(argv)

    try:
        config = load_eval_config(args.config)
        if args.proposal_width is not None:
            if args.proposal_width < 1:
                raise WorkflowError("--proposal-width must be >= 1")
            config = dataclasses.replace(
                config, run=dataclasses.replace(config.run, proposal_width=args.proposal_width)
            )
        if args.max_candidate_evaluations is not None:
            if args.max_candidate_evaluations < 0:
                raise WorkflowError("--max-candidate-evaluations must be >= 0")
            config = dataclasses.replace(
                config,
                run=dataclasses.replace(
                    config.run, max_candidate_evaluations=args.max_candidate_evaluations
                ),
            )
        if args.max_rounds is not None:
            if args.max_rounds < 1:
                raise WorkflowError("--max-rounds must be >= 1")
            config = dataclasses.replace(
                config, run=dataclasses.replace(config.run, max_rounds=args.max_rounds)
            )
        if args.proposer_command:
            config = dataclasses.replace(
                config,
                run=dataclasses.replace(config.run, proposer_command=args.proposer_command),
            )
        if args.proposer_response:
            config = dataclasses.replace(
                config,
                run=dataclasses.replace(
                    config.run, proposer_response=args.proposer_response.expanduser().resolve()
                ),
            )
        orchestrator = Orchestrator(
            config=config,
            run_dir=args.run_dir,
            run_id=args.run_id,
            reuse_existing=args.reuse_existing,
        )
        if args.command == "init":
            summary = orchestrator.cmd_init()
        elif args.command == "baseline":
            summary = orchestrator.cmd_baseline()
        elif args.command == "round":
            summary = orchestrator.cmd_round()
        elif args.command == "run":
            summary = orchestrator.cmd_run(max_rounds=args.max_rounds)
        elif args.command == "resume":
            summary = orchestrator.cmd_resume(max_rounds=args.max_rounds)
        elif args.command == "report":
            summary = orchestrator.cmd_report()
        else:
            if not args.version_id:
                raise WorkflowError("the 'eval' command requires --version-id")
            summary = orchestrator.cmd_eval_version(args.version_id)
    except (WorkflowError, ResumeMismatch) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
