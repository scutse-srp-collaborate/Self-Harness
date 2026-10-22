#!/usr/bin/env python3
"""Offline SWE-bench fixture smoke test (runner or full workflow).

Builds local git repositories, a pinned-format dataset export, a fake agent and
a fake evaluator, then exercises the real runner (and optionally the real
workflow) end-to-end. No network, Docker image or model endpoint is required.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("eval/src", "eval/fixtures", "diagnosis/src", "proposer/src", "acceptance/src", "workflow/src"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from swebench_fixture import build_environment, render_swebench_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline SWE-bench fixture smoke test.")
    parser.add_argument("--work-dir", type=Path, help="Persist fixtures/artifacts under this directory.")
    parser.add_argument(
        "--mode",
        choices=("runner", "workflow"),
        default="runner",
        help="Exercise only the evaluation runner or the full propose/eval/gate/merge workflow.",
    )
    args = parser.parse_args(argv)

    if args.work_dir:
        work_dir = args.work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="self-harness-smoke-"))
    fixture_env = build_environment(work_dir / "fixture")
    config_path = render_swebench_config(
        env=fixture_env, config_path=work_dir / "config.toml", repeats=2
    )

    from self_harness_eval.config import load_eval_config

    config = load_eval_config(config_path)
    if args.mode == "runner":
        from self_harness_eval.swebench import run_swebench_eval

        output_dir = work_dir / "eval"
        result = run_swebench_eval(
            config=config,
            output_dir=output_dir,
            run_id="smoke-run",
            result_id="smoke-baseline",
            harness_version_id="smoke-baseline",
        )
        summary = {
            "mode": "runner",
            "result_path": str(output_dir / "result.json"),
            "status": result["status"],
            "splits": {
                split: {
                    "mean_repeat_pass_rate": block["mean_repeat_pass_rate"],
                    "pooled": [block["pooled_passed"], block["pooled_total"]],
                }
                for split, block in result["summary"].items()
            },
            "evaluator": result["benchmark"]["evaluator"],
            "fixture_notice": (
                "Fixture smoke test only (fake agent + fake evaluator). These are not "
                "SWE-bench measurements."
            ),
        }
    else:
        from self_harness_workflow import Orchestrator

        orchestrator = Orchestrator(config=config, run_id="smoke-workflow")
        orchestrator.cmd_init()
        orchestrator.ensure_baseline()
        orchestrator.run_round()
        orchestrator.finalize_run(reason="smoke_complete")
        summary = {
            "mode": "workflow",
            "run_root": str(orchestrator.paths.root),
            "final_report_json": str(orchestrator.paths.final_report_json),
            "final_report_md": str(orchestrator.paths.final_report_md),
            "events": str(orchestrator.paths.events),
            "fixture_notice": (
                "Fixture smoke test only (fake proposer/agent/evaluator). These are not "
                "SWE-bench measurements."
            ),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
