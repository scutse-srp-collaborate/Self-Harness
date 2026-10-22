"""Shared CLI plumbing for the benchmark runner scripts."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import EvalConfig, ModelSettings, load_eval_config
from .protocol import summarize_result


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="TOML evaluation config.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for this evaluation.")
    parser.add_argument("--run-root", type=Path, help="Self-Harness run root (defaults to output dir).")
    parser.add_argument("--run-id", help="Self-Harness run id shared by baseline and candidates.")
    parser.add_argument("--result-id", help="Unique result id inside the run.")
    parser.add_argument("--harness-snapshot", type=Path, help="Materialized harness snapshot directory.")
    parser.add_argument("--harness-version-id", help="Harness version id for provenance.")
    parser.add_argument(
        "--harness-kind",
        default="baseline",
        choices=("baseline", "candidate", "merged", "final"),
    )
    parser.add_argument(
        "--parent-version-id",
        action="append",
        default=[],
        help="Parent harness version id. Repeatable.",
    )
    parser.add_argument("--split", action="append", help="Run only this split. Repeatable.")
    parser.add_argument("--task-id", action="append", help="Run only this task id. Repeatable.")
    parser.add_argument("--repeats", type=int, help="Override [eval].repeats.")
    parser.add_argument("--case-concurrency", type=int, help="Override [eval].case_concurrency.")
    parser.add_argument("--model", help="Override the model id (provider/decoding stay configured).")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse a validated existing result.")
    parser.add_argument("--candidate-workspace", type=Path, help="Harness workspace exported to the agent.")
    parser.add_argument("--json", action="store_true", help="Print the result summary as JSON.")


def load_config_for_args(args: argparse.Namespace) -> EvalConfig:
    config = load_eval_config(args.config)
    if args.model:
        config = dataclasses.replace(
            config,
            model=ModelSettings(
                provider=config.model.provider,
                model_id=str(args.model),
                endpoint_type=config.model.endpoint_type,
                base_url=config.model.base_url,
                temperature=config.model.temperature,
                top_p=config.model.top_p,
                max_tokens=config.model.max_tokens,
                reasoning_budget=config.model.reasoning_budget,
                max_retries=config.model.max_retries,
                deployment=config.model.deployment,
            ),
        )
    if args.repeats is not None:
        if args.repeats < 1:
            raise SystemExit("--repeats must be >= 1")
        config = dataclasses.replace(
            config, eval=dataclasses.replace(config.eval, repeats=args.repeats)
        )
    if args.case_concurrency is not None:
        if args.case_concurrency < 1:
            raise SystemExit("--case-concurrency must be >= 1")
        config = dataclasses.replace(
            config, eval=dataclasses.replace(config.eval, case_concurrency=args.case_concurrency)
        )
    return config


def resolve_run_id(args: argparse.Namespace, config: EvalConfig) -> str:
    explicit = getattr(args, "run_id", None) or config.run.run_id
    if explicit:
        return str(explicit)
    env_value = os.environ.get("SELF_HARNESS_RUN_ID")
    if env_value:
        return env_value
    # Deterministic fallback: baseline and candidate evaluations driven from the
    # same config share one experiment run id (important for legacy callers that
    # do not pass --run-id explicitly).
    seed = f"{config.path}:{config.sha256}".encode("utf-8")
    return f"run-{hashlib.sha256(seed).hexdigest()[:16]}"


def resolve_result_id(args: argparse.Namespace, output_dir: Path) -> str:
    explicit = getattr(args, "result_id", None)
    if explicit:
        return str(explicit)
    return output_dir.name or "result"


def run_args_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "splits": tuple(args.split) if args.split else None,
        "task_ids": tuple(args.task_id) if args.task_id else None,
        "repeats": args.repeats,
        "reuse_existing": bool(args.reuse_existing) or bool(getattr(args, "config_reuse_existing", False)),
        "harness_snapshot": args.harness_snapshot,
        "harness_version_id": args.harness_version_id,
        "harness_kind": args.harness_kind,
        "parent_version_ids": tuple(args.parent_version_id or ()),
        "candidate_workspace": args.candidate_workspace,
    }


def print_result_summary(result: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        import json

        print(json.dumps(summarize_result(result), indent=2, sort_keys=True))
        return
    summary = summarize_result(result)
    print(
        f"result_id={summary['result_id']} status={summary['status']} "
        f"benchmark={(result.get('benchmark') or {}).get('id')}"
    )
    for split, block in summary["splits"].items():
        print(
            f"  {split}: mean_repeat_pass_rate={block['mean_repeat_pass_rate']:.6f} "
            f"pooled={block['pooled_passed']}/{block['pooled_total']}"
        )
    if summary["errors"]:
        print(f"  errors={summary['errors']}")


def fail(message: str, *, code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def result_path_for(output_dir: Path) -> Path:
    return Path(output_dir).expanduser().resolve() / "result.json"


def cleanup_unused(arguments: Sequence[str]) -> None:
    del arguments
