#!/usr/bin/env python3
"""Terminal-Bench 2.0 (Harbor + pytest) evaluation runner.

This script keeps the original CLI flags used by the legacy workflow
(``--config --output-dir --reuse-existing``) and writes the unified
``self_harness.eval_result.v1`` result.  The parsed pytest/Harbor behavior is
unchanged; the legacy aggregate is available as ``result.legacy.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_harness_eval.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

from self_harness_eval.cli import (  # noqa: E402
    add_common_arguments,
    fail,
    load_config_for_args,
    print_result_summary,
    resolve_result_id,
    resolve_run_id,
    run_args_from_namespace,
)
from self_harness_eval.runner_common import RunnerError  # noqa: E402
from self_harness_eval.tb2 import run_tb2_eval  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Terminal-Bench 2.0 Harbor/pytest evaluation.")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        config = load_config_for_args(args)
        if config.eval.benchmark != "terminal_bench_2":
            return fail(f"config benchmark is {config.eval.benchmark!r}, expected 'terminal_bench_2'")
        output_dir = args.output_dir.expanduser().resolve()
        result = run_tb2_eval(
            config=config,
            output_dir=output_dir,
            run_root=args.run_root,
            run_id=resolve_run_id(args, config),
            result_id=resolve_result_id(args, output_dir),
            **run_args_from_namespace(args),
        )
    except RunnerError as exc:
        return fail(str(exc))
    print_result_summary(result, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
