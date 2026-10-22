"""Shared evaluation primitives for the Self-Harness experiment harness.

The package contains the unified evaluation result protocol
(``self_harness.eval_result.v1``) plus the configuration, dataset, manifest and
execution helpers shared by the Terminal-Bench 2 and SWE-bench Verified
runners.  Nothing in this package constructs a model client or reads a secret
value; credentials are injected by the caller through environment variables.
"""

from .hashing import (
    canonical_json,
    file_hashes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    snapshot_tree_hash,
    tree_file_hashes,
)
from .protocol import (
    BENCHMARK_SWEBENCH,
    BENCHMARK_TB2,
    SCHEMA_VERSION,
    EvalResultError,
    build_benchmark_block,
    build_config_block,
    build_dataset_block,
    build_eval_result,
    build_harness_block,
    build_model_block,
    case_passed,
    compute_repeat_metrics,
    compute_split_summary,
    make_agent_block,
    make_case_result,
    make_evaluator_block,
    make_repeat_result,
    recompute_result_metrics,
    utc_now,
    validate_result,
    write_result_immutable,
)

__all__ = [
    "BENCHMARK_SWEBENCH",
    "BENCHMARK_TB2",
    "SCHEMA_VERSION",
    "EvalResultError",
    "build_benchmark_block",
    "build_config_block",
    "build_dataset_block",
    "build_eval_result",
    "build_harness_block",
    "build_model_block",
    "canonical_json",
    "case_passed",
    "compute_repeat_metrics",
    "compute_split_summary",
    "file_hashes",
    "make_agent_block",
    "make_case_result",
    "make_evaluator_block",
    "make_repeat_result",
    "recompute_result_metrics",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "snapshot_tree_hash",
    "tree_file_hashes",
    "utc_now",
    "validate_result",
    "write_result_immutable",
]
