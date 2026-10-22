"""Unified evaluation result protocol tests."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from self_harness_eval.protocol import (
    CASE_STATUS_FAILED,
    CASE_STATUS_INFRA_ERROR,
    CASE_STATUS_NOT_RUN,
    CASE_STATUS_PASSED,
    CASE_STATUS_RESOLVED,
    CASE_STATUS_UNRESOLVED,
    EvalResultError,
    build_eval_result,
    case_passed,
    compute_repeat_metrics,
    make_agent_block,
    make_case_result,
    make_evaluator_block,
    make_repeat_result,
    recompute_result_metrics,
    validate_result,
    write_result_immutable,
)

from tests.helpers import make_result, statuses


class ProtocolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_swebench_result_is_schema_valid(self) -> None:
        result, _ = make_result(
            self.root,
            benchmark="swe_bench_verified",
            splits=statuses(train=["resolved", "unresolved"], heldout=["resolved"]),
        )
        validate_result(result, expected_repeats=1)
        self.assertEqual(result["schema_version"], "self_harness.eval_result.v1")
        status = result["splits"]["train"][0]["case_results"]
        self.assertEqual([case["status"] for case in status], ["resolved", "unresolved"])
        self.assertEqual([case["passed"] for case in status], [True, False])
        self.assertEqual(result["summary"]["train"]["passed_per_repeat"], [1])
        self.assertEqual(result["summary"]["train"]["total_per_repeat"], [2])
        self.assertAlmostEqual(result["summary"]["train"]["mean_repeat_pass_rate"], 0.5)

    def test_terminal_bench_result_uses_passed_and_null_patch(self) -> None:
        result, _ = make_result(
            self.root,
            benchmark="terminal_bench_2",
            splits=statuses(train=["passed", "failed"], heldout=["passed"]),
        )
        validate_result(result)
        for repeat in result["splits"]["train"]:
            for case in repeat["case_results"]:
                self.assertIsNone(case["patch"])
        self.assertEqual(
            [case["status"] for case in result["splits"]["train"][0]["case_results"]],
            ["passed", "failed"],
        )

    def test_passed_tristate_mapping(self) -> None:
        self.assertTrue(case_passed(CASE_STATUS_RESOLVED))
        self.assertTrue(case_passed(CASE_STATUS_PASSED))
        self.assertFalse(case_passed(CASE_STATUS_UNRESOLVED))
        self.assertFalse(case_passed(CASE_STATUS_FAILED))
        self.assertIsNone(case_passed(CASE_STATUS_INFRA_ERROR))
        self.assertIsNone(case_passed(CASE_STATUS_NOT_RUN))

    def test_metrics_recompute_from_case_results(self) -> None:
        result, _ = make_result(
            self.root,
            splits={
                "train": [
                    [("t1", "resolved"), ("t2", "unresolved")],
                    [("t1", "resolved"), ("t2", "resolved")],
                ],
                "heldout": [
                    [("h1", "resolved")],
                    [("h1", "unresolved")],
                ],
            },
        )
        recomputed = recompute_result_metrics(result)
        train_summary = recomputed["splits"]["train"]["summary"]
        heldout_summary = recomputed["splits"]["heldout"]["summary"]
        self.assertEqual(train_summary["passed_per_repeat"], [1, 2])
        self.assertAlmostEqual(train_summary["mean_repeat_pass_rate"], (0.5 + 1.0) / 2)
        self.assertEqual(heldout_summary["pooled_passed"], 1)
        self.assertEqual(heldout_summary["pooled_total"], 2)

    def test_tampered_stored_metrics_are_rejected(self) -> None:
        result, _ = make_result(
            self.root, splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        tampered = copy.deepcopy(result)
        tampered["splits"]["train"][0]["passed"] = 0
        with self.assertRaises(EvalResultError):
            validate_result(tampered)
        tampered = copy.deepcopy(result)
        tampered["summary"]["train"]["mean_repeat_pass_rate"] = 0.99
        with self.assertRaises(EvalResultError):
            validate_result(tampered)

    def test_duplicate_repeat_ids_rejected(self) -> None:
        metrics = make_repeat_result(
            split="train",
            repeat=1,
            expected_task_ids=["t1"],
            case_results=[
                make_case_result(
                    task_id="t1",
                    split="train",
                    repeat=1,
                    status="resolved",
                    duration_s=1.0,
                    agent=make_agent_block(
                        status="complete", timed_out=False, tool_calls=1, trace_path=None
                    ),
                    evaluator=make_evaluator_block(
                        status="complete",
                        name="e",
                        resolved=True,
                        failure_summary=None,
                        log_path=None,
                    ),
                )
            ],
        )
        self.assertEqual(metrics["total"], 1)
        with self.assertRaises(EvalResultError):
            compute_repeat_metrics(
                split="train",
                repeat=1,
                expected_task_ids=["t1", "t2"],
                case_results=metrics["case_results"],
            )

    def test_unscoreable_case_requires_error_and_marks_partial(self) -> None:
        result, _ = make_result(
            self.root,
            splits=statuses(train=["infra_error", "resolved"], heldout=["resolved"]),
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["splits"]["train"][0]["status"], "partial")
        self.assertTrue(result["errors"])
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertIsNone(case["passed"])
        self.assertIn("infra_error", case["error"])

    def test_result_immutability(self) -> None:
        result, _ = make_result(
            self.root, splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        target = self.root / "immutable" / "result.json"
        write_result_immutable(target, result)
        with self.assertRaises(EvalResultError):
            write_result_immutable(target, result)

    def test_absolute_artifact_paths_rejected(self) -> None:
        result, _ = make_result(
            self.root, splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        result["artifacts"] = {"splits_dir": "/tmp/absolute"}
        with self.assertRaises(EvalResultError):
            validate_result(result)

    def test_secret_keys_rejected(self) -> None:
        result, _ = make_result(
            self.root, splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        result["model"]["api_key"] = "sk-should-not-be-stored"
        with self.assertRaises(EvalResultError):
            validate_result(result)

    def test_schema_version_is_enforced(self) -> None:
        result, _ = make_result(
            self.root, splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        result["schema_version"] = "self_harness.eval_result.v0"
        with self.assertRaises(EvalResultError):
            validate_result(result)

    def test_benchmark_specific_statuses(self) -> None:
        swebench, _ = make_result(
            self.root, benchmark="swe_bench_verified", splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        swebench["splits"]["train"][0]["case_results"][0]["status"] = "passed"
        with self.assertRaises(EvalResultError):
            validate_result(swebench)

    def test_tb2_patch_must_be_null(self) -> None:
        result, _ = make_result(
            self.root, benchmark="terminal_bench_2", splits=statuses(train=["passed"], heldout=["passed"])
        )
        result["splits"]["train"][0]["case_results"][0]["patch"] = {
            "path": "patch.diff",
            "sha256": "0" * 64,
        }
        with self.assertRaises(EvalResultError):
            validate_result(result)

    def test_success_requires_evaluator_resolved_true(self) -> None:
        with self.assertRaises(EvalResultError):
            make_case_result(
                task_id="t",
                split="train",
                repeat=1,
                status="resolved",
                duration_s=1.0,
                agent=make_agent_block(
                    status="complete", timed_out=False, tool_calls=1, trace_path=None
                ),
                evaluator=make_evaluator_block(
                    status="complete",
                    name="e",
                    resolved=False,
                    failure_summary=None,
                    log_path=None,
                ),
            )

    def test_builder_requires_consistent_split_repeat_counts(self) -> None:
        result, _ = make_result(
            self.root, splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        with self.assertRaises(EvalResultError):
            build_eval_result(
                run_id=result["run_id"],
                result_id="another",
                benchmark=result["benchmark"],
                model=result["model"],
                harness=result["harness"],
                dataset=result["dataset"],
                config=result["config"],
                started_at=result["started_at"],
                finished_at=result["finished_at"],
                splits={
                    "train": result["splits"]["train"],
                    "heldout": result["splits"]["heldout"] + result["splits"]["heldout"],
                },
            )


if __name__ == "__main__":
    unittest.main()
