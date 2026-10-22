"""Acceptance gate tests: Pareto non-regression, comparability and completeness."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from self_harness_acceptance import (
    DECISION_ACCEPTED,
    DECISION_INCONCLUSIVE,
    DECISION_INVALID,
    DECISION_REJECTED,
    AcceptanceError,
    run_acceptance_gate,
    write_acceptance,
)

from tests.helpers import make_result, statuses


class AcceptanceGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _pair(
        self,
        *,
        baseline_train: list[str],
        baseline_heldout: list[str],
        candidate_train: list[str],
        candidate_heldout: list[str],
        candidate_kwargs: dict | None = None,
        baseline_extra: dict | None = None,
    ) -> tuple[dict, dict, dict]:
        baseline, _ = make_result(
            self.root,
            result_id="baseline",
            splits=statuses(train=baseline_train, heldout=baseline_heldout),
            **(baseline_extra or {}),
        )
        candidate, _ = make_result(
            self.root,
            result_id="candidate",
            harness_version_id="v001_candidate",
            harness_kind="candidate",
            parent_version_ids=["v000_baseline"],
            splits=statuses(train=candidate_train, heldout=candidate_heldout),
            tag="b",
            **(candidate_kwargs or {}),
        )
        acceptance = run_acceptance_gate(
            baseline_result=baseline,
            candidate_result=candidate,
            splits=("train", "heldout"),
            expected_repeats=1,
        )
        return baseline, candidate, acceptance

    def test_accepts_train_improvement_heldout_flat(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["unresolved", "unresolved"],
            baseline_heldout=["unresolved"],
            candidate_train=["resolved", "resolved"],
            candidate_heldout=["unresolved"],
        )
        self.assertEqual(acceptance["decision"], DECISION_ACCEPTED)
        self.assertTrue(acceptance["accepted"])
        self.assertAlmostEqual(acceptance["deltas"]["train"], 1.0)
        self.assertAlmostEqual(acceptance["deltas"]["heldout"], 0.0)
        self.assertEqual(acceptance["rule_version"], "self_harness.acceptance.v1")
        self.assertEqual(acceptance["baseline_result_id"], "baseline")
        self.assertEqual(acceptance["candidate_result_id"], "candidate")

    def test_accepts_heldout_improvement_train_flat(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["resolved"],
            baseline_heldout=["unresolved"],
            candidate_train=["resolved"],
            candidate_heldout=["resolved"],
        )
        self.assertEqual(acceptance["decision"], DECISION_ACCEPTED)
        self.assertAlmostEqual(acceptance["deltas"]["heldout"], 1.0)
        self.assertAlmostEqual(acceptance["deltas"]["train"], 0.0)

    def test_rejects_train_regression(self) -> None:
        # two train tasks: baseline 2/2, candidate 1/2; heldout improves.
        _, _, acceptance = self._pair(
            baseline_train=["resolved", "resolved"],
            baseline_heldout=["unresolved"],
            candidate_train=["resolved", "unresolved"],
            candidate_heldout=["resolved"],
        )
        self.assertEqual(acceptance["decision"], DECISION_REJECTED)
        self.assertFalse(acceptance["accepted"])
        self.assertAlmostEqual(acceptance["deltas"]["train"], -0.5)
        self.assertIn("train", acceptance["reason"])

    def test_rejects_heldout_regression(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["unresolved"],
            baseline_heldout=["resolved"],
            candidate_train=["resolved"],
            candidate_heldout=["unresolved"],
        )
        self.assertEqual(acceptance["decision"], DECISION_REJECTED)
        self.assertAlmostEqual(acceptance["deltas"]["heldout"], -1.0)

    def test_rejects_all_flat(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["resolved"],
            baseline_heldout=["resolved"],
            candidate_train=["resolved"],
            candidate_heldout=["resolved"],
        )
        self.assertEqual(acceptance["decision"], DECISION_REJECTED)
        self.assertIn("no split improved", acceptance["reason"])

    def test_rejects_both_drop(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["resolved"],
            baseline_heldout=["resolved"],
            candidate_train=["unresolved"],
            candidate_heldout=["unresolved"],
        )
        self.assertEqual(acceptance["decision"], DECISION_REJECTED)

    def test_task_transitions_are_counted(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["resolved", "unresolved", "resolved", "unresolved"],
            baseline_heldout=["unresolved"],
            candidate_train=["resolved", "resolved", "unresolved", "unresolved"],
            candidate_heldout=["resolved"],
        )
        train = acceptance["task_transitions"]["train"]
        self.assertEqual(train["both_pass"], 1)
        self.assertEqual(train["baseline_pass_candidate_fail"], 1)
        self.assertEqual(train["baseline_fail_candidate_pass"], 1)
        self.assertEqual(train["both_fail"], 1)
        self.assertEqual(acceptance["task_transitions"]["heldout"]["baseline_fail_candidate_pass"], 1)
        self.assertTrue(acceptance["splits"]["train"]["baseline"]["repeats"])

    def test_inconclusive_when_candidate_has_infra_error(self) -> None:
        # Same task set; candidate contains an unscoreable case.
        baseline, _ = make_result(
            self.root,
            result_id="baseline",
            splits=statuses(train=["resolved", "unresolved"], heldout=["resolved"]),
        )
        candidate, _ = make_result(
            self.root,
            run_id="run-test",
            result_id="candidate",
            harness_version_id="v1",
            harness_kind="candidate",
            parent_version_ids=["baseline"],
            splits=statuses(train=["resolved", "infra_error"], heldout=["resolved"]),
            tag="b",
        )
        acceptance = run_acceptance_gate(
            baseline_result=baseline,
            candidate_result=candidate,
            expected_repeats=1,
        )
        self.assertEqual(acceptance["decision"], DECISION_INCONCLUSIVE)
        self.assertFalse(acceptance["accepted"])
        self.assertIn("not scoreable", acceptance["reason"])
        self.assertFalse(acceptance["completeness"]["ok"])

    def test_invalid_when_comparability_hash_differs(self) -> None:
        baseline, _ = make_result(
            self.root,
            result_id="baseline",
            splits=statuses(train=["resolved"], heldout=["resolved"]),
        )
        candidate, _ = make_result(
            self.root,
            result_id="candidate",
            harness_version_id="v1",
            harness_kind="candidate",
            parent_version_ids=["baseline"],
            splits=statuses(train=["resolved"], heldout=["resolved"]),
            tools_variant="different-tools",
            tag="b",
        )
        acceptance = run_acceptance_gate(
            baseline_result=baseline, candidate_result=candidate, expected_repeats=1
        )
        self.assertEqual(acceptance["decision"], DECISION_INVALID)
        self.assertIn("comparability.tools", acceptance["reason"])
        self.assertFalse(acceptance["accepted"])

    def test_invalid_when_model_differs(self) -> None:
        baseline, _ = make_result(
            self.root, result_id="baseline", splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        candidate, _ = make_result(
            self.root,
            result_id="candidate",
            harness_version_id="v1",
            harness_kind="candidate",
            parent_version_ids=["baseline"],
            model_id="different-model",
            splits=statuses(train=["resolved"], heldout=["resolved"]),
            tag="b",
        )
        acceptance = run_acceptance_gate(
            baseline_result=baseline, candidate_result=candidate, expected_repeats=1
        )
        self.assertEqual(acceptance["decision"], DECISION_INVALID)
        self.assertIn("model", acceptance["reason"])

    def test_invalid_when_task_sets_differ(self) -> None:
        baseline, _ = make_result(
            self.root,
            result_id="baseline",
            splits={
                "train": [[("task-a", "resolved")], [("task-a", "resolved")]],
                "heldout": [[("heldout-a", "resolved")], [("heldout-a", "resolved")]],
            },
        )
        candidate, _ = make_result(
            self.root,
            result_id="candidate",
            harness_version_id="v1",
            harness_kind="candidate",
            parent_version_ids=["baseline"],
            splits={
                "train": [[("task-b", "resolved")], [("task-b", "resolved")]],
                "heldout": [[("heldout-a", "resolved")], [("heldout-a", "resolved")]],
            },
            tag="b",
        )
        acceptance = run_acceptance_gate(
            baseline_result=baseline, candidate_result=candidate, expected_repeats=2
        )
        self.assertEqual(acceptance["decision"], DECISION_INVALID)
        self.assertIn("task ids", acceptance["reason"])

    def test_invalid_when_repeat_counts_differ(self) -> None:
        baseline, _ = make_result(
            self.root,
            result_id="baseline",
            splits={
                "train": [[("t1", "resolved")], [("t1", "resolved")]],
                "heldout": [[("h1", "resolved")], [("h1", "resolved")]],
            },
        )
        candidate, _ = make_result(
            self.root,
            result_id="candidate",
            harness_version_id="v1",
            harness_kind="candidate",
            parent_version_ids=["baseline"],
            splits={
                "train": [[("t1", "resolved")]],
                "heldout": [[("h1", "resolved")]],
            },
            tag="b",
        )
        acceptance = run_acceptance_gate(
            baseline_result=baseline, candidate_result=candidate, expected_repeats=2
        )
        self.assertEqual(acceptance["decision"], DECISION_INVALID)

    def test_invalid_when_result_schema_is_broken(self) -> None:
        baseline, _ = make_result(
            self.root, result_id="baseline", splits=statuses(train=["resolved"], heldout=["resolved"])
        )
        candidate, _ = make_result(
            self.root,
            result_id="candidate",
            harness_version_id="v1",
            harness_kind="candidate",
            parent_version_ids=["baseline"],
            splits=statuses(train=["resolved"], heldout=["resolved"]),
            tag="b",
        )
        broken = copy.deepcopy(candidate)
        broken["summary"]["train"]["mean_repeat_pass_rate"] = 0.123456
        acceptance = run_acceptance_gate(
            baseline_result=baseline, candidate_result=broken, expected_repeats=1
        )
        self.assertEqual(acceptance["decision"], DECISION_INVALID)
        self.assertFalse(acceptance["accepted"])

    def test_acceptance_file_is_immutable(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["unresolved"],
            baseline_heldout=["unresolved"],
            candidate_train=["resolved"],
            candidate_heldout=["unresolved"],
        )
        path = self.root / "acceptance.json"
        write_acceptance(path, acceptance)
        with self.assertRaises(AcceptanceError):
            write_acceptance(path, acceptance)

    def test_acceptance_payload_fields_present(self) -> None:
        _, _, acceptance = self._pair(
            baseline_train=["unresolved"],
            baseline_heldout=["unresolved"],
            candidate_train=["resolved"],
            candidate_heldout=["unresolved"],
        )
        for key in (
            "format",
            "rule_version",
            "decision",
            "accepted",
            "reason",
            "decided_at",
            "baseline_result_id",
            "candidate_result_id",
            "splits",
            "task_transitions",
        ):
            self.assertIn(key, acceptance)
        split_block = acceptance["splits"]["train"]
        self.assertIn("baseline", split_block)
        self.assertIn("candidate", split_block)
        self.assertIn("delta", split_block)
        self.assertIn("repeats", split_block["baseline"])


if __name__ == "__main__":
    unittest.main()
