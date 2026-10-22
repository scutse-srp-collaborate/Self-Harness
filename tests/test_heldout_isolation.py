"""Held-out information isolation tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from self_harness_diagnosis.evidence import (
    HeldoutLeakError,
    assert_no_heldout_leak,
    load_train_evidence,
    render_diagnosis_brief,
    write_diagnosis_brief,
)
from self_harness_proposer import (
    EditableSurface,
    HeldoutLeakError as ProposerHeldoutLeakError,
    MultiProposerRequest,
    audit_text,
    build_multi_proposer_prompt,
    validate_proposal_metadata,
)

from tests.helpers import make_result

REPO_ROOT = Path(__file__).resolve().parents[1]
HELDOUT_SECRET = "HELDOUT_SECRET_FAILURE_DETAIL_1234"
TRAIN_FAILURE = "train verifier assertion failure"


def _surface() -> EditableSurface:
    return EditableSurface(
        name="baseline",
        kind="workspace_file",
        target="repo_baseline.py",
        filename="repo_baseline.py",
        current_value=(
            "def build_system_prompt() -> str:\n    return 'baseline prompt'\n"
            "\n\ndef build_verification_instruction() -> str:\n    return 'verify'\n"
        ),
    )


class HeldoutIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train_ids = ["train-1", "train-2"]
        self.heldout_ids = ["heldout-1"]
        result, run_root = make_result(
            self.root,
            splits={
                "train": [[("train-1", "unresolved"), ("train-2", "agent_timeout")]],
                "heldout": [[("heldout-1", "unresolved")]],
            },
            failure_summary=TRAIN_FAILURE,
        )
        for repeat in result["splits"]["heldout"]:
            for case in repeat["case_results"]:
                case["evaluator"]["failure_summary"] = HELDOUT_SECRET
                case["error"] = None
                case["agent"]["trace_path"] = "heldout/trace.json"
        self.result = result
        self.run_root = run_root
        self.result_path = run_root / "result.json"
        self.result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _bundle(self):
        return load_train_evidence(
            result=self.result,
            result_path=self.result_path,
            run_root=self.run_root,
            heldout_task_ids=self.heldout_ids,
            train_task_ids=self.train_ids,
        )

    def test_diagnosis_uses_train_split_only(self) -> None:
        bundle = self._bundle()
        self.assertEqual(bundle["split"], "train")
        self.assertEqual(
            [item["task_id"] for item in bundle["case_evidence"]], ["train-1", "train-2"]
        )
        brief = render_diagnosis_brief(bundle, heldout_task_ids=self.heldout_ids)
        self.assertIn("Train Diagnosis Brief", brief)
        self.assertIn("train-1", brief)
        self.assertNotIn("heldout-1", brief)
        self.assertNotIn(HELDOUT_SECRET, brief)
        self.assertIn("Train Passing Tasks To Preserve", brief)
        self.assertIn("Held-out Isolation Statement", brief)
        self.assertIn("A repository/domain with many failures is not by itself", brief)

    def test_written_brief_and_evidence_contain_no_heldout_detail(self) -> None:
        bundle = write_diagnosis_brief(
            result=self.result,
            result_path=self.result_path,
            output_path=self.root / "diagnosis" / "brief.md",
            evidence_path=self.root / "diagnosis" / "evidence.json",
            heldout_task_ids=self.heldout_ids,
            train_task_ids=self.train_ids,
            run_root=self.run_root,
        )
        brief = (self.root / "diagnosis" / "brief.md").read_text(encoding="utf-8")
        evidence = (self.root / "diagnosis" / "evidence.json").read_text(encoding="utf-8")
        for text in (brief, evidence):
            self.assertNotIn("heldout-1", text.replace('"heldout_task_ids_validated": 1', ""))
            self.assertNotIn(HELDOUT_SECRET, text)
        self.assertEqual(bundle["isolation"]["heldout_task_ids_validated"], 1)

    def test_heldout_task_in_train_split_is_rejected(self) -> None:
        tampered = json.loads(json.dumps(self.result))
        tampered["splits"]["train"][0]["case_results"][0]["task_id"] = "heldout-1"
        with self.assertRaises(HeldoutLeakError):
            load_train_evidence(
                result=tampered,
                result_path=self.result_path,
                run_root=self.run_root,
                heldout_task_ids=self.heldout_ids,
            )

    def test_unknown_train_task_is_rejected(self) -> None:
        with self.assertRaises(HeldoutLeakError):
            load_train_evidence(
                result=self.result,
                result_path=self.result_path,
                run_root=self.run_root,
                heldout_task_ids=self.heldout_ids,
                train_task_ids=["train-1"],
            )

    def test_missing_evidence_is_marked_not_invented(self) -> None:
        result = json.loads(json.dumps(self.result))
        for case in result["splits"]["train"][0]["case_results"]:
            case["evaluator"]["failure_summary"] = None
            case["error"] = None
            case["artifacts"] = {}
            case["evaluator"]["log_path"] = None
            case["agent"]["trace_path"] = None
        bundle = load_train_evidence(
            result=result,
            result_path=self.result_path,
            run_root=self.run_root,
            heldout_task_ids=self.heldout_ids,
        )
        statuses = {item["evidence_status"] for item in bundle["case_evidence"]}
        self.assertTrue(statuses)
        self.assertTrue(statuses <= {"partial", "missing"})
        self.assertTrue(all(item["evidence_missing_reason"] for item in bundle["case_evidence"]))

    def test_unscoreable_evidence_is_not_reported_as_failure_cluster(self) -> None:
        result = json.loads(json.dumps(self.result))
        for repeat in result["splits"]["train"]:
            case = repeat["case_results"][0]
            case["status"] = "infra_error"
            case["passed"] = None
            case["error"] = "infra boom"
            case["evaluator"]["failure_summary"] = None
            case["evaluator"]["log_path"] = None
            case["artifacts"] = {}
            case["agent"]["trace_path"] = None
        bundle = load_train_evidence(
            result=result,
            result_path=self.result_path,
            run_root=self.run_root,
            heldout_task_ids=self.heldout_ids,
        )
        self.assertTrue(bundle["unscoreable_cases"])
        self.assertTrue(bundle["evidence_gaps"])
        self.assertNotIn(
            "train-1", [item["task_id"] for item in bundle["case_evidence"]]
        )

    def test_proposer_prompt_audit_blocks_heldout_ids(self) -> None:
        bundle = self._bundle()
        brief = render_diagnosis_brief(bundle, heldout_task_ids=self.heldout_ids)
        request = MultiProposerRequest(diagnosis=brief, surfaces=(_surface(),), route_count=2)
        prompt = build_multi_proposer_prompt(request=request)
        audit = audit_text(
            prompt, heldout_task_ids=self.heldout_ids, train_task_ids=self.train_ids,
            label="proposer_prompt",
        )
        self.assertEqual(audit["leaks"], [])
        self.assertNotIn("heldout-1", prompt)
        with self.assertRaises(ProposerHeldoutLeakError):
            audit_text(
                prompt + "\nAlso inspect heldout-1",
                heldout_task_ids=self.heldout_ids,
                label="proposer_prompt",
            )
        assert_no_heldout_leak(prompt, self.heldout_ids)

    def test_proposal_metadata_cannot_reference_heldout_cases(self) -> None:
        proposal = {
            "proposal_id": "p",
            "metadata": {
                "expected_affected_cases": ["train-1"],
                "protected_passing_cases": ["train-2"],
            },
        }
        validate_proposal_metadata(
            proposal=proposal, train_task_ids=self.train_ids, heldout_task_ids=self.heldout_ids
        )
        with self.assertRaises(ProposerHeldoutLeakError):
            validate_proposal_metadata(
                proposal={
                    "proposal_id": "p",
                    "expected_affected_cases": ["heldout-1"],
                },
                train_task_ids=self.train_ids,
                heldout_task_ids=self.heldout_ids,
            )
        with self.assertRaises(ProposerHeldoutLeakError):
            validate_proposal_metadata(
                proposal={"proposal_id": "p", "expected_affected_cases": ["unknown-task"]},
                train_task_ids=self.train_ids,
                heldout_task_ids=self.heldout_ids,
            )

    def test_proposer_cli_refuses_leaked_diagnosis(self) -> None:
        surface_path = self.root / "repo_baseline.py"
        surface_path.write_text(_surface().current_value, encoding="utf-8")
        heldout_file = self.root / "heldout.json"
        heldout_file.write_text(json.dumps(self.heldout_ids), encoding="utf-8")
        leaked_diagnosis = self.root / "leaked.md"
        leaked_diagnosis.write_text("Failure cluster includes heldout-1 details", encoding="utf-8")
        output = self.root / "prompt.txt"
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "proposer" / "scripts" / "run_multi_proposer.py"),
                "--diagnosis",
                str(leaked_diagnosis),
                "--route-count",
                "2",
                "--output",
                str(output),
                "--heldout-ids-file",
                str(heldout_file),
                "--strict-isolation",
                "--surface",
                f"baseline={surface_path}",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("held-out", (completed.stderr or "").lower())


if __name__ == "__main__":
    unittest.main()
