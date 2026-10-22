"""End-to-end workflow tests: snapshots, versions, events, gates, reports, resume."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from self_harness_eval.config import load_eval_config
from self_harness_workflow import Orchestrator, ResumeMismatch
from self_harness_workflow.events import REQUIRED_EVENT_TYPES, read_events

from swebench_fixture import build_environment, render_swebench_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="sh-workflow-tests-")
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.case_tmp = Path(tempfile.mkdtemp(dir=self.tmp))
        self.env = build_environment(self.case_tmp / "fixture")
        config_path = render_swebench_config(
            env=self.env, config_path=self.case_tmp / "config.toml", repeats=2
        )
        self.config = load_eval_config(config_path)

    def _orchestrator(self, *, run_id: str, config=None, reuse_existing: bool = False) -> Orchestrator:
        return Orchestrator(
            config=config or self.config,
            run_dir=self.case_tmp / "runs" / run_id,
            run_id=run_id,
            reuse_existing=reuse_existing,
        )

    def test_full_run_produces_auditable_artifacts(self) -> None:
        orchestrator = self._orchestrator(run_id="full-run")
        orchestrator.cmd_init()
        summary = orchestrator.cmd_run(max_rounds=1)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["stop_reason"], "max_rounds_reached")
        run_root = orchestrator.paths.root

        versions = sorted((run_root / "versions").iterdir())
        version_ids = [path.name for path in versions]
        self.assertGreaterEqual(len(version_ids), 4)
        self.assertEqual(version_ids[0], "v000_baseline")
        self.assertTrue(any("merged" in version_id for version_id in version_ids))

        snapshots: set[str] = set()
        for version_dir in versions:
            record = json.loads((version_dir / "version.json").read_text(encoding="utf-8"))
            self.assertEqual(record["format"], "self_harness.version.v1")
            self.assertEqual(record["run_id"], "full-run")
            self.assertIn(record["status"], {"evaluated", "accepted", "rejected", "merged", "final"})
            self.assertIn("snapshot_sha256", record["harness"])
            self.assertTrue(record["harness"]["files"])
            for file_entry in record["harness"]["files"]:
                self.assertFalse(file_entry["path"].startswith("/"))
            snapshots.add(record["harness"]["snapshot_sha256"])
            self.assertTrue((run_root / record["harness_snapshot_path"]).is_dir())
            if record["kind"] == "baseline":
                self.assertEqual(record["parent_version_ids"], [])
            else:
                self.assertTrue(record["parent_version_ids"])
            if record["kind"] in {"candidate", "merged"}:
                self.assertTrue((version_dir / "eval" / "result.json").exists())
                self.assertTrue((version_dir / "acceptance.json").exists())
        self.assertGreater(len(snapshots), 1)
        baseline = json.loads(
            (run_root / "versions" / "v000_baseline" / "version.json").read_text(encoding="utf-8")
        )
        merged_candidates = [
            json.loads((run_root / "versions" / version_id / "version.json").read_text(encoding="utf-8"))
            for version_id in version_ids
            if "merged" in version_id
        ]
        self.assertEqual(len(merged_candidates), 1)
        merged = merged_candidates[0]
        self.assertEqual(merged["kind"], "merged")
        self.assertIn(baseline["version_id"], merged["lineage"])
        self.assertGreaterEqual(len(merged["parent_version_ids"]), 3)

        events = read_events(orchestrator.paths.events)
        event_types = {event["event_type"] for event in events}
        required_here = [name for name in REQUIRED_EVENT_TYPES if name != "candidate_rejected"]
        for required in required_here:
            self.assertIn(required, event_types)
        for event in events:
            self.assertIn("timestamp", event)
            self.assertIn("artifacts", event)
            for artifact in event["artifacts"].values():
                if artifact:
                    self.assertFalse(str(artifact).startswith("/"))

        diagnosis_text = (
            run_root / "versions" / "v000_baseline" / "diagnosis" / "diagnosis.md"
        ).read_text(encoding="utf-8")
        prompt_text = (run_root / "rounds" / "round-001" / "proposer_prompt.txt").read_text(
            encoding="utf-8"
        )
        heldout_ids = json.loads(
            (run_root / "dataset" / "heldout_task_ids.json").read_text(encoding="utf-8")
        )
        for heldout_id in heldout_ids:
            self.assertNotIn(heldout_id, diagnosis_text)
            self.assertNotIn(heldout_id, prompt_text)
            self.assertNotIn(heldout_id, orchestrator.paths.events.read_text(encoding="utf-8"))

        report = json.loads(orchestrator.paths.final_report_json.read_text(encoding="utf-8"))
        for key in (
            "run_id",
            "benchmark",
            "fixed_parameters",
            "baseline",
            "final",
            "splits",
            "repository_domain",
            "candidates",
            "task_transitions",
            "versions",
            "totals",
            "errors",
            "scope",
            "not_executed",
            "claims",
        ):
            self.assertIn(key, report)
        self.assertEqual(report["run_id"], "full-run")
        self.assertIn("task_manifest", report)
        self.assertEqual(len(report["task_manifest"]["splits"]["train"]), 2)
        self.assertEqual(len(report["task_manifest"]["splits"]["heldout"]), 1)
        self.assertIn("Task Manifest (actually evaluated)", orchestrator.paths.final_report_md.read_text(encoding="utf-8"))
        self.assertFalse(report["claims"]["reproduces_paper_results"])
        self.assertTrue(report["claims"]["scores_are_measured_from_results"])
        self.assertEqual(report["baseline"]["splits"]["train"]["mean_repeat_pass_rate"], 0.0)
        self.assertEqual(report["final"]["splits"]["train"]["mean_repeat_pass_rate"], 1.0)
        self.assertTrue(orchestrator.paths.final_report_md.exists())
        self.assertEqual(report["final"]["version_id"], summary["final_version_id"])
        candidate_versions = [item for item in report["candidates"] if item["kind"] == "candidate"]
        self.assertEqual(len(candidate_versions), 2)
        for candidate in candidate_versions:
            self.assertIn("proposal_summary", candidate)
            self.assertIn("deltas", candidate)
            self.assertEqual(candidate["acceptance_decision"], "accepted")

    def test_decline_proposals_stop_with_no_candidates(self) -> None:
        with mock.patch.dict(
            os.environ, {"SELF_HARNESS_FAKE_PROPOSER_MODE": "decline"}, clear=False
        ):
            orchestrator = self._orchestrator(run_id="decline-run")
            orchestrator.cmd_init()
            summary = orchestrator.cmd_run(max_rounds=2)
        self.assertEqual(summary["stop_reason"], "no_candidates")
        version_dirs = [path for path in (orchestrator.paths.root / "versions").iterdir()]
        self.assertEqual(len(version_dirs), 1)
        self.assertTrue((orchestrator.paths.final_report_json).exists())

    def test_duplicate_proposals_are_reported_not_silently_dropped(self) -> None:
        with mock.patch.dict(
            os.environ, {"SELF_HARNESS_FAKE_PROPOSER_MODE": "duplicate"}, clear=False
        ):
            orchestrator = self._orchestrator(run_id="duplicate-run")
            orchestrator.cmd_init()
            orchestrator.ensure_baseline()
            orchestrator.run_round()
        self.assertGreaterEqual(orchestrator.run_manifest["state"]["proposer_failures"], 1)
        self.assertTrue(
            (orchestrator.paths.root / "rounds" / "round-001" / "proposer_failure.json").exists()
        )
        self.assertIn(
            "proposer_failed",
            {event["event_type"] for event in read_events(orchestrator.paths.events)},
        )
        candidate_versions = [
            path for path in (orchestrator.paths.root / "versions").iterdir() if "candidate" in path.name
        ]
        self.assertEqual(candidate_versions, [])

    def test_resume_rejects_changed_model(self) -> None:
        orchestrator = self._orchestrator(run_id="resume-model")
        orchestrator.cmd_init()
        orchestrator.ensure_baseline()
        changed = dataclasses.replace(
            self.config,
            model=dataclasses.replace(self.config.model, model_id="changed-model"),
        )
        with self.assertRaises(ResumeMismatch):
            self._orchestrator(run_id="resume-model", config=changed)

    def test_resume_rejects_changed_dataset_revision(self) -> None:
        orchestrator = self._orchestrator(run_id="resume-dataset")
        orchestrator.cmd_init()
        orchestrator.ensure_baseline()
        changed = dataclasses.replace(
            self.config,
            dataset=dataclasses.replace(self.config.dataset, revision="different-revision"),
        )
        with self.assertRaises(ResumeMismatch):
            self._orchestrator(run_id="resume-dataset", config=changed)

    def test_resume_rejects_tampered_snapshot(self) -> None:
        orchestrator = self._orchestrator(run_id="resume-snapshot")
        orchestrator.cmd_init()
        orchestrator.ensure_baseline()
        snapshot_file = (
            orchestrator.paths.root / "versions" / "v000_baseline" / "harness" / "repo_baseline.py"
        )
        snapshot_file.write_text(snapshot_file.read_text(encoding="utf-8") + "\n# tampered\n")
        with self.assertRaises(ResumeMismatch):
            self._orchestrator(run_id="resume-snapshot")

    def test_reuse_existing_result_for_completed_version(self) -> None:
        orchestrator = self._orchestrator(run_id="reuse-run")
        orchestrator.cmd_init()
        baseline_id = orchestrator.ensure_baseline()
        result_path = orchestrator.paths.version_result(baseline_id)
        before = result_path.read_bytes()
        summary = orchestrator.cmd_eval_version(baseline_id)
        self.assertTrue(summary["reused"])
        self.assertEqual(result_path.read_bytes(), before)

    def _config_with_evaluator_command(self, command: tuple[str, ...], *, name: str) -> object:
        from shlex import join as shell_join

        path = render_swebench_config(
            env=self.env,
            config_path=self.case_tmp / name,
            evaluator_kind="command",
        )
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        in_evaluator = False
        for index, line in enumerate(lines):
            if line.strip() == "[evaluator]":
                in_evaluator = True
                continue
            if in_evaluator and line.startswith("command = "):
                lines[index] = f"command = {json.dumps(shell_join(command))}"
                break
            if in_evaluator and line.startswith("[") and line.strip() != "[evaluator]":
                break
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return load_eval_config(path)

    def test_baseline_failure_stops_run_and_writes_report(self) -> None:
        config = self._config_with_evaluator_command(
            (sys.executable, "-c", "import sys; sys.exit(9)"),
            name="broken-evaluator.toml",
        )
        orchestrator = self._orchestrator(run_id="baseline-failure", config=config)
        orchestrator.cmd_init()
        summary = orchestrator.cmd_run(max_rounds=1)
        self.assertEqual(summary["stop_reason"], "baseline_not_complete")
        self.assertTrue(orchestrator.paths.final_report_json.exists())
        baseline_result = json.loads(
            (orchestrator.paths.root / "versions" / "v000_baseline" / "eval" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(baseline_result["status"], "partial")

    def test_candidate_evaluation_failure_is_recorded(self) -> None:
        evaluator_script = self.case_tmp / "conditional_evaluator.py"
        evaluator_script.write_text(
            "import argparse, json, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--predictions')\n"
            "parser.add_argument('--output')\n"
            "args = parser.parse_args()\n"
            "preds = [json.loads(line) for line in open(args.predictions) if line.strip()]\n"
            "if any('# SOLVED' in (p.get('model_patch') or '') "
            "and '# NOT_RESOLVED' not in (p.get('model_patch') or '') for p in preds):\n"
            "    print('official evaluator crashed', file=sys.stderr)\n"
            "    raise SystemExit(9)\n"
            "report = {p['instance_id']: {'resolved': False, 'error': 'fixture unsolved'}\n"
            "          for p in preds}\n"
            "json.dump(report, open(args.output, 'w'))\n",
            encoding="utf-8",
        )
        config = self._config_with_evaluator_command(
            (
                sys.executable,
                str(evaluator_script),
                "--predictions",
                "{predictions}",
                "--output",
                "{evaluation_output}",
            ),
            name="conditional-evaluator.toml",
        )
        orchestrator = self._orchestrator(run_id="candidate-failure", config=config)
        orchestrator.cmd_init()
        orchestrator.ensure_baseline()
        orchestrator.run_round()
        candidate_versions = [
            path for path in (orchestrator.paths.root / "versions").iterdir() if "candidate" in path.name
        ]
        self.assertEqual(len(candidate_versions), 2)
        for version_dir in candidate_versions:
            result = json.loads((version_dir / "eval" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "partial")
            acceptance = json.loads((version_dir / "acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(acceptance["decision"], "inconclusive")
            self.assertFalse(acceptance["accepted"])

    def test_candidate_budget_zero_stops_before_round(self) -> None:
        config = dataclasses.replace(
            self.config,
            run=dataclasses.replace(self.config.run, max_candidate_evaluations=0),
        )
        orchestrator = self._orchestrator(run_id="budget-run", config=config)
        orchestrator.cmd_init()
        summary = orchestrator.cmd_run(max_rounds=2)
        self.assertEqual(summary["stop_reason"], "candidate_budget_exhausted")

    def test_no_improvement_stop_condition(self) -> None:
        config = dataclasses.replace(
            self.config,
            run=dataclasses.replace(
                self.config.run, max_rounds=5, max_consecutive_no_improvement=1
            ),
        )
        with mock.patch.dict(
            os.environ, {"SELF_HARNESS_FAKE_AGENT_ALWAYS_RESOLVE": "1"}, clear=False
        ):
            orchestrator = self._orchestrator(run_id="no-improvement", config=config)
            orchestrator.cmd_init()
            summary = orchestrator.cmd_run()
        self.assertEqual(summary["stop_reason"], "no_improvement")
        self.assertEqual(orchestrator.run_manifest["state"]["consecutive_no_improvement"], 1)

    def test_cli_run_command_writes_report(self) -> None:
        summary = self._run_cli(
            "run", "--run-id", "cli-run", "--max-rounds", "1", "--json"
        )
        self.assertEqual(summary["stop_reason"], "max_rounds_reached")
        run_dir = self.case_tmp / "fixture" / "runs" / "cli-run"
        self.assertTrue((run_dir / "final_report.md").exists())
        self.assertTrue((run_dir / "final_report.json").exists())

    def _run_cli(self, *args: str) -> dict:
        env = dict(os.environ)
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    [
                        "eval/src",
                        "eval/fixtures",
                        "diagnosis/src",
                        "proposer/src",
                        "acceptance/src",
                        "workflow/src",
                    ]
                )
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "workflow" / "scripts" / "run_self_harness.py"),
                *args,
                "--config",
                str(self.case_tmp / "config.toml"),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)


if __name__ == "__main__":
    unittest.main()
