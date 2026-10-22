"""SWE-bench pilot selection, validation, summary and blocked-pilot tests."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from self_harness_eval.config import load_eval_config
from self_harness_eval.manifest import load_swebench_manifest
from self_harness_eval.pilot import (
    PILOT_MAX_TASKS,
    PilotError,
    pilot_manifest_payload,
    render_pilot_markdown,
    select_pilot_tasks,
    summarize_pilot_result,
    validate_pilot_task_ids,
    write_pilot_outputs,
)

from tests.helpers import make_result

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "Self-Harness-data"
API_KEY_ENV = "SELF_HARNESS_MODEL_API_KEY"

PILOT_IDS = (
    "astropy__astropy-13398",
    "django__django-14017",
    "matplotlib__matplotlib-24627",
    "pydata__xarray-6461",
    "sphinx-doc__sphinx-8548",
)


class PilotSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = load_swebench_manifest()
        self.manifest_hash = self.manifest.sha256

    def test_deterministic_five_task_selection(self) -> None:
        first = select_pilot_tasks(self.manifest)
        second = select_pilot_tasks(self.manifest)
        self.assertEqual(first, second)
        self.assertEqual(len(first), PILOT_MAX_TASKS)
        self.assertEqual([task["task_id"] for task in first], list(PILOT_IDS))
        self.assertTrue(all(task["split"] == "train" for task in first))
        self.assertEqual(len({task["repository"] for task in first}), 5)
        self.assertEqual(len({task["domain"] for task in first}), 5)
        self.assertFalse({task["task_id"] for task in first} & set(self.manifest.held_out_task_ids))
        self.assertEqual(self.manifest.sha256, self.manifest_hash)

    def test_validation_accepts_subset_and_rejects_invalid(self) -> None:
        subset = validate_pilot_task_ids(self.manifest, PILOT_IDS[:2])
        self.assertEqual([task["task_id"] for task in subset], list(PILOT_IDS[:2]))
        with self.assertRaises(PilotError):
            validate_pilot_task_ids(self.manifest, PILOT_IDS[:2] + (PILOT_IDS[0],))
        with self.assertRaises(PilotError):
            validate_pilot_task_ids(self.manifest, tuple(PILOT_IDS) + ("django__django-14017-extra",))
        with self.assertRaises(PilotError):
            validate_pilot_task_ids(self.manifest, (PILOT_IDS[0], "not-a-real-task"))
        heldout = self.manifest.held_out_task_ids[0]
        with self.assertRaises(PilotError):
            validate_pilot_task_ids(self.manifest, (heldout,))
        with self.assertRaises(PilotError):
            validate_pilot_task_ids(self.manifest, ())
        six = [task_id for task_id in self.manifest.held_in_task_ids[:6]]
        with self.assertRaises(PilotError):
            validate_pilot_task_ids(self.manifest, six)

    def test_pilot_payload_is_explicitly_not_a_benchmark(self) -> None:
        tasks = select_pilot_tasks(self.manifest)
        payload = pilot_manifest_payload(self.manifest, tasks, repeats=2)
        self.assertFalse(payload["heldout_run"])
        self.assertFalse(payload["candidate_iteration"])
        self.assertEqual(payload["max_tasks"], 5)
        self.assertEqual(payload["task_count"], 5)
        self.assertEqual(payload["repeats"], 2)
        self.assertTrue(any("100-task" in item for item in payload["limitations"]))
        self.assertTrue(any("DeepSeek Flash" in item for item in payload["limitations"]))

    def test_pilot_config_fixed_parameters(self) -> None:
        config = load_eval_config(REPO_ROOT / "eval" / "configs" / "swebench_pilot_5.toml")
        self.assertEqual(config.eval.benchmark, "swe_bench_verified")
        self.assertTrue(config.eval.pilot)
        self.assertEqual(config.eval.splits, ("train",))
        self.assertEqual(config.eval.task_ids, PILOT_IDS)
        self.assertEqual(config.eval.repeats, 2)
        self.assertEqual(config.eval.task_timeout_s, 3600)
        self.assertEqual(config.eval.tool_timeout_s, 120)
        self.assertEqual(config.eval.case_concurrency, 1)
        self.assertEqual(config.model.model_id, "deepseek-flash")
        self.assertEqual(config.model.base_url, "https://api.qadg.xyz/v1")
        self.assertEqual(config.model.temperature, 0.0)
        self.assertEqual(config.model.top_p, 1.0)
        self.assertEqual(config.model.max_tokens, 8192)
        self.assertEqual(config.run.proposal_width, 4)
        self.assertEqual(config.run.max_rounds, 5)
        self.assertEqual(config.run.max_candidate_evaluations, 8)
        self.assertEqual(config.dataset.revision, "c104f840cc67f8b6eec6f759ebc8b2693d585d4a")
        self.assertEqual(config.evaluator.kind, "official")
        self.assertIn("run_deepagent_swebench.py", str(config.agent.command))
        self.assertIn("run_openai_compatible_proposer.py", str(config.run.proposer_command))

    def test_full_example_config_fixed_parameters(self) -> None:
        config = load_eval_config(
            REPO_ROOT / "eval" / "configs" / "swebench_verified_100.example.toml"
        )
        self.assertFalse(config.eval.pilot)
        self.assertEqual(config.eval.splits, ("train", "heldout"))
        self.assertEqual(config.model.model_id, "deepseek-flash")
        self.assertEqual(config.model.base_url, "https://api.qadg.xyz/v1")
        self.assertEqual(config.run.proposal_width, 4)
        self.assertEqual(config.run.max_rounds, 5)
        self.assertEqual(config.run.max_candidate_evaluations, 8)


class PilotSummaryTest(unittest.TestCase):
    def test_summary_counts_only_recorded_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _ = make_result(
                root,
                splits={
                    "train": [
                        [
                            ("astropy__astropy-13398", "resolved"),
                            ("django__django-14017", "unresolved"),
                            ("matplotlib__matplotlib-24627", "evaluator_error"),
                        ]
                    ]
                },
            )
            tasks = [
                {
                    "task_id": "astropy__astropy-13398",
                    "repository": "astropy/astropy",
                    "domain": "Scientific_Symbolic",
                    "manifest_position": 0,
                },
                {
                    "task_id": "django__django-14017",
                    "repository": "django/django",
                    "domain": "Web_Networking",
                    "manifest_position": 13,
                },
                {
                    "task_id": "matplotlib__matplotlib-24627",
                    "repository": "matplotlib/matplotlib",
                    "domain": "Visualization",
                    "manifest_position": 45,
                },
            ]
            manifest = load_swebench_manifest()
            payload = pilot_manifest_payload(manifest, tasks, repeats=1)
            summary = summarize_pilot_result(result, pilot=payload)
            self.assertEqual(summary["counts"]["resolved"], 1)
            self.assertEqual(summary["counts"]["unresolved_or_failed"], 1)
            self.assertEqual(summary["counts"]["unscoreable"], 1)
            self.assertEqual(summary["counts"]["case_attempts"], 3)
            self.assertFalse(summary["scope"]["heldout_run"])
            self.assertFalse(summary["scope"]["is_full_benchmark"])
            self.assertFalse(summary["claims"]["is_full_swe_bench_verified"])
            json_path, md_path = write_pilot_outputs(result=result, pilot=payload, output_dir=root)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            markdown = render_pilot_markdown(summary)
            self.assertIn("Pilot only", markdown)
            self.assertIn("`astropy__astropy-13398`", markdown)

    def test_summary_does_not_invent_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = make_result(
                Path(tmp),
                splits={"train": [[("astropy__astropy-13398", "unresolved")]]},
            )
            manifest = load_swebench_manifest()
            tasks = validate_pilot_task_ids(manifest, ("astropy__astropy-13398",))
            payload = pilot_manifest_payload(manifest, tasks, repeats=1)
            summary = summarize_pilot_result(result, pilot=payload)
            self.assertEqual(summary["counts"]["resolved"], 0)
            self.assertNotIn("score", summary)
            self.assertNotIn("benchmark_score", summary)


@unittest.skipUnless(
    (DATA_ROOT / "datasets" / "swe_bench_verified_100.json").exists(),
    "prepared ../Self-Harness-data dataset is required for the blocked-pilot CLI test",
)
class BlockedPilotCliTest(unittest.TestCase):
    def test_pilot_stops_without_api_key_and_writes_blocked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "pilot"
            env = dict(os.environ)
            env.pop(API_KEY_ENV, None)
            completed = subprocess.run(
                [
                    str(REPO_ROOT / ".venv" / "bin" / "python"),
                    str(REPO_ROOT / "eval" / "scripts" / "run_swebench_pilot.py"),
                    "--config",
                    str(REPO_ROOT / "eval" / "configs" / "swebench_pilot_5.toml"),
                    "--output-dir",
                    str(output_dir),
                    "--run-id",
                    "pilot-blocked-test",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(API_KEY_ENV, completed.stderr)
            blocked_path = output_dir / "pilot_blocked.json"
            self.assertTrue(blocked_path.exists())
            blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
            self.assertFalse(blocked["model_calls_made"])
            self.assertFalse(blocked["result_written"])
            self.assertIn(API_KEY_ENV, " ".join(blocked["blockers"]))
            self.assertFalse((output_dir / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
