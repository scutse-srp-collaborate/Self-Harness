"""Regression test for the pre-existing Terminal-Bench workflow loop.

The legacy ``workflow/scripts/run_self_harness_loop.py`` entry point must keep
working with the unified result protocol: it drives the preserved Harbor/pytest
runner CLI, the proposer/materialization scripts and the acceptance gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from swebench_fixture import FAKE_PROPOSER, FIXTURE_HARNESS_SOURCE
from tb2_fixture import build_environment, write_legacy_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class LegacyWorkflowCompatibilityTest(unittest.TestCase):
    def test_legacy_loop_runs_baseline_candidate_and_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sh-legacy-compat-") as tmp_name:
            tmp = Path(tmp_name)
            env = build_environment(tmp / "fixture")
            (env["harness_dir"] / "repo_baseline.py").write_text(
                FIXTURE_HARNESS_SOURCE, encoding="utf-8"
            )
            config_path = write_legacy_config(env, tmp / "legacy.toml")
            diagnosis_path = tmp / "diagnosis.md"
            diagnosis_path.write_text("# legacy diagnosis\n- fixture evidence\n", encoding="utf-8")
            response_path = tmp / "proposer_response.json"
            proposer = subprocess.run(
                [
                    sys.executable,
                    str(FAKE_PROPOSER),
                    "--prompt",
                    str(tmp / "prompt.txt"),
                    "--response",
                    str(response_path),
                    "--width",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proposer.returncode, 0, proposer.stderr)
            work_dir = tmp / "work"
            env_vars = dict(os.environ)
            env_vars["PYTHONPATH"] = os.pathsep.join(
                ["eval/src", "eval/fixtures", "diagnosis/src", "proposer/src", "acceptance/src", "workflow/src"]
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "workflow" / "scripts" / "run_self_harness_loop.py"),
                    "--eval-config",
                    str(config_path),
                    "--work-dir",
                    str(work_dir),
                    "--surface",
                    f"baseline={env['harness_dir'] / 'repo_baseline.py'}",
                    "--diagnosis",
                    str(diagnosis_path),
                    "--proposer-response",
                    str(response_path),
                    "--route-count",
                    "1",
                    "--max-candidates",
                    "1",
                ],
                cwd=str(REPO_ROOT),
                env=env_vars,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            baseline_result = json.loads(
                (work_dir / "baseline_eval" / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(baseline_result["schema_version"], "self_harness.eval_result.v1")
            self.assertEqual(baseline_result["benchmark"]["id"], "terminal_bench_2")
            queue = json.loads(
                (work_dir / "candidate_queue.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(queue["candidates"]), 1)
            candidate = queue["candidates"][0]
            self.assertIn(candidate["status"], {"rejected", "accepted", "accepted_pending_merge"})
            acceptance_path = work_dir / candidate["acceptance_result"]
            acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
            self.assertEqual(acceptance["format"], "self_harness.acceptance_gate.v1")
            self.assertIn("train", acceptance["splits"])
            self.assertIn("heldout", acceptance["splits"])
            # Unified results must not overwrite the legacy aggregate behaviour.
            self.assertTrue((work_dir / "baseline_eval" / "result.legacy.json").exists())


if __name__ == "__main__":
    unittest.main()
