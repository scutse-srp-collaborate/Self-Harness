"""Terminal-Bench runner regression tests (fixture pytest, no Harbor)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from self_harness_eval.config import load_eval_config
from self_harness_eval.runner_common import ReuseMismatch
from self_harness_eval.tb2 import run_tb2_eval

from tb2_fixture import build_environment, render_tb2_config, write_legacy_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class Tb2RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="sh-tb2-tests-")
        cls.tmp = Path(cls._tmp.name)
        cls.env = build_environment(cls.tmp / "fixture")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.case_tmp = Path(tempfile.mkdtemp(dir=self.tmp))

    def _config(self, *, name: str = "config.toml", **kwargs):
        path = render_tb2_config(env=self.env, config_path=self.case_tmp / name, **kwargs)
        return load_eval_config(path)

    def _run(self, config, output_name: str = "eval", **kwargs):
        output_dir = self.case_tmp / output_name
        return run_tb2_eval(
            config=config,
            output_dir=output_dir,
            run_id=kwargs.pop("run_id", "tb2-run"),
            result_id=kwargs.pop("result_id", output_name),
            harness_version_id=kwargs.pop("harness_version_id", output_name),
            **kwargs,
        )

    def test_complete_run_uses_unified_protocol(self) -> None:
        result = self._run(self._config(), "baseline")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["benchmark"]["id"], "terminal_bench_2")
        self.assertEqual(result["benchmark"]["evaluator"], "harbor+pytest-junit")
        self.assertEqual(result["splits"]["train"][0]["status"], "complete")
        self.assertEqual(result["summary"]["train"]["passed_per_repeat"], [2, 2])
        self.assertEqual(result["summary"]["heldout"]["total_per_repeat"], [1, 1])
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "passed")
        self.assertTrue(case["passed"])
        self.assertIsNone(case["patch"])
        self.assertEqual(case["evaluator"]["name"], "harbor+pytest-junit")
        self.assertTrue(case["agent"]["trace_path"])
        self.assertIsNotNone(case["agent"]["tool_calls"])
        self.assertEqual(_artifact_paths_are_relative(result), [])
        # The legacy aggregate remains available for older tooling.
        legacy_path = self.case_tmp / "baseline" / "result.legacy.json"
        self.assertTrue(legacy_path.exists())
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        self.assertEqual(legacy["passed"], 6)

    def test_tool_timeout_and_candidate_workspace_reach_execution_layer(self) -> None:
        result = self._run(self._config(), "env-check")
        case_dir = self.case_tmp / "env-check" / "splits" / "train" / "repeat-01" / "cases" / "extract_elf"
        env_capture = json.loads((case_dir / "env_capture.json").read_text(encoding="utf-8"))
        self.assertEqual(env_capture["SELF_HARNESS_TOOL_TIMEOUT_S"], "15")
        self.assertEqual(env_capture["SELF_HARNESS_AGENT_TIMEOUT_SEC"], "60")
        self.assertTrue(env_capture["SELF_HARNESS_CANDIDATE_WORKSPACE"])
        snapshot = result["harness"]["snapshot_path"]
        self.assertTrue((self.case_tmp / "env-check" / snapshot / "repo_baseline.py").exists())

    def test_failing_verifier_is_failed_not_error(self) -> None:
        with mock.patch.dict(os.environ, {"SELF_HARNESS_FAKE_PYTEST_BEHAVIOR": "fail"}, clear=False):
            result = self._run(self._config(), "fail")
        self.assertEqual(result["status"], "complete")
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "failed")
        self.assertFalse(case["passed"])
        self.assertEqual(case["evaluator"]["status"], "complete")
        self.assertFalse(case["evaluator"]["resolved"])
        self.assertIn("assertion", (case["evaluator"]["failure_summary"] or "").lower())

    def test_skipped_verifier_is_unscoreable(self) -> None:
        with mock.patch.dict(os.environ, {"SELF_HARNESS_FAKE_PYTEST_BEHAVIOR": "skip"}, clear=False):
            result = self._run(self._config(), "skip")
        self.assertEqual(result["status"], "partial")
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "evaluator_error")
        self.assertIsNone(case["passed"])
        self.assertTrue(result["errors"])

    def test_missing_junit_is_unscoreable_not_failed(self) -> None:
        with mock.patch.dict(
            os.environ, {"SELF_HARNESS_FAKE_PYTEST_BEHAVIOR": "no_junit"}, clear=False
        ):
            result = self._run(self._config(), "no-junit")
        self.assertEqual(result["status"], "partial")
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "evaluator_error")
        self.assertIsNone(case["passed"])
        self.assertIn("JUnit", case["error"])

    def test_task_timeout_is_agent_timeout(self) -> None:
        with mock.patch.dict(os.environ, {"SELF_HARNESS_FAKE_PYTEST_BEHAVIOR": "timeout"}, clear=False):
            result = self._run(self._config(task_timeout_s=1), "timeout")
        self.assertEqual(result["status"], "complete")
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "agent_timeout")
        self.assertTrue(case["agent"]["timed_out"])
        self.assertEqual(case["evaluator"]["status"], "not_run")

    def test_bandwidth_limit_enforcement_is_recorded(self) -> None:
        result = self._run(self._config(), "bandwidth")
        network = result["config"]["network_limit"]
        self.assertEqual(network["requested_mbps"], 2.0)
        self.assertIn("enforced", network)
        self.assertFalse(network["enforced"])
        self.assertIn("disabled", network["detail"])

    def test_legacy_config_is_supported(self) -> None:
        path = write_legacy_config(self.env, self.case_tmp / "legacy.toml")
        config = load_eval_config(path)
        result = self._run(config, "legacy")
        self.assertEqual(set(result["splits"]), {"train", "heldout"})
        self.assertEqual(
            [case["task_id"] for case in result["splits"]["train"][0]["case_results"]],
            ["extract_elf"],
        )
        self.assertEqual(
            [case["task_id"] for case in result["splits"]["heldout"][0]["case_results"]],
            ["headless_terminal"],
        )

    def test_shipped_example_configs_load(self) -> None:
        # The TB2 example uses env-injected paths; set the required variables.
        with mock.patch.dict(
            os.environ,
            {
                "SELF_HARNESS_MODEL": "fixture-model",
                "SELF_HARNESS_TB2_EVAL_PROJECT": str(self.env["evals_project"]),
            },
            clear=False,
        ):
            config = load_eval_config(REPO_ROOT / "eval" / "configs" / "tb2_clean64.example.toml")
        self.assertEqual(config.eval.benchmark, "terminal_bench_2")
        self.assertEqual(config.dataset.manifest_path, REPO_ROOT / "eval" / "manifests" / "tb2_clean64.json")
        self.assertEqual(config.eval.task_timeout_s, 900)
        self.assertEqual(config.eval.tool_timeout_s, 120)
        self.assertEqual(config.constraints.outbound_bandwidth_mbps, 2.0)
        with mock.patch.dict(
            os.environ, {"SELF_HARNESS_MODEL": "fixture-model"}, clear=False
        ):
            legacy = load_eval_config(REPO_ROOT / "eval" / "configs" / "harbor_eval.example.toml")
        self.assertEqual(legacy.eval.raw.get("repeats"), 2)
        self.assertEqual(len(legacy.eval.legacy.get("cases") or []), 1)

    def test_dispatcher_cli_runs_configured_subset(self) -> None:
        config = self._config(name="dispatch.toml")
        output_dir = self.case_tmp / "dispatch-eval"
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "scripts" / "run_eval.py"),
                "--config",
                str(config.path),
                "--output-dir",
                str(output_dir),
                "--result-id",
                "dispatched",
                "--split",
                "train",
                "--task-id",
                "extract_elf",
                "--repeats",
                "1",
                "--json",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["result_id"], "dispatched")
        self.assertEqual(set(result["splits"]), {"train"})
        self.assertEqual(
            [case["task_id"] for case in result["splits"]["train"][0]["case_results"]],
            ["extract_elf"],
        )

    def test_legacy_runner_cli_preserves_flags(self) -> None:
        legacy_path = write_legacy_config(self.env, self.case_tmp / "legacy-cli.toml")
        output_dir = self.case_tmp / "legacy-cli"
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "eval" / "scripts" / "run_harbor_eval.py"),
                "--config",
                str(legacy_path),
                "--output-dir",
                str(output_dir),
                "--repeats",
                "1",
                "--reuse-existing",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["schema_version"], "self_harness.eval_result.v1")
        self.assertTrue((output_dir / "result.legacy.json").exists())

    def test_reuse_existing_rejects_changed_timeout(self) -> None:
        output = "reuse-timeout"
        self._run(self._config(name="first.toml"), output)
        changed = self._config(name="second.toml", task_timeout_s=61)
        with self.assertRaises(ReuseMismatch):
            self._run(changed, output, reuse_existing=True)


def _artifact_paths_are_relative(result: dict) -> list[str]:
    offenders: list[str] = []
    for key, value in (result.get("artifacts") or {}).items():
        if isinstance(value, str) and value.startswith("/"):
            offenders.append(f"artifacts.{key}")
    for split in result["splits"].values():
        for repeat in split:
            for case in repeat["case_results"]:
                for key, value in (case.get("artifacts") or {}).items():
                    if isinstance(value, str) and value.startswith("/"):
                        offenders.append(f"case.artifacts.{key}")
                if (case.get("agent") or {}).get("trace_path", "").startswith("/"):
                    offenders.append("trace_path")
    return offenders


if __name__ == "__main__":
    unittest.main()
