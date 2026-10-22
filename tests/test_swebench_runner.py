"""SWE-bench runner integration tests with local fake fixtures."""

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

from self_harness_eval.config import ConfigError, load_eval_config
from self_harness_eval.dataset import assert_no_protected_fields, load_swebench_dataset
from self_harness_eval.runner_common import ReuseMismatch
from self_harness_eval.dataset import SwebenchInstance
from self_harness_eval.swebench import (
    _evaluator_env,
    _parse_report_entry,
    _validate_harbor_verification,
    run_swebench_eval,
)

from swebench_fixture import (
    GOLD_PATCH_MARKER,
    GOLD_TEST_MARKER,
    build_environment,
    render_swebench_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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
                patch = case.get("patch")
                if isinstance(patch, dict) and patch.get("path", "").startswith("/"):
                    offenders.append("case.patch.path")
                trace = (case.get("agent") or {}).get("trace_path")
                if isinstance(trace, str) and trace.startswith("/"):
                    offenders.append("case.agent.trace_path")
    return offenders


class SwebenchRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="sh-swebench-tests-")
        cls.tmp = Path(cls._tmp.name)
        cls.env = build_environment(cls.tmp / "fixture")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        self.case_tmp = Path(tempfile.mkdtemp(dir=self.tmp))

    def _config(self, *, name: str = "config.toml", **kwargs):
        path = render_swebench_config(
            env=self.env, config_path=self.case_tmp / name, **kwargs
        )
        return load_eval_config(path)

    def _run(self, config, output_name: str = "eval", **kwargs):
        output_dir = self.case_tmp / output_name
        return run_swebench_eval(
            config=config,
            output_dir=output_dir,
            run_id=kwargs.pop("run_id", "test-run"),
            result_id=kwargs.pop("result_id", output_name),
            harness_version_id=kwargs.pop("harness_version_id", output_name),
            **kwargs,
        )

    def test_complete_run_uses_official_shape_and_evaluator_resolved(self) -> None:
        with mock.patch.dict(
            os.environ, {"SELF_HARNESS_FAKE_AGENT_ALWAYS_RESOLVE": "1"}, clear=False
        ):
            result = self._run(self._config(), "baseline")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["schema_version"], "self_harness.eval_result.v1")
        self.assertEqual(result["benchmark"]["id"], "swe_bench_verified")
        self.assertEqual(result["benchmark"]["evaluator"], "fake-command-evaluator")
        self.assertEqual(
            result["splits"]["train"][0]["case_results"][0]["status"], "resolved"
        )
        self.assertEqual(
            result["summary"]["train"]["passed_per_repeat"], [2, 2]
        )
        self.assertEqual(
            result["summary"]["heldout"]["passed_per_repeat"], [1, 1]
        )
        self.assertEqual(result["dataset"]["revision"], "fixture-revision-2024-01-01")
        self.assertEqual(_artifact_paths_are_relative(result), [])
        dataset = load_swebench_dataset(
            name=result["dataset"]["name"],
            revision=result["dataset"]["revision"],
            source=self.env["dataset_path"],
            split="test",
        )
        self.assertTrue(dataset.source_sha256)

    def test_patch_artifact_is_collected(self) -> None:
        result = self._run(self._config(), "patches")
        for split in result["splits"].values():
            for repeat in split:
                for case in repeat["case_results"]:
                    patch = case["patch"]
                    self.assertIsNotNone(patch)
                    patch_path = self.case_tmp / "patches" / patch["path"]
                    self.assertTrue(patch_path.exists())
                    from self_harness_eval.hashing import sha256_file

                    self.assertEqual(patch["sha256"], sha256_file(patch_path))

    def test_passed_only_follows_evaluator_resolved(self) -> None:
        env = dict(os.environ)
        env["SELF_HARNESS_FAKE_RESOLVED_TASKS"] = "astropy__astropy-13398"
        with mock.patch.dict(os.environ, env, clear=False):
            result = self._run(self._config(), "partial-resolved")
        outcomes = {}
        for repeat in result["splits"]["train"]:
            for case in repeat["case_results"]:
                outcomes.setdefault(case["task_id"], set()).add(case["status"])
        self.assertEqual(outcomes["astropy__astropy-13398"], {"resolved"})
        self.assertEqual(outcomes["sympy__sympy-21930"], {"unresolved"})
        self.assertEqual(result["splits"]["heldout"][0]["case_results"][0]["status"], "unresolved")

    def test_agent_timeout_is_a_scoreable_failure(self) -> None:
        env = dict(os.environ)
        env["SELF_HARNESS_FAKE_AGENT_BEHAVIOR"] = "timeout"
        with mock.patch.dict(os.environ, env, clear=False):
            result = self._run(
                self._config(task_timeout_s=1), "timeout"
            )
        self.assertEqual(result["status"], "complete")
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "agent_timeout")
        self.assertFalse(case["passed"])
        self.assertTrue(case["agent"]["timed_out"])
        self.assertEqual(case["evaluator"]["status"], "not_run")
        self.assertIsNone(case["patch"])

    def test_agent_error_records_retry_attempts(self) -> None:
        env = dict(os.environ)
        env["SELF_HARNESS_FAKE_AGENT_BEHAVIOR"] = "error"
        with mock.patch.dict(os.environ, env, clear=False):
            result = self._run(self._config(), "agent-error")
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "agent_error")
        attempts = case["agent"]["attempts"]
        self.assertEqual(len(attempts), 2)  # one retry
        self.assertEqual([item["status"] for item in attempts], ["error", "error"])
        self.assertTrue(all(item["reason"] for item in attempts))
        self.assertIs(case["passed"], False)
        self.assertTrue(case["error"])

    def test_evaluator_failure_is_unscoreable_partial_result(self) -> None:
        config = self._config(name="bad-evaluator.toml")
        broken_evaluator = dataclasses.replace(
            config.evaluator,
            command=(sys.executable, "-c", "import sys; sys.exit(9)"),
        )
        config = dataclasses.replace(config, evaluator=broken_evaluator)
        result = self._run(config, "evaluator-error")
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["errors"])
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "evaluator_error")
        self.assertIsNone(case["passed"])
        self.assertTrue(case["error"])

    def test_missing_dataset_task_becomes_infra_error(self) -> None:
        config = self._config(
            task_ids=("astropy__astropy-13398", "django__django-14017", "astropy__astropy-7336")
        )
        result = self._run(config, "missing-task")
        self.assertEqual(result["status"], "partial")
        cases = {
            case["task_id"]: case
            for repeat in result["splits"]["train"]
            for case in repeat["case_results"]
        }
        self.assertEqual(cases["django__django-14017"]["status"], "infra_error")
        self.assertIsNone(cases["django__django-14017"]["passed"])
        self.assertTrue(cases["django__django-14017"]["error"])

    def test_agent_view_excludes_gold_and_hidden_tests(self) -> None:
        self._run(self._config(), "gold-check")
        case_dirs = list((self.case_tmp / "gold-check" / "splits").rglob("cases/*"))
        self.assertTrue(case_dirs)
        for case_dir in case_dirs:
            instruction = (case_dir / "instruction.md").read_text(encoding="utf-8")
            agent_input = json.loads((case_dir / "agent_input.json").read_text(encoding="utf-8"))
            assert_no_protected_fields(agent_input)
            for forbidden in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "resolved"):
                self.assertNotIn(forbidden, agent_input)
            self.assertNotIn(GOLD_PATCH_MARKER, instruction)
            self.assertNotIn(GOLD_TEST_MARKER, instruction)
            agent_env = json.loads((case_dir / "agent_env.json").read_text(encoding="utf-8"))
            self.assertTrue(agent_env["candidate_workspace"])
            self.assertEqual(agent_env["tool_timeout_s"], "10")

    def test_dataset_agent_view_has_no_protected_fields(self) -> None:
        dataset = load_swebench_dataset(
            name="princeton-nlp/SWE-bench_Verified",
            revision="fixture-revision-2024-01-01",
            source=self.env["dataset_path"],
            split="test",
        )
        for instance in dataset.instances.values():
            view = instance.agent_view()
            assert_no_protected_fields(view)
            self.assertNotIn("patch", view)
            self.assertNotIn("test_patch", view)

    @unittest.skipUnless(
        __import__("tests.test_deepagent_adapter", fromlist=["DOCKER_READY"]).DOCKER_READY,
        "Docker/python:3.12 image required for the DeepAgent runner integration test",
    )
    def test_deepagent_adapter_integration_with_runner_and_fake_evaluator(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        calls: list[int] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: ANN002
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                self.rfile.read(length)
                calls.append(len(calls) + 1)
                if len(calls) == 1:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "execute",
                                    "arguments": __import__("json").dumps(
                                        {
                                            "command": "printf '%s\\n' '# SOLVED' >> /testbed/solution.py"
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": "done"}
                    finish = "stop"
                data = __import__("json").dumps(
                    {
                        "id": f"stub-{len(calls)}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "deepseek-flash",
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = self._config(
                name="deepagent-runner.toml",
                task_ids=("astropy__astropy-13398",),
                repeats=1,
                case_concurrency=1,
            )
            adapter_command = (
                "{repo_root}/.venv/bin/python {repo_root}/eval/scripts/run_deepagent_swebench.py "
                "--instruction {instruction_file} --workspace {workspace} --case-dir {case_dir} "
                "--task-id {task_id} --repo {repo} --base-commit {base_commit} "
                "--split {split} --repeat {repeat}"
            )
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(
                    config.model, base_url=f"http://127.0.0.1:{server.server_port}/v1"
                ),
                harness=dataclasses.replace(
                    config.harness,
                    baseline_dir=REPO_ROOT / "eval" / "harness_workspace",
                ),
                agent=dataclasses.replace(
                    config.agent,
                    command=adapter_command,
                    env_passthrough=(
                        "SELF_HARNESS_MODEL_API_KEY",
                        "SELF_HARNESS_MODEL",
                        "OPENAI_BASE_URL",
                    ),
                ),
            )
            env = {
                "SELF_HARNESS_MODEL_API_KEY": "fake-runner-key-not-real",
                "SELF_HARNESS_MODEL": "deepseek-flash",
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "SELF_HARNESS_FAKE_RESOLVED_TASKS": "astropy__astropy-13398",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                result = self._run(config, "deepagent-runner", splits=("train",))
            case = result["splits"]["train"][0]["case_results"][0]
            self.assertEqual(case["status"], "resolved")
            self.assertTrue(case["agent"]["trace_path"])
            self.assertIsNotNone(case["patch"])
            self.assertEqual(case["evaluator"]["name"], "fake-command-evaluator")
            result_json = (self.case_tmp / "deepagent-runner" / "result.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("fake-runner-key-not-real", result_json)
            trace = (
                self.case_tmp
                / "deepagent-runner"
                / str(case["agent"]["trace_path"])
            ).read_text(encoding="utf-8")
            self.assertNotIn("fake-runner-key-not-real", trace)
            self.assertIn("agent_runtime", case)
        finally:
            server.shutdown()

    def test_split_and_task_filtering(self) -> None:
        result = self._run(
            self._config(),
            "filtered",
            splits=("train",),
            task_ids=("astropy__astropy-13398",),
            repeats=1,
        )
        self.assertEqual(set(result["splits"]), {"train"})
        self.assertEqual(len(result["splits"]["train"]), 1)
        self.assertEqual(
            [case["task_id"] for case in result["splits"]["train"][0]["case_results"]],
            ["astropy__astropy-13398"],
        )
        self.assertEqual(result["config"]["repeats"], 1)

    def test_case_level_resume_reuses_matching_cases(self) -> None:
        config = self._config()
        self._run(config, "resume")
        result_path = self.case_tmp / "resume" / "result.json"
        result_path.unlink()
        second = self._run(config, "resume", reuse_existing=True)
        self.assertEqual(second["status"], "complete")
        resumed = [
            case
            for split in second["splits"].values()
            for repeat in split
            for case in repeat["case_results"]
            if (case.get("legacy") or {}).get("resumed")
        ]
        self.assertTrue(resumed)

    def test_reuse_existing_rejects_changed_provenance(self) -> None:
        output = "reuse-validation"
        self._run(self._config(name="first.toml"), output)
        changed = self._config(name="second.toml")
        changed = dataclasses.replace(
            changed,
            model=dataclasses.replace(changed.model, model_id="different-model"),
        )
        with self.assertRaises(ReuseMismatch):
            self._run(changed, output, reuse_existing=True)

    def test_official_report_parsing_supports_classic_and_current_shapes(self) -> None:
        task_id = "astropy__astropy-13398"
        per_instance = {
            task_id: {
                "resolved": True,
                "patch_successfully_applied": True,
                "tests_status": {"FAIL_TO_PASS": {"success": ["t"], "failure": []}},
            }
        }
        self.assertTrue(_parse_report_entry(per_instance, task_id)["resolved"])
        summary = {
            "resolved_ids": [task_id],
            "unresolved_ids": [],
            "error_ids": [],
            "completed_ids": [task_id],
            "failure_reasons": {},
        }
        self.assertTrue(_parse_report_entry(summary, task_id)["resolved"])
        summary["resolved_ids"] = []
        summary["unresolved_ids"] = [task_id]
        self.assertFalse(_parse_report_entry(summary, task_id)["resolved"])
        summary["error_ids"] = [task_id]
        self.assertIsNone(_parse_report_entry(summary, task_id))
        self.assertIsNone(_parse_report_entry({"nothing": 1}, task_id))

    def test_evaluator_environment_never_contains_model_credentials(self) -> None:
        env = dict(os.environ)
        env.update(
            {
                "SELF_HARNESS_MODEL_API_KEY": "should-not-appear",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
                "SELF_HARNESS_MODEL": "deepseek-flash",
            }
        )
        with mock.patch.dict(os.environ, env, clear=False):
            evaluator_env = _evaluator_env()
        self.assertNotIn("SELF_HARNESS_MODEL_API_KEY", evaluator_env)
        self.assertNotIn("should-not-appear", str(evaluator_env))
        self.assertNotIn("OPENAI_BASE_URL", evaluator_env)
        self.assertNotIn("SELF_HARNESS_MODEL", evaluator_env)

    def test_harbor_workspace_verification_is_required(self) -> None:
        instance = SwebenchInstance(
            task_id="astropy__astropy-13398",
            repo="astropy/astropy",
            base_commit="a" * 40,
            problem_statement="fixture",
        )
        self.assertIn(
            "must write harbor_workspace_verification.json",
            _validate_harbor_verification(None, instance=instance) or "",
        )
        self.assertIn(
            "does not match",
            _validate_harbor_verification(
                {"base_commit": "b" * 40, "head_commit": "b" * 40}, instance=instance
            )
            or "",
        )
        self.assertIsNone(
            _validate_harbor_verification(
                {"base_commit": "a" * 40, "head_commit": "a" * 40}, instance=instance
            )
        )

    def test_fixture_evaluator_requires_smoke_test(self) -> None:
        path = render_swebench_config(
            env=self.env,
            config_path=self.case_tmp / "fixture-eval.toml",
            evaluator_kind="fixture",
            smoke_test=False,
        )
        with self.assertRaises(ConfigError):
            load_eval_config(path)

    def test_official_evaluator_adapter_uses_report_resolved(self) -> None:
        path = render_swebench_config(
            env=self.env,
            config_path=self.case_tmp / "official.toml",
            evaluator_kind="official",
            smoke_test=True,
        )
        previous_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(self.env["fake_swebench_pkg"]), previous_pythonpath or ""]
        ).strip(os.pathsep)
        sys.path.insert(0, str(self.env["fake_swebench_pkg"]))
        try:
            with mock.patch(
                "self_harness_eval.swebench.shutil.which", return_value="/usr/bin/docker"
            ):
                with mock.patch.dict(
                    os.environ, {"SELF_HARNESS_FAKE_AGENT_ALWAYS_RESOLVE": "1"}, clear=False
                ):
                    result = self._run(load_eval_config(path), "official-resolved")
                with mock.patch.dict(
                    os.environ, {"SELF_HARNESS_FAKE_AGENT_ALWAYS_RESOLVE": "0"}, clear=False
                ):
                    unresolved = self._run(load_eval_config(path), "official-unresolved")
        finally:
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath
            if str(self.env["fake_swebench_pkg"]) in sys.path:
                sys.path.remove(str(self.env["fake_swebench_pkg"]))
        self.assertEqual(result["benchmark"]["evaluator"], "swebench-official-harness")
        self.assertEqual(
            result["splits"]["train"][0]["case_results"][0]["status"], "resolved"
        )
        self.assertEqual(
            unresolved["splits"]["train"][0]["case_results"][0]["status"], "unresolved"
        )

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
                "swe-dispatched",
                "--split",
                "train",
                "--task-id",
                "astropy__astropy-13398",
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
        self.assertEqual(result["result_id"], "swe-dispatched")
        self.assertEqual(set(result["splits"]), {"train"})
        self.assertEqual(
            [case["task_id"] for case in result["splits"]["train"][0]["case_results"]],
            ["astropy__astropy-13398"],
        )

    def test_shipped_example_config_loads(self) -> None:
        env = {
            "SELF_HARNESS_MODEL": "fixture-model",
            "SELF_HARNESS_SWEBENCH_REVISION": "fixture-revision",
            "SELF_HARNESS_SWEBENCH_DATA": str(self.env["dataset_path"]),
            "SELF_HARNESS_REPO_CACHE": str(self.env["repos_dir"]),
            "SELF_HARNESS_AGENT_COMMAND": f"{sys.executable} {self.env['fake_agent']} "
            "--workspace {workspace} --case-dir {case_dir} --instruction {instruction_file}",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            config = load_eval_config(
                REPO_ROOT / "eval" / "configs" / "swebench_verified_100.example.toml"
            )
        self.assertEqual(config.eval.benchmark, "swe_bench_verified")
        self.assertEqual(config.eval.task_timeout_s, 3600)
        self.assertEqual(config.eval.tool_timeout_s, 120)
        self.assertEqual(config.eval.repeats, 2)
        self.assertEqual(
            config.dataset.manifest_path,
            REPO_ROOT / "swe_bench_verified_stratified_100.json",
        )

    def test_official_evaluator_missing_dependency_fails_result(self) -> None:
        path = render_swebench_config(
            env=self.env,
            config_path=self.case_tmp / "official-missing.toml",
            evaluator_kind="official",
            smoke_test=True,
        )
        with mock.patch(
            "self_harness_eval.swebench._evaluator_preflight_error",
            return_value="official swebench evaluator unavailable in test",
        ):
            result = self._run(load_eval_config(path), "official-failed")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["errors"])
        case = result["splits"]["train"][0]["case_results"][0]
        self.assertEqual(case["status"], "not_run")
        self.assertIsNone(case["passed"])


if __name__ == "__main__":
    unittest.main()
