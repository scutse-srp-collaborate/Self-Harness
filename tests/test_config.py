"""Configuration loading, defaults and secret handling tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from self_harness_eval.config import ConfigError, load_eval_config, write_resolved_config
from self_harness_eval.dotenv import DISABLE_ENV_VAR, load_dotenv


class DotenvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.environment_snapshot = dict(os.environ)
        self.addCleanup(self._restore_environment)

    def _restore_environment(self) -> None:
        os.environ.clear()
        os.environ.update(self.environment_snapshot)

    def test_loads_names_without_returning_values(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text(
            "# comment\nexport SELF_HARNESS_TEST_KEY=super-secret-value\n"
            "SELF_HARNESS_TEST_MODEL=deepseek-flash\nQUOTED='with space'\n",
            encoding="utf-8",
        )
        os.environ.pop(DISABLE_ENV_VAR, None)
        applied = load_dotenv(env_file)
        self.assertIn("SELF_HARNESS_TEST_KEY", applied)
        self.assertEqual(os.environ["SELF_HARNESS_TEST_KEY"], "super-secret-value")
        self.assertEqual(os.environ["SELF_HARNESS_TEST_MODEL"], "deepseek-flash")
        self.assertEqual(os.environ["QUOTED"], "with space")
        self.assertNotIn("super-secret-value", " ".join(applied))

    def test_does_not_override_existing_environment(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text("SELF_HARNESS_TEST_KEY=file-value\n", encoding="utf-8")
        os.environ.pop(DISABLE_ENV_VAR, None)
        os.environ["SELF_HARNESS_TEST_KEY"] = "process-value"
        applied = load_dotenv(env_file)
        self.assertNotIn("SELF_HARNESS_TEST_KEY", applied)
        self.assertEqual(os.environ["SELF_HARNESS_TEST_KEY"], "process-value")

    def test_disabled_for_tests(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text("SELF_HARNESS_TEST_KEY=file-value\n", encoding="utf-8")
        os.environ["SELF_HARNESS_DISABLE_DOTENV"] = "1"
        self.assertEqual(load_dotenv(env_file), [])


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write(self, text: str, name: str = "config.toml") -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def _minimal(
        self,
        extra: str = "",
        dataset_extra: str = "",
        model_extra: str = "",
        constraints_extra: str = "",
    ) -> str:
        return f"""
[eval]
benchmark = "swe_bench_verified"
name = "config-test"
repeats = 2
split = ["train", "heldout"]
task_timeout_s = 3600
tool_timeout_s = 120
{extra}

[dataset]
name = "princeton-nlp/SWE-bench_Verified"
revision = "pinned-rev"
manifest_path = "{Path(__file__).resolve().parents[1] / 'swe_bench_verified_stratified_100.json'}"
source = "{self.root / 'dataset.json'}"
repos_dir = "{self.root / 'repos'}"
{dataset_extra}

[agent]
kind = "command"
command = "python3 agent.py --workspace {{workspace}}"

[model]
provider = "provider"
model_id = "model"
endpoint_type = "hosted_api"
temperature = 0.0
top_p = 1.0
max_tokens = 512
{model_extra}

[constraints]
{constraints_extra}

[harness]
baseline_dir = "{self.root}"
"""

    def test_swebench_defaults_and_required_revision(self) -> None:
        config = load_eval_config(self._write(self._minimal()))
        self.assertEqual(config.eval.repeats, 2)
        self.assertEqual(config.eval.task_timeout_s, 3600)
        self.assertEqual(config.eval.tool_timeout_s, 120)
        self.assertEqual(config.agent.kind, "command")
        self.assertEqual(config.dataset.revision, "pinned-rev")

    def test_tb2_defaults(self) -> None:
        path = self._write(
            f"""
[eval]
benchmark = "terminal_bench_2"
name = "tb2"
evals_project = "{self.root}"
harness_workspace = "{self.root}"
agent_import_path = "module:Agent"
[model]
provider = "provider"
model_id = "model"
endpoint_type = "hosted_api"
[dataset]
manifest_path = "{Path(__file__).resolve().parents[1] / 'eval' / 'manifests' / 'tb2_clean64.json'}"

[harness]
baseline_dir = "{self.root}"
"""
        )
        config = load_eval_config(path)
        self.assertEqual(config.eval.task_timeout_s, 900)
        self.assertEqual(config.eval.tool_timeout_s, 120)
        self.assertEqual(config.eval.repeats, 2)
        self.assertEqual(config.agent.kind, "harbor")

    def test_missing_revision_is_rejected(self) -> None:
        path = self._write(self._minimal().replace('revision = "pinned-rev"\n', ""))
        with self.assertRaises(ConfigError):
            load_eval_config(path)

    def test_missing_environment_variable_is_rejected(self) -> None:
        text = self._minimal(model_extra='model_id = "${SELF_HARNESS_TEST_MISSING_VAR}"')
        text = text.replace('provider = "provider"\nmodel_id = "model"', 'provider = "provider"')
        path = self._write(text)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SELF_HARNESS_TEST_MISSING_VAR", None)
            with self.assertRaises(ConfigError):
                load_eval_config(path)

    def test_secret_config_keys_are_rejected(self) -> None:
        path = self._write(
            self._minimal(dataset_extra='OPENAI_API_KEY = "sk-should-not-be-in-config"')
        )
        with self.assertRaises(ConfigError):
            load_eval_config(path)
        path = self._write(
            self._minimal(model_extra='password = "plain-text-password"'),
            name="password.toml",
        )
        with self.assertRaises(ConfigError):
            load_eval_config(path)

    def test_resolved_config_redacts_secret_environment_values(self) -> None:
        text = self._minimal(
            constraints_extra='cache_dir = "${SELF_HARNESS_SECRET_TOKEN}/cache"'
        )
        path = self._write(text)
        with mock.patch.dict(
            os.environ, {"SELF_HARNESS_SECRET_TOKEN": "super-secret-token-value"}, clear=False
        ):
            config = load_eval_config(path)
            self.assertIn("SELF_HARNESS_SECRET_TOKEN", config.env_refs)
            resolved_path = self.root / "config.resolved.json"
            write_resolved_config(resolved_path, config)
        resolved = resolved_path.read_text(encoding="utf-8")
        self.assertNotIn("super-secret-token-value", resolved)
        self.assertIn("<redacted:SELF_HARNESS_SECRET_TOKEN>", resolved)

    def test_invalid_benchmark_and_splits_are_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_eval_config(self._write(self._minimal().replace("swe_bench_verified", "other")))
        with self.assertRaises(ConfigError):
            load_eval_config(
                self._write(
                    self._minimal().replace('split = ["train", "heldout"]', 'split = ["validation"]'),
                    name="bad-split.toml",
                )
            )


if __name__ == "__main__":
    unittest.main()
