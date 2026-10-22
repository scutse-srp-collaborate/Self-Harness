"""DeepAgents adapter, dynamic harness loading and Docker sandbox tests."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    from self_harness_eval.deepagent_adapter import (
        API_KEY_ENV,
        DeepAgentModelSettings,
        MissingApiKeyError,
        TraceWriter,
        api_key_from_environment,
        build_chat_model,
        load_candidate_harness,
        run_deepagent_task,
        settings_from_environment,
    )
    from self_harness_eval.deepagent_sandbox import (
        DockerSandbox,
        DockerSandboxSettings,
        SandboxError,
    )

    ADAPTER_DEPENDENCIES_AVAILABLE = True
    ADAPTER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - system python collection
    ADAPTER_DEPENDENCIES_AVAILABLE = False
    ADAPTER_IMPORT_ERROR = exc
    API_KEY_ENV = "SELF_HARNESS_MODEL_API_KEY"
    DeepAgentModelSettings = MissingApiKeyError = TraceWriter = object  # type: ignore[assignment,misc]
    api_key_from_environment = build_chat_model = load_candidate_harness = None  # type: ignore[assignment]
    run_deepagent_task = settings_from_environment = None  # type: ignore[assignment]
    DockerSandbox = DockerSandboxSettings = SandboxError = object  # type: ignore[assignment,misc]

ADAPTER_SKIP_REASON = (
    "DeepAgents adapter dependencies are not installed in this interpreter; "
    "run tests with .venv/bin/pytest"
)


def adapter_test(cls):
    return unittest.skipUnless(ADAPTER_DEPENDENCIES_AVAILABLE, ADAPTER_SKIP_REASON)(cls)


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_SNAPSHOT = REPO_ROOT / "eval" / "harness_workspace"
FAKE_KEY = "fake-key-for-adapter-tests-not-real"


def _docker_ready(image: str = "python:3.12") -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


DOCKER_READY = _docker_ready()


def _git_repo(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "fixture@example.invalid"],
        ["git", "config", "user.name", "Fixture"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    (root / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=root, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _candidate_harness(directory: Path, *, marker: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "repo_baseline.py").write_text(
        "from __future__ import annotations\n\n"
        "def build_system_prompt() -> str:\n"
        f"    return {marker!r}\n\n"
        "def build_fixed_harness_agent(model, *, backend=None):\n"
        "    from deepagents import create_deep_agent\n"
        "    return create_deep_agent(model=model, system_prompt=build_system_prompt(), backend=backend)\n",
        encoding="utf-8",
    )
    return directory


@adapter_test
class DeepAgentSettingsTest(unittest.TestCase):
    def test_chat_model_construction_parameters_and_redaction(self) -> None:
        env = {
            API_KEY_ENV: FAKE_KEY,
            "SELF_HARNESS_MODEL": "deepseek-flash",
            "OPENAI_BASE_URL": "https://api.qadg.xyz/v1",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            settings = settings_from_environment()
            model = build_chat_model(settings)
        self.assertEqual(settings.model, "deepseek-flash")
        self.assertEqual(settings.base_url, "https://api.qadg.xyz/v1")
        self.assertEqual(settings.temperature, 0.0)
        self.assertEqual(settings.top_p, 1.0)
        self.assertEqual(settings.max_tokens, 8192)
        self.assertEqual(model.model_name, "deepseek-flash")
        self.assertEqual(model.openai_api_base, "https://api.qadg.xyz/v1")
        self.assertEqual(model.temperature, 0.0)
        self.assertEqual(model.top_p, 1.0)
        self.assertFalse(FAKE_KEY in repr(model))
        self.assertFalse(FAKE_KEY in str(model))

    def test_missing_api_key_has_clear_safe_error(self) -> None:
        with mock.patch.dict(os.environ, {API_KEY_ENV: ""}, clear=False):
            with self.assertRaises(MissingApiKeyError) as caught:
                api_key_from_environment()
        message = str(caught.exception)
        self.assertIn(API_KEY_ENV, message)
        self.assertIn("Refusing", message)
        self.assertNotIn(FAKE_KEY, message)

    def test_adapter_cli_refuses_without_key(self) -> None:
        env = dict(os.environ)
        env.pop(API_KEY_ENV, None)
        with tempfile.TemporaryDirectory() as tmp:
            instruction = Path(tmp) / "instruction.md"
            instruction.write_text("fixture task", encoding="utf-8")
            completed = subprocess.run(
                [
                    str(REPO_ROOT / ".venv" / "bin" / "python"),
                    str(REPO_ROOT / "eval" / "scripts" / "run_deepagent_swebench.py"),
                    "--instruction",
                    str(instruction),
                    "--workspace",
                    tmp,
                    "--case-dir",
                    str(Path(tmp) / "case"),
                    "--task-id",
                    "t",
                    "--repo",
                    "a/b",
                    "--base-commit",
                    "0" * 40,
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(API_KEY_ENV, completed.stderr)
            self.assertFalse((Path(tmp) / "case" / "agent_trace.jsonl").exists())


@adapter_test
class DynamicHarnessLoadingTest(unittest.TestCase):
    def test_each_candidate_snapshot_loads_its_own_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _candidate_harness(root / "snapshot-a", marker="PROMPT-A")
            second = _candidate_harness(root / "snapshot-b", marker="PROMPT-B")
            loaded_a = load_candidate_harness(first)
            loaded_b = load_candidate_harness(second)
            self.assertNotEqual(loaded_a.module_name, loaded_b.module_name)
            self.assertIsNot(loaded_a.module, loaded_b.module)
            self.assertEqual(loaded_a.module.build_system_prompt(), "PROMPT-A")
            self.assertEqual(loaded_b.module.build_system_prompt(), "PROMPT-B")
            self.assertNotIn("repo_baseline", sys.modules)

    def test_builds_deepagent_from_real_harness_snapshot(self) -> None:
        from deepagents.backends import StateBackend

        with mock.patch.dict(
            os.environ,
            {
                API_KEY_ENV: FAKE_KEY,
                "SELF_HARNESS_MODEL": "deepseek-flash",
                "OPENAI_BASE_URL": "https://api.qadg.xyz/v1",
            },
            clear=False,
        ):
            loaded = load_candidate_harness(HARNESS_SNAPSHOT)
            model = build_chat_model()
            agent = loaded.build_agent(model, backend=StateBackend())
        self.assertTrue(hasattr(agent, "ainvoke"))
        self.assertIsNotNone(loaded.harness_file_sha256)

    def test_missing_harness_file_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                load_candidate_harness(Path(tmp))


@adapter_test
class TraceTest(unittest.TestCase):
    def test_trace_redacts_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            writer = TraceWriter(path, secrets=(FAKE_KEY,))
            writer.write("event", {"message": f"prefix {FAKE_KEY} suffix"})
            writer.close()
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(FAKE_KEY, text)
            self.assertIn("<redacted>", text)


@adapter_test
class CasesTimeoutTest(unittest.TestCase):
    def test_case_timeout_recorded(self) -> None:
        class DummySandbox:
            class Settings:
                workspace_mount = "/testbed"

            settings = Settings()
            _container_id = "dummy-container"
            image_digest = "sha256:dummy"

            def execute(self, command: str, *, timeout=None):
                class Result:
                    exit_code = 0
                    output = ""

                return Result()

            def close(self) -> None:
                pass

        class SlowAgent:
            async def ainvoke(self, *args, **kwargs):
                await asyncio.sleep(30)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = _candidate_harness(
                root / "harness",
                marker="prompt",
            )
            # Replace build_fixed_harness_agent with one that returns a slow agent.
            (harness / "repo_baseline.py").write_text(
                "import asyncio\n\n"
                "class SlowAgent:\n"
                "    async def ainvoke(self, *args, **kwargs):\n"
                "        await asyncio.sleep(30)\n\n"
                "def build_fixed_harness_agent(model, *, backend=None):\n"
                "    return SlowAgent()\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {API_KEY_ENV: FAKE_KEY}, clear=False):
                result = run_deepagent_task(
                    instruction="task",
                    workspace=root,
                    case_dir=root / "case",
                    harness_snapshot=harness,
                    trace_path=root / "case" / "trace.jsonl",
                    task_id="t",
                    repo="a/b",
                    base_commit="0" * 40,
                    split="train",
                    repeat=1,
                    task_timeout_s=1,
                    tool_timeout_s=5,
                    chat_model=object(),
                    sandbox=DummySandbox(),  # type: ignore[arg-type]
                )
            self.assertEqual(result.status, "timeout")
            trace = (root / "case" / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("agent_timeout", trace)
            self.assertFalse((root / "case" / "final_message.txt").read_text())


@unittest.skipUnless(
    ADAPTER_DEPENDENCIES_AVAILABLE and DOCKER_READY,
    "DeepAgents dependencies and Docker/python:3.12 image are required for sandbox tests",
)
class DockerSandboxTest(unittest.TestCase):
    def test_workspace_mount_and_harness_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            head = _git_repo(workspace)
            harness = root / "harness"
            harness.mkdir()
            secret_harness_file = harness / "repo_baseline.py"
            secret_harness_file.write_text("# harness sentinel\n", encoding="utf-8")
            events: list[dict] = []
            with mock.patch.dict(os.environ, {API_KEY_ENV: FAKE_KEY}, clear=False):
                with DockerSandbox(
                    workspace=workspace,
                    settings=DockerSandboxSettings(tool_timeout_s=10),
                    expected_base_commit=head,
                    trace=events.append,
                    secrets=(FAKE_KEY,),
                ) as sandbox:
                    self.assertEqual(
                        sandbox.execute("git rev-parse HEAD").output.strip(), head
                    )
                    self.assertNotEqual(
                        sandbox.execute(f"cat {secret_harness_file}").exit_code, 0
                    )
                    upload = sandbox.upload_files(
                        [("/testbed/added.py", b"# added by test\n")]
                    )
                    self.assertIsNone(upload[0].error)
                    self.assertTrue((workspace / "added.py").exists())
                    mounts = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{json .Mounts}}",
                            sandbox.id,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                    self.assertIn(str(workspace.resolve()), mounts)
                    self.assertNotIn(str(harness.resolve()), mounts)
                    container_env = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{json .Config.Env}}",
                            sandbox.id,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                    self.assertNotIn(API_KEY_ENV, container_env)
                    self.assertNotIn(FAKE_KEY, container_env)
            self.assertEqual(secret_harness_file.read_text(encoding="utf-8"), "# harness sentinel\n")

    def test_tool_timeout_returns_explicit_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            head = _git_repo(workspace)
            with DockerSandbox(
                workspace=workspace,
                settings=DockerSandboxSettings(tool_timeout_s=2),
                expected_base_commit=head,
            ) as sandbox:
                result = sandbox.execute("sleep 10", timeout=1)
            self.assertEqual(result.exit_code, 124)
            self.assertIn("[timeout]", result.output)


@unittest.skipUnless(
    ADAPTER_DEPENDENCIES_AVAILABLE and DOCKER_READY,
    "DeepAgents dependencies and Docker/python:3.12 image are required for integration test",
)
class DeepAgentIntegrationTest(unittest.TestCase):
    def test_adapter_cli_runs_real_deepagent_loop_against_local_stub_endpoint(self) -> None:
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
                                    "arguments": json.dumps(
                                        {
                                            "command": "printf '%s\\n' 'EDIT_MARKER' >> /testbed/hello.py"
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": "Finished editing."}
                    finish = "stop"
                payload = {
                    "id": f"stub-{len(calls)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "deepseek-flash",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                workspace = root / "workspace"
                head = _git_repo(workspace)
                case_dir = root / "case"
                case_dir.mkdir()
                instruction = root / "instruction.md"
                instruction.write_text("Append EDIT_MARKER to hello.py.", encoding="utf-8")
                env = dict(os.environ)
                env.update(
                    {
                        API_KEY_ENV: FAKE_KEY,
                        "SELF_HARNESS_MODEL": "deepseek-flash",
                        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                        "SELF_HARNESS_TOOL_TIMEOUT_S": "30",
                    }
                )
                completed = subprocess.run(
                    [
                        str(REPO_ROOT / ".venv" / "bin" / "python"),
                        str(REPO_ROOT / "eval" / "scripts" / "run_deepagent_swebench.py"),
                        "--instruction",
                        str(instruction),
                        "--workspace",
                        str(workspace),
                        "--case-dir",
                        str(case_dir),
                        "--task-id",
                        "astropy__astropy-13398",
                        "--repo",
                        "astropy/astropy",
                        "--base-commit",
                        head,
                        "--split",
                        "train",
                        "--repeat",
                        "1",
                        "--harness-snapshot",
                        str(HARNESS_SNAPSHOT),
                        "--trace",
                        str(case_dir / "agent_trace.jsonl"),
                        "--task-timeout",
                        "120",
                        "--tool-timeout",
                        "30",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])
                self.assertIn("EDIT_MARKER", (workspace / "hello.py").read_text(encoding="utf-8"))
                trace_text = (case_dir / "agent_trace.jsonl").read_text(encoding="utf-8")
                self.assertIn("model_start", trace_text)
                self.assertIn("tool_command", trace_text)
                self.assertNotIn(FAKE_KEY, trace_text)
                agent_result = json.loads((case_dir / "agent_result.json").read_text(encoding="utf-8"))
                self.assertEqual(agent_result["status"], "complete")
                self.assertNotIn(FAKE_KEY, json.dumps(agent_result))
                self.assertTrue((case_dir / "final_message.txt").exists())
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
