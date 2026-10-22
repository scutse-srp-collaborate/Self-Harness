"""DeepAgents execution adapter for SWE-bench Verified tasks.

The adapter is the real agent entry point used by the SWE-bench runner:

* OpenAI-compatible chat model built from ``SELF_HARNESS_MODEL`` (default
  ``deepseek-flash``), ``OPENAI_BASE_URL`` (default ``https://api.qadg.xyz/v1``)
  and ``SELF_HARNESS_MODEL_API_KEY``.
* The candidate harness ``repo_baseline.py`` is imported dynamically by file
  path for every run, so a promoted/candidate harness snapshot really takes
  effect and cannot be shadowed by Python's import cache.
* Filesystem/shell tools execute inside a disposable task-scoped Docker
  container that only bind-mounts the task workspace.
* Every model call, tool call, command, timeout and error is written to a JSONL
  trace with secret redaction.  The API key is never passed to the container,
  written to disk, or included in exception text.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .deepagent_sandbox import DockerSandbox, DockerSandboxSettings, redact_secrets

from .openai_compat import (  # noqa: F401 - re-exported for adapter consumers
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MODEL_ENV,
    MISSING_API_KEY_MESSAGE,
    MissingApiKeyError,
    OpenAIModelSettings,
    api_key_from_environment,
    build_chat_model,
    settings_from_environment,
)

DeepAgentModelSettings = OpenAIModelSettings
TRACE_ENV = "SELF_HARNESS_AGENT_TRACE"
CANDIDATE_WORKSPACE_ENV = "SELF_HARNESS_CANDIDATE_WORKSPACE"
TASK_TIMEOUT_ENV = "SELF_HARNESS_TASK_TIMEOUT_S"
TOOL_TIMEOUT_ENV = "SELF_HARNESS_TOOL_TIMEOUT_S"

TRACE_FORMAT = "self_harness.deepagent_trace.v1"


# ---------------------------------------------------------------------------
# Dynamic candidate harness loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedHarness:
    snapshot_dir: Path
    module: Any
    module_name: str
    harness_file_sha256: str

    def build_agent(self, model: Any, *, backend: Any) -> Any:
        builder = getattr(self.module, "build_fixed_harness_agent", None)
        if not callable(builder):
            raise RuntimeError(
                f"candidate harness {self.snapshot_dir / 'repo_baseline.py'} does not expose "
                "build_fixed_harness_agent(model, backend=...)"
            )
        return builder(model, backend=backend)


def load_candidate_harness(snapshot_dir: Path | str) -> LoadedHarness:
    """Import ``repo_baseline.py`` from exactly this snapshot.

    A unique module name and a cache purge guarantee that a later candidate
    never reuses an earlier candidate's module object.
    """

    import importlib.util

    snapshot = Path(snapshot_dir).expanduser().resolve()
    harness_file = snapshot / "repo_baseline.py"
    if not harness_file.is_file():
        raise RuntimeError(f"harness snapshot has no repo_baseline.py: {snapshot}")
    digest = hashlib.sha256(harness_file.read_bytes()).hexdigest()[:16]
    module_name = f"self_harness_candidate_{digest}_{uuid.uuid4().hex[:8]}"
    for loaded_name in [name for name in sys.modules if name.startswith("self_harness_candidate_")]:
        sys.modules.pop(loaded_name, None)
    sys.modules.pop("repo_baseline", None)
    spec = importlib.util.spec_from_file_location(module_name, harness_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate harness: {harness_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_path = False
    if str(snapshot) not in sys.path:
        sys.path.insert(0, str(snapshot))
        added_path = True
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        if added_path and sys.path and sys.path[0] == str(snapshot):
            sys.path.pop(0)
        raise
    harness = LoadedHarness(
        snapshot_dir=snapshot,
        module=module,
        module_name=module_name,
        harness_file_sha256=digest,
    )
    return harness


# ---------------------------------------------------------------------------
# Trace writer / callbacks
# ---------------------------------------------------------------------------


class TraceWriter:
    """Append-only JSONL trace with secret redaction and thread safety."""

    def __init__(self, path: Path, *, secrets: Sequence[str] = ()) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._secrets = tuple(secret for secret in secrets if secret)
        import threading

        self._lock = threading.Lock()
        self._handle = self.path.open("w", encoding="utf-8")
        self._closed = False

    def write(self, event_type: str, payload: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        record = {
            "format": TRACE_FORMAT,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": event_type,
        }
        if payload:
            record.update(_sanitize_payload(payload, self._secrets))
        line = redact_secrets(json.dumps(record, ensure_ascii=False, default=str), secrets=self._secrets)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.close()
        except OSError:
            pass


def _sanitize_payload(payload: Mapping[str, Any], secrets: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            result[str(key)] = redact_secrets(_truncate(value, 4000), secrets=secrets)
        elif isinstance(value, (int, float, bool)) or value is None:
            result[str(key)] = value
        elif isinstance(value, (list, tuple)):
            result[str(key)] = [
                redact_secrets(_truncate(str(item), 500), secrets=secrets) for item in value[:50]
            ]
        elif isinstance(value, Mapping):
            result[str(key)] = {
                str(k): redact_secrets(_truncate(str(v), 500), secrets=secrets)
                for k, v in list(value.items())[:30]
            }
        else:
            result[str(key)] = redact_secrets(_truncate(str(value), 500), secrets=secrets)
    return result


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


try:  # pragma: no cover - import availability is checked by the adapter self-test
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
except Exception:  # pragma: no cover
    _BaseCallbackHandler = object  # type: ignore[assignment,misc]


class TracingCallbackHandler(_BaseCallbackHandler):  # type: ignore[misc,valid-type]
    """LangChain callback handler that records model/tool events without secrets."""

    def __init__(self, trace: TraceWriter) -> None:
        super().__init__()
        self.trace = trace

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.trace.write(
            "model_start",
            {
                "model": _serialized_model_name(serialized),
                "message_count": len(messages[0]) if messages else 0,
            },
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.trace.write(
            "model_end",
            {
                "response_preview": _truncate(str(response), 1500),
            },
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.trace.write("model_error", {"error": _safe_error(error)})

    def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") if isinstance(serialized, Mapping) else str(serialized)
        self.trace.write(
            "tool_start",
            {"tool": name, "input_preview": _truncate(str(input_str), 1200)},
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self.trace.write("tool_end", {"output_preview": _truncate(str(output), 1500)})

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self.trace.write("tool_error", {"error": _safe_error(error)})


def _serialized_model_name(serialized: Any) -> str:
    if isinstance(serialized, Mapping):
        for key in ("name", "model_name", "model"):
            value = serialized.get(key)
            if value:
                return str(value)
    return "unknown"


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


@dataclass
class AgentRunResult:
    status: str
    final_text: str
    timed_out: bool
    duration_s: float
    error: str | None
    tool_calls: int | None
    model: str
    base_url: str
    harness_file_sha256: str
    container_id: str | None
    image_digest: str | None
    image: str | None = None
    container_args: list[str] | None = None
    readonly_mounts: list[list[str]] | None = None
    read_only_root: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_text": self.final_text,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "tool_calls": self.tool_calls,
            "model": self.model,
            "base_url": self.base_url,
            "harness_file_sha256": self.harness_file_sha256,
            "container_id": self.container_id,
            "image_digest": self.image_digest,
            "image": self.image,
            "container_args": self.container_args,
            "readonly_mounts": self.readonly_mounts,
            "read_only_root": self.read_only_root,
        }


def run_deepagent_task(
    *,
    instruction: str,
    workspace: Path,
    case_dir: Path,
    harness_snapshot: Path,
    trace_path: Path,
    task_id: str,
    repo: str,
    base_commit: str,
    split: str,
    repeat: int,
    task_timeout_s: float,
    tool_timeout_s: float,
    settings: DeepAgentModelSettings | None = None,
    chat_model: Any | None = None,
    sandbox: DockerSandbox | None = None,
    sandbox_settings: DockerSandboxSettings | None = None,
) -> AgentRunResult:
    """Run one SWE-bench task with the candidate harness and return agent facts."""

    settings = settings or settings_from_environment()
    key = api_key_from_environment()
    secrets = (key,)
    os.environ.setdefault("SELF_HARNESS_BENCHMARK", "swe_bench_verified")
    trace = TraceWriter(trace_path, secrets=secrets)
    started = time.time()
    container_id: str | None = None
    image_digest: str | None = None
    sandbox_backend = sandbox
    try:
        trace.write(
            "agent_start",
            {
                "task_id": task_id,
                "repo": repo,
                "base_commit": base_commit,
                "split": split,
                "repeat": repeat,
                "model": settings.model,
                "base_url": settings.base_url,
                "task_timeout_s": task_timeout_s,
                "tool_timeout_s": tool_timeout_s,
                "harness_snapshot": str(harness_snapshot),
                "workspace": str(workspace),
            },
        )
        try:
            loaded = load_candidate_harness(harness_snapshot)
        except Exception as exc:  # noqa: BLE001
            error = _safe_error(exc)
            trace.write("harness_load_error", {"error": error})
            return AgentRunResult(
                status="error",
                final_text="",
                timed_out=False,
                duration_s=time.time() - started,
                error=error,
                tool_calls=None,
                model=settings.model,
                base_url=settings.base_url,
                harness_file_sha256="",
                container_id=None,
                image_digest=None,
            )
        trace.write(
            "harness_loaded",
            {
                "module_name": loaded.module_name,
                "harness_file_sha256": loaded.harness_file_sha256,
            },
        )
        if sandbox_backend is None:
            effective_settings = sandbox_settings or DockerSandboxSettings.from_environment()
            readonly_mounts = list(effective_settings.readonly_mounts)
            if harness_snapshot.is_dir():
                harness_mount = (str(harness_snapshot.resolve()), "/harness")
                if harness_mount not in readonly_mounts:
                    readonly_mounts.append(harness_mount)
            effective_settings = dataclasses.replace(
                effective_settings,
                tool_timeout_s=max(1, int(tool_timeout_s)),
                readonly_mounts=tuple(readonly_mounts),
            )
            sandbox_backend = DockerSandbox(
                workspace=workspace,
                settings=effective_settings,
                expected_base_commit=base_commit,
                trace=lambda event: trace.write(str(event.get("event_type")), event),
                secrets=secrets,
            )
            sandbox_settings = effective_settings
        container_id = sandbox_backend._container_id  # noqa: SLF001 - adapter-owned backend
        image_digest = sandbox_backend.image_digest
        runtime_fields = {
            "image": getattr(sandbox_backend, "image", None),
            "container_args": list(getattr(sandbox_backend, "startup_args", ())),
            "readonly_mounts": [
                list(item)
                for item in getattr(sandbox_backend.settings, "readonly_mounts", ())
            ],
            "read_only_root": getattr(sandbox_backend.settings, "read_only_root", None),
        }
        _prepare_container(sandbox_backend, trace=trace)

        if chat_model is None:
            chat_model = build_chat_model(settings)
        agent = loaded.build_agent(chat_model, backend=sandbox_backend)
        augmented_instruction = _augment_instruction(
            instruction=instruction,
            mount=sandbox_backend.settings.workspace_mount,
            repo=repo,
        )
        trace.write("agent_invoke_start", {"instruction_preview": augmented_instruction[:3000]})
        callbacks = [TracingCallbackHandler(trace)]
        invoke_config = {
            "configurable": {"thread_id": f"swebench-{task_id}-r{repeat}-{uuid.uuid4().hex[:8]}"},
            "callbacks": callbacks,
            "recursion_limit": 400,
        }
        result: Any = None
        timed_out = False
        try:
            async def _invoke() -> Any:
                return await agent.ainvoke(
                    {"messages": [{"role": "user", "content": augmented_instruction}]},
                    config=invoke_config,
                )

            result = asyncio.run(asyncio.wait_for(_invoke(), timeout=task_timeout_s))
        except (asyncio.TimeoutError, TimeoutError):
            timed_out = True
            trace.write("agent_timeout", {"limit_s": task_timeout_s})
        except Exception as exc:  # noqa: BLE001 - errors are mapped into the result
            error = _safe_error(exc)
            trace.write("agent_error", {"error": error})
            _write_final(case_dir, "")
            return AgentRunResult(
                status="error",
                final_text="",
                timed_out=False,
                duration_s=time.time() - started,
                error=error,
                tool_calls=None,
                model=settings.model,
                base_url=settings.base_url,
                harness_file_sha256=loaded.harness_file_sha256,
                container_id=container_id,
                image_digest=image_digest,
                **runtime_fields,
            )
        final_text = _final_text(result)
        tool_calls = _count_tool_calls(result)
        _write_final(case_dir, final_text)
        trace.write(
            "agent_end",
            {
                "final_text": final_text,
                "tool_calls": tool_calls,
                "timed_out": timed_out,
            },
        )
        return AgentRunResult(
            status="timeout" if timed_out else "complete",
            final_text=final_text,
            timed_out=timed_out,
            duration_s=time.time() - started,
            error=None if not timed_out else f"case exceeded {task_timeout_s}s timeout",
            tool_calls=tool_calls,
            model=settings.model,
            base_url=settings.base_url,
            harness_file_sha256=loaded.harness_file_sha256,
            container_id=container_id,
            image_digest=image_digest,
            **runtime_fields,
        )
    finally:
        trace.write(
            "agent_finish",
            {
                "duration_s": round(time.time() - started, 3),
            },
        )
        trace.close()
        if sandbox is None and sandbox_backend is not None:
            try:
                sandbox_backend.close()
            except Exception:  # noqa: BLE001
                pass


def _prepare_container(sandbox: DockerSandbox, *, trace: TraceWriter) -> None:
    # With a read-only root only /tmp, /root and the task workspace are writable.
    # The instruction already describes the workspace; avoid touching the repo
    # (an untracked AGENTS.md would pollute the collected patch).
    read_only_root = bool(getattr(sandbox.settings, "read_only_root", False))
    target = "/tmp/AGENTS.md" if read_only_root else "/AGENTS.md"
    script = (
        "printf '%s\\n' 'Repository workspace: /testbed' "
        f"'Use shell and file tools to inspect and edit files under /testbed.' > {target}"
    )
    result = sandbox.execute(script, timeout=30)
    trace.write(
        "container_prepare",
        {"exit_code": result.exit_code, "output_preview": result.output[-500:]},
    )


def _augment_instruction(*, instruction: str, mount: str, repo: str) -> str:
    return (
        f"{instruction.rstrip()}\n\n"
        "[Execution environment]\n"
        f"- You are editing the repository {repo} inside a disposable Linux container.\n"
        f"- The repository workspace is mounted at {mount}. Use shell/file tools with "
        f"absolute paths under {mount} (for example `{mount}/src`).\n"
        "- Your final answer is not the deliverable: the runner collects the file diff you "
        "leave in the workspace and an official SWE-bench evaluator decides resolved/unresolved.\n"
        "- The harness snapshot is mounted read-only at /harness for inspection; you cannot "
        "modify it. The dataset, credentials and repository cache are not available inside "
        "this container.\n"
        "- The task image already contains the project's test dependencies. You may run the "
        "repository tests to verify your fix; network access is disabled, so do not attempt "
        "to install packages.\n"
    )


def _final_text(result: Any) -> str:
    if not isinstance(result, Mapping):
        return ""
    messages = result.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        text = getattr(message, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [str(item.get("text")) for item in content if isinstance(item, Mapping) and item.get("text")]
            if parts:
                return "\n".join(parts)
    return ""


def _count_tool_calls(result: Any) -> int | None:
    if not isinstance(result, Mapping):
        return None
    messages = result.get("messages")
    if not isinstance(messages, list):
        return None
    total = 0
    found = False
    for message in messages:
        calls = getattr(message, "tool_calls", None)
        if isinstance(calls, list):
            found = True
            total += len(calls)
    return total if found else None


def _write_final(case_dir: Path, text: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "final_message.txt").write_text(text, encoding="utf-8")


def _safe_error(error: BaseException) -> str:
    """Exception text without secrets or authorization headers."""

    text = f"{error.__class__.__name__}: {error}"
    lowered = text.lower()
    if any(marker in lowered for marker in ("authorization", "bearer ", "api_key", "api-key")):
        return f"{error.__class__.__name__}: model request failed (details redacted)"
    key = os.environ.get(API_KEY_ENV, "")
    return redact_secrets(text, secrets=(key,))[:2000]
