#!/usr/bin/env python3
"""DeepAgents SWE-bench agent entry point used by the SWE-bench runner.

This script is invoked by ``eval/scripts/run_swebench_eval.py`` through the
``[agent].command`` template.  It reads the task instruction/workspace from
explicit CLI arguments (no shell ``$VARIABLE`` expansion), builds an
OpenAI-compatible chat model from environment variables, loads the current
candidate Harness snapshot's ``repo_baseline.py`` dynamically, runs DeepAgents
inside a task-scoped Docker container, and writes a JSONL trace.

Exit codes: 0 = agent run completed; 2 = missing configuration/API key;
3 = agent/runtime error; 124 = case timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_harness_eval.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

from self_harness_eval.deepagent_adapter import (  # noqa: E402
    API_KEY_ENV,
    CANDIDATE_WORKSPACE_ENV,
    MissingApiKeyError,
    load_candidate_harness,
    run_deepagent_task,
    settings_from_environment,
)
from self_harness_eval.deepagent_sandbox import SandboxError  # noqa: E402

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_AGENT_ERROR = 3
EXIT_TIMEOUT = 124


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one SWE-bench task with DeepAgents.")
    parser.add_argument("--instruction", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--harness-snapshot", type=Path)
    parser.add_argument(
        "--image-map",
        type=Path,
        help="Preflight-authored task image map (self_harness.task_images.v1).",
    )
    parser.add_argument(
        "--require-image-map",
        action="store_true",
        help="Refuse to run unless the task image map supplies the exact task image.",
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--task-timeout", type=_positive_float)
    parser.add_argument("--tool-timeout", type=_positive_float)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Only validate imports/ChatOpenAI/DeepAgent construction (no model call).",
    )
    args = parser.parse_args(argv)

    case_dir = args.case_dir.expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_path = (
        args.trace.expanduser().resolve()
        if args.trace
        else Path(os.environ.get("SELF_HARNESS_AGENT_TRACE", case_dir / "agent_trace.jsonl"))
    )
    harness_snapshot = (
        args.harness_snapshot.expanduser().resolve()
        if args.harness_snapshot
        else Path(
            os.environ.get(
                CANDIDATE_WORKSPACE_ENV,
                str(Path(__file__).resolve().parents[2] / "eval" / "harness_workspace"),
            )
        ).resolve()
    )
    task_timeout = args.task_timeout or _env_timeout("SELF_HARNESS_TASK_TIMEOUT_S", 3600.0)
    tool_timeout = args.tool_timeout or _env_timeout("SELF_HARNESS_TOOL_TIMEOUT_S", 120.0)

    settings = settings_from_environment()
    if args.self_check:
        return _self_check(harness_snapshot=harness_snapshot, settings=settings)

    if args.image_map is not None or args.require_image_map:
        if args.image_map is None:
            print("error: --require-image-map needs --image-map", file=sys.stderr)
            return EXIT_CONFIG
        try:
            _apply_task_image_map(args.image_map, args.task_id)
        except ValueError as exc:
            print(f"error: task image map: {exc}", file=sys.stderr)
            return EXIT_CONFIG

    if not os.environ.get(API_KEY_ENV, "").strip():
        print(
            f"error: {API_KEY_ENV} is not set in the environment. Refusing to make real model "
            "calls. Set the variable in the runtime environment; do not paste the key into "
            "chat, config files or command arguments.",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    instruction = args.instruction.expanduser().resolve().read_text(encoding="utf-8")
    try:
        result = run_deepagent_task(
            instruction=instruction,
            workspace=args.workspace.expanduser().resolve(),
            case_dir=case_dir,
            harness_snapshot=harness_snapshot,
            trace_path=trace_path,
            task_id=args.task_id,
            repo=args.repo,
            base_commit=args.base_commit,
            split=args.split,
            repeat=args.repeat,
            task_timeout_s=task_timeout,
            tool_timeout_s=tool_timeout,
            settings=settings,
        )
    except MissingApiKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except SandboxError as exc:
        print(f"error: task sandbox could not be prepared: {exc}", file=sys.stderr)
        return EXIT_AGENT_ERROR
    except Exception as exc:  # noqa: BLE001 - never expose secrets in failure text
        print(
            f"error: deepagent adapter failed: {exc.__class__.__name__}: {_redact_exception(exc)}",
            file=sys.stderr,
        )
        return EXIT_AGENT_ERROR
    (case_dir / "agent_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"deepagent completed: status={result.status} tool_calls={result.tool_calls} "
        f"duration_s={result.duration_s:.1f}"
    )
    if result.status == "timeout":
        return EXIT_TIMEOUT
    if result.status != "complete":
        return EXIT_AGENT_ERROR
    return EXIT_OK


def _apply_task_image_map(path: Path, task_id: str) -> None:
    """Apply the preflight task image entry to the adapter environment."""

    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "self_harness.task_images.v1":
        raise ValueError(f"unexpected format in {path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"task map has no tasks object: {path}")
    entry = tasks.get(task_id)
    if not isinstance(entry, dict):
        raise ValueError(f"no image entry for task {task_id!r}")
    image = str(entry.get("image") or "").strip()
    if not image:
        raise ValueError(f"empty image name for task {task_id!r}")
    digest = str(entry.get("image_digest") or "").strip()
    os.environ["SELF_HARNESS_AGENT_IMAGE"] = image
    if digest:
        os.environ["SELF_HARNESS_AGENT_IMAGE_DIGEST"] = digest
    container_env = entry.get("container_env")
    if isinstance(container_env, dict) and container_env:
        os.environ["SELF_HARNESS_AGENT_CONTAINER_ENV"] = json.dumps(
            {str(k): str(v) for k, v in container_env.items()}, sort_keys=True
        )
    if entry.get("read_only_root") is not None:
        os.environ["SELF_HARNESS_AGENT_READ_ONLY"] = (
            "1" if entry.get("read_only_root") else "0"
        )
    if entry.get("bootstrap_workspace_from_image"):
        os.environ["SELF_HARNESS_AGENT_BOOTSTRAP_FROM_IMAGE"] = "1"
        os.environ["SELF_HARNESS_AGENT_IMAGE_REPO"] = str(
            entry.get("image_repo_path") or "/testbed"
        )


def _self_check(*, harness_snapshot: Path, settings: Any) -> int:
    """Validate real imports/interfaces without a network request or container."""

    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from langchain_openai import ChatOpenAI

    problems: list[str] = []
    try:
        loaded = load_candidate_harness(harness_snapshot)
    except Exception as exc:  # noqa: BLE001
        loaded = None
        problems.append(f"harness_load: {exc.__class__.__name__}: {_redact_exception(exc)}")
    api_key_present = bool(os.environ.get(API_KEY_ENV, "").strip())
    try:
        from self_harness_eval.deepagent_adapter import build_chat_model

        # Interface validation does not need a real credential; a placeholder is
        # used only when the environment has no key, and the report says so.
        model = build_chat_model(
            settings,
            api_key_override=None if api_key_present else "self-check-placeholder-not-a-key",
        )
        if not isinstance(model, ChatOpenAI):
            problems.append("chat_model: unexpected type")
        if model.model_name != settings.model:
            problems.append("chat_model: model name mismatch")
    except Exception as exc:  # noqa: BLE001
        model = None
        problems.append(f"chat_model: {exc.__class__.__name__}: {_redact_exception(exc)}")
    try:
        if loaded is not None and model is not None:
            agent = loaded.build_agent(model, backend=StateBackend())
            if not hasattr(agent, "ainvoke"):
                problems.append("deepagent: object has no ainvoke")
        else:
            from langchain_core.language_models.fake_chat_models import FakeListChatModel

            create_deep_agent(model=FakeListChatModel(responses=["ok"]), backend=StateBackend())
    except Exception as exc:  # noqa: BLE001
        problems.append(f"deepagent: {exc.__class__.__name__}: {_redact_exception(exc)}")
    summary = {
        "check": "deepagent_self_check",
        "ok": not problems,
        "problems": problems,
        "provider": "openai_compatible",
        "model": settings.model,
        "base_url": settings.base_url,
        "harness_snapshot": str(harness_snapshot),
        "api_key_present": api_key_present,
        "note": (
            "Interface/import validation only; no real model call is made. A real pilot "
            "requires SELF_HARNESS_MODEL_API_KEY in the environment."
        ),
        "imports": {
            "deepagents": True,
            "langchain_openai": True,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_OK if not problems else EXIT_AGENT_ERROR


def _env_timeout(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _redact_exception(exc: BaseException) -> str:
    text = str(exc)
    lowered = text.lower()
    if any(marker in lowered for marker in ("authorization", "bearer ", "api_key", "api-key")):
        return "details redacted"
    key = os.environ.get(API_KEY_ENV, "")
    if key:
        text = text.replace(key, "<redacted>")
    return text[:1000]


if __name__ == "__main__":
    raise SystemExit(main())
