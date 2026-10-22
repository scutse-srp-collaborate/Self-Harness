from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from deepagents import create_deep_agent

try:  # Newer DeepAgents versions require middleware to implement the base class.
    from langchain.agents.middleware import AgentMiddleware as _AgentMiddlewareBase
except Exception:  # pragma: no cover - older environments accept duck-typed middleware
    _AgentMiddlewareBase = object  # type: ignore[assignment,misc]


def _swe_bench_mode() -> bool:
    return os.environ.get("SELF_HARNESS_BENCHMARK", "").strip() == "swe_bench_verified"


def build_system_prompt() -> str:
    if _swe_bench_mode():
        return """
You are fixing a GitHub issue in a SWE-bench repository. The repository is
mounted at /testbed. Work only inside /testbed and use the shell/file tools to
inspect code, edit files, and run targeted tests.

Read the issue, locate the relevant implementation, make the smallest correct
change, run the most targeted tests you can, and stop. Do not search the host
filesystem or other directories. Your final message is not graded: leave the
code changes in the workspace; an official evaluator tests them.
""".strip()
    return """
You are running inside a Terminal Bench 2 Harbor task environment.

Use the built-in filesystem and shell tools to inspect the workspace, make
concrete edits, and verify outcomes against the actual task environment.

Do not assume synthetic datasets, domain-specific tools, or hidden fixtures
unless you discover them in the repo or runtime.
""".strip()


BASELINE_SYSTEM_PROMPT = build_system_prompt()


def build_memory_sources() -> list[str]:
    """Repo-level always-on context sources."""
    if _swe_bench_mode():
        return []
    return ["/AGENTS.md"]


def build_subagents() -> list[dict[str, Any]]:
    """Optional editable subagent definitions."""
    return []


def build_skills() -> list[str]:
    """Optional Deep Agents skill sources."""
    return []


def build_permissions() -> list[Any]:
    """Optional filesystem permission rules."""
    return []


def build_interrupt_on() -> dict[str, bool] | None:
    """Optional human-in-the-loop tool interruption config."""
    return None


def build_bootstrap_instruction() -> str:
    if _swe_bench_mode():
        return (
            "Read the issue, then locate the relevant files under /testbed. "
            "Do not search outside /testbed."
        )
    return "Start by inspecting the workspace and identifying the smallest relevant edit surface."


def build_execution_instruction() -> str:
    if _swe_bench_mode():
        return (
            "Make concrete edits and run targeted tests. Keep the patch minimal; "
            "stop once the fix is implemented and verified."
        )
    return "Prefer concrete repo changes over generic advice, and keep edits tightly scoped to the task."


def build_verification_instruction() -> str:
    if _swe_bench_mode():
        return "Run the most relevant tests and inspect any failure before concluding."
    return "Before concluding, verify the result with the most targeted command, file read, or test you can run."


def build_failure_recovery_instruction() -> str:
    if _swe_bench_mode():
        return "If a command or test fails, inspect the error, change approach, and retry differently."
    return "If a tool call fails, inspect the error and adapt; do not blindly retry the same action."


def build_multimodal_instruction() -> str:
    if _swe_bench_mode():
        return ""
    return "If the user supplies images, ground conclusions in visible evidence from those images."


def _message_blocks(message: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(message, str):
        return blocks

    content_blocks = getattr(message, "content_blocks", None)
    if isinstance(content_blocks, list):
        blocks.extend(block for block in content_blocks if isinstance(block, dict))

    content = getattr(message, "content", None)
    if isinstance(content, list):
        blocks.extend(block for block in content if isinstance(block, dict))

    return blocks


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in _message_blocks(message):
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    if parts:
        return "\n".join(parts)

    text = getattr(message, "text", None)
    return text if isinstance(text, str) else ""


def _is_tool_message(message: Any) -> bool:
    return str(getattr(message, "type", "")).lower() == "tool" or message.__class__.__name__ == "ToolMessage"


def _has_tool_error(messages: Sequence[Any]) -> bool:
    for message in messages:
        if not _is_tool_message(message):
            continue
        if getattr(message, "status", None) == "error":
            return True
        if "error" in _message_text(message).lower():
            return True
    return False


def _has_image_input(messages: Sequence[Any]) -> bool:
    for message in messages:
        for block in _message_blocks(message):
            block_type = str(block.get("type", "")).lower()
            if block_type in {"image", "input_image"}:
                return True
            if block_type == "file" and str(block.get("mime_type", "")).lower().startswith("image/"):
                return True
    return False


def _append_instruction(system_message: Any, instruction: str) -> str:
    current = _message_text(system_message)
    return f"{current}\n\n{instruction}" if current else instruction


def _build_prompt_middleware(
    name: str,
    instruction_builder: Callable[[], str],
    *,
    predicate: Callable[[Sequence[Any]], bool] | None = None,
) -> Any | None:
    class _PromptMiddleware(_AgentMiddlewareBase):  # type: ignore[misc,valid-type]
        def _modify_request(self, request: Any) -> Any:
            messages = list(getattr(request, "messages", ()) or ())
            if predicate is not None and not predicate(messages):
                return request
            instruction = instruction_builder().strip()
            if not instruction:
                return request
            system_message = _append_instruction(request.system_message, instruction)
            try:
                from langchain_core.messages import SystemMessage

                patched_system_message: Any = SystemMessage(content=system_message)
            except Exception:  # pragma: no cover - older LangChain accepts strings
                patched_system_message = system_message
            return request.override(system_message=patched_system_message)

        def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
            return handler(self._modify_request(request))

        async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
            return await handler(self._modify_request(request))

    _PromptMiddleware.__name__ = name
    _PromptMiddleware.name = name
    return _PromptMiddleware()


def build_bootstrap_middleware() -> Sequence[Any]:
    middleware = _build_prompt_middleware(
        "TB2BootstrapMiddleware",
        build_bootstrap_instruction,
        predicate=lambda messages: not any(_is_tool_message(message) for message in messages),
    )
    return (middleware,) if middleware is not None else ()


def build_execution_middleware() -> Sequence[Any]:
    middleware = _build_prompt_middleware(
        "TB2ExecutionMiddleware",
        build_execution_instruction,
    )
    return (middleware,) if middleware is not None else ()


def build_verification_middleware() -> Sequence[Any]:
    middleware = _build_prompt_middleware(
        "TB2VerificationMiddleware",
        build_verification_instruction,
    )
    return (middleware,) if middleware is not None else ()


def build_failure_recovery_middleware() -> Sequence[Any]:
    middleware = _build_prompt_middleware(
        "TB2FailureRecoveryMiddleware",
        build_failure_recovery_instruction,
        predicate=_has_tool_error,
    )
    return (middleware,) if middleware is not None else ()


def build_multimodal_middleware() -> Sequence[Any]:
    middleware = _build_prompt_middleware(
        "TB2MultimodalMiddleware",
        build_multimodal_instruction,
        predicate=_has_image_input,
    )
    return (middleware,) if middleware is not None else ()


def build_agent_kwargs(model: Any, *, backend: Any | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "system_prompt": build_system_prompt(),
        "memory": build_memory_sources(),
        "subagents": build_subagents(),
    }

    middleware: list[Any] = []
    for builder in (
        build_bootstrap_middleware,
        build_execution_middleware,
        build_verification_middleware,
        build_failure_recovery_middleware,
        build_multimodal_middleware,
    ):
        middleware.extend(builder())
    if middleware:
        kwargs["middleware"] = middleware

    skills = build_skills()
    if skills:
        kwargs["skills"] = skills

    permissions = build_permissions()
    if permissions:
        kwargs["permissions"] = permissions

    interrupt_on = build_interrupt_on()
    if interrupt_on is not None:
        kwargs["interrupt_on"] = interrupt_on

    if backend is not None:
        kwargs["backend"] = backend


    return kwargs


def build_fixed_harness_agent(model: Any, *, backend: Any | None = None) -> Any:
    return create_deep_agent(**build_agent_kwargs(model, backend=backend))
