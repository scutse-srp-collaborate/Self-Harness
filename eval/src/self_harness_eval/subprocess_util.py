"""Subprocess execution helpers with explicit timeouts and attempt records."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.error is None and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 3),
            "timed_out": self.timed_out,
            "error": self.error,
        }


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    status: str
    reason: str | None
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "status": self.status,
            "reason": self.reason,
            "duration_s": round(self.duration_s, 3),
        }


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    stdin_text: str | None = None,
) -> CommandResult:
    """Run a command, never raising for non-zero exit codes.

    A timeout always produces a result with ``timed_out=True``; an OSError
    (missing binary, permission problem) produces ``error``.  Callers decide how
    to map those facts into unified case statuses.
    """

    materialized = [str(item) for item in argv]
    started = time.time()
    try:
        completed = subprocess.run(
            materialized,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            input=stdin_text,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=timeout_s,
        )
        return CommandResult(
            argv=materialized,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_s=time.time() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=materialized,
            returncode=None,
            stdout=_normalize_output(exc.stdout),
            stderr=_normalize_output(exc.stderr),
            duration_s=time.time() - started,
            timed_out=True,
            error=f"command exceeded timeout of {timeout_s}s",
        )
    except OSError as exc:
        return CommandResult(
            argv=materialized,
            returncode=None,
            stdout="",
            stderr="",
            duration_s=time.time() - started,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def run_command_with_retries(
    *,
    argv_factory: Any,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
    max_retries: int,
    retry_on: Sequence[str] = ("error",),
    accept: Any | None = None,
) -> tuple[CommandResult, list[AttemptRecord]]:
    """Run a command with bounded retries and a full attempt audit trail.

    ``retry_on`` may contain ``"error"`` (non-zero exit / spawn failure) and/or
    ``"timeout"``.  Attempts are always capped at ``max_retries + 1``; the
    attempt list is recorded in the case result.
    """

    attempts: list[AttemptRecord] = []
    limit = max(0, int(max_retries))
    last: CommandResult | None = None
    for attempt in range(1, limit + 2):
        argv = argv_factory(attempt)
        last = run_command(argv, cwd=cwd, env=env, timeout_s=timeout_s)
        if last.timed_out:
            status = "timeout"
            reason = last.error
        elif last.error is not None or last.returncode != 0:
            status = "error"
            reason = last.error or f"returncode={last.returncode}"
        else:
            status = "complete"
            reason = None
        attempts.append(
            AttemptRecord(
                attempt=attempt,
                status=status,
                reason=reason,
                duration_s=last.duration_s,
            )
        )
        if accept is not None and accept(last):
            break
        if status == "complete" and accept is None:
            break
        if status == "timeout" and "timeout" not in retry_on:
            break
        if status == "error" and "error" not in retry_on:
            break
        if attempt > limit:
            break
    assert last is not None
    return last, attempts


def expand_command_template(
    template: str | Sequence[str],
    *,
    placeholders: Mapping[str, Any],
) -> list[str]:
    """Render a configured command template into an argv list.

    String templates are shell-split first; list templates are used verbatim.
    Placeholders use ``{name}`` syntax and are substituted per token so paths
    with spaces survive.
    """

    if isinstance(template, str):
        tokens = shlex.split(template)
    else:
        tokens = [str(item) for item in template]
    mapping = {key: str(value) for key, value in placeholders.items()}
    rendered: list[str] = []
    for token in tokens:
        try:
            rendered.append(token.format(**mapping))
        except KeyError as exc:
            raise ValueError(f"unknown command placeholder in {token!r}: {exc}") from exc
    if not rendered:
        raise ValueError("command template resolved to an empty argv")
    return rendered


def preview_command(argv: Sequence[str], *, limit: int = 240) -> str:
    text = shlex.join([str(item) for item in argv])
    return text if len(text) <= limit else text[: limit - 3] + "..."


def tail_text(value: str | bytes | None, *, max_chars: int = 8000) -> str:
    text = _normalize_output(value)
    return text if len(text) <= max_chars else text[-max_chars:]


def write_command_logs(
    *,
    case_dir: Path,
    stdout_name: str,
    stderr_name: str,
    result: CommandResult,
) -> dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = case_dir / stdout_name
    stderr_path = case_dir / stderr_name
    stdout_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(result.stderr, encoding="utf-8", errors="replace")
    return {"stdout": str(stdout_path), "stderr": str(stderr_path)}


def env_with_overrides(
    *,
    base_env: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
    passthrough: Sequence[str] = (),
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    for name in passthrough:
        if name in os.environ:
            env[name] = os.environ[name]
    for key, value in (overrides or {}).items():
        env[str(key)] = str(value)
    return env


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:  # pragma: no cover - helper
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except OSError:
        process.terminate()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
