"""Unified evaluation result protocol: ``self_harness.eval_result.v1``.

Every baseline, candidate and final evaluation writes one immutable
``result.json`` in this schema.  Candidate admission decisions live in a
separate ``acceptance.json``; the result file only records measurable
evaluation facts.

The module is intentionally dependency-free (standard library only) so the
Terminal-Bench, SWE-bench, diagnosis, acceptance and workflow components can
all validate the same facts.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import (
    sha256_json,
    snapshot_tree_hash,
    tree_file_hashes,
)

SCHEMA_VERSION = "self_harness.eval_result.v1"
BENCHMARK_TB2 = "terminal_bench_2"
BENCHMARK_SWEBENCH = "swe_bench_verified"
BENCHMARKS = (BENCHMARK_TB2, BENCHMARK_SWEBENCH)
HARNESS_KINDS = ("baseline", "candidate", "merged", "final")
SPLIT_NAMES = ("train", "heldout")
RESULT_STATUSES = ("complete", "partial", "failed")

CASE_STATUS_RESOLVED = "resolved"
CASE_STATUS_PASSED = "passed"
CASE_STATUS_UNRESOLVED = "unresolved"
CASE_STATUS_FAILED = "failed"
CASE_STATUS_AGENT_TIMEOUT = "agent_timeout"
CASE_STATUS_AGENT_ERROR = "agent_error"
CASE_STATUS_INFRA_ERROR = "infra_error"
CASE_STATUS_EVALUATOR_ERROR = "evaluator_error"
CASE_STATUS_NOT_RUN = "not_run"

CASE_STATUSES = (
    CASE_STATUS_RESOLVED,
    CASE_STATUS_PASSED,
    CASE_STATUS_UNRESOLVED,
    CASE_STATUS_FAILED,
    CASE_STATUS_AGENT_TIMEOUT,
    CASE_STATUS_AGENT_ERROR,
    CASE_STATUS_INFRA_ERROR,
    CASE_STATUS_EVALUATOR_ERROR,
    CASE_STATUS_NOT_RUN,
)
#: Evaluation outcomes that still count as a valid task-level failure.
SCOREABLE_FAILURE_STATUSES = (
    CASE_STATUS_UNRESOLVED,
    CASE_STATUS_FAILED,
    CASE_STATUS_AGENT_TIMEOUT,
    CASE_STATUS_AGENT_ERROR,
)
#: Outcomes that mean the task could not be scored at all.
UNSCOREABLE_STATUSES = (
    CASE_STATUS_INFRA_ERROR,
    CASE_STATUS_EVALUATOR_ERROR,
    CASE_STATUS_NOT_RUN,
)
SUCCESS_STATUS_BY_BENCHMARK = {
    BENCHMARK_TB2: CASE_STATUS_PASSED,
    BENCHMARK_SWEBENCH: CASE_STATUS_RESOLVED,
}
FAILURE_STATUS_BY_BENCHMARK = {
    BENCHMARK_TB2: CASE_STATUS_FAILED,
    BENCHMARK_SWEBENCH: CASE_STATUS_UNRESOLVED,
}

AGENT_STATUSES = ("complete", "timeout", "error", "not_run")
EVALUATOR_STATUSES = ("complete", "error", "not_run")

COMPARABILITY_KEYS = (
    "benchmark",
    "dataset",
    "decoding",
    "environment",
    "evaluator",
    "manifest",
    "model",
    "task_ids",
    "tools",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY_RE = re.compile(
    r"^(api[_-]?key|apikey|secret|client[_-]?secret|password|passwd|authorization|"
    r"auth[_-]?header|access[_-]?token|refresh[_-]?token|bearer[_-]?token|"
    r"credentials?|private[_-]?key)$|(_secret|_password|_api_key|_access_token)$",
    re.IGNORECASE,
)
_ABSOLUTE_PATH_RE = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")


class EvalResultError(ValueError):
    """Raised when an evaluation result violates the unified protocol."""


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise EvalResultError("expected an ISO-8601 UTC timestamp")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EvalResultError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def case_passed(status: str) -> bool | None:
    """Map a case status to the unified ``passed`` tri-state."""

    if status in (CASE_STATUS_RESOLVED, CASE_STATUS_PASSED):
        return True
    if status in SCOREABLE_FAILURE_STATUSES:
        return False
    if status in UNSCOREABLE_STATUSES:
        return None
    raise EvalResultError(f"unknown case status: {status!r}")


def is_unscoreable(status: str) -> bool:
    return status in UNSCOREABLE_STATUSES


def is_scoreable(status: str) -> bool:
    return status not in UNSCOREABLE_STATUSES


# ---------------------------------------------------------------------------
# Path and secret validation
# ---------------------------------------------------------------------------


def validate_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalResultError(f"{field} must be a non-empty relative path string")
    if _ABSOLUTE_PATH_RE.match(value):
        raise EvalResultError(f"{field} must be relative to the run root, got absolute-looking path {value!r}")
    if "\\" in value:
        raise EvalResultError(f"{field} must use POSIX separators, got {value!r}")
    parts = [part for part in value.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise EvalResultError(f"{field} must be a normalized relative path, got {value!r}")
    return "/".join(parts)


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise EvalResultError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _validate_optional_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvalResultError(f"{field} must be a string or null")
    return value


def find_secret_like_keys(value: Any, *, path: str = "$") -> list[str]:
    """Return dotted paths of keys that look like credential carriers."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{path}.{key_text}"
            if _SECRET_KEY_RE.search(key_text):
                found.append(child)
            found.extend(find_secret_like_keys(item, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(find_secret_like_keys(item, path=f"{path}[{index}]"))
    return found


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------


def _normalize_decoding(decoding: Mapping[str, Any]) -> dict[str, Any]:
    required = ("temperature", "top_p", "max_tokens")
    missing = [key for key in required if key not in decoding]
    if missing:
        raise EvalResultError(f"decoding block is missing: {', '.join(missing)}")
    normalized: dict[str, Any] = {
        "temperature": float(decoding["temperature"]),
        "top_p": float(decoding["top_p"]),
        "max_tokens": int(decoding["max_tokens"]),
    }
    for key, value in sorted(decoding.items()):
        if key not in normalized:
            normalized[str(key)] = value
    return normalized


def build_model_block(
    *,
    provider: str,
    model_id: str,
    endpoint_type: str,
    decoding: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``model`` block.

    Only non-sensitive deployment identifiers belong here.  Credentials are
    read from the environment by the runner and never persisted.
    """

    for name, value in (("provider", provider), ("model_id", model_id), ("endpoint_type", endpoint_type)):
        if not isinstance(value, str) or not value.strip():
            raise EvalResultError(f"model.{name} must be a non-empty string")
    normalized_decoding = _normalize_decoding(decoding)
    payload: dict[str, Any] = {
        "provider": provider,
        "model_id": model_id,
        "endpoint_type": endpoint_type,
        "decoding": normalized_decoding,
    }
    if extra:
        for key, value in sorted(extra.items()):
            payload.setdefault(str(key), value)
    payload["config_sha256"] = sha256_json(payload)
    return payload


def build_benchmark_block(
    *,
    benchmark_id: str,
    version: str,
    evaluator: str,
    evaluator_version: str,
    evaluator_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if benchmark_id not in BENCHMARKS:
        raise EvalResultError(f"unknown benchmark id: {benchmark_id!r}")
    for name, value in (
        ("version", version),
        ("evaluator", evaluator),
        ("evaluator_version", evaluator_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise EvalResultError(f"benchmark.{name} must be a non-empty string")
    block: dict[str, Any] = {
        "id": benchmark_id,
        "version": version,
        "evaluator": evaluator,
        "evaluator_version": evaluator_version,
    }
    if evaluator_config is not None:
        block["evaluator_config_sha256"] = sha256_json(dict(evaluator_config))
    return block


def build_dataset_block(
    *,
    run_root: Path,
    name: str,
    revision: str,
    manifest_path: Path,
    train_task_ids: Sequence[str],
    heldout_task_ids: Sequence[str],
    source_sha256: str | None = None,
) -> dict[str, Any]:
    for field_name, value in (("name", name), ("revision", revision)):
        if not isinstance(value, str) or not value.strip():
            raise EvalResultError(f"dataset.{field_name} must be a non-empty string")
    manifest = Path(manifest_path)
    block: dict[str, Any] = {
        "name": name,
        "revision": revision,
        "manifest_path": validate_relative_path(
            _relative_to_run(manifest, run_root), field="dataset.manifest_path"
        ),
        "manifest_sha256": _hash_file(manifest),
        "task_ids_sha256": task_ids_sha256(train_task_ids, heldout_task_ids),
    }
    if source_sha256 is not None:
        block["source_sha256"] = _validate_sha256(source_sha256, field="dataset.source_sha256")
    return block


def task_ids_sha256(train_task_ids: Sequence[str], heldout_task_ids: Sequence[str]) -> str:
    """Hash split membership and order; stable for run provenance."""

    return sha256_json(
        {
            "train": [str(item) for item in train_task_ids],
            "heldout": [str(item) for item in heldout_task_ids],
        }
    )


def build_harness_block(
    *,
    run_root: Path,
    snapshot_dir: Path,
    version_id: str,
    parent_version_ids: Sequence[str],
    kind: str,
) -> dict[str, Any]:
    if not isinstance(version_id, str) or not version_id.strip():
        raise EvalResultError("harness.version_id must be a non-empty string")
    if kind not in HARNESS_KINDS:
        raise EvalResultError(f"harness.kind must be one of {HARNESS_KINDS}")
    snapshot_dir = Path(snapshot_dir).resolve()
    if not snapshot_dir.is_dir():
        raise EvalResultError(f"harness snapshot directory does not exist: {snapshot_dir}")
    relative_files = tree_file_hashes(snapshot_dir)
    if not relative_files:
        raise EvalResultError(f"harness snapshot is empty: {snapshot_dir}")
    return {
        "version_id": version_id,
        "parent_version_ids": [str(item) for item in parent_version_ids],
        "kind": kind,
        "snapshot_path": validate_relative_path(
            _relative_to_run(snapshot_dir, run_root), field="harness.snapshot_path"
        ),
        "snapshot_sha256": snapshot_tree_hash(relative_files),
        "files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(relative_files.items())
        ],
    }


def build_config_block(
    *,
    run_root: Path,
    resolved_config_path: Path,
    repeats: int,
    task_timeout_s: float,
    tool_timeout_s: float,
    comparability: Mapping[str, str],
    network_limit: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(repeats) < 1:
        raise EvalResultError("config.repeats must be >= 1")
    if float(task_timeout_s) <= 0:
        raise EvalResultError("config.task_timeout_s must be positive")
    if float(tool_timeout_s) <= 0:
        raise EvalResultError("config.tool_timeout_s must be positive")
    normalized: dict[str, str] = {}
    for key in COMPARABILITY_KEYS:
        if key not in comparability:
            raise EvalResultError(f"config.comparability is missing required key {key!r}")
        normalized[key] = _validate_sha256(comparability[key], field=f"config.comparability.{key}")
    for key, value in sorted(comparability.items()):
        if key not in normalized:
            normalized[str(key)] = _validate_sha256(value, field=f"config.comparability.{key}")
    block: dict[str, Any] = {
        "resolved_config_path": validate_relative_path(
            _relative_to_run(Path(resolved_config_path), run_root), field="config.resolved_config_path"
        ),
        "resolved_config_sha256": _hash_file(resolved_config_path),
        "repeats": int(repeats),
        "task_timeout_s": float(task_timeout_s),
        "tool_timeout_s": float(tool_timeout_s),
        "comparability": normalized,
        "comparability_sha256": sha256_json(normalized),
    }
    if network_limit is not None:
        block["network_limit"] = dict(network_limit)
    if extra:
        for key, value in sorted(extra.items()):
            block.setdefault(str(key), value)
    return block


def _relative_to_run(path: Path, run_root: Path) -> str:
    resolved_root = Path(run_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise EvalResultError(
            f"artifacts must live under the run root: {resolved} is outside {resolved_root}"
        ) from exc


def _hash_file(path: Path) -> str:
    from .hashing import sha256_file

    return sha256_file(Path(path))


# ---------------------------------------------------------------------------
# Case / repeat builders
# ---------------------------------------------------------------------------


def make_agent_block(
    *,
    status: str,
    timed_out: bool,
    tool_calls: int | None,
    trace_path: str | None,
    attempts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if status not in AGENT_STATUSES:
        raise EvalResultError(f"agent.status must be one of {AGENT_STATUSES}")
    if trace_path is not None:
        trace_path = validate_relative_path(trace_path, field="case.agent.trace_path")
    block: dict[str, Any] = {
        "status": status,
        "timed_out": bool(timed_out),
        "tool_calls": int(tool_calls) if tool_calls is not None else None,
        "trace_path": trace_path,
    }
    if attempts is not None:
        block["attempts"] = [dict(item) for item in attempts]
    return block


def make_evaluator_block(
    *,
    status: str,
    name: str,
    resolved: bool | None,
    failure_summary: str | None,
    log_path: str | None,
) -> dict[str, Any]:
    if status not in EVALUATOR_STATUSES:
        raise EvalResultError(f"evaluator.status must be one of {EVALUATOR_STATUSES}")
    if not isinstance(name, str) or not name.strip():
        raise EvalResultError("evaluator.name must be a non-empty string")
    if resolved is not None and not isinstance(resolved, bool):
        raise EvalResultError("evaluator.resolved must be a bool or null")
    if status == "complete" and resolved is None:
        raise EvalResultError("evaluator.status=complete requires a boolean resolved value")
    if status != "complete" and resolved is not None:
        raise EvalResultError("evaluator.resolved must be null unless evaluator.status=complete")
    if log_path is not None:
        log_path = validate_relative_path(log_path, field="case.evaluator.log_path")
    return {
        "status": status,
        "name": name,
        "resolved": resolved,
        "failure_summary": _validate_optional_text(failure_summary, field="evaluator.failure_summary"),
        "log_path": log_path,
    }


def make_case_result(
    *,
    task_id: str,
    split: str,
    repeat: int,
    status: str,
    duration_s: float,
    agent: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    patch: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str | None] | None = None,
    repository: str | None = None,
    domain: str | None = None,
    error: str | None = None,
    passed: bool | None = None,
) -> dict[str, Any]:
    """Build and validate one task-level case result."""

    if not isinstance(task_id, str) or not task_id.strip():
        raise EvalResultError("case.task_id must be a non-empty string")
    if split not in SPLIT_NAMES:
        raise EvalResultError(f"case.split must be one of {SPLIT_NAMES}")
    if int(repeat) < 1:
        raise EvalResultError("case.repeat must be >= 1")
    if status not in CASE_STATUSES:
        raise EvalResultError(f"unknown case status: {status!r}")
    expected_passed = case_passed(status)
    if passed is not None and bool(passed) is not expected_passed and expected_passed is not None:
        raise EvalResultError(
            f"case.passed={passed!r} contradicts case.status={status!r} (expected {expected_passed!r})"
        )
    if expected_passed is None and passed is not None:
        raise EvalResultError(f"case.status={status!r} is not scoreable; case.passed must be null")
    duration = float(duration_s)
    if duration < 0:
        raise EvalResultError("case.duration_s must be >= 0")

    agent_block = _validate_agent_block(agent)
    evaluator_block = _validate_evaluator_block(evaluator)
    _assert_case_consistency(status=status, agent=agent_block, evaluator=evaluator_block)

    patch_block: dict[str, str] | None = None
    if patch is not None:
        if not isinstance(patch, Mapping):
            raise EvalResultError("case.patch must be an object or null")
        patch_block = {
            "path": validate_relative_path(patch.get("path"), field="case.patch.path"),
            "sha256": _validate_sha256(patch.get("sha256"), field="case.patch.sha256"),
        }
    artifact_block: dict[str, str | None] = {}
    for key in ("case_dir", "agent_stdout", "agent_stderr", "summary"):
        value = (artifacts or {}).get(key)
        if value is None:
            artifact_block[key] = None
        else:
            artifact_block[key] = validate_relative_path(value, field=f"case.artifacts.{key}")
    for key, value in sorted((artifacts or {}).items()):
        if key in artifact_block or value is None:
            continue
        artifact_block[str(key)] = validate_relative_path(value, field=f"case.artifacts.{key}")

    error_text = _validate_optional_text(error, field="case.error")
    if expected_passed is None and not (error_text or "").strip():
        raise EvalResultError(
            f"case.status={status!r} is not scoreable and must carry an explicit case.error message"
        )

    return {
        "task_id": task_id,
        "split": split,
        "repeat": int(repeat),
        "repository": _validate_optional_text(repository, field="case.repository"),
        "domain": _validate_optional_text(domain, field="case.domain"),
        "status": status,
        "passed": expected_passed,
        "duration_s": round(duration, 3),
        "agent": agent_block,
        "evaluator": evaluator_block,
        "patch": patch_block,
        "artifacts": artifact_block,
        "error": error_text,
    }


def _validate_agent_block(agent: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(agent, Mapping):
        raise EvalResultError("case.agent must be an object")
    return make_agent_block(
        status=str(agent.get("status")),
        timed_out=bool(agent.get("timed_out")),
        tool_calls=agent.get("tool_calls"),
        trace_path=agent.get("trace_path"),
        attempts=agent.get("attempts") if isinstance(agent.get("attempts"), list) else None,
    )


def _validate_evaluator_block(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evaluator, Mapping):
        raise EvalResultError("case.evaluator must be an object")
    return make_evaluator_block(
        status=str(evaluator.get("status")),
        name=str(evaluator.get("name") or ""),
        resolved=evaluator.get("resolved"),
        failure_summary=evaluator.get("failure_summary"),
        log_path=evaluator.get("log_path"),
    )


def _assert_case_consistency(*, status: str, agent: Mapping[str, Any], evaluator: Mapping[str, Any]) -> None:
    if status in (CASE_STATUS_RESOLVED, CASE_STATUS_PASSED):
        if evaluator["status"] != "complete" or evaluator["resolved"] is not True:
            raise EvalResultError(
                f"case.status={status!r} requires evaluator.status=complete and evaluator.resolved=true"
            )
        if agent["status"] != "complete":
            raise EvalResultError(f"case.status={status!r} requires agent.status=complete")
    elif status in (CASE_STATUS_UNRESOLVED, CASE_STATUS_FAILED):
        if evaluator["status"] != "complete" or evaluator["resolved"] is not False:
            raise EvalResultError(
                f"case.status={status!r} requires evaluator.status=complete and evaluator.resolved=false"
            )
        if agent["status"] != "complete":
            raise EvalResultError(f"case.status={status!r} requires agent.status=complete")
    elif status == CASE_STATUS_AGENT_TIMEOUT:
        if agent["status"] != "timeout":
            raise EvalResultError("case.status=agent_timeout requires agent.status=timeout")
        if evaluator["resolved"] is not None:
            raise EvalResultError("agent_timeout cases must not carry an evaluator resolved value")
    elif status == CASE_STATUS_AGENT_ERROR:
        if agent["status"] != "error":
            raise EvalResultError("case.status=agent_error requires agent.status=error")
        if evaluator["resolved"] is not None:
            raise EvalResultError("agent_error cases must not carry an evaluator resolved value")
    elif status == CASE_STATUS_INFRA_ERROR:
        if evaluator["status"] == "complete":
            raise EvalResultError("infra_error cases must not carry a complete evaluator result")
    elif status == CASE_STATUS_EVALUATOR_ERROR:
        if evaluator["status"] != "error":
            raise EvalResultError("case.status=evaluator_error requires evaluator.status=error")
    elif status == CASE_STATUS_NOT_RUN:
        if evaluator["status"] == "complete":
            raise EvalResultError("not_run cases must not carry a complete evaluator result")
        if agent["status"] != "not_run":
            raise EvalResultError("not_run cases require agent.status=not_run")


def compute_repeat_metrics(
    *,
    split: str,
    repeat: int,
    expected_task_ids: Sequence[str],
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute repeat metrics from case results.  Single source of truth."""

    if split not in SPLIT_NAMES:
        raise EvalResultError(f"repeat.split must be one of {SPLIT_NAMES}")
    expected = [str(item) for item in expected_task_ids]
    if not expected:
        raise EvalResultError("repeat must be configured with at least one task id")
    if len(set(expected)) != len(expected):
        raise EvalResultError("repeat expected_task_ids contains duplicates")

    by_task: dict[str, Mapping[str, Any]] = {}
    for case in case_results:
        if not isinstance(case, Mapping):
            raise EvalResultError("case_results entries must be objects")
        task_id = str(case.get("task_id"))
        if task_id in by_task:
            raise EvalResultError(f"repeat {split}/{repeat} contains duplicate task_id {task_id!r}")
        by_task[task_id] = case
    missing = [task_id for task_id in expected if task_id not in by_task]
    unexpected = sorted(set(by_task) - set(expected))
    if missing or unexpected:
        raise EvalResultError(
            f"repeat {split}/{repeat} task id set does not match the fixed manifest; "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    ordered = [by_task[task_id] for task_id in expected]
    passed = 0
    unscoreable = 0
    durations: list[float] = []
    for case in ordered:
        if str(case.get("split")) != split or int(case.get("repeat", 0)) != int(repeat):
            raise EvalResultError(
                f"case {case.get('task_id')!r} split/repeat does not match its repeat record"
            )
        status = str(case.get("status"))
        expected_passed = case_passed(status)
        if case.get("passed") is not expected_passed:
            raise EvalResultError(
                f"case {case.get('task_id')!r} passed={case.get('passed')!r} contradicts status={status!r}"
            )
        if expected_passed is True:
            passed += 1
        elif expected_passed is None:
            unscoreable += 1
        durations.append(float(case.get("duration_s") or 0.0))

    total = len(expected)
    if unscoreable:
        all_not_run = all(
            str(case.get("status")) == CASE_STATUS_NOT_RUN for case in ordered
        )
        repeat_status = "failed" if all_not_run else "partial"
    else:
        repeat_status = "complete"
    mean_duration = round(sum(durations) / total, 3) if total else 0.0
    return {
        "split": split,
        "repeat": int(repeat),
        "status": repeat_status,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total,
        "mean_case_duration_s": mean_duration,
        "duration_s": round(sum(durations), 3),
        "case_results": [dict(case) for case in ordered],
    }


def make_repeat_result(
    *,
    split: str,
    repeat: int,
    expected_task_ids: Sequence[str],
    case_results: Sequence[Mapping[str, Any]],
    duration_s: float | None = None,
) -> dict[str, Any]:
    record = compute_repeat_metrics(
        split=split,
        repeat=repeat,
        expected_task_ids=expected_task_ids,
        case_results=case_results,
    )
    if duration_s is not None:
        measured = float(duration_s)
        if measured < 0:
            raise EvalResultError("repeat.duration_s must be >= 0")
        record["duration_s"] = round(measured, 3)
    return record


def compute_split_summary(
    *,
    split: str,
    repeats: Sequence[Mapping[str, Any]],
    expected_repeats: int,
) -> dict[str, Any]:
    if expected_repeats < 1:
        raise EvalResultError("expected_repeats must be >= 1")
    repeat_ids = sorted(int(item["repeat"]) for item in repeats)
    if repeat_ids != list(range(1, expected_repeats + 1)):
        raise EvalResultError(
            f"split {split!r} repeat ids must be exactly 1..{expected_repeats}, got {repeat_ids}"
        )
    ordered = sorted(repeats, key=lambda item: int(item["repeat"]))
    totals = [int(item["total"]) for item in ordered]
    if len(set(totals)) != 1:
        raise EvalResultError(f"split {split!r} repeats have inconsistent totals: {totals}")
    passed_per_repeat = [int(item["passed"]) for item in ordered]
    pass_rate_per_repeat = [int(item["passed"]) / int(item["total"]) for item in ordered]
    infrastructure_error_count = 0
    evaluator_error_count = 0
    not_run_count = 0
    for repeat in ordered:
        for case in repeat["case_results"]:
            status = str(case["status"])
            if status == CASE_STATUS_INFRA_ERROR:
                infrastructure_error_count += 1
            elif status == CASE_STATUS_EVALUATOR_ERROR:
                evaluator_error_count += 1
            elif status == CASE_STATUS_NOT_RUN:
                not_run_count += 1
    pooled_passed = sum(passed_per_repeat)
    pooled_total = sum(totals)
    return {
        "repeats": expected_repeats,
        "passed_per_repeat": passed_per_repeat,
        "total_per_repeat": totals,
        "pass_rate_per_repeat": pass_rate_per_repeat,
        "mean_repeat_pass_rate": sum(pass_rate_per_repeat) / len(pass_rate_per_repeat),
        "pooled_passed": pooled_passed,
        "pooled_total": pooled_total,
        "pooled_pass_rate": pooled_passed / pooled_total if pooled_total else 0.0,
        "infrastructure_error_count": infrastructure_error_count,
        "evaluator_error_count": evaluator_error_count,
        "not_run_count": not_run_count,
    }


# ---------------------------------------------------------------------------
# Assembly, validation, recomputation
# ---------------------------------------------------------------------------


def _derive_result_status(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    statuses: list[str] = []
    for repeat_list in splits.values():
        for repeat in repeat_list:
            statuses.append(str(repeat.get("status")))
    if not statuses:
        return "failed"
    if all(status == "complete" for status in statuses):
        return "complete"
    if all(status == "failed" for status in statuses):
        return "failed"
    return "partial"


def build_eval_result(
    *,
    run_id: str,
    result_id: str,
    benchmark: Mapping[str, Any],
    model: Mapping[str, Any],
    harness: Mapping[str, Any],
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    started_at: str,
    finished_at: str,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    artifacts: Mapping[str, str | None] | None = None,
    errors: Sequence[Mapping[str, Any]] | None = None,
    status: str | None = None,
    expected_repeats: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise EvalResultError("run_id must be a non-empty string")
    if not isinstance(result_id, str) or not result_id.strip():
        raise EvalResultError("result_id must be a non-empty string")
    start = parse_utc(started_at)
    finish = parse_utc(finished_at)
    duration = (finish - start).total_seconds()
    if duration < 0:
        raise EvalResultError("finished_at precedes started_at")

    normalized_splits: dict[str, list[dict[str, Any]]] = {}
    repeats_expected = expected_repeats
    for split in SPLIT_NAMES:
        if split not in splits:
            continue
        raw_repeats = list(splits[split])
        if not raw_repeats:
            raise EvalResultError(f"split {split!r} must contain at least one repeat")
        if repeats_expected is None:
            repeats_expected = len(raw_repeats)
        normalized_splits[split] = [dict(item) for item in raw_repeats]
    if not normalized_splits:
        raise EvalResultError("result must contain at least one split")
    for split in normalized_splits:
        if len(normalized_splits[split]) != repeats_expected:
            raise EvalResultError(
                f"split {split!r} must contain exactly {repeats_expected} repeats"
            )

    summary = {
        split: compute_split_summary(
            split=split,
            repeats=repeat_list,
            expected_repeats=int(repeats_expected),
        )
        for split, repeat_list in normalized_splits.items()
    }
    derived_status = status or _derive_result_status(normalized_splits)
    if derived_status not in RESULT_STATUSES:
        raise EvalResultError(f"result.status must be one of {RESULT_STATUSES}")

    artifact_block: dict[str, str | None] = {}
    for key, value in sorted((artifacts or {}).items()):
        artifact_block[str(key)] = (
            None if value is None else validate_relative_path(value, field=f"artifacts.{key}")
        )

    error_block = _normalize_errors(errors or [], normalized_splits)

    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "result_id": result_id,
        "status": derived_status,
        "benchmark": dict(benchmark),
        "model": dict(model),
        "harness": dict(harness),
        "dataset": dict(dataset),
        "config": dict(config),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration, 3),
        "splits": normalized_splits,
        "summary": summary,
        "artifacts": artifact_block,
        "errors": error_block,
    }
    if extra:
        for key, value in sorted(extra.items()):
            if key in doc:
                raise EvalResultError(f"extra result field would overwrite reserved key {key!r}")
            doc[str(key)] = value
    validate_result(doc, expected_repeats=repeats_expected)
    return doc


def _normalize_errors(
    errors: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in errors:
        if not isinstance(item, Mapping):
            raise EvalResultError("errors entries must be objects")
        entry: dict[str, Any] = {}
        for key in ("split", "repeat", "task_id", "status", "message"):
            value = item.get(key)
            if value is None:
                entry[key] = None
            elif key == "repeat":
                entry[key] = int(value)
            else:
                entry[key] = str(value)
        if not (entry["message"] or "").strip():
            raise EvalResultError("errors entries must include a non-empty message")
        normalized.append(entry)

    explicit = {
        (entry.get("split"), entry.get("repeat"), entry.get("task_id"))
        for entry in normalized
        if entry.get("task_id") is not None
    }
    for split, repeat_list in splits.items():
        for repeat in repeat_list:
            repeat_id = int(repeat.get("repeat", 0))
            for case in repeat.get("case_results", []):
                status = str(case.get("status"))
                if not is_unscoreable(status):
                    continue
                key = (split, repeat_id, str(case.get("task_id")))
                if key in explicit:
                    continue
                if (case.get("error") or "").strip():
                    normalized.append(
                        {
                            "split": split,
                            "repeat": repeat_id,
                            "task_id": key[2],
                            "status": status,
                            "message": str(case["error"]),
                        }
                    )
                else:
                    raise EvalResultError(
                        f"unscoreable case {key[2]!r} ({status}) must carry an error message"
                    )
    return normalized


def recompute_result_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every repeat/split metric from ``case_results`` only."""

    if not isinstance(result, Mapping):
        raise EvalResultError("result must be an object")
    splits = result.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        raise EvalResultError("result.splits must be a non-empty object")
    recomputed: dict[str, Any] = {"splits": {}, "status": None}
    repeat_counts: set[int] = set()
    for split in SPLIT_NAMES:
        if split not in splits:
            continue
        raw_repeats = splits[split]
        if not isinstance(raw_repeats, list) or not raw_repeats:
            raise EvalResultError(f"result.splits.{split} must be a non-empty list")
        repeat_records: list[dict[str, Any]] = []
        for raw in raw_repeats:
            if not isinstance(raw, Mapping):
                raise EvalResultError(f"result.splits.{split} entries must be objects")
            cases = raw.get("case_results")
            if not isinstance(cases, list):
                raise EvalResultError(f"repeat {split}/{raw.get('repeat')} is missing case_results")
            expected_ids = [str(case.get("task_id")) for case in cases]
            repeat_record = compute_repeat_metrics(
                split=split,
                repeat=int(raw.get("repeat", 0)),
                expected_task_ids=expected_ids,
                case_results=cases,
            )
            repeat_record["stored_duration_s"] = raw.get("duration_s")
            repeat_records.append(repeat_record)
        repeat_ids = sorted(int(item["repeat"]) for item in repeat_records)
        if len(set(repeat_ids)) != len(repeat_ids):
            raise EvalResultError(f"split {split!r} contains duplicate repeat ids")
        repeat_counts.add(len(repeat_records))
        recomputed["splits"][split] = {
            "repeats": repeat_records,
            "summary": compute_split_summary(
                split=split,
                repeats=repeat_records,
                expected_repeats=len(repeat_records),
            ),
        }
    if len(repeat_counts) != 1:
        raise EvalResultError("all splits must contain the same number of repeats")
    recomputed["status"] = _derive_result_status(splits)
    return recomputed


def validate_result(
    result: Mapping[str, Any],
    *,
    expected_repeats: int | None = None,
    expected_splits: Iterable[str] | None = None,
    expected_task_ids: Mapping[str, Sequence[str]] | None = None,
    require_complete: bool = False,
) -> None:
    """Validate a result document, recomputing all derived metrics."""

    if not isinstance(result, Mapping):
        raise EvalResultError("result must be a JSON object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise EvalResultError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {result.get('schema_version')!r}"
        )
    for key in (
        "run_id",
        "result_id",
        "status",
        "benchmark",
        "model",
        "harness",
        "dataset",
        "config",
        "started_at",
        "finished_at",
        "duration_s",
        "splits",
        "summary",
        "artifacts",
        "errors",
    ):
        if key not in result:
            raise EvalResultError(f"result is missing required field {key!r}")
    if not isinstance(result["run_id"], str) or not result["run_id"].strip():
        raise EvalResultError("result.run_id must be a non-empty string")
    if not isinstance(result["result_id"], str) or not result["result_id"].strip():
        raise EvalResultError("result.result_id must be a non-empty string")
    if result["status"] not in RESULT_STATUSES:
        raise EvalResultError(f"result.status must be one of {RESULT_STATUSES}")

    _validate_benchmark_block(result["benchmark"])
    _validate_model_block(result["model"])
    _validate_harness_block(result["harness"])
    _validate_dataset_block(result["dataset"])
    _validate_config_block(result["config"])

    started = parse_utc(result["started_at"])
    finished = parse_utc(result["finished_at"])
    if finished < started - timedelta(milliseconds=1):
        raise EvalResultError("result.finished_at must not precede result.started_at")
    duration = float(result["duration_s"])
    if duration < 0:
        raise EvalResultError("result.duration_s must be >= 0")
    computed_duration = (finished - started).total_seconds()
    if abs(duration - computed_duration) > 1.0:
        raise EvalResultError(
            f"result.duration_s={duration} disagrees with timestamps ({computed_duration:.3f})"
        )

    spl = result["splits"]
    if not isinstance(spl, Mapping) or not spl:
        raise EvalResultError("result.splits must be a non-empty object")
    for key in spl:
        if key not in SPLIT_NAMES:
            raise EvalResultError(f"unknown split name: {key!r}")
    if expected_splits is not None:
        missing = sorted(set(expected_splits) - set(spl))
        if missing:
            raise EvalResultError(f"result is missing required splits: {missing}")
    if expected_repeats is not None and expected_repeats < 1:
        raise EvalResultError("expected_repeats must be >= 1")

    recomputed = recompute_result_metrics(result)
    for split in spl:
        stored_summary = result["summary"].get(split) if isinstance(result.get("summary"), Mapping) else None
        if not isinstance(stored_summary, Mapping):
            raise EvalResultError(f"result.summary is missing split {split!r}")
        _compare_summary(split=split, stored=stored_summary, computed=recomputed["splits"][split]["summary"])
        if require_complete and stored_summary.get("infrastructure_error_count", 0):
            raise EvalResultError(f"split {split!r} contains infrastructure errors")
        if require_complete and stored_summary.get("evaluator_error_count", 0):
            raise EvalResultError(f"split {split!r} contains evaluator errors")
        if require_complete and stored_summary.get("not_run_count", 0):
            raise EvalResultError(f"split {split!r} contains not_run cases")

    _validate_repeat_records(spl)

    expected_repeats_actual = {len(value) for value in spl.values()}
    if len(expected_repeats_actual) != 1:
        raise EvalResultError("all splits must contain the same number of repeats")
    actual_repeats = next(iter(expected_repeats_actual))
    if expected_repeats is not None and actual_repeats != expected_repeats:
        raise EvalResultError(
            f"result must contain exactly {expected_repeats} repeats per split, got {actual_repeats}"
        )
    for split, repeat_list in spl.items():
        ids = sorted(int(item.get("repeat", 0)) for item in repeat_list)
        if ids != list(range(1, actual_repeats + 1)):
            raise EvalResultError(f"split {split!r} repeat ids must be exactly 1..{actual_repeats}")

    derived_status = _derive_result_status(spl)
    if result["status"] != derived_status:
        raise EvalResultError(
            f"result.status={result['status']!r} contradicts case outcomes ({derived_status!r})"
        )
    if require_complete and derived_status != "complete":
        raise EvalResultError(f"result.status must be complete, got {derived_status!r}")

    if expected_task_ids is not None:
        for split, expected_ids in expected_task_ids.items():
            if split not in spl:
                continue
            expected_list = [str(item) for item in expected_ids]
            for repeat in spl[split]:
                case_ids = [str(case.get("task_id")) for case in repeat.get("case_results", [])]
                if case_ids != expected_list:
                    raise EvalResultError(
                        f"split {split!r} repeat {repeat.get('repeat')} task ids do not match the "
                        "fixed manifest order"
                    )

    _validate_case_records(result, benchmark_id=str(result["benchmark"].get("id")))
    _validate_summary_keys(result)
    _validate_artifact_block(result.get("artifacts"))
    errors = result.get("errors")
    if not isinstance(errors, list):
        raise EvalResultError("result.errors must be a list")
    for entry in errors:
        if not isinstance(entry, Mapping):
            raise EvalResultError("result.errors entries must be objects")
        if not str(entry.get("message") or "").strip():
            raise EvalResultError("result.errors entries require a message")
    _assert_unscoreable_errors_present(result)
    _assert_no_secrets(result)


def _validate_repeat_records(splits: Mapping[str, Any]) -> None:
    """Recompute every repeat record from its own case_results and compare."""

    for split, repeat_list in splits.items():
        for stored in repeat_list:
            if not isinstance(stored, Mapping):
                raise EvalResultError(f"split {split!r} repeat entries must be objects")
            cases = stored.get("case_results")
            if not isinstance(cases, list) or not cases:
                raise EvalResultError(
                    f"repeat {split}/{stored.get('repeat')} must contain case_results"
                )
            task_ids = [str(case.get("task_id")) for case in cases]
            computed = compute_repeat_metrics(
                split=split,
                repeat=int(stored.get("repeat", 0)),
                expected_task_ids=task_ids,
                case_results=cases,
            )
            for key in ("passed", "total", "status"):
                if stored.get(key) != computed[key]:
                    raise EvalResultError(
                        f"repeat {split}/{stored.get('repeat')} {key}="
                        f"{stored.get(key)!r} disagrees with case_results ({computed[key]!r})"
                    )
            if abs(float(stored.get("pass_rate", -1.0)) - float(computed["pass_rate"])) > 1e-9:
                raise EvalResultError(
                    f"repeat {split}/{stored.get('repeat')} pass_rate disagrees with case_results"
                )
            if (
                abs(
                    float(stored.get("mean_case_duration_s", -1.0))
                    - float(computed["mean_case_duration_s"])
                )
                > 1e-6
            ):
                raise EvalResultError(
                    f"repeat {split}/{stored.get('repeat')} mean_case_duration_s disagrees "
                    "with case_results"
                )


def _validate_summary_keys(result: Mapping[str, Any]) -> None:
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise EvalResultError("result.summary must be an object")
    required = {
        "repeats",
        "passed_per_repeat",
        "total_per_repeat",
        "pass_rate_per_repeat",
        "mean_repeat_pass_rate",
        "pooled_passed",
        "pooled_total",
        "pooled_pass_rate",
        "infrastructure_error_count",
        "evaluator_error_count",
    }
    for split, block in summary.items():
        if not isinstance(block, Mapping):
            raise EvalResultError(f"result.summary.{split} must be an object")
        missing = required - set(block)
        if missing:
            raise EvalResultError(f"result.summary.{split} is missing {sorted(missing)}")


def _compare_summary(*, split: str, stored: Mapping[str, Any], computed: Mapping[str, Any]) -> None:
    for key in (
        "repeats",
        "passed_per_repeat",
        "total_per_repeat",
        "infrastructure_error_count",
        "evaluator_error_count",
        "pooled_passed",
        "pooled_total",
    ):
        if stored.get(key) != computed.get(key):
            raise EvalResultError(
                f"summary {split}.{key}={stored.get(key)!r} disagrees with case_results "
                f"({computed.get(key)!r})"
            )
    for key in ("mean_repeat_pass_rate", "pooled_pass_rate"):
        if abs(float(stored.get(key, -1)) - float(computed[key])) > 1e-9:
            raise EvalResultError(
                f"summary {split}.{key} disagrees with case_results"
            )
    stored_rates = stored.get("pass_rate_per_repeat")
    computed_rates = computed.get("pass_rate_per_repeat")
    if not isinstance(stored_rates, list) or len(stored_rates) != len(computed_rates):
        raise EvalResultError(f"summary {split}.pass_rate_per_repeat is malformed")
    for index, value in enumerate(stored_rates):
        if abs(float(value) - float(computed_rates[index])) > 1e-9:
            raise EvalResultError(f"summary {split}.pass_rate_per_repeat disagrees with case_results")


def _validate_case_records(result: Mapping[str, Any], *, benchmark_id: str) -> None:
    allowed_success = SUCCESS_STATUS_BY_BENCHMARK.get(benchmark_id)
    allowed_failure = FAILURE_STATUS_BY_BENCHMARK.get(benchmark_id)
    for split, repeat_list in result["splits"].items():
        for repeat in repeat_list:
            repeat_id = int(repeat.get("repeat", 0))
            cases = repeat.get("case_results")
            if not isinstance(cases, list):
                raise EvalResultError(f"repeat {split}/{repeat_id} is missing case_results")
            for case in cases:
                if not isinstance(case, Mapping):
                    raise EvalResultError("case_results entries must be objects")
                for key in (
                    "task_id",
                    "split",
                    "repeat",
                    "repository",
                    "domain",
                    "status",
                    "passed",
                    "duration_s",
                    "agent",
                    "evaluator",
                    "patch",
                    "artifacts",
                    "error",
                ):
                    if key not in case:
                        raise EvalResultError(f"case is missing required field {key!r}")
                status = str(case["status"])
                if status not in CASE_STATUSES:
                    raise EvalResultError(f"unknown case status: {status!r}")
                if status == CASE_STATUS_RESOLVED and benchmark_id != BENCHMARK_SWEBENCH:
                    raise EvalResultError("status=resolved is only valid for SWE-bench results")
                if status == CASE_STATUS_PASSED and benchmark_id != BENCHMARK_TB2:
                    raise EvalResultError("status=passed is only valid for Terminal-Bench results")
                if status in (CASE_STATUS_RESOLVED, CASE_STATUS_PASSED) and status != allowed_success:
                    raise EvalResultError(f"status={status!r} is not the success status for this benchmark")
                if status in (CASE_STATUS_UNRESOLVED, CASE_STATUS_FAILED) and status != allowed_failure:
                    raise EvalResultError(f"status={status!r} is not the failure status for this benchmark")
                if str(case["split"]) != split or int(case["repeat"]) != repeat_id:
                    raise EvalResultError("case split/repeat does not match its repeat record")
                if case["passed"] is not case_passed(status):
                    raise EvalResultError(
                        f"case {case.get('task_id')!r} passed value contradicts status"
                    )
                if benchmark_id == BENCHMARK_TB2 and case["patch"] is not None:
                    raise EvalResultError("Terminal-Bench cases must set patch to null")
                if case["patch"] is not None:
                    patch_block = case["patch"]
                    if not isinstance(patch_block, Mapping):
                        raise EvalResultError("case.patch must be an object or null")
                    validate_relative_path(patch_block.get("path"), field="case.patch.path")
                    _validate_sha256(patch_block.get("sha256"), field="case.patch.sha256")
                _validate_agent_block(case["agent"])
                _validate_evaluator_block(case["evaluator"])
                _assert_case_consistency(
                    status=status, agent=case["agent"], evaluator=case["evaluator"]
                )
                if is_unscoreable(status) and not (case.get("error") or "").strip():
                    raise EvalResultError(
                        f"unscoreable case {case.get('task_id')!r} must carry case.error"
                    )


def _validate_benchmark_block(block: Any) -> None:
    if not isinstance(block, Mapping):
        raise EvalResultError("result.benchmark must be an object")
    if block.get("id") not in BENCHMARKS:
        raise EvalResultError(f"benchmark.id must be one of {BENCHMARKS}")
    for key in ("version", "evaluator", "evaluator_version"):
        if not isinstance(block.get(key), str) or not block[key].strip():
            raise EvalResultError(f"benchmark.{key} must be a non-empty string")


def _validate_model_block(block: Any) -> None:
    if not isinstance(block, Mapping):
        raise EvalResultError("result.model must be an object")
    for key in ("provider", "model_id", "endpoint_type"):
        if not isinstance(block.get(key), str) or not block[key].strip():
            raise EvalResultError(f"model.{key} must be a non-empty string")
    decoding = block.get("decoding")
    if not isinstance(decoding, Mapping):
        raise EvalResultError("model.decoding must be an object")
    for key in ("temperature", "top_p", "max_tokens"):
        if key not in decoding:
            raise EvalResultError(f"model.decoding is missing {key!r}")
    _validate_sha256(block.get("config_sha256"), field="model.config_sha256")


def _validate_harness_block(block: Any) -> None:
    if not isinstance(block, Mapping):
        raise EvalResultError("result.harness must be an object")
    if not isinstance(block.get("version_id"), str) or not block["version_id"].strip():
        raise EvalResultError("harness.version_id must be a non-empty string")
    if block.get("kind") not in HARNESS_KINDS:
        raise EvalResultError(f"harness.kind must be one of {HARNESS_KINDS}")
    parents = block.get("parent_version_ids")
    if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
        raise EvalResultError("harness.parent_version_ids must be a list of strings")
    if block["kind"] != "baseline" and not parents:
        raise EvalResultError("non-baseline harness versions must declare parent_version_ids")
    validate_relative_path(block.get("snapshot_path"), field="harness.snapshot_path")
    files = block.get("files")
    if not isinstance(files, list) or not files:
        raise EvalResultError("harness.files must be a non-empty list")
    file_map: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise EvalResultError("harness.files entries must be objects")
        relative = validate_relative_path(item.get("path"), field="harness.files[].path")
        digest = _validate_sha256(item.get("sha256"), field="harness.files[].sha256")
        if relative in file_map:
            raise EvalResultError(f"harness.files contains duplicate path {relative!r}")
        file_map[relative] = digest
    expected_tree = snapshot_tree_hash(file_map)
    if block.get("snapshot_sha256") != expected_tree:
        raise EvalResultError("harness.snapshot_sha256 does not match harness.files")


def _validate_dataset_block(block: Any) -> None:
    if not isinstance(block, Mapping):
        raise EvalResultError("result.dataset must be an object")
    for key in ("name", "revision"):
        if not isinstance(block.get(key), str) or not block[key].strip():
            raise EvalResultError(f"dataset.{key} must be a non-empty string")
    validate_relative_path(block.get("manifest_path"), field="dataset.manifest_path")
    _validate_sha256(block.get("manifest_sha256"), field="dataset.manifest_sha256")
    _validate_sha256(block.get("task_ids_sha256"), field="dataset.task_ids_sha256")


def _validate_config_block(block: Any) -> None:
    if not isinstance(block, Mapping):
        raise EvalResultError("result.config must be an object")
    validate_relative_path(block.get("resolved_config_path"), field="config.resolved_config_path")
    _validate_sha256(block.get("resolved_config_sha256"), field="config.resolved_config_sha256")
    if int(block.get("repeats", 0)) < 1:
        raise EvalResultError("config.repeats must be >= 1")
    for key in ("task_timeout_s", "tool_timeout_s"):
        if float(block.get(key, 0)) <= 0:
            raise EvalResultError(f"config.{key} must be positive")
    comparability = block.get("comparability")
    if not isinstance(comparability, Mapping):
        raise EvalResultError("config.comparability must be an object")
    for key in COMPARABILITY_KEYS:
        _validate_sha256(comparability.get(key), field=f"config.comparability.{key}")
    if block.get("comparability_sha256") != sha256_json(dict(comparability)):
        raise EvalResultError("config.comparability_sha256 does not match config.comparability")
    network = block.get("network_limit")
    if network is not None:
        if not isinstance(network, Mapping):
            raise EvalResultError("config.network_limit must be an object when present")
        if "enforced" not in network:
            raise EvalResultError("config.network_limit must record whether the limit was enforced")


def _validate_artifact_block(block: Any) -> None:
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise EvalResultError("result.artifacts must be an object")
    for key, value in block.items():
        if value is not None:
            validate_relative_path(value, field=f"artifacts.{key}")


def _assert_unscoreable_errors_present(result: Mapping[str, Any]) -> None:
    errors = result.get("errors") or []
    covered = {
        (entry.get("split"), int(entry["repeat"]) if entry.get("repeat") is not None else None, entry.get("task_id"))
        for entry in errors
        if entry.get("task_id") is not None
    }
    for split, repeat_list in result["splits"].items():
        for repeat in repeat_list:
            for case in repeat.get("case_results", []):
                if not is_unscoreable(str(case.get("status"))):
                    continue
                key = (split, int(case.get("repeat", 0)), case.get("task_id"))
                if key not in covered and not (case.get("error") or "").strip():
                    raise EvalResultError(
                        f"unscoreable case {key[2]!r} is not recorded in result.errors"
                    )


def _assert_no_secrets(result: Mapping[str, Any]) -> None:
    found = find_secret_like_keys(result)
    if found:
        raise EvalResultError(
            "evaluation result must not persist credentials; suspicious keys: "
            + ", ".join(found[:10])
        )


def write_result_immutable(path: Path, result: Mapping[str, Any]) -> Path:
    """Validate ``result`` and write it once.  Existing files are never replaced."""

    validate_result(result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise EvalResultError(
            f"refusing to overwrite immutable evaluation result: {target}"
        ) from exc
    return target


def summarize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Small helper used by reports and logs."""

    return {
        "run_id": result.get("run_id"),
        "result_id": result.get("result_id"),
        "status": result.get("status"),
        "splits": {
            split: {
                "mean_repeat_pass_rate": block.get("mean_repeat_pass_rate"),
                "pooled_pass_rate": block.get("pooled_pass_rate"),
                "pooled_passed": block.get("pooled_passed"),
                "pooled_total": block.get("pooled_total"),
            }
            for split, block in (result.get("summary") or {}).items()
        },
        "errors": len(result.get("errors") or []),
    }
