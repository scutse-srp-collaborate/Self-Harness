"""Terminal-Bench 2.0 runner that preserves Harbor + pytest semantics.

The runner executes the configured pytest eval project (which drives a Harbor
agent), reads JUnit XML, and converts every task into the unified
``self_harness.eval_result.v1`` protocol.  It keeps the historical per-repeat
and per-case artifact layout and additionally writes ``result.legacy.json`` for
older tooling.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import importlib.metadata
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bandwidth import bandwidth_plan_from_settings
from .config import EvalConfig
from .hashing import sha256_file
from .protocol import (
    CASE_STATUS_AGENT_TIMEOUT,
    CASE_STATUS_EVALUATOR_ERROR,
    CASE_STATUS_FAILED,
    CASE_STATUS_NOT_RUN,
    CASE_STATUS_PASSED,
    SPLIT_NAMES,
    make_agent_block,
    make_case_result,
    make_evaluator_block,
    make_repeat_result,
    utc_now,
)
from .runner_common import (
    RunContext,
    assert_reuse_compatible,
    expected_fingerprint_from_context,
    finalize_result,
    load_existing_result,
    prepare_run_context,
    relative_to_run,
    RunnerError,
)
from .subprocess_util import run_command, tail_text
from .tb2_manifest import (
    TB2_BENCHMARK_ID,
    Tb2Manifest,
    select_tb2_tasks,
    tb2_dataset_manifest_dict,
)


@dataclass(frozen=True)
class Tb2Case:
    task_id: str
    split: str
    stratum: str
    selector: str
    repository: str | None = None
    domain: str | None = None


def run_tb2_eval(
    *,
    config: EvalConfig,
    output_dir: Path,
    run_root: Path | None = None,
    run_id: str,
    result_id: str,
    harness_snapshot: Path | None = None,
    harness_version_id: str | None = None,
    harness_kind: str = "baseline",
    parent_version_ids: Sequence[str] = (),
    splits: Sequence[str] | None = None,
    task_ids: Sequence[str] | None = None,
    repeats: int | None = None,
    reuse_existing: bool = False,
    candidate_workspace: Path | None = None,
) -> dict[str, Any]:
    """Run a Terminal-Bench evaluation and return the unified result."""

    if config.eval.benchmark != TB2_BENCHMARK_ID:
        raise RunnerError(f"run_tb2_eval requires benchmark {TB2_BENCHMARK_ID!r}")
    output_dir = Path(output_dir).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve() if run_root is not None else output_dir
    run_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_splits = tuple(splits or config.eval.splits)
    selected_task_ids = tuple(task_ids) if task_ids is not None else config.eval.task_ids
    repeat_count = int(repeats if repeats is not None else config.eval.repeats)
    if repeat_count < 1:
        raise RunnerError("repeats must be >= 1")
    if repeat_count != config.eval.repeats:
        config = dataclasses.replace(
            config, eval=dataclasses.replace(config.eval, repeats=repeat_count)
        )

    if not config.eval.legacy.get("cases"):
        manifest = _load_manifest(config)
        if selected_task_ids:
            requested = set(selected_task_ids)
            selected_splits = tuple(
                split
                for split in selected_splits
                if any(task_id in requested for task_id in manifest.split_task_ids(split))
            )
        selected = select_tb2_tasks(
            manifest,
            splits=selected_splits,
            task_ids=selected_task_ids,
        )
        selected_splits = tuple(selected)
        cases_by_split = {
            split: tuple(
                Tb2Case(
                    task_id=task.task_id,
                    split=split,
                    stratum=task.stratum,
                    selector=_render_selector(config, task.selector, task.task_id),
                )
                for task in tasks
            )
            for split, tasks in selected.items()
        }
        dataset_payload = tb2_dataset_manifest_dict(
            manifest,
            selected=selected,
            evaluator="harbor+pytest-junit",
            evaluator_version=_tb2_evaluator_version(config),
        )
        benchmark_version = manifest.version
        manifest_audit = manifest.to_audit_dict()
        evaluator_name = "harbor+pytest-junit"
    else:
        cases_by_split = _legacy_cases(config, selected_splits, selected_task_ids)
        if not cases_by_split:
            raise RunnerError("no Terminal-Bench cases selected")
        selected_splits = tuple(split for split in selected_splits if split in cases_by_split)
        dataset_payload, benchmark_version, manifest_audit = _legacy_dataset_payload(
            config, cases_by_split
        )
        evaluator_name = "harbor+pytest-junit"

    split_task_ids = {
        split: [case.task_id for case in cases] for split, cases in cases_by_split.items()
    }
    preflight_errors = _preflight_errors(
        config=config,
        output_dir=output_dir,
        candidate_workspace=candidate_workspace,
    )
    bandwidth_plan = bandwidth_plan_from_settings(
        config.constraints, raw=config.constraints.raw
    )
    harness_version = harness_version_id or f"{run_id}-harness"
    context: RunContext | None = None
    context_error: str | None = None
    try:
        context = prepare_run_context(
            config=config,
            output_dir=output_dir,
            run_root=run_root,
            run_id=run_id,
            result_id=result_id,
            benchmark_id=TB2_BENCHMARK_ID,
            benchmark_version=benchmark_version,
            evaluator_name=evaluator_name,
            evaluator_version=_tb2_evaluator_version(config),
            evaluator_config={
                "model_flag": config.eval.legacy.get("model_flag"),
                "summary_flag": config.eval.legacy.get("summary_flag"),
                "pytest_args": config.eval.legacy.get("pytest_args"),
                "evals_project": str(config.eval.legacy.get("evals_project")),
                "agent_import_path": config.eval.legacy.get("agent_import_path"),
            },
            harness_snapshot=harness_snapshot,
            harness_version_id=harness_version,
            harness_kind=harness_kind,
            parent_version_ids=parent_version_ids,
            dataset_manifest_payload=dataset_payload,
            split_task_ids=split_task_ids,
            environment_extra={"runner": "terminal_bench_2.harbor_pytest"},
            bandwidth=bandwidth_plan,
            reuse_existing=reuse_existing,
        )
    except RunnerError as exc:
        context_error = str(exc)
        context = prepare_run_context(
            config=config,
            output_dir=output_dir,
            run_root=run_root,
            run_id=run_id,
            result_id=result_id,
            benchmark_id=TB2_BENCHMARK_ID,
            benchmark_version=benchmark_version,
            evaluator_name=f"{evaluator_name}-unavailable",
            evaluator_version="unavailable",
            harness_snapshot=_fallback_harness_snapshot(output_dir, context_error),
            harness_version_id=f"{harness_version}-unavailable",
            harness_kind="baseline",
            parent_version_ids=(),
            dataset_manifest_payload=dataset_payload,
            split_task_ids=split_task_ids,
            environment_extra={"runner": "terminal_bench_2.harbor_pytest", "preflight_error": context_error},
            bandwidth=bandwidth_plan,
        )

    assert context is not None
    target_result = output_dir / "result.json"
    if reuse_existing and target_result.exists():
        existing = load_existing_result(target_result)
        expected = expected_fingerprint_from_context(context, split_task_ids=split_task_ids)
        assert_reuse_compatible(existing=existing, expected=expected, existing_path=target_result)
        _audit_reuse(output_dir, target_result)
        return existing
    if target_result.exists():
        raise RunnerError(
            f"evaluation result already exists and --reuse-existing was not requested: {target_result}"
        )

    errors: list[dict[str, Any]] = []
    if context_error:
        errors.append({"message": f"preflight failure: {context_error}"})
    errors.extend(preflight_errors)

    if context_error or preflight_errors:
        splits_records = _all_not_run_splits(
            cases_by_split=cases_by_split,
            repeats=repeat_count,
            run_root=run_root,
            reason=context_error or "; ".join(item["message"] for item in preflight_errors),
        )
        result, _ = finalize_result(
            context=context,
            splits=splits_records,
            errors=errors,
            artifacts={
                "splits_dir": context.rel(output_dir / "splits"),
                "failure_marker": "preflight",
            },
            status="failed",
        )
        _write_legacy_aggregate(output_dir, result)
        return result

    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in selected_splits:
        cases = cases_by_split[split]
        if not cases:
            raise RunnerError(f"no cases configured for split {split!r}")
        split_records[split] = []
        for repeat_index in range(1, repeat_count + 1):
            repeat_dir = output_dir / "splits" / split / f"repeat-{repeat_index:02d}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            repeat_started = time.time()
            outcomes: list[dict[str, Any]] = []
            max_workers = min(config.eval.case_concurrency, len(cases))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _run_case,
                        config=config,
                        context=context,
                        case=case,
                        repeat_index=repeat_index,
                        repeat_dir=repeat_dir,
                        bandwidth_prefix=bandwidth_plan,
                        candidate_workspace=candidate_workspace,
                    )
                    for case in cases
                ]
                for future in concurrent.futures.as_completed(futures):
                    outcomes.append(future.result())
            order = {case.task_id: index for index, case in enumerate(cases)}
            outcomes.sort(key=lambda item: order.get(str(item["case"]["task_id"]), 0))
            case_results = [item["case"] for item in outcomes]
            repeat_record = make_repeat_result(
                split=split,
                repeat=repeat_index,
                expected_task_ids=[case.task_id for case in cases],
                case_results=case_results,
                duration_s=time.time() - repeat_started,
            )
            (repeat_dir / "result.json").write_text(
                json.dumps(repeat_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            errors.extend(item["error"] for item in outcomes if item["error"] is not None)
            split_records[split].append(repeat_record)

    result, _ = finalize_result(
        context=context,
        splits=split_records,
        errors=errors,
        artifacts={
            "splits_dir": context.rel(output_dir / "splits"),
            "legacy_result": context.rel(output_dir / "result.legacy.json"),
            "bandwidth_log": context.rel(output_dir / "network_limit.json"),
        },
    )
    (output_dir / "network_limit.json").write_text(
        json.dumps(bandwidth_plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_legacy_aggregate(output_dir, result)
    return result


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------


def _run_case(
    *,
    config: EvalConfig,
    context: RunContext,
    case: Tb2Case,
    repeat_index: int,
    repeat_dir: Path,
    bandwidth_prefix: Any,
    candidate_workspace: Path | None,
) -> dict[str, Any]:
    case_dir = repeat_dir / "cases" / _safe_slug(case.task_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = _case_env(
        config=config,
        context=context,
        split=case.split,
        repeat_index=repeat_index,
        candidate_workspace=candidate_workspace,
    )
    try:
        command = _build_pytest_command(config=config, case=case, case_dir=case_dir)
    except Exception as exc:
        error_message = f"cannot build pytest command: {exc}"
        case_result = _case(
            case=case,
            repeat_index=repeat_index,
            status=CASE_STATUS_EVALUATOR_ERROR,
            duration_s=0.0,
            agent_status="not_run",
            timed_out=False,
            evaluator_status="error",
            evaluator_resolved=None,
            failure_summary=error_message,
            case_dir=case_dir,
            context=context,
            error=error_message,
        )
        return _finish_case(case_dir, case_result, context, error_message, split=case.split, repeat=repeat_index)
    command = bandwidth_prefix.wrap(command) if bandwidth_prefix is not None else command
    (case_dir / "command.json").write_text(
        json.dumps(
            {
                "argv": command,
                "cwd": str(config.eval.legacy.get("evals_project")),
                "timeout_s": config.eval.task_timeout_s,
                "tool_timeout_s": config.eval.tool_timeout_s,
                "network_limit": bandwidth_prefix.to_dict() if bandwidth_prefix is not None else None,
                "split": case.split,
                "repeat": repeat_index,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = run_command(
        command,
        cwd=config.eval.legacy.get("evals_project"),
        env=env,
        timeout_s=config.eval.task_timeout_s,
    )
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    junit_path = case_dir / "junit.xml"
    summary_path = case_dir / "summary.json"
    summary_payload = _read_json(summary_path)
    trace_metadata = _extract_trace_metadata(summary_payload)
    trace_rel = _archive_trace(trace_metadata, case_dir=case_dir, run_root=context.run_root)
    tool_calls = _count_tool_calls(trace_metadata)

    if completed.timed_out:
        message = (
            f"pytest task exceeded task_timeout_s={config.eval.task_timeout_s}s; "
            f"stderr tail: {tail_text(completed.stderr, max_chars=1200)}"
        )
        case_result = _case(
            case=case,
            repeat_index=repeat_index,
            status=CASE_STATUS_AGENT_TIMEOUT,
            duration_s=time.time() - started,
            agent_status="timeout",
            timed_out=True,
            evaluator_status="not_run",
            evaluator_resolved=None,
            failure_summary=message,
            case_dir=case_dir,
            context=context,
            trace_path=trace_rel,
            tool_calls=tool_calls,
            error=message,
        )
        return _finish_case(case_dir, case_result, context, message, split=case.split, repeat=repeat_index)

    if completed.error is not None:
        message = f"cannot execute pytest command: {completed.error}"
        case_result = _case(
            case=case,
            repeat_index=repeat_index,
            status=CASE_STATUS_EVALUATOR_ERROR,
            duration_s=time.time() - started,
            agent_status="not_run",
            timed_out=False,
            evaluator_status="error",
            evaluator_resolved=None,
            failure_summary=message,
            case_dir=case_dir,
            context=context,
            trace_path=trace_rel,
            tool_calls=tool_calls,
            error=message,
        )
        return _finish_case(case_dir, case_result, context, message, split=case.split, repeat=repeat_index)

    if not junit_path.exists():
        message = (
            "pytest produced no JUnit XML; the task cannot be scored. "
            f"returncode={completed.returncode}. stderr tail: {tail_text(completed.stderr, max_chars=1600)}"
        )
        case_result = _case(
            case=case,
            repeat_index=repeat_index,
            status=CASE_STATUS_EVALUATOR_ERROR,
            duration_s=time.time() - started,
            agent_status="complete" if completed.returncode is not None else "not_run",
            timed_out=False,
            evaluator_status="error",
            evaluator_resolved=None,
            failure_summary=message,
            case_dir=case_dir,
            context=context,
            trace_path=trace_rel,
            tool_calls=tool_calls,
            error=message,
        )
        return _finish_case(case_dir, case_result, context, message, split=case.split, repeat=case.repeat if False else repeat_index)

    try:
        junit = _parse_junit(junit_path)
    except (ET.ParseError, OSError) as exc:
        message = f"cannot parse JUnit XML: {exc}"
        case_result = _case(
            case=case,
            repeat_index=repeat_index,
            status=CASE_STATUS_EVALUATOR_ERROR,
            duration_s=time.time() - started,
            agent_status="complete",
            timed_out=False,
            evaluator_status="error",
            evaluator_resolved=None,
            failure_summary=message,
            case_dir=case_dir,
            context=context,
            trace_path=trace_rel,
            tool_calls=tool_calls,
            error=message,
        )
        return _finish_case(case_dir, case_result, context, message, split=case.split, repeat=repeat_index)

    passed = junit["passed"] and completed.returncode == 0
    evaluator_status = "complete"
    if passed:
        case_status = CASE_STATUS_PASSED
        evaluator_resolved = True
        failure_summary = None
        error = None
    elif junit["tests"] == 0 or junit["skipped"] > 0 or junit["passed"] and completed.returncode != 0:
        case_status = CASE_STATUS_EVALUATOR_ERROR
        evaluator_status = "error"
        evaluator_resolved = None
        failure_summary = (
            f"JUnit XML is not scoreable: tests={junit['tests']} skipped={junit['skipped']} "
            f"returncode={completed.returncode}"
        )
        error = failure_summary
    else:
        case_status = CASE_STATUS_FAILED
        evaluator_resolved = False
        failure_summary = junit["failure_message"] or tail_text(completed.stderr, max_chars=2000)
        error = None
    case_result = _case(
        case=case,
        repeat_index=repeat_index,
        status=case_status,
        duration_s=time.time() - started,
        agent_status="complete",
        timed_out=False,
        evaluator_status=evaluator_status,
        evaluator_resolved=evaluator_resolved,
        failure_summary=failure_summary,
        case_dir=case_dir,
        context=context,
        trace_path=trace_rel,
        tool_calls=tool_calls,
        error=error,
        evaluator_log=str(junit_path),
        legacy={
            "case_id": case.selector,
            "stratum": case.stratum,
            "returncode": completed.returncode,
            "failure_message": failure_summary,
            "junit": junit,
        },
    )
    return _finish_case(case_dir, case_result, context, error, split=case.split, repeat=repeat_index)


def _finish_case(
    case_dir: Path,
    case_result: dict[str, Any],
    context: RunContext,
    error: str | None,
    *,
    split: str,
    repeat: int,
) -> dict[str, Any]:
    (case_dir / "summary.json").write_text(
        json.dumps(case_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    error_record = None
    if case_result["passed"] is None:
        error_record = {
            "split": split,
            "repeat": repeat,
            "task_id": case_result["task_id"],
            "status": case_result["status"],
            "message": error or case_result["evaluator"].get("failure_summary") or "unscored",
        }
    return {"case": case_result, "error": error_record, "dir": case_dir}


def _case(
    *,
    case: Tb2Case,
    repeat_index: int,
    status: str,
    duration_s: float,
    agent_status: str,
    timed_out: bool,
    evaluator_status: str,
    evaluator_resolved: bool | None,
    failure_summary: str | None,
    case_dir: Path,
    context: RunContext,
    trace_path: str | None = None,
    tool_calls: int | None = None,
    error: str | None = None,
    evaluator_log: str | None = None,
    legacy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = make_case_result(
        task_id=case.task_id,
        split=case.split,
        repeat=repeat_index,
        repository=case.repository,
        domain=case.domain,
        status=status,
        duration_s=duration_s,
        agent=make_agent_block(
            status=agent_status,
            timed_out=timed_out,
            tool_calls=tool_calls,
            trace_path=trace_path,
        ),
        evaluator=make_evaluator_block(
            status=evaluator_status,
            name="harbor+pytest-junit",
            resolved=evaluator_resolved,
            failure_summary=failure_summary,
            log_path=_relative_or_none(case_dir / "junit.xml", context.run_root)
            if evaluator_log
            else None,
        ),
        patch=None,
        artifacts={
            "case_dir": relative_to_run(case_dir, context.run_root),
            "agent_stdout": relative_to_run(case_dir / "stdout.log", context.run_root)
            if (case_dir / "stdout.log").exists()
            else None,
            "agent_stderr": relative_to_run(case_dir / "stderr.log", context.run_root)
            if (case_dir / "stderr.log").exists()
            else None,
            "summary": relative_to_run(case_dir / "summary.json", context.run_root)
            if (case_dir / "summary.json").exists()
            else None,
        },
        error=error,
    )
    if legacy:
        record["legacy"] = dict(legacy)
    return record


# ---------------------------------------------------------------------------
# Config / manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest(config: EvalConfig) -> Tb2Manifest:
    from .tb2_manifest import load_tb2_manifest

    return load_tb2_manifest(config.dataset.manifest_path)


def _render_selector(config: EvalConfig, selector: str | None, task_id: str) -> str:
    template = config.eval.raw.get("selector_template") or "{task_id}"
    raw = selector or str(template)
    return raw.format(task_id=task_id, model=config.model.model_id)


def _legacy_cases(
    config: EvalConfig,
    splits: Sequence[str],
    task_ids: Sequence[str] | None,
) -> dict[str, tuple[Tb2Case, ...]]:
    cases_by_split: dict[str, list[Tb2Case]] = {split: [] for split in splits}
    requested = set(task_ids) if task_ids else None
    for index, raw in enumerate(config.eval.legacy.get("cases") or []):
        if not isinstance(raw, Mapping):
            continue
        case_id = str(raw.get("case_id") or raw.get("selector") or "").strip()
        if not case_id:
            raise RunnerError(f"legacy case {index} is missing case_id")
        split = str(raw.get("split") or "train")
        if split not in SPLIT_NAMES:
            raise RunnerError(f"legacy case {index} has invalid split {split!r}")
        if split not in cases_by_split:
            continue
        task_id = str(raw.get("task_id") or case_id)
        if requested is not None and task_id not in requested:
            continue
        cases_by_split[split].append(
            Tb2Case(
                task_id=task_id,
                split=split,
                stratum=str(raw.get("stratum") or "default"),
                selector=case_id,
                repository=str(raw["repository"]) if raw.get("repository") else None,
                domain=str(raw["domain"]) if raw.get("domain") else None,
            )
        )
    return {
        split: tuple(cases)
        for split, cases in cases_by_split.items()
        if cases
    }


def _legacy_dataset_payload(
    config: EvalConfig,
    cases_by_split: Mapping[str, Sequence[Tb2Case]],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from .protocol import task_ids_sha256

    splits = {
        split: [
            {"task_id": case.task_id, "split": split, "stratum": case.stratum, "selector": case.selector}
            for case in cases
        ]
        for split, cases in cases_by_split.items()
    }
    payload = {
        "benchmark": TB2_BENCHMARK_ID,
        "version": str(config.dataset.revision or "legacy-config"),
        "name": config.eval.name,
        "manifest": {
            "path": str(config.path),
            "sha256": sha256_file(config.path),
        },
        "evaluator": {"name": "harbor+pytest-junit", "version": _tb2_evaluator_version(config)},
        "selected_splits": splits,
        "selected_task_ids_sha256": task_ids_sha256(
            (case.task_id for case in cases_by_split.get("train", ())),
            (case.task_id for case in cases_by_split.get("heldout", ())),
        ),
        "clean64_note": (
            "Legacy [[cases]] configuration is supported for backward compatibility. The "
            "shipped tb2_clean64 config selects the complete fixed 64-task split."
        ),
    }
    return payload, str(config.dataset.revision or "legacy-config"), payload


def _preflight_errors(
    *,
    config: EvalConfig,
    output_dir: Path,
    candidate_workspace: Path | None,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    evals_project = config.eval.legacy.get("evals_project")
    if not evals_project:
        errors.append(
            {"message": "[eval].evals_project is required: point it at the Harbor pytest eval project"}
        )
    elif not Path(evals_project).is_dir():
        errors.append(
            {"message": f"evals_project does not exist: {evals_project}"}
        )
    harness_workspace = config.eval.legacy.get("harness_workspace")
    if harness_workspace and not Path(harness_workspace).is_dir():
        errors.append({"message": f"harness_workspace does not exist: {harness_workspace}"})
    if not config.eval.legacy.get("agent_import_path"):
        errors.append({"message": "[eval].agent_import_path is required for TB2 Harbor runs"})
    return errors


def _all_not_run_splits(
    *,
    cases_by_split: Mapping[str, Sequence[Tb2Case]],
    repeats: int,
    run_root: Path,
    reason: str,
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for split, cases in cases_by_split.items():
        records[split] = []
        for repeat_index in range(1, repeats + 1):
            case_results = []
            for case in cases:
                case_results.append(
                    make_case_result(
                        task_id=case.task_id,
                        split=split,
                        repeat=repeat_index,
                        repository=case.repository,
                        domain=case.domain,
                        status=CASE_STATUS_NOT_RUN,
                        duration_s=0.0,
                        agent=make_agent_block(
                            status="not_run", timed_out=False, tool_calls=None, trace_path=None
                        ),
                        evaluator=make_evaluator_block(
                            status="not_run",
                            name="harbor+pytest-junit",
                            resolved=None,
                            failure_summary=reason,
                            log_path=None,
                        ),
                        patch=None,
                        artifacts={},
                        error=reason,
                    )
                )
            records[split].append(
                make_repeat_result(
                    split=split,
                    repeat=repeat_index,
                    expected_task_ids=[case.task_id for case in cases],
                    case_results=case_results,
                )
            )
    return records


def _fallback_harness_snapshot(output_dir: Path, reason: str) -> Path:
    snapshot = output_dir / "harness_unavailable"
    snapshot.mkdir(parents=True, exist_ok=True)
    marker = snapshot / "HARNESS_UNAVAILABLE.txt"
    marker.write_text(
        "The configured harness snapshot could not be prepared. This evaluation result is\n"
        "recorded as failed; no harness code was executed.\n\n"
        f"reason: {reason}\n",
        encoding="utf-8",
    )
    return snapshot


# ---------------------------------------------------------------------------
# Command construction / environment
# ---------------------------------------------------------------------------


def _build_pytest_command(*, config: EvalConfig, case: Tb2Case, case_dir: Path) -> list[str]:
    from .subprocess_util import expand_command_template

    legacy = config.eval.legacy
    base = legacy.get("pytest_command")
    if base:
        tokens = expand_command_template(
            base,
            placeholders={
                "evals_project": legacy.get("evals_project") or "",
                "case_dir": case_dir,
                "selector": case.selector,
                "model": config.model.model_id,
                "junitxml": case_dir / "junit.xml",
                "summary": case_dir / "summary.json",
            },
        )
    else:
        evals_project = legacy.get("evals_project")
        if not evals_project:
            raise RunnerError("[eval].evals_project is required")
        tokens = ["uv", "run", "--project", str(evals_project), "pytest"]
    model_flag = legacy.get("model_flag")
    if model_flag:
        tokens.extend([str(model_flag), config.model.model_id])
    summary_flag = legacy.get("summary_flag")
    if summary_flag:
        tokens.extend([str(summary_flag), str(case_dir / "summary.json")])
    tokens.extend(["--junitxml", str(case_dir / "junit.xml")])
    tokens.extend(str(item) for item in legacy.get("pytest_args") or [])
    tokens.append(case.selector)
    return tokens


def _case_env(
    *,
    config: EvalConfig,
    context: RunContext,
    split: str,
    repeat_index: int,
    candidate_workspace: Path | None,
) -> dict[str, str]:
    env = os.environ.copy()
    legacy_env = config.eval.legacy.get("env") or {}
    for key, value in legacy_env.items():
        env[str(key)] = str(value)
    env["SELF_HARNESS_AGENT_IMPORT_PATH"] = str(
        config.eval.legacy.get("agent_import_path") or ""
    )
    env["SELF_HARNESS_EVAL_SPLIT"] = split
    env["SELF_HARNESS_EVAL_REPEAT"] = str(repeat_index)
    env["SELF_HARNESS_AGENT_TIMEOUT_SEC"] = f"{config.eval.task_timeout_s:g}"
    env["SELF_HARNESS_TOOL_TIMEOUT_S"] = f"{config.eval.tool_timeout_s:g}"
    env["SELF_HARNESS_RUN_ID"] = context.run_id
    env["SELF_HARNESS_RESULT_ID"] = context.result_id
    pythonpath_parts = [
        str(config.eval.legacy.get("harness_workspace") or ""),
        str(Path(config.eval.legacy.get("evals_project") or ".").parent),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)
    workspace = candidate_workspace
    if workspace is None and not env.get("SELF_HARNESS_CANDIDATE_WORKSPACE"):
        workspace = context.run_root / context.harness["snapshot_path"]
    if workspace is not None:
        env["SELF_HARNESS_CANDIDATE_WORKSPACE"] = str(Path(workspace).resolve())
    return env


# ---------------------------------------------------------------------------
# JUnit / trace helpers
# ---------------------------------------------------------------------------


def _parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    testcases = root.findall(".//testcase")
    failures = root.findall(".//failure")
    errors = root.findall(".//error")
    skipped = root.findall(".//skipped")
    messages: list[str] = []
    for node in [*failures, *errors]:
        message = node.attrib.get("message") or (node.text or "")
        if message.strip():
            messages.append(message.strip())
    return {
        "tests": len(testcases),
        "failures": len(failures),
        "errors": len(errors),
        "skipped": len(skipped),
        "passed": bool(testcases) and not failures and not errors and not skipped,
        "failure_message": "\n".join(messages) if messages else None,
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _extract_trace_metadata(payload: Any) -> dict[str, Any] | None:
    metadata = _find_trace_metadata(payload)
    if not metadata:
        return None
    messages = _existing_path(metadata.get("messages_path"))
    if messages is None:
        return None
    result = dict(metadata)
    result["messages_path"] = str(messages)
    events = _existing_path(result.get("events_path"))
    if events is not None:
        result["events_path"] = str(events)
    return result


def _find_trace_metadata(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        normalized = _normalize_trace_metadata(value)
        if normalized:
            return normalized
        for item in value.values():
            found = _find_trace_metadata(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_trace_metadata(item)
            if found:
                return found
    return None


def _normalize_trace_metadata(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if "messages_path" in value:
        return dict(value)
    agent_result = value.get("agent_result")
    if isinstance(agent_result, Mapping):
        metadata = agent_result.get("metadata")
        if isinstance(metadata, Mapping) and "messages_path" in metadata:
            return dict(metadata)
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping) and "messages_path" in metadata:
        return dict(metadata)
    return None


def _existing_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    try:
        return path.resolve() if path.exists() else None
    except OSError:
        return None


def _archive_trace(
    metadata: Mapping[str, Any] | None,
    *,
    case_dir: Path,
    run_root: Path,
) -> str | None:
    if not metadata:
        return None
    trace_dir = case_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    archived_messages: Path | None = None
    for key in ("messages_path", "events_path"):
        source = _existing_path(metadata.get(key))
        if source is None:
            continue
        target = trace_dir / source.name
        try:
            shutil.copy2(source, target)
        except OSError:
            continue
        if key == "messages_path":
            archived_messages = target
    if archived_messages is None:
        return None
    return relative_to_run(archived_messages, run_root)


def _count_tool_calls(metadata: Mapping[str, Any] | None) -> int | None:
    if not metadata:
        return None
    path = _existing_path(metadata.get("messages_path"))
    if path is None:
        return None
    payload = _read_json(path)
    if not isinstance(payload, list):
        return None
    total = 0
    found = False
    for message in payload:
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            found = True
            total += len(calls)
    return total if found else None


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _tb2_evaluator_version(config: EvalConfig) -> str:
    parts: list[str] = []
    for package in ("harbor", "pytest"):
        try:
            parts.append(f"{package}={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package}=unavailable")
    return ";".join(parts)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")[:180] or "case"


def _relative_or_none(path: Path, run_root: Path) -> str | None:
    try:
        return relative_to_run(path, run_root)
    except RunnerError:
        return None


def _write_legacy_aggregate(output_dir: Path, result: Mapping[str, Any]) -> None:
    splits = result.get("splits") or {}
    total_passed = 0
    total_cases = 0
    legacy_splits: dict[str, list[dict[str, Any]]] = {}
    for split, repeat_list in splits.items():
        legacy_splits[split] = []
        for repeat in repeat_list:
            total_passed += int(repeat.get("passed", 0))
            total_cases += int(repeat.get("total", 0))
            legacy_splits[split].append(
                {
                    "split": split,
                    "repeat": int(repeat.get("repeat", 0)),
                    "passed": int(repeat.get("passed", 0)),
                    "total": int(repeat.get("total", 0)),
                    "correctness": (
                        int(repeat.get("passed", 0)) / int(repeat.get("total", 1))
                        if int(repeat.get("total", 0))
                        else 0.0
                    ),
                    "duration_s": repeat.get("duration_s"),
                    "case_results": repeat.get("case_results"),
                }
            )
    payload = {
        "name": result.get("benchmark", {}).get("version"),
        "model": result.get("model", {}).get("model_id"),
        "repeats": result.get("config", {}).get("repeats"),
        "passed": total_passed,
        "total": total_cases,
        "correctness": total_passed / total_cases if total_cases else 0.0,
        "splits": legacy_splits,
        "deprecated": True,
        "superseded_by": "result.json",
    }
    (output_dir / "result.legacy.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _audit_reuse(output_dir: Path, result_path: Path) -> None:
    (output_dir / "reuse.json").write_text(
        json.dumps(
            {
                "reused_result": result_path.name,
                "result_sha256": sha256_file(result_path),
                "reused_at": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
