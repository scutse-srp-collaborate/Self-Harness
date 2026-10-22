"""SWE-bench Verified runner.

The runner is deliberately *not* a pytest runner: it prepares the correct
repository/base-commit workspace, runs an agent adapter, collects the model
patch, and delegates the resolved/unresolved decision to the SWE-bench
evaluator.  Gold patches, test patches, FAIL_TO_PASS/PASS_TO_PASS metadata and
the resolved label never reach the agent.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import os
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bandwidth import bandwidth_plan_from_settings
from .config import EvalConfig
from .dataset import (
    PROTECTED_FIELDS,
    SwebenchDataset,
    SwebenchInstance,
    assert_no_protected_fields,
    load_swebench_dataset,
)
from .hashing import sha256_file, sha256_text
from .manifest import (
    SwebenchManifest,
    dataset_manifest_dict,
    load_swebench_manifest,
    select_task_ids,
)
from .protocol import (
    BENCHMARK_SWEBENCH,
    CASE_STATUS_AGENT_ERROR,
    CASE_STATUS_AGENT_TIMEOUT,
    CASE_STATUS_EVALUATOR_ERROR,
    CASE_STATUS_INFRA_ERROR,
    CASE_STATUS_NOT_RUN,
    CASE_STATUS_RESOLVED,
    CASE_STATUS_UNRESOLVED,
    make_agent_block,
    make_case_result,
    make_evaluator_block,
    make_repeat_result,
)
from .runner_common import (
    RunContext,
    RunnerError,
    assert_reuse_compatible,
    expected_fingerprint_from_context,
    finalize_result,
    load_existing_result,
    prepare_run_context,
    relative_to_run,
)
from .subprocess_util import (
    AttemptRecord,
    CommandResult,
    env_with_overrides,
    expand_command_template,
    run_command,
    tail_text,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_NAME_OFFICIAL = "swebench-official-harness"
EVALUATOR_NAME_FIXTURE = "fixture-fake-swebench-evaluator"


@dataclass
class AgentOutcome:
    status: str
    timed_out: bool
    duration_s: float
    stdout: str
    stderr: str
    returncode: int | None
    error: str | None
    command: list[str]
    attempts: list[AttemptRecord] = field(default_factory=list)
    trace_path: Path | None = None
    tool_calls: int | None = None
    patch_override: str | None = None
    patch_source: str = "workspace_git"
    verification: dict[str, Any] | None = None
    extra_artifacts: dict[str, Path] = field(default_factory=dict)


@dataclass
class EvaluationOutcome:
    status: str
    resolved: bool | None
    failure_summary: str | None
    duration_s: float
    command: list[str] = field(default_factory=list)
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    report_path: Path | None = None
    error: str | None = None


def run_swebench_eval(
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
    pilot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config.eval.benchmark != BENCHMARK_SWEBENCH:
        raise RunnerError(f"run_swebench_eval requires benchmark {BENCHMARK_SWEBENCH!r}")
    output_dir = Path(output_dir).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve() if run_root is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    selected_splits = tuple(splits or config.eval.splits)
    repeat_count = int(repeats if repeats is not None else config.eval.repeats)
    if repeat_count < 1:
        raise RunnerError("repeats must be >= 1")
    if repeat_count != config.eval.repeats:
        config = dataclasses.replace(
            config, eval=dataclasses.replace(config.eval, repeats=repeat_count)
        )

    manifest = load_swebench_manifest(config.dataset.manifest_path)
    requested_task_ids = task_ids if task_ids is not None else config.eval.task_ids
    if requested_task_ids:
        requested = set(requested_task_ids)
        selected_splits = tuple(
            split
            for split in selected_splits
            if any(task_id in requested for task_id in manifest.split_task_ids(split))
        )
    selected = select_task_ids(
        manifest,
        splits=selected_splits,
        task_ids=requested_task_ids,
    )
    selected_splits = tuple(selected)
    split_task_ids = {split: list(ids) for split, ids in selected.items()}

    dataset_audit: dict[str, Any]
    dataset: SwebenchDataset | None = None
    dataset_error: str | None = None
    try:
        dataset = load_swebench_dataset(
            name=config.dataset.name,
            revision=config.dataset.revision,
            source=config.dataset.source,
            cache_dir=config.dataset.cache_dir,
            split=config.dataset.split,
            allow_hf_download=config.dataset.allow_hf_download,
        )
        dataset_audit = dataset.audit_dict()
    except Exception as exc:
        dataset_error = f"{exc.__class__.__name__}: {exc}"
        dataset_audit = {
            "name": config.dataset.name,
            "revision": config.dataset.revision,
            "split": config.dataset.split,
            "source": config.dataset.source,
            "source_sha256": None,
            "unavailable_reason": dataset_error,
        }

    dataset_payload = dataset_manifest_dict(
        manifest,
        selected=selected,
        dataset_name=config.dataset.name,
        dataset_revision=config.dataset.revision,
        source=config.dataset.source,
        source_sha256=dataset_audit.get("source_sha256"),
    )
    dataset_payload["dataset_source"] = dataset_audit

    bandwidth_plan = bandwidth_plan_from_settings(
        config.constraints, raw=config.constraints.raw
    )
    evaluator_meta = _evaluator_meta(config)
    environment_extra = {
        "runner": "swe_bench_verified",
        "repos_dir": str(config.dataset.repos_dir) if config.dataset.repos_dir else None,
        "evaluator_kind": config.evaluator.kind,
    }
    context, context_error = _prepare_context(
        config=config,
        output_dir=output_dir,
        run_root=run_root,
        run_id=run_id,
        result_id=result_id,
        harness_snapshot=harness_snapshot,
        harness_version_id=harness_version_id or f"{run_id}-harness",
        harness_kind=harness_kind,
        parent_version_ids=parent_version_ids,
        dataset_payload=dataset_payload,
        split_task_ids=split_task_ids,
        dataset_source_sha256=dataset_audit.get("source_sha256"),
        environment_extra=environment_extra,
        bandwidth=bandwidth_plan,
        evaluator_meta=evaluator_meta,
        reuse_existing=reuse_existing,
    )

    target_result = output_dir / "result.json"
    if reuse_existing and target_result.exists():
        existing = load_existing_result(target_result)
        expected = expected_fingerprint_from_context(context, split_task_ids=split_task_ids)
        assert_reuse_compatible(existing=existing, expected=expected, existing_path=target_result)
        _audit_reuse(output_dir, target_result, kind="result")
        return existing
    if target_result.exists():
        raise RunnerError(
            f"evaluation result already exists and --reuse-existing was not requested: {target_result}"
        )

    errors: list[dict[str, Any]] = []
    if context_error:
        errors.append({"message": f"preflight failure: {context_error}"})
    if dataset_error:
        errors.append({"message": f"dataset unavailable: {dataset_error}"})
    evaluator_preflight_error = _evaluator_preflight_error(config)
    if evaluator_preflight_error:
        errors.append({"message": evaluator_preflight_error})

    blocker = context_error or dataset_error or evaluator_preflight_error
    if blocker:
        split_records = _not_run_splits(
            manifest=manifest,
            split_task_ids=split_task_ids,
            repeats=repeat_count,
            reason=blocker,
        )
        result, _ = finalize_result(
            context=context,
            splits=split_records,
            errors=errors,
            artifacts={"splits_dir": context.rel(output_dir / "splits")},
            status="failed",
            extra=_result_extra(
                config=config,
                split_task_ids=split_task_ids,
                repeat_count=repeat_count,
                selected=selected,
                pilot=pilot,
            ),
        )
        return result

    assert dataset is not None
    split_records: dict[str, list[dict[str, Any]]] = {}
    case_fingerprint = _case_fingerprint(config, context, dataset)
    for split in selected_splits:
        task_id_list = split_task_ids[split]
        if not task_id_list:
            raise RunnerError(f"no tasks selected for split {split!r}")
        instances = [dataset.instance(task_id) for task_id in task_id_list]
        split_records[split] = []
        for repeat_index in range(1, repeat_count + 1):
            repeat_dir = output_dir / "splits" / split / f"repeat-{repeat_index:02d}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            repeat_started = time.time()
            outcomes = _run_repeat_cases(
                config=config,
                context=context,
                manifest=manifest,
                dataset=dataset,
                instances=instances,
                task_ids=task_id_list,
                split=split,
                repeat_index=repeat_index,
                repeat_dir=repeat_dir,
                case_fingerprint=case_fingerprint,
                reuse_existing=reuse_existing,
                candidate_workspace=candidate_workspace,
                bandwidth_plan=bandwidth_plan,
            )
            order = {task_id: index for index, task_id in enumerate(task_id_list)}
            outcomes.sort(key=lambda item: order.get(str(item["case"]["task_id"]), 0))
            case_results = [item["case"] for item in outcomes]
            errors.extend(item["error"] for item in outcomes if item["error"] is not None)
            repeat_record = make_repeat_result(
                split=split,
                repeat=repeat_index,
                expected_task_ids=task_id_list,
                case_results=case_results,
                duration_s=time.time() - repeat_started,
            )
            (repeat_dir / "result.json").write_text(
                json.dumps(repeat_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            split_records[split].append(repeat_record)

    result, _ = finalize_result(
        context=context,
        splits=split_records,
        errors=errors,
        artifacts={
            "splits_dir": context.rel(output_dir / "splits"),
            "dataset_audit": context.rel(context.dataset_manifest_path),
            "bandwidth_log": context.rel(output_dir / "network_limit.json"),
        },
        extra=_result_extra(
            config=config,
            split_task_ids=split_task_ids,
            repeat_count=repeat_count,
            selected=selected,
            pilot=pilot,
        ),
    )
    (output_dir / "network_limit.json").write_text(
        json.dumps(bandwidth_plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _prepare_context(
    *,
    config: EvalConfig,
    output_dir: Path,
    run_root: Path,
    run_id: str,
    result_id: str,
    harness_snapshot: Path | None,
    harness_version_id: str,
    harness_kind: str,
    parent_version_ids: Sequence[str],
    dataset_payload: Mapping[str, Any],
    split_task_ids: Mapping[str, Sequence[str]],
    dataset_source_sha256: str | None,
    environment_extra: Mapping[str, Any],
    bandwidth: Any,
    evaluator_meta: Mapping[str, Any],
    reuse_existing: bool,
) -> tuple[RunContext, str | None]:
    try:
        context = prepare_run_context(
            config=config,
            output_dir=output_dir,
            run_root=run_root,
            run_id=run_id,
            result_id=result_id,
            benchmark_id=BENCHMARK_SWEBENCH,
            benchmark_version=config.dataset.revision,
            evaluator_name=str(evaluator_meta["name"]),
            evaluator_version=str(evaluator_meta["version"]),
            evaluator_config=dict(evaluator_meta["config"]),
            harness_snapshot=harness_snapshot,
            harness_version_id=harness_version_id,
            harness_kind=harness_kind,
            parent_version_ids=parent_version_ids,
            dataset_manifest_payload=dataset_payload,
            split_task_ids=split_task_ids,
            dataset_source_sha256=dataset_source_sha256,
            environment_extra=environment_extra,
            bandwidth=bandwidth,
            reuse_existing=reuse_existing,
        )
        return context, None
    except RunnerError as exc:
        reason = str(exc)
        fallback_snapshot = _fallback_harness_snapshot(output_dir, reason)
        context = prepare_run_context(
            config=config,
            output_dir=output_dir,
            run_root=run_root,
            run_id=run_id,
            result_id=result_id,
            benchmark_id=BENCHMARK_SWEBENCH,
            benchmark_version=config.dataset.revision,
            evaluator_name=f"{evaluator_meta['name']}-unavailable",
            evaluator_version="unavailable",
            evaluator_config=dict(evaluator_meta["config"]),
            harness_snapshot=fallback_snapshot,
            harness_version_id=f"{harness_version_id}-unavailable",
            harness_kind="baseline",
            parent_version_ids=(),
            dataset_manifest_payload=dataset_payload,
            split_task_ids=split_task_ids,
            dataset_source_sha256=dataset_source_sha256,
            environment_extra={**dict(environment_extra), "preflight_error": reason},
            bandwidth=bandwidth,
        )
        return context, reason


def _run_repeat_cases(
    *,
    config: EvalConfig,
    context: RunContext,
    manifest: SwebenchManifest,
    dataset: SwebenchDataset,
    instances: Sequence[SwebenchInstance | None],
    task_ids: Sequence[str],
    split: str,
    repeat_index: int,
    repeat_dir: Path,
    case_fingerprint: Mapping[str, Any],
    reuse_existing: bool,
    candidate_workspace: Path | None,
    bandwidth_plan: Any,
) -> list[dict[str, Any]]:
    del dataset
    max_workers = min(config.eval.case_concurrency, len(instances))
    outcomes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_case,
                config=config,
                context=context,
                manifest=manifest,
                instance=instance,
                task_id=task_id,
                split=split,
                repeat_index=repeat_index,
                repeat_dir=repeat_dir,
                case_fingerprint=case_fingerprint,
                reuse_existing=reuse_existing,
                candidate_workspace=candidate_workspace,
                bandwidth_plan=bandwidth_plan,
            )
            for task_id, instance in zip(task_ids, instances)
        ]
        for future in as_completed(futures):
            outcomes.append(future.result())
    return outcomes


def _run_case(
    *,
    config: EvalConfig,
    context: RunContext,
    manifest: SwebenchManifest,
    instance: SwebenchInstance | None,
    task_id: str,
    split: str,
    repeat_index: int,
    repeat_dir: Path,
    case_fingerprint: Mapping[str, Any],
    reuse_existing: bool,
    candidate_workspace: Path | None,
    bandwidth_plan: Any,
) -> dict[str, Any]:
    case_dir = repeat_dir / "cases" / _safe_slug(task_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repository = manifest.repository_of(task_id)
    domain = manifest.domain_of(task_id)
    provenance = {
        "task_id": task_id,
        "split": split,
        "repeat": repeat_index,
        **case_fingerprint,
    }
    provenance_path = case_dir / "case_provenance.json"
    result_path = case_dir / "case_result.json"
    if reuse_existing and provenance_path.exists() and result_path.exists():
        try:
            stored_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            stored_case = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stored_provenance = None
            stored_case = None
        if stored_provenance == provenance and isinstance(stored_case, dict):
            stored_case.setdefault("legacy", {})
            stored_case["legacy"]["resumed"] = True
            return {"case": stored_case, "error": _case_error(stored_case)}
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def complete(case: Mapping[str, Any], error: dict[str, Any] | None) -> dict[str, Any]:
        result_path.write_text(
            json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"case": dict(case), "error": error}

    if instance is None:
        message = (
            "pinned SWE-bench dataset does not contain this task id; the task cannot be "
            "prepared or scored"
        )
        case = _case(
            config=config,
            context=context,
            instance=_missing_instance(task_id),
            split=split,
            repeat_index=repeat_index,
            case_dir=case_dir,
            status=CASE_STATUS_INFRA_ERROR,
            duration_s=time.time() - started,
            agent_status="not_run",
            timed_out=False,
            evaluator_status="not_run",
            evaluator_resolved=None,
            failure_summary=message,
            repository=repository or "unknown",
            domain=domain,
            error=message,
        )
        return complete(case, _case_error(case))

    workspace = case_dir / "workspace"
    try:
        workspace_info = prepare_workspace(
            instance=instance,
            workspace=workspace,
            repos_dir=config.dataset.repos_dir,
            run_root=context.run_root,
            case_dir=case_dir,
        )
    except RunnerError as exc:
        message = f"workspace preparation failed: {exc}"
        case = _case(
            config=config,
            context=context,
            instance=instance,
            split=split,
            repeat_index=repeat_index,
            case_dir=case_dir,
            status=CASE_STATUS_INFRA_ERROR,
            duration_s=time.time() - started,
            agent_status="not_run",
            timed_out=False,
            evaluator_status="not_run",
            evaluator_resolved=None,
            failure_summary=message,
            repository=repository,
            domain=domain,
            error=message,
        )
        return complete(case, _case_error(case))

    (case_dir / "instruction.md").write_text(instance.instruction_text(), encoding="utf-8")
    agent_input = instance.agent_view()
    assert_no_protected_fields(agent_input)
    (case_dir / "agent_input.json").write_text(
        json.dumps(agent_input, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "task_id": instance.task_id,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "workspace": str(workspace),
                "workspace_head": workspace_info.get("head_commit"),
                "repository_cache": workspace_info.get("cache_path"),
                "tool_timeout_s": config.eval.tool_timeout_s,
                "task_timeout_s": config.eval.task_timeout_s,
                "docker_image": config.constraints.docker_image,
                "docker_image_digest": config.constraints.docker_image_digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        agent = run_agent(
            config=config,
            context=context,
            instance=instance,
            case_dir=case_dir,
            workspace=workspace,
            split=split,
            repeat_index=repeat_index,
            candidate_workspace=candidate_workspace,
            bandwidth_plan=bandwidth_plan,
        )
    except RunnerError as exc:
        message = f"agent adapter configuration error: {exc}"
        case = _case(
            config=config,
            context=context,
            instance=instance,
            split=split,
            repeat_index=repeat_index,
            case_dir=case_dir,
            status=CASE_STATUS_AGENT_ERROR,
            duration_s=time.time() - started,
            agent_status="error",
            timed_out=False,
            evaluator_status="not_run",
            evaluator_resolved=None,
            failure_summary=message,
            repository=repository,
            domain=domain,
            error=message,
        )
        return complete(case, _case_error(case))

    trace_path = _relative_or_none(agent.trace_path, context.run_root)
    if agent.status == "timeout":
        message = (
            f"agent exceeded task_timeout_s={config.eval.task_timeout_s}s"
            + (f": {agent.error}" if agent.error else "")
        )
        case = _case(
            config=config,
            context=context,
            instance=instance,
            split=split,
            repeat_index=repeat_index,
            case_dir=case_dir,
            status=CASE_STATUS_AGENT_TIMEOUT,
            duration_s=time.time() - started,
            agent_status="timeout",
            timed_out=True,
            evaluator_status="not_run",
            evaluator_resolved=None,
            failure_summary=message,
            repository=repository,
            domain=domain,
            trace_path=trace_path,
            tool_calls=agent.tool_calls,
            attempts=agent.attempts,
            error=message,
        )
        return complete(case, _case_error(case))
    if agent.status != "complete":
        message = f"agent failed: {agent.error or 'unknown agent error'}"
        case = _case(
            config=config,
            context=context,
            instance=instance,
            split=split,
            repeat_index=repeat_index,
            case_dir=case_dir,
            status=CASE_STATUS_AGENT_ERROR,
            duration_s=time.time() - started,
            agent_status="error",
            timed_out=False,
            evaluator_status="not_run",
            evaluator_resolved=None,
            failure_summary=message,
            repository=repository,
            domain=domain,
            trace_path=trace_path,
            tool_calls=agent.tool_calls,
            attempts=agent.attempts,
            error=message,
        )
        return complete(case, _case_error(case))

    patch_result = collect_patch(
        workspace=workspace,
        case_dir=case_dir,
        override=agent.patch_override,
        patch_source=agent.patch_source,
    )
    if patch_result.get("error"):
        message = f"patch collection failed: {patch_result['error']}"
        case = _case(
            config=config,
            context=context,
            instance=instance,
            split=split,
            repeat_index=repeat_index,
            case_dir=case_dir,
            status=CASE_STATUS_INFRA_ERROR,
            duration_s=time.time() - started,
            agent_status="complete",
            timed_out=False,
            evaluator_status="not_run",
            evaluator_resolved=None,
            failure_summary=message,
            repository=repository,
            domain=domain,
            trace_path=trace_path,
            tool_calls=agent.tool_calls,
            attempts=agent.attempts,
            error=message,
        )
        return complete(case, _case_error(case))

    patch_path = Path(str(patch_result["patch_path"]))
    predictions_path = repeat_dir / "predictions.jsonl"
    prediction = {
        "instance_id": instance.task_id,
        "model_name_or_path": config.model.model_id,
        "model_patch": patch_result["patch_text"],
    }
    with predictions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    case_predictions = case_dir / "predictions.jsonl"
    case_predictions.write_text(
        json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    evaluation = run_evaluator(
        config=config,
        context=context,
        instance=instance,
        case_dir=case_dir,
        predictions_path=case_predictions,
        repeat_index=repeat_index,
        split=split,
    )
    if evaluation.status != "complete":
        message = evaluation.error or "SWE-bench evaluator failed"
        case = _case(
            config=config,
            context=context,
            instance=instance,
            split=split,
            repeat_index=repeat_index,
            case_dir=case_dir,
            status=CASE_STATUS_EVALUATOR_ERROR,
            duration_s=time.time() - started,
            agent_status="complete",
            timed_out=False,
            evaluator_status="error",
            evaluator_resolved=None,
            failure_summary=tail_text(evaluation.failure_summary or message, max_chars=2000),
            repository=repository,
            domain=domain,
            trace_path=trace_path,
            tool_calls=agent.tool_calls,
            attempts=agent.attempts,
            patch_path=patch_path,
            error=message,
        )
        return complete(case, _case_error(case))

    resolved = bool(evaluation.resolved)
    case = _case(
        config=config,
        context=context,
        instance=instance,
        split=split,
        repeat_index=repeat_index,
        case_dir=case_dir,
        status=CASE_STATUS_RESOLVED if resolved else CASE_STATUS_UNRESOLVED,
        duration_s=time.time() - started,
        agent_status="complete",
        timed_out=False,
        evaluator_status="complete",
        evaluator_resolved=resolved,
        failure_summary=evaluation.failure_summary if not resolved else None,
        repository=repository,
        domain=domain,
        trace_path=trace_path,
        tool_calls=agent.tool_calls,
        attempts=agent.attempts,
        patch_path=patch_path,
        error=None,
    )
    case["legacy"] = {
        "patch_source": agent.patch_source,
        "workspace_head": workspace_info.get("head_commit"),
        "evaluator_command": evaluation.command,
    }
    return complete(case, None)


# ---------------------------------------------------------------------------
# Workspace preparation and patch collection
# ---------------------------------------------------------------------------


def prepare_workspace(
    *,
    instance: SwebenchInstance,
    workspace: Path,
    repos_dir: Path | None,
    run_root: Path,
    case_dir: Path,
) -> dict[str, Any]:
    if workspace.exists():
        shutil.rmtree(workspace)
    if repos_dir is None:
        raise RunnerError(
            "[dataset].repos_dir must point at local repository clones so the runner can "
            f"check out {instance.repo}@{instance.base_commit} without network access"
        )
    cache = _resolve_repo_cache(repos_dir, instance.repo)
    if cache is None:
        raise RunnerError(
            f"no local clone of {instance.repo} under repos_dir={repos_dir}; expected one of "
            f"{_repo_cache_candidates(repos_dir, instance.repo)}"
        )
    if not _commit_exists(cache, instance.base_commit):
        raise RunnerError(
            f"local clone {cache} does not contain base commit {instance.base_commit} for "
            f"{instance.task_id}; fetch/pin the repository history before running"
        )
    clone = run_command(
        ["git", "clone", "--shared", "--no-checkout", str(cache), str(workspace)],
        cwd=case_dir,
        timeout_s=300,
    )
    if not clone.ok:
        raise RunnerError(f"git clone failed: {tail_text(clone.stderr, max_chars=1000)}")
    checkout = run_command(
        ["git", "-c", "advice.detachedHead=false", "checkout", "--detach", instance.base_commit],
        cwd=workspace,
        timeout_s=300,
    )
    if not checkout.ok:
        raise RunnerError(f"git checkout failed: {tail_text(checkout.stderr, max_chars=1000)}")
    head = _git_head(workspace)
    if head != instance.base_commit:
        raise RunnerError(
            f"workspace HEAD {head!r} does not equal the task base commit {instance.base_commit!r}"
        )
    return {
        "cache_path": _display_cache_path(cache, run_root, case_dir),
        "head_commit": head,
        "workspace": str(workspace),
    }


def collect_patch(
    *,
    workspace: Path,
    case_dir: Path,
    override: str | None = None,
    patch_source: str = "workspace_git",
) -> dict[str, Any]:
    patch_path = case_dir / "agent_patch.diff"
    if override is not None:
        patch_text = override
        patch_path.write_text(patch_text, encoding="utf-8")
        return {
            "patch_path": patch_path,
            "patch_text": patch_text,
            "patch_sha256": sha256_text(patch_text),
            "patch_source": patch_source,
        }
    add = run_command(["git", "add", "-A", "-N"], cwd=workspace, timeout_s=120)
    if not add.ok:
        return {"error": f"git add -A -N failed: {tail_text(add.stderr, max_chars=1000)}"}
    diff = run_command(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=workspace,
        timeout_s=300,
    )
    if not diff.ok:
        return {"error": f"git diff failed: {tail_text(diff.stderr, max_chars=1000)}"}
    patch_text = diff.stdout
    patch_path.write_text(patch_text, encoding="utf-8")
    return {
        "patch_path": patch_path,
        "patch_text": patch_text,
        "patch_sha256": sha256_text(patch_text),
        "patch_source": patch_source,
    }


def _resolve_repo_cache(repos_dir: Path, repo: str) -> Path | None:
    for candidate in _repo_cache_candidates(repos_dir, repo):
        if candidate.is_dir():
            return candidate
    return None


def _repo_cache_candidates(repos_dir: Path, repo: str) -> list[Path]:
    return [
        repos_dir / repo,
        repos_dir / repo.replace("/", "__"),
        repos_dir / repo.replace("/", "_"),
        repos_dir / repo.split("/")[-1],
    ]


def _commit_exists(cache: Path, commit: str) -> bool:
    result = run_command(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=cache, timeout_s=60)
    return result.ok


def _git_head(workspace: Path) -> str | None:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=workspace, timeout_s=60)
    if not result.ok:
        return None
    return result.stdout.strip()


def _display_cache_path(cache: Path, run_root: Path, case_dir: Path) -> str:
    try:
        return relative_to_run(cache, run_root)
    except RunnerError:
        meta = case_dir / "repository_cache.txt"
        meta.write_text(str(cache) + "\n", encoding="utf-8")
        return str(meta)


# ---------------------------------------------------------------------------
# Agent adapters
# ---------------------------------------------------------------------------


def run_agent(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    case_dir: Path,
    workspace: Path,
    split: str,
    repeat_index: int,
    candidate_workspace: Path | None,
    bandwidth_plan: Any,
) -> AgentOutcome:
    if config.agent.kind == "command":
        return _run_command_agent(
            config=config,
            context=context,
            instance=instance,
            case_dir=case_dir,
            workspace=workspace,
            split=split,
            repeat_index=repeat_index,
            candidate_workspace=candidate_workspace,
            bandwidth_plan=bandwidth_plan,
        )
    if config.agent.kind in {"harbor", "harbor_exec"}:
        return _run_harbor_agent(
            config=config,
            context=context,
            instance=instance,
            case_dir=case_dir,
            workspace=workspace,
            split=split,
            repeat_index=repeat_index,
            candidate_workspace=candidate_workspace,
            bandwidth_plan=bandwidth_plan,
        )
    raise RunnerError(f"unknown agent kind: {config.agent.kind!r}")


def _agent_placeholders(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    case_dir: Path,
    workspace: Path,
    split: str,
    repeat: str | int,
) -> dict[str, Any]:
    return {
        "instruction_file": case_dir / "instruction.md",
        "instruction": case_dir / "instruction.md",
        "workspace": workspace,
        "repo_workspace": workspace,
        "case_dir": case_dir,
        "task_id": instance.task_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "split": split,
        "timeout_s": f"{config.eval.task_timeout_s:g}",
        "tool_timeout_s": f"{config.eval.tool_timeout_s:g}",
        "harness_snapshot": context.run_root / str(context.harness["snapshot_path"]),
        "candidate_workspace": (context.run_root / str(context.harness["snapshot_path"])),
        "run_root": context.run_root,
        "run_id": context.run_id,
        "result_id": context.result_id,
        "model": config.model.model_id,
        "python": sys.executable,
        "patch_path": case_dir / "agent_patch.diff",
        "trace_path": case_dir / "agent_trace.jsonl",
        "verification_path": case_dir / "harbor_workspace_verification.json",
        "repo_root": REPO_ROOT,
        "repeat": str(repeat),
    }


def _agent_env(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    case_dir: Path,
    workspace: Path,
    split: str,
    repeat_index: int,
    candidate_workspace: Path | None,
) -> dict[str, str]:
    candidate = (
        Path(candidate_workspace).resolve()
        if candidate_workspace is not None
        else context.run_root / str(context.harness["snapshot_path"])
    )
    overrides = {
        "SELF_HARNESS_MODEL": config.model.model_id,
        "SELF_HARNESS_TASK_ID": instance.task_id,
        "SELF_HARNESS_REPO": instance.repo,
        "SELF_HARNESS_BASE_COMMIT": instance.base_commit,
        "SELF_HARNESS_SPLIT": split,
        "SELF_HARNESS_REPEAT": str(repeat_index),
        "SELF_HARNESS_WORKSPACE": str(workspace),
        "SELF_HARNESS_CASE_DIR": str(case_dir),
        "SELF_HARNESS_TASK_TIMEOUT_S": f"{config.eval.task_timeout_s:g}",
        "SELF_HARNESS_TOOL_TIMEOUT_S": f"{config.eval.tool_timeout_s:g}",
        "SELF_HARNESS_AGENT_TIMEOUT_SEC": f"{config.eval.task_timeout_s:g}",
        "SELF_HARNESS_CANDIDATE_WORKSPACE": str(candidate),
        "SELF_HARNESS_AGENT_TRACE": str(case_dir / "agent_trace.jsonl"),
        "SELF_HARNESS_RUN_ID": context.run_id,
        "SELF_HARNESS_RESULT_ID": context.result_id,
        "SELF_HARNESS_REPO_ROOT": str(REPO_ROOT),
        "OPENAI_BASE_URL": config.model.base_url,
        "SELF_HARNESS_BENCHMARK": BENCHMARK_SWEBENCH,
    }
    if config.constraints.docker_image:
        overrides["SELF_HARNESS_AGENT_IMAGE"] = config.constraints.docker_image
    if config.constraints.docker_image_digest:
        overrides["SELF_HARNESS_AGENT_IMAGE_DIGEST"] = config.constraints.docker_image_digest
    overrides.update(dict(config.agent.extra_env))
    return env_with_overrides(
        overrides=overrides,
        passthrough=config.agent.env_passthrough,
    )


def _run_command_agent(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    case_dir: Path,
    workspace: Path,
    split: str,
    repeat_index: int,
    candidate_workspace: Path | None,
    bandwidth_plan: Any,
) -> AgentOutcome:
    if config.agent.command is None:
        raise RunnerError("[agent].command is required")
    placeholders = _agent_placeholders(
        config=config,
        context=context,
        instance=instance,
        case_dir=case_dir,
        workspace=workspace,
        split=split,
        repeat=repeat_index,
    )
    env = _agent_env(
        config=config,
        context=context,
        instance=instance,
        case_dir=case_dir,
        workspace=workspace,
        split=split,
        repeat_index=repeat_index,
        candidate_workspace=candidate_workspace,
    )
    attempts_path = case_dir / "agent_attempts"
    attempts_path.mkdir(parents=True, exist_ok=True)
    max_retries = max(config.eval.max_retries, config.agent.max_retries)
    records: list[AttemptRecord] = []
    last: CommandResult | None = None
    for attempt in range(1, max_retries + 2):
        argv = expand_command_template(config.agent.command, placeholders=placeholders)
        argv = bandwidth_plan.wrap(argv) if bandwidth_plan is not None else argv
        last = run_command(argv, cwd=workspace, env=env, timeout_s=config.eval.task_timeout_s)
        (attempts_path / f"attempt-{attempt:02d}.stdout.log").write_text(
            last.stdout, encoding="utf-8", errors="replace"
        )
        (attempts_path / f"attempt-{attempt:02d}.stderr.log").write_text(
            last.stderr, encoding="utf-8", errors="replace"
        )
        if last.timed_out:
            status = "timeout"
            reason = last.error
        elif last.error is not None or last.returncode != 0:
            status = "error"
            reason = last.error or f"returncode={last.returncode}"
        else:
            status = "complete"
            reason = None
        records.append(
            AttemptRecord(
                attempt=attempt,
                status=status,
                duration_s=last.duration_s,
                reason=reason,
            )
        )
        if status != "error":
            break
    assert last is not None
    case_dir.joinpath("agent_stdout.log").write_text(
        last.stdout, encoding="utf-8", errors="replace"
    )
    case_dir.joinpath("agent_stderr.log").write_text(
        last.stderr, encoding="utf-8", errors="replace"
    )
    trace = case_dir / "agent_trace.jsonl"
    if last.timed_out:
        status = "timeout"
        error = last.error
    elif last.error is not None:
        status = "error"
        error = last.error
    elif last.returncode != 0:
        status = "error"
        error = f"agent command exited with returncode={last.returncode}"
    else:
        status = "complete"
        error = None
    return AgentOutcome(
        status=status,
        timed_out=last.timed_out,
        duration_s=sum(record.duration_s for record in records),
        stdout=last.stdout,
        stderr=last.stderr,
        returncode=last.returncode,
        error=error,
        command=last.argv,
        attempts=records,
        trace_path=trace if trace.exists() else None,
        tool_calls=count_trace_tool_calls(trace),
        patch_source="workspace_git",
    )


def _run_harbor_agent(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    case_dir: Path,
    workspace: Path,
    split: str,
    repeat_index: int,
    candidate_workspace: Path | None,
    bandwidth_plan: Any,
) -> AgentOutcome:
    if config.agent.harbor_command is None:
        raise RunnerError("[agent].harbor_command is required")
    placeholders = _agent_placeholders(
        config=config,
        context=context,
        instance=instance,
        case_dir=case_dir,
        workspace=workspace,
        split=split,
        repeat=repeat_index,
    )
    env = _agent_env(
        config=config,
        context=context,
        instance=instance,
        case_dir=case_dir,
        workspace=workspace,
        split=split,
        repeat_index=repeat_index,
        candidate_workspace=candidate_workspace,
    )
    argv = expand_command_template(config.agent.harbor_command, placeholders=placeholders)
    argv = bandwidth_plan.wrap(argv) if bandwidth_plan is not None else argv
    (case_dir / "agent_command.json").write_text(
        json.dumps({"argv": argv, "cwd": str(workspace)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = run_command(argv, cwd=workspace, env=env, timeout_s=config.eval.task_timeout_s)
    (case_dir / "agent_stdout.log").write_text(
        completed.stdout, encoding="utf-8", errors="replace"
    )
    (case_dir / "agent_stderr.log").write_text(
        completed.stderr, encoding="utf-8", errors="replace"
    )
    trace = case_dir / "agent_trace.jsonl"
    verification = _read_json_file(case_dir / "harbor_workspace_verification.json")
    if completed.timed_out:
        return AgentOutcome(
            status="timeout",
            timed_out=True,
            duration_s=completed.duration_s,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            error=completed.error,
            command=completed.argv,
            attempts=[AttemptRecord(1, "timeout", completed.error, completed.duration_s)],
            trace_path=trace if trace.exists() else None,
            tool_calls=count_trace_tool_calls(trace),
            verification=verification,
        )
    if not completed.ok:
        return AgentOutcome(
            status="error",
            timed_out=False,
            duration_s=completed.duration_s,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            error=completed.error or f"harbor command returncode={completed.returncode}",
            command=completed.argv,
            attempts=[
                AttemptRecord(
                    1,
                    "error",
                    completed.error or f"returncode={completed.returncode}",
                    completed.duration_s,
                )
            ],
            trace_path=trace if trace.exists() else None,
            tool_calls=count_trace_tool_calls(trace),
            verification=verification,
        )
    verification_error = _validate_harbor_verification(verification, instance=instance)
    if verification_error:
        return AgentOutcome(
            status="error",
            timed_out=False,
            duration_s=completed.duration_s,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            error=verification_error,
            command=completed.argv,
            attempts=[AttemptRecord(1, "error", verification_error, completed.duration_s)],
            trace_path=trace if trace.exists() else None,
            tool_calls=count_trace_tool_calls(trace),
            verification=verification,
        )
    patch_file = case_dir / "agent_patch.diff"
    patch_override = None
    patch_source = "workspace_git"
    if patch_file.exists():
        patch_override = patch_file.read_text(encoding="utf-8", errors="replace")
        patch_source = "harbor_patch_artifact"
    return AgentOutcome(
        status="complete",
        timed_out=False,
        duration_s=completed.duration_s,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        error=None,
        command=completed.argv,
        attempts=[AttemptRecord(1, "complete", None, completed.duration_s)],
        trace_path=trace if trace.exists() else None,
        tool_calls=count_trace_tool_calls(trace),
        patch_override=patch_override,
        patch_source=patch_source,
        verification=verification,
    )


def _validate_harbor_verification(
    verification: Mapping[str, Any] | None,
    *,
    instance: SwebenchInstance,
) -> str | None:
    if not verification:
        return (
            "Harbor SWE-bench adapter must write harbor_workspace_verification.json with "
            "base_commit/head_commit after preparing the task workspace; refusing to trust "
            "the run without that verification"
        )
    head = str(verification.get("head_commit") or "").strip()
    base = str(verification.get("base_commit") or "").strip()
    if not head or not base:
        return "harbor_workspace_verification.json must contain base_commit and head_commit"
    if head != instance.base_commit:
        return (
            f"Harbor workspace HEAD {head!r} does not match task base commit "
            f"{instance.base_commit!r}"
        )
    if base != instance.base_commit:
        return (
            f"Harbor workspace verification base_commit {base!r} does not match task base "
            f"commit {instance.base_commit!r}"
        )
    return None


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def run_evaluator(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    case_dir: Path,
    predictions_path: Path,
    repeat_index: int,
    split: str,
) -> EvaluationOutcome:
    evaluator_dir = case_dir / "evaluator"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    kind = config.evaluator.kind
    if kind == "official":
        return _run_official_evaluator(
            config=config,
            context=context,
            instance=instance,
            evaluator_dir=evaluator_dir,
            predictions_path=predictions_path,
            repeat_index=repeat_index,
            split=split,
        )
    if kind == "command":
        return _run_command_evaluator(
            config=config,
            context=context,
            instance=instance,
            evaluator_dir=evaluator_dir,
            predictions_path=predictions_path,
            repeat_index=repeat_index,
            split=split,
        )
    if kind == "fixture":
        return _run_fixture_evaluator(
            config=config,
            instance=instance,
            evaluator_dir=evaluator_dir,
        )
    return EvaluationOutcome(
        status="error",
        resolved=None,
        failure_summary=f"unknown evaluator kind: {kind!r}",
        duration_s=0.0,
        error=f"unknown evaluator kind: {kind!r}",
    )


def _evaluator_meta(config: EvalConfig) -> dict[str, Any]:
    kind = config.evaluator.kind
    if kind == "official":
        name = EVALUATOR_NAME_OFFICIAL
        version = _swebench_package_version()
    elif kind == "fixture":
        name = EVALUATOR_NAME_FIXTURE
        version = "fixture"
    else:
        name = config.evaluator.name or "swebench-evaluator-command"
        version = config.evaluator.version or "configured"
    evaluator_config = {
        "kind": kind,
        "dataset": config.evaluator.dataset,
        "split": config.evaluator.split,
        "command_template": config.evaluator.command,
        "max_workers": config.evaluator.max_workers,
        "report_dir": str(config.evaluator.report_dir) if config.evaluator.report_dir else None,
    }
    return {"name": name, "version": version, "config": evaluator_config}


def _evaluator_preflight_error(config: EvalConfig) -> str | None:
    kind = config.evaluator.kind
    if kind == "fixture":
        if not config.eval.smoke_test:
            return "fixture evaluator is only allowed for smoke tests"
        return None
    if kind == "command":
        if not config.evaluator.command:
            return "[evaluator].command is required when [evaluator].kind='command'"
        return None
    try:
        import importlib.util

        if importlib.util.find_spec("swebench") is None:
            return (
                "the official SWE-bench evaluator package is not installed. Install the pinned "
                "`swebench` package (see README) or configure [evaluator].kind='command' to invoke "
                "a containerized official harness. Results are not fabricated without an evaluator."
            )
    except (ImportError, ValueError) as exc:
        return f"cannot import the official SWE-bench evaluator: {exc}"
    if shutil.which("docker") is None:
        return "the official SWE-bench evaluator requires a working Docker runtime"
    return None


def _run_official_evaluator(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    evaluator_dir: Path,
    predictions_path: Path,
    repeat_index: int,
    split: str,
) -> EvaluationOutcome:
    started = time.time()
    report_dir = evaluator_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_tag = _safe_slug(
        f"{context.result_id}-{instance.task_id}-r{repeat_index}-{int(time.time())}"
    )[:120]
    dataset_arg = _resolve_evaluator_dataset_arg(config)
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_arg,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max(1, config.evaluator.max_workers)),
        "--run_id",
        run_tag,
        "--instance_ids",
        instance.task_id,
        "--report_dir",
        str(report_dir),
        "--timeout",
        str(int(config.eval.task_timeout_s)),
    ]
    if config.evaluator.split:
        command.extend(["--split", config.evaluator.split])
    extra_args = config.evaluator.raw.get("args")
    if isinstance(extra_args, list):
        command.extend(str(item) for item in extra_args)
    completed = run_command(
        command,
        cwd=report_dir,
        env=env_with_overrides(
            base_env=_evaluator_env(),
            overrides={
                "SELF_HARNESS_TOOL_TIMEOUT_S": f"{config.eval.tool_timeout_s:g}",
            },
        ),
        timeout_s=config.eval.task_timeout_s,
    )
    stdout_path = evaluator_dir / "stdout.log"
    stderr_path = evaluator_dir / "stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    if completed.timed_out:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary=f"official SWE-bench evaluator timed out after {config.eval.task_timeout_s}s",
            duration_s=time.time() - started,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error="official evaluator timed out",
        )
    report_path = _find_report(report_dir, instance.task_id)
    if report_path is None:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary=(
                "official SWE-bench evaluator produced no report for the instance. "
                f"stdout tail: {tail_text(completed.stdout, max_chars=1500)}"
            ),
            duration_s=time.time() - started,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error="missing official evaluator report",
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary=f"cannot parse official SWE-bench report: {exc}",
            duration_s=time.time() - started,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            report_path=report_path,
            error=f"invalid evaluator report: {exc}",
        )
    parsed = _parse_report_entry(report, instance.task_id)
    if parsed is None:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary="official SWE-bench report does not contain the task instance",
            duration_s=time.time() - started,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            report_path=report_path,
            error="official evaluator report missing instance",
        )
    return EvaluationOutcome(
        status="complete",
        resolved=bool(parsed["resolved"]),
        failure_summary=parsed.get("failure_summary"),
        duration_s=time.time() - started,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        report_path=report_path,
    )


def _official_dataset_argument(config: EvalConfig) -> str:
    if config.dataset.source:
        return str(Path(config.dataset.source).expanduser())
    return config.dataset.name


def _resolve_evaluator_dataset_arg(config: EvalConfig) -> str:
    raw = config.evaluator.dataset
    if not raw:
        return _official_dataset_argument(config)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (config.path.parent / candidate).resolve()
    if candidate.exists():
        return str(candidate)
    return raw


def _evaluator_env() -> dict[str, str]:
    """Environment for evaluator subprocesses: never pass model credentials."""

    env = dict(os.environ)
    env.pop("SELF_HARNESS_MODEL_API_KEY", None)
    env.pop("OPENAI_BASE_URL", None)
    env.pop("SELF_HARNESS_MODEL", None)
    return env


def _find_report(report_dir: Path, task_id: str) -> Path | None:
    candidates = sorted(report_dir.rglob("*.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and task_id in payload:
            return path
    return candidates[0] if candidates else None


def _parse_report_entry(report: Any, task_id: str) -> dict[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    entry = report.get(task_id)
    if entry is None:
        for value in report.values():
            if isinstance(value, Mapping) and str(value.get("instance_id") or "") == task_id:
                entry = value
                break
    if not isinstance(entry, Mapping):
        # Classic SWE-bench evaluator writes a run summary instead of a
        # per-instance dict when no per-instance report file is available.
        for resolved_key, error_key in (
            ("resolved_ids", "error_ids"),
            ("completed_ids", "error_ids"),
        ):
            resolved_ids = report.get(resolved_key)
            error_ids = report.get(error_key)
            if isinstance(resolved_ids, list):
                resolved = task_id in {str(item) for item in resolved_ids}
                if not resolved and isinstance(error_ids, list) and task_id in {
                    str(item) for item in error_ids
                }:
                    return None
                reasons = report.get("failure_reasons")
                summary = None
                if isinstance(reasons, Mapping):
                    summary = str(reasons.get(task_id) or "") or None
                return {"resolved": resolved, "failure_summary": summary}
        return None
    resolved = entry.get("resolved")
    if not isinstance(resolved, bool):
        return None
    failure_summary = None
    for key in ("error", "failure_summary", "solution"):
        value = entry.get(key)
        if value:
            failure_summary = str(value)[:2000]
            break
    tests_status = entry.get("tests_status")
    if failure_summary is None and isinstance(tests_status, Mapping):
        failed = [
            name
            for name, status in tests_status.items()
            if isinstance(status, str) and status.upper() not in {"PASSED", "PASS", "SUCCESS"}
        ]
        if failed:
            failure_summary = "failing test groups: " + ", ".join(failed[:10])
    return {"resolved": resolved, "failure_summary": failure_summary}


def _run_command_evaluator(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    evaluator_dir: Path,
    predictions_path: Path,
    repeat_index: int,
    split: str,
) -> EvaluationOutcome:
    started = time.time()
    if config.evaluator.command is None:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary="[evaluator].command is not configured",
            duration_s=0.0,
            error="missing evaluator command",
        )
    output_path = evaluator_dir / "evaluation.json"
    placeholders = {
        "task_id": instance.task_id,
        "instance_id": instance.task_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "predictions": predictions_path,
        "prediction_file": predictions_path,
        "patch": evaluator_dir.parent / "agent_patch.diff",
        "case_dir": evaluator_dir.parent,
        "evaluator_dir": evaluator_dir,
        "output": output_path,
        "evaluation_output": output_path,
        "report": evaluator_dir / "report.json",
        "dataset": config.dataset.source or config.dataset.name,
        "dataset_revision": config.dataset.revision,
        "split": config.evaluator.split or config.dataset.split,
        "run_id": context.run_id,
        "result_id": context.result_id,
        "workspace": evaluator_dir.parent / "workspace",
        "repeat": str(repeat_index),
        "eval_split": split,
    }
    argv = expand_command_template(config.evaluator.command, placeholders=placeholders)
    completed = run_command(
        argv,
        cwd=evaluator_dir,
        env=env_with_overrides(
            base_env=_evaluator_env(),
            overrides={
                "SELF_HARNESS_TASK_ID": instance.task_id,
                "SELF_HARNESS_TOOL_TIMEOUT_S": f"{config.eval.tool_timeout_s:g}",
            },
            passthrough=config.agent.env_passthrough,
        ),
        timeout_s=config.eval.task_timeout_s,
    )
    stdout_path = evaluator_dir / "stdout.log"
    stderr_path = evaluator_dir / "stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    if completed.timed_out:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary=f"evaluator command timed out after {config.eval.task_timeout_s}s",
            duration_s=time.time() - started,
            command=argv,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error="evaluator command timed out",
        )
    payload = _read_json_file(output_path)
    if payload is None:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary=(
                f"evaluator command did not write {output_path.name}. "
                f"stdout tail: {tail_text(completed.stdout, max_chars=1500)}"
            ),
            duration_s=time.time() - started,
            command=argv,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error="missing evaluator output",
        )
    parsed = _parse_report_entry(payload, instance.task_id)
    if parsed is None and isinstance(payload, Mapping) and "resolved" in payload:
        parsed = _parse_report_entry({instance.task_id: payload}, instance.task_id)
    if parsed is None:
        return EvaluationOutcome(
            status="error",
            resolved=None,
            failure_summary=f"evaluator output does not record a boolean resolved for {instance.task_id}",
            duration_s=time.time() - started,
            command=argv,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error="malformed evaluator output",
        )
    return EvaluationOutcome(
        status="complete",
        resolved=bool(parsed["resolved"]),
        failure_summary=parsed.get("failure_summary"),
        duration_s=time.time() - started,
        command=argv,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        report_path=output_path,
    )


def _run_fixture_evaluator(
    *,
    config: EvalConfig,
    instance: SwebenchInstance,
    evaluator_dir: Path,
) -> EvaluationOutcome:
    raw = config.evaluator.raw
    resolved_ids: set[str] = set()
    if isinstance(raw.get("resolved_task_ids"), list):
        resolved_ids = {str(item) for item in raw["resolved_task_ids"]}
    elif raw.get("resolved_task_ids_file"):
        path = Path(str(raw["resolved_task_ids_file"]))
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                resolved_ids = {str(item) for item in payload}
    resolved = instance.task_id in resolved_ids
    output = {
        instance.task_id: {
            "resolved": resolved,
            "error": None if resolved else "fixture evaluator marks this task unresolved",
        }
    }
    (evaluator_dir / "evaluation.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return EvaluationOutcome(
        status="complete",
        resolved=resolved,
        failure_summary=None if resolved else "fixture evaluator marks this task unresolved",
        duration_s=0.0,
        command=["<fixture-swebench-evaluator>"],
        report_path=evaluator_dir / "evaluation.json",
    )


def _swebench_package_version() -> str:
    try:
        return importlib.metadata.version("swebench")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


# ---------------------------------------------------------------------------
# Case construction / misc helpers
# ---------------------------------------------------------------------------


def _case(
    *,
    config: EvalConfig,
    context: RunContext,
    instance: SwebenchInstance,
    split: str,
    repeat_index: int,
    case_dir: Path,
    status: str,
    duration_s: float,
    agent_status: str,
    timed_out: bool,
    evaluator_status: str,
    evaluator_resolved: bool | None,
    failure_summary: str | None,
    repository: str | None,
    domain: str | None,
    trace_path: str | None = None,
    tool_calls: int | None = None,
    attempts: Sequence[AttemptRecord] = (),
    patch_path: Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    patch_block = None
    if patch_path is not None and patch_path.exists():
        patch_block = {
            "path": relative_to_run(patch_path, context.run_root),
            "sha256": sha256_file(patch_path),
        }
    attempts_payload = [record.to_dict() for record in attempts]
    record = make_case_result(
        task_id=instance.task_id,
        split=split,
        repeat=repeat_index,
        repository=repository,
        domain=domain,
        status=status,
        duration_s=duration_s,
        agent=make_agent_block(
            status=agent_status,
            timed_out=timed_out,
            tool_calls=tool_calls,
            trace_path=trace_path,
            attempts=attempts_payload or None,
        ),
        evaluator=make_evaluator_block(
            status=evaluator_status,
            name=str(_evaluator_meta(config)["name"]),
            resolved=evaluator_resolved,
            failure_summary=failure_summary,
            log_path=_relative_or_none(case_dir / "evaluator" / "stdout.log", context.run_root)
            if evaluator_status != "not_run"
            else None,
        ),
        patch=patch_block,
        artifacts={
            "case_dir": relative_to_run(case_dir, context.run_root),
            "agent_stdout": _relative_or_none(case_dir / "agent_stdout.log", context.run_root),
            "agent_stderr": _relative_or_none(case_dir / "agent_stderr.log", context.run_root),
            "agent_result": _relative_or_none(case_dir / "agent_result.json", context.run_root),
            "summary": relative_to_run(case_dir / "case_result.json", context.run_root),
        },
        error=error,
    )
    agent_runtime = _read_json_file(case_dir / "agent_result.json")
    if agent_runtime:
        record["agent_runtime"] = {
            key: agent_runtime.get(key)
            for key in (
                "status",
                "container_id",
                "image",
                "image_digest",
                "container_args",
                "readonly_mounts",
                "read_only_root",
                "model",
                "base_url",
                "duration_s",
                "tool_calls",
                "harness_file_sha256",
            )
        }
    return record


def _case_error(case: Mapping[str, Any]) -> dict[str, Any] | None:
    if case.get("passed") is not None:
        return None
    return {
        "split": case.get("split"),
        "repeat": case.get("repeat"),
        "task_id": case.get("task_id"),
        "status": case.get("status"),
        "message": str(case.get("error") or "unscored task"),
    }


def _not_run_splits(
    *,
    manifest: SwebenchManifest,
    split_task_ids: Mapping[str, Sequence[str]],
    repeats: int,
    reason: str,
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for split, task_ids in split_task_ids.items():
        records[split] = []
        for repeat_index in range(1, repeats + 1):
            case_results = []
            for task_id in task_ids:
                case_results.append(
                    make_case_result(
                        task_id=task_id,
                        split=split,
                        repeat=repeat_index,
                        repository=manifest.repository_of(task_id),
                        domain=manifest.domain_of(task_id),
                        status=CASE_STATUS_NOT_RUN,
                        duration_s=0.0,
                        agent=make_agent_block(
                            status="not_run", timed_out=False, tool_calls=None, trace_path=None
                        ),
                        evaluator=make_evaluator_block(
                            status="not_run",
                            name="swebench-evaluator-unavailable",
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
                    expected_task_ids=list(task_ids),
                    case_results=case_results,
                )
            )
    return records


def _run_options(
    config: EvalConfig,
    split_task_ids: Mapping[str, Sequence[str]],
    repeats: int,
    selected: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "benchmark": config.eval.benchmark,
        "dataset_name": config.dataset.name,
        "dataset_revision": config.dataset.revision,
        "dataset_split": config.dataset.split,
        "splits": list(selected),
        "task_ids": {split: list(ids) for split, ids in split_task_ids.items()},
        "repeats": repeats,
        "task_timeout_s": config.eval.task_timeout_s,
        "tool_timeout_s": config.eval.tool_timeout_s,
        "case_concurrency": config.eval.case_concurrency,
        "max_retries": max(config.eval.max_retries, config.agent.max_retries),
        "evaluator": {
            "kind": config.evaluator.kind,
            "name": config.evaluator.name,
            "version": config.evaluator.version,
        },
        "agent_kind": config.agent.kind,
        "docker_image": config.constraints.docker_image,
        "docker_image_digest": config.constraints.docker_image_digest,
        "gold_patch_exposed_to_agent": False,
    }


def _result_extra(
    *,
    config: EvalConfig,
    split_task_ids: Mapping[str, Sequence[str]],
    repeat_count: int,
    selected: Mapping[str, Sequence[str]],
    pilot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "run_options": _run_options(config, split_task_ids, repeat_count, selected)
    }
    if pilot:
        extra["pilot"] = dict(pilot)
    return extra


def _case_fingerprint(
    config: EvalConfig,
    context: RunContext,
    dataset: SwebenchDataset,
) -> dict[str, Any]:
    return {
        "schema_version": "self_harness.swebench_case.v1",
        "config_sha256": config.sha256,
        "resolved_config_sha256": context.config_block["resolved_config_sha256"],
        "comparability_sha256": context.config_block["comparability_sha256"],
        "model_config_sha256": context.model["config_sha256"],
        "harness_snapshot_sha256": context.harness["snapshot_sha256"],
        "dataset_revision": dataset.revision,
        "dataset_source_sha256": dataset.source_sha256,
        "task_ids_sha256": context.dataset["task_ids_sha256"],
    }


def _fallback_harness_snapshot(output_dir: Path, reason: str) -> Path:
    snapshot = output_dir / "harness_unavailable"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "HARNESS_UNAVAILABLE.txt").write_text(
        "The configured harness snapshot could not be prepared. This evaluation result is\n"
        "recorded as failed; no harness code was executed.\n\n"
        f"reason: {reason}\n",
        encoding="utf-8",
    )
    return snapshot


def _audit_reuse(output_dir: Path, result_path: Path, *, kind: str) -> None:
    (output_dir / "reuse.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "reused_result": result_path.name,
                "result_sha256": sha256_file(result_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def count_trace_tool_calls(trace_path: Path | None) -> int | None:
    if trace_path is None or not trace_path.exists():
        return None
    total = 0
    found = False
    try:
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, Mapping):
                calls = event.get("tool_calls")
                if isinstance(calls, list):
                    found = True
                    total += len(calls)
                elif str(event.get("type") or "") in {"tool_call", "tool_use"}:
                    found = True
                    total += 1
    except OSError:
        return None
    return total if found else None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _relative_or_none(path: Path | None, run_root: Path) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return relative_to_run(path, run_root)
    except RunnerError:
        return None


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")[:180] or "case"


# Keep imported protocol constants referenced for clarity/introspection.
_ = (CASE_STATUS_EVALUATOR_ERROR, PROTECTED_FIELDS, sha256_text)


def _missing_instance(task_id: str) -> SwebenchInstance:
    repo = task_id.split("__", 1)[0] + "/" + task_id.split("__", 1)[1].rsplit("-", 1)[0]
    return SwebenchInstance(
        task_id=task_id,
        repo=repo,
        base_commit="missing",
        problem_statement="",
    )
