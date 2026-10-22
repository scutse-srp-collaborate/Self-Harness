"""SWE-bench pilot helpers: deterministic task selection and honest summaries.

A pilot is intentionally *not* a benchmark run: it evaluates at most five fixed
train tasks (default two repeats) with no candidates and no heldout tasks.  The
pilot task list is derived deterministically from the frozen manifest.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import SwebenchManifest

PILOT_MAX_TASKS = 5
PILOT_SELECTION_RULE = (
    "Iterate held_in_task_ids in manifest order and select the first task for each "
    "five-domain stratum (domain first-appearance order), so the pilot covers five "
    "domains and five distinct repositories. The heldout split is never considered."
)


class PilotError(ValueError):
    """Raised when pilot task selection/validation is invalid."""


def select_pilot_tasks(
    manifest: SwebenchManifest,
    *,
    max_tasks: int = PILOT_MAX_TASKS,
    domains: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministically choose one train task per domain (repo-diverse)."""

    if max_tasks < 1:
        raise PilotError("max_tasks must be >= 1")
    domain_order: list[str] = []
    if domains:
        domain_order = [str(item) for item in domains]
    else:
        domain_order = list(dict.fromkeys(manifest.metadata.get("domain_quotas", {}).keys()))
    if not domain_order:
        # Fall back to first-appearance order derived from the manifest itself.
        domain_order = []
        for task_id in manifest.held_in_task_ids:
            domain = manifest.domain_of(task_id)
            if domain and domain not in domain_order:
                domain_order.append(domain)
    selected: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    seen_repos: set[str] = set()
    for position, task_id in enumerate(manifest.held_in_task_ids):
        if len(selected) >= max_tasks:
            break
        domain = manifest.domain_of(task_id)
        repo = manifest.repository_of(task_id)
        if domain is None or repo is None:
            raise PilotError(f"cannot resolve domain/repository for task {task_id!r}")
        if domain in seen_domains:
            continue
        if domain not in domain_order:
            domain_order.append(domain)
        seen_domains.add(domain)
        seen_repos.add(repo)
        selected.append(
            {
                "task_id": task_id,
                "repository": repo,
                "domain": domain,
                "manifest_position": position,
                "split": "train",
            }
        )
    if len(selected) < min(max_tasks, len(set(domain_order))):
        raise PilotError(
            "could not select the requested number of domain-diverse train tasks; "
            f"selected {len(selected)} of {max_tasks}"
        )
    return selected


def validate_pilot_task_ids(
    manifest: SwebenchManifest,
    task_ids: Sequence[str],
    *,
    max_tasks: int = PILOT_MAX_TASKS,
) -> list[dict[str, Any]]:
    """Validate explicit pilot ids: train-only, unique, at most max_tasks."""

    materialized = [str(item).strip() for item in task_ids]
    if not materialized:
        raise PilotError("pilot requires at least one task id")
    if any(not item for item in materialized):
        raise PilotError("pilot task ids must be non-empty strings")
    if len(set(materialized)) != len(materialized):
        raise PilotError("pilot task ids must be unique")
    if len(materialized) > max_tasks:
        raise PilotError(
            f"pilot may run at most {max_tasks} tasks, got {len(materialized)}"
        )
    heldout = set(manifest.held_out_task_ids)
    train = set(manifest.held_in_task_ids)
    leaked = sorted(set(materialized) & heldout)
    if leaked:
        raise PilotError(f"pilot task ids must not include heldout tasks: {leaked[:10]}")
    unknown = sorted(set(materialized) - train)
    if unknown:
        raise PilotError(f"pilot task ids are not in the train split: {unknown[:10]}")
    train_order = {task_id: index for index, task_id in enumerate(manifest.held_in_task_ids)}
    return [
        {
            "task_id": task_id,
            "repository": manifest.repository_of(task_id),
            "domain": manifest.domain_of(task_id),
            "manifest_position": train_order[task_id],
            "split": "train",
        }
        for task_id in sorted(materialized, key=lambda item: train_order[item])
    ]


def pilot_manifest_payload(
    manifest: SwebenchManifest,
    tasks: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> dict[str, Any]:
    return {
        "format": "self_harness.swebench_pilot.v1",
        "benchmark": "SWE-bench_Verified",
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "selection_rule": PILOT_SELECTION_RULE,
        "max_tasks": PILOT_MAX_TASKS,
        "task_count": len(tasks),
        "repeats": repeats,
        "heldout_run": False,
        "candidate_iteration": False,
        "tasks": [
            {
                "task_id": str(task["task_id"]),
                "repository": task.get("repository"),
                "domain": task.get("domain"),
                "split": "train",
                "manifest_position": task.get("manifest_position"),
            }
            for task in tasks
        ],
        "limitations": [
            "Pilot scope only: at most five fixed train tasks with repeats, not the 100-task "
            "SWE-bench Verified benchmark.",
            "No heldout evaluation and no candidate acceptance in the pilot.",
            "Executed on DeepSeek Flash; this is an evaluation of the Self-Harness method on a "
            "different model than the paper's MiniMax/GLM/Qwen configurations, not a "
            "model-for-model reproduction.",
        ],
    }


def docker_runtime_blockers(image: str) -> list[str]:
    """Return concrete blockers when a task-scoped Docker runtime is unavailable."""

    if shutil.which("docker") is None:
        return ["Docker CLI is not available; task-scoped containers cannot be created."]
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"Docker runtime check failed: {exc.__class__.__name__}: {exc}"]
    if info.returncode != 0:
        detail = (info.stderr or info.stdout or "").strip().replace("\n", " ")[:500]
        return [
            "Docker runtime is not usable for task-scoped containers: "
            f"{detail or f'exit {info.returncode}'}"
        ]
    image_check = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if image_check.returncode != 0:
        return [
            f"agent sandbox image {image!r} is not available locally and cannot be pulled "
            "through the current Docker runtime; pre-pull it or fix Docker access."
        ]
    return []


def case_status_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": case.get("task_id"),
        "repository": case.get("repository"),
        "domain": case.get("domain"),
        "repeat": case.get("repeat"),
        "status": case.get("status"),
        "passed": case.get("passed"),
        "duration_s": case.get("duration_s"),
        "agent_status": (case.get("agent") or {}).get("status"),
        "timed_out": (case.get("agent") or {}).get("timed_out"),
        "evaluator_status": (case.get("evaluator") or {}).get("status"),
        "resolved": (case.get("evaluator") or {}).get("resolved"),
        "patch_sha256": (case.get("patch") or {}).get("sha256"),
        "trace_path": (case.get("agent") or {}).get("trace_path"),
        "error": case.get("error"),
    }


def summarize_pilot_result(
    result: Mapping[str, Any],
    *,
    pilot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a pilot summary from the unified result (no metric invention)."""

    per_task: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    for split in ("train",):
        for repeat in (result.get("splits") or {}).get(split, []):
            for case in repeat.get("case_results", []):
                summary = case_status_summary(case)
                attempts.append(summary)
                entry = per_task.setdefault(
                    str(case.get("task_id")),
                    {
                        "task_id": case.get("task_id"),
                        "repository": case.get("repository"),
                        "domain": case.get("domain"),
                        "repeats": [],
                    },
                )
                entry["repeats"].append(
                    {
                        "repeat": summary["repeat"],
                        "status": summary["status"],
                        "resolved": summary["resolved"],
                        "passed": summary["passed"],
                        "patch_sha256": summary["patch_sha256"],
                        "trace_path": summary["trace_path"],
                        "error": summary["error"],
                    }
                )
    resolved = sum(1 for attempt in attempts if attempt["passed"] is True)
    total = len(attempts)
    unresolved = sum(1 for attempt in attempts if attempt["passed"] is False)
    unscoreable = sum(1 for attempt in attempts if attempt["passed"] is None)
    summary = {
        "format": "self_harness.swebench_pilot_summary.v1",
        "run_id": result.get("run_id"),
        "result_id": result.get("result_id"),
        "result_status": result.get("status"),
        "benchmark": (result.get("benchmark") or {}).get("id"),
        "benchmark_version": (result.get("benchmark") or {}).get("version"),
        "evaluator": (result.get("benchmark") or {}).get("evaluator"),
        "evaluator_version": (result.get("benchmark") or {}).get("evaluator_version"),
        "dataset_revision": (result.get("dataset") or {}).get("revision"),
        "dataset_manifest_sha256": (result.get("dataset") or {}).get("manifest_sha256"),
        "model": {
            "provider": (result.get("model") or {}).get("provider"),
            "model_id": (result.get("model") or {}).get("model_id"),
            "decoding": (result.get("model") or {}).get("decoding"),
        },
        "harness": {
            "version_id": (result.get("harness") or {}).get("version_id"),
            "snapshot_path": (result.get("harness") or {}).get("snapshot_path"),
            "snapshot_sha256": (result.get("harness") or {}).get("snapshot_sha256"),
            "files": (result.get("harness") or {}).get("files"),
        },
        "pilot": dict(pilot),
        "scope": {
            "pilot_task_count": len(per_task),
            "repeats": len({attempt["repeat"] for attempt in attempts}),
            "case_attempts": total,
            "heldout_run": False,
            "candidate_iteration": False,
            "is_full_benchmark": False,
        },
        "counts": {
            "resolved": resolved,
            "unresolved_or_failed": unresolved,
            "unscoreable": unscoreable,
            "case_attempts": total,
        },
        "per_task": [per_task[task_id] for task_id in sorted(per_task)],
        "attempts": attempts,
        "errors": list(result.get("errors") or []),
        "claims": {
            "scores_are_measured_from_recorded_result": True,
            "is_full_swe_bench_verified": False,
            "model": (result.get("model") or {}).get("model_id"),
            "note": (
                "Pilot only: at most five fixed train tasks. This is an evaluation of the "
                "Self-Harness method on the configured model, not the paper's exact model "
                "configuration and not a 100-task benchmark result."
            ),
        },
    }
    return summary


def render_pilot_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# SWE-bench Verified Pilot: `{summary.get('run_id')}`",
        "",
        f"- Result: `{summary.get('result_id')}` status `{summary.get('result_status')}`",
        f"- Benchmark: `{summary.get('benchmark')}` version `{summary.get('benchmark_version')}`",
        f"- Dataset revision: `{summary.get('dataset_revision')}`",
        f"- Evaluator: `{summary.get('evaluator')}` (version `{summary.get('evaluator_version')}`)",
        f"- Model: `{(summary.get('model') or {}).get('model_id')}` "
        f"(provider `{(summary.get('model') or {}).get('provider')}`)",
        f"- Harness snapshot: `{(summary.get('harness') or {}).get('snapshot_sha256')}`",
        "",
        "## Scope",
        "",
        f"- Pilot tasks: `{summary['scope']['pilot_task_count']}` (train only)",
        f"- Repeats: `{summary['scope']['repeats']}`",
        f"- Case attempts: `{summary['scope']['case_attempts']}`",
        "- Heldout run: `false`",
        "- Candidate iteration / acceptance: `false`",
        "- Full 100-task benchmark: `false`",
        "",
        "## Results",
        "",
        "| Task | Repository | Domain | Repeat | Status | Resolved |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for attempt in summary.get("attempts") or []:
        lines.append(
            f"| `{attempt['task_id']}` | `{attempt['repository']}` | `{attempt['domain']}` | "
            f"{attempt['repeat']} | `{attempt['status']}` | `{attempt['resolved']}` |"
        )
    counts = summary.get("counts") or {}
    lines.extend(
        [
            "",
            f"- Resolved case attempts: `{counts.get('resolved')}` / `{counts.get('case_attempts')}`",
            f"- Unresolved/failed: `{counts.get('unresolved_or_failed')}`",
            f"- Unscoreable (infra/evaluator/not_run): `{counts.get('unscoreable')}`",
            "",
            "## Claims And Limitations",
            "",
            f"- {(summary.get('claims') or {}).get('note')}",
            "- This pilot is not the full SWE-bench Verified benchmark and no heldout result "
            "was used to update the harness.",
            "",
        ]
    )
    errors = summary.get("errors") or []
    if errors:
        lines.extend(["## Errors", "", "```json", json.dumps(errors, indent=2)[:8000], "```", ""])
    return "\n".join(lines)


def write_pilot_outputs(
    *,
    result: Mapping[str, Any],
    pilot: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    summary = summarize_pilot_result(result, pilot=pilot)
    output_dir = Path(output_dir)
    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = output_dir / "pilot_summary.md"
    markdown_path.write_text(render_pilot_markdown(summary), encoding="utf-8")
    return summary_path, markdown_path
