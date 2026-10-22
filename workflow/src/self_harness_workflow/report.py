"""Final run report generation (JSON and Markdown)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_harness_eval.protocol import UNSCOREABLE_STATUSES

from .runstate import RunPaths, WorkflowError, read_json, utc_now
from .versions import load_version

REPORT_FORMAT = "self_harness.final_report.v1"


def build_final_report(
    *,
    paths: RunPaths,
    run_manifest: Mapping[str, Any],
    task_ids_payload: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    state = run_manifest.get("state") or {}
    version_records: list[dict[str, Any]] = []
    for item in run_manifest.get("versions") or []:
        try:
            record = load_version(paths, str(item["version_id"]))
        except WorkflowError:
            continue
        version_records.append(record)

    baseline_id = state.get("baseline_version_id")
    final_id = state.get("final_version_id") or state.get("active_version_id")
    baseline = _load_result(paths, baseline_id)
    final = _load_result(paths, final_id)
    if baseline is None:
        raise WorkflowError("cannot write a final report without a baseline evaluation result")

    candidate_blocks = []
    for record in version_records:
        if record.get("kind") == "baseline":
            continue
        result = _load_result(paths, record.get("version_id"))
        acceptance = _load_acceptance(paths, record.get("version_id"))
        candidate_blocks.append(
            {
                "version_id": record.get("version_id"),
                "kind": record.get("kind"),
                "status": record.get("status"),
                "parent_version_ids": record.get("parent_version_ids"),
                "lineage": record.get("lineage"),
                "proposal_id": record.get("proposal_id"),
                "candidate_id": record.get("candidate_id"),
                "mechanism_family": record.get("mechanism_family"),
                "proposal_summary": _proposal_summary(paths, record.get("version_id")),
                "eval_result_id": (record.get("eval") or {}).get("result_id"),
                "eval_result_path": (record.get("eval") or {}).get("result_path"),
                "evaluation_status": (result or {}).get("status"),
                "splits": _split_metrics(result),
                "deltas": (acceptance or {}).get("deltas"),
                "acceptance_decision": (acceptance or {}).get("decision")
                or (record.get("acceptance") or {}).get("decision"),
                "accepted": (acceptance or {}).get("accepted"),
                "acceptance_reason": (acceptance or {}).get("reason"),
                "task_transitions": (acceptance or {}).get("task_transitions"),
                "harness_snapshot_sha256": (record.get("harness") or {}).get("snapshot_sha256"),
            }
        )

    error_counts = _aggregate_error_counts(
        [baseline, final] + [(_load_result(paths, record.get("version_id")) or {}) for record in version_records]
    )
    transitions = _task_transitions(baseline, final or baseline)
    wall_seconds = _wall_seconds(state)

    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "generated_at": utc_now(),
        "run_id": run_manifest.get("run_id"),
        "benchmark": run_manifest.get("benchmark"),
        "measurement": _measurement_block(
            run_manifest=run_manifest,
            task_ids_payload=task_ids_payload,
            config=config,
            final_result=final,
        ),
        "task_manifest": {
            "benchmark": task_ids_payload.get("benchmark"),
            "splits": task_ids_payload.get("splits"),
            "counts": task_ids_payload.get("counts"),
            "distributions": task_ids_payload.get("distributions"),
            "manifest_metadata": task_ids_payload.get("manifest_metadata"),
        },
        "fixed_parameters": {
            "model": run_manifest.get("model"),
            "eval": run_manifest.get("eval"),
            "dataset": run_manifest.get("dataset"),
            "budget": run_manifest.get("budget"),
            "experiment_fingerprint": run_manifest.get("experiment_fingerprint"),
        },
        "baseline": {
            "version_id": baseline_id,
            "result_id": baseline.get("result_id"),
            "status": baseline.get("status"),
            "splits": _split_metrics(baseline),
            "harness_snapshot_sha256": (baseline.get("harness") or {}).get("snapshot_sha256"),
            "harness_files": (baseline.get("harness") or {}).get("files"),
        },
        "final": {
            "version_id": final_id,
            "result_id": (final or {}).get("result_id"),
            "status": (final or {}).get("status"),
            "splits": _split_metrics(final),
            "harness_snapshot_sha256": ((final or {}).get("harness") or {}).get("snapshot_sha256"),
            "harness_files": ((final or {}).get("harness") or {}).get("files"),
            "lineage": next(
                (record.get("lineage") for record in version_records if record.get("version_id") == final_id),
                [],
            ),
        },
        "benchmark_counts": _benchmark_counts(run_manifest.get("benchmark"), final or baseline),
        "splits": {
            "baseline": _split_metrics(baseline),
            "final": _split_metrics(final),
        },
        "repository_domain": {
            "final": _repo_domain_metrics(final),
            "baseline": _repo_domain_metrics(baseline),
        },
        "candidates": candidate_blocks,
        "task_transitions": transitions,
        "versions": [
            {
                "version_id": record.get("version_id"),
                "kind": record.get("kind"),
                "status": record.get("status"),
                "parent_version_ids": record.get("parent_version_ids"),
                "lineage": record.get("lineage"),
                "harness_snapshot_path": record.get("harness_snapshot_path"),
                "harness_snapshot_sha256": record.get("harness_snapshot_sha256"),
                "eval_result_path": (record.get("eval") or {}).get("result_path"),
                "acceptance_path": (record.get("acceptance") or {}).get("path"),
                "proposal_id": record.get("proposal_id"),
                "mechanism_family": record.get("mechanism_family"),
            }
            for record in version_records
        ],
        "totals": {
            "rounds_completed": state.get("rounds_completed"),
            "proposals_created": state.get("proposals_created"),
            "candidate_evaluations": state.get("candidate_evaluations"),
            "versions": len(version_records),
            "stop_reason": state.get("stop_reason"),
            "wall_time_s": wall_seconds,
            "started_at": state.get("started_at"),
            "stopped_at": state.get("stopped_at"),
        },
        "errors": error_counts,
        "scope": _scope_block(config=config, task_ids_payload=task_ids_payload, final=final),
        "not_executed": _not_executed_notes(config=config, task_ids_payload=task_ids_payload),
        "claims": {
            "scores_are_measured_from_results": True,
            "reproduces_paper_results": False,
            "note": (
                "All pass rates in this report are computed from the recorded unified "
                "result.json case_results. No benchmark score is entered by hand and no "
                "claim of reproducing the paper's reported numbers is made."
            ),
        },
    }
    return report


def write_final_report(
    *,
    paths: RunPaths,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    paths.final_report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.final_report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths.final_report_md.write_text(render_markdown(report), encoding="utf-8")
    return paths.final_report_json, paths.final_report_md


def _load_result(paths: RunPaths, version_id: Any) -> dict[str, Any] | None:
    if not version_id:
        return None
    path = paths.version_result(str(version_id))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_acceptance(paths: RunPaths, version_id: Any) -> dict[str, Any] | None:
    if not version_id:
        return None
    path = paths.version_acceptance(str(version_id))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _proposal_summary(paths: RunPaths, version_id: Any) -> str | None:
    if not version_id:
        return None
    path = paths.version_dir(str(version_id)) / "proposal.json"
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    return str(payload.get("summary") or payload.get("title") or "")[:2000] or None


def _split_metrics(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    metrics: dict[str, Any] = {}
    for split, summary in (result.get("summary") or {}).items():
        repeats = [
            {
                "repeat": int(repeat.get("repeat", 0)),
                "passed": int(repeat.get("passed", 0)),
                "total": int(repeat.get("total", 0)),
                "pass_rate": float(repeat.get("pass_rate", 0.0)),
                "status": repeat.get("status"),
            }
            for repeat in (result.get("splits") or {}).get(split, [])
        ]
        metrics[split] = {
            "repeats": repeats,
            "mean_repeat_pass_rate": summary.get("mean_repeat_pass_rate"),
            "pooled_passed": summary.get("pooled_passed"),
            "pooled_total": summary.get("pooled_total"),
            "pooled_pass_rate": summary.get("pooled_pass_rate"),
            "infrastructure_error_count": summary.get("infrastructure_error_count"),
            "evaluator_error_count": summary.get("evaluator_error_count"),
        }
    return metrics


def _benchmark_counts(benchmark: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for split, summary in (result.get("summary") or {}).items():
        counts[split] = {
            "passed": summary.get("pooled_passed"),
            "total": summary.get("pooled_total"),
            "pass_rate": summary.get("pooled_pass_rate"),
        }
    return {
        "benchmark": benchmark,
        "metric": "resolved_rate" if benchmark == "swe_bench_verified" else "pass_rate",
        "splits": counts,
    }


def _repo_domain_metrics(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    repo: dict[str, dict[str, int]] = {}
    domain: dict[str, dict[str, int]] = {}
    for split in ("train", "heldout"):
        for repeat in (result.get("splits") or {}).get(split, []):
            for case in repeat.get("case_results", []):
                passed = case.get("passed")
                if passed is None:
                    continue
                repo_key = str(case.get("repository") or "unknown")
                domain_key = str(case.get("domain") or "unknown")
                for bucket, key in ((repo, repo_key), (domain, domain_key)):
                    entry = bucket.setdefault(key, {"passed": 0, "total": 0})
                    entry["total"] += 1
                    if passed:
                        entry["passed"] += 1
    for bucket in (repo, domain):
        for entry in bucket.values():
            entry["pass_rate"] = entry["passed"] / entry["total"] if entry["total"] else 0.0
    return {
        "by_repository": dict(sorted(repo.items())),
        "by_domain": dict(sorted(domain.items())),
        "note": (
            "Repository/domain pass rates are descriptive; a repository with many failures "
            "is not by itself evidence of a reusable mechanism."
        ),
    }


def _task_transitions(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "heldout"):
        baseline_tasks = _stable_outcomes(baseline, split)
        final_tasks = _stable_outcomes(final, split)
        counts = {
            "both_pass": 0,
            "baseline_pass_final_fail": 0,
            "baseline_fail_final_pass": 0,
            "both_fail": 0,
            "mixed_or_unscoreable": 0,
        }
        for task_id in sorted(set(baseline_tasks) & set(final_tasks)):
            before = baseline_tasks[task_id]
            after = final_tasks[task_id]
            if before is None or after is None:
                counts["mixed_or_unscoreable"] += 1
            elif before and after:
                counts["both_pass"] += 1
            elif before and not after:
                counts["baseline_pass_final_fail"] += 1
            elif not before and after:
                counts["baseline_fail_final_pass"] += 1
            else:
                counts["both_fail"] += 1
        result[split] = counts
    return result


def _stable_outcomes(result: Mapping[str, Any], split: str) -> dict[str, bool | None]:
    outcomes: dict[str, list[bool | None]] = {}
    allowed = {"resolved": True, "passed": True, "unresolved": False, "failed": False,
               "agent_timeout": False, "agent_error": False}
    for repeat in (result.get("splits") or {}).get(split, []):
        for case in repeat.get("case_results", []):
            status = str(case.get("status"))
            outcomes.setdefault(str(case.get("task_id")), []).append(allowed.get(status))
    stable: dict[str, bool | None] = {}
    for task_id, values in outcomes.items():
        if any(value is None for value in values):
            stable[task_id] = None
        else:
            stable[task_id] = bool(values) and all(values)
    return stable


def _aggregate_error_counts(results: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    counts = {
        "infra_errors": 0,
        "evaluator_errors": 0,
        "agent_errors": 0,
        "agent_timeouts": 0,
        "not_run": 0,
        "retries": 0,
        "unscoreable_tasks": 0,
        "results": 0,
    }
    seen: set[str] = set()
    for result in results:
        if not result:
            continue
        result_id = str(result.get("result_id"))
        if result_id in seen:
            continue
        seen.add(result_id)
        counts["results"] += 1
        for split in (result.get("splits") or {}).values():
            for repeat in split:
                for case in repeat.get("case_results", []):
                    status = str(case.get("status"))
                    if status == "infra_error":
                        counts["infra_errors"] += 1
                    elif status == "evaluator_error":
                        counts["evaluator_errors"] += 1
                    elif status == "agent_error":
                        counts["agent_errors"] += 1
                    elif status == "agent_timeout":
                        counts["agent_timeouts"] += 1
                    elif status == "not_run":
                        counts["not_run"] += 1
                    if status in UNSCOREABLE_STATUSES:
                        counts["unscoreable_tasks"] += 1
                    attempts = (case.get("agent") or {}).get("attempts")
                    if isinstance(attempts, list) and len(attempts) > 1:
                        counts["retries"] += len(attempts) - 1
    return counts


def _wall_seconds(state: Mapping[str, Any]) -> float | None:
    from .runstate import utc_now
    from datetime import datetime

    started = state.get("started_at")
    stopped = state.get("stopped_at") or utc_now()
    if not started:
        return None
    try:
        return (
            datetime.fromisoformat(str(stopped).replace("Z", "+00:00"))
            - datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        ).total_seconds()
    except ValueError:
        return None


def _measurement_block(
    *,
    run_manifest: Mapping[str, Any],
    task_ids_payload: Mapping[str, Any],
    config: Any,
    final_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    splits = task_ids_payload.get("splits") or {}
    evaluator = ((final_result or {}).get("benchmark") or {})
    return {
        "tasks_configured": {split: len(ids) for split, ids in splits.items()},
        "task_ids_sha256": (final_result or {}).get("dataset", {}).get("task_ids_sha256"),
        "dataset_revision": (run_manifest.get("dataset") or {}).get("revision"),
        "dataset_manifest_sha256": (run_manifest.get("dataset") or {}).get("manifest_sha256"),
        "evaluator": evaluator.get("evaluator"),
        "evaluator_version": evaluator.get("evaluator_version"),
        "repeats": (run_manifest.get("eval") or {}).get("repeats"),
        "smoke_test": bool((run_manifest.get("eval") or {}).get("smoke_test")),
        "evaluator_kind": config.evaluator.kind,
        "network_limit": ((final_result or {}).get("config") or {}).get("network_limit"),
    }


def _scope_block(
    *,
    config: Any,
    task_ids_payload: Mapping[str, Any],
    final: Mapping[str, Any] | None,
) -> dict[str, Any]:
    counts = task_ids_payload.get("counts") or {}
    benchmark = task_ids_payload.get("benchmark")
    smoke = bool(config.eval.smoke_test) or config.evaluator.kind == "fixture"
    if benchmark == "swe_bench_verified":
        configured_full = counts.get("train") == 67 and counts.get("heldout") == 33
        official = config.evaluator.kind == "official"
        full_claim = configured_full and official and not smoke
        scope = {
            "configured_scope": (
                "swe_bench_verified_full_100_split" if configured_full else "swe_bench_verified_subset"
            ),
            "official_evaluator": official,
        }
    else:
        configured_full = counts.get("train") == 43 and counts.get("heldout") == 21
        full_claim = configured_full and not smoke
        scope = {
            "configured_scope": (
                "terminal_bench_2_clean64_full_split" if configured_full else "terminal_bench_2_subset"
            ),
            "official_evaluator": False,
        }
    scope.update(
        {
            "smoke_test": smoke,
            "full_benchmark_claim_allowed": bool(full_claim),
            "claim_note": (
                "Configured scope only. External prerequisites (Harbor eval project, container "
                "images, official SWE-bench evaluator/Docker, pinned dataset and model endpoint) "
                "are not verified by this repository; see not_executed."
                if full_claim
                else "This run does not cover the full fixed benchmark split with the official "
                "evaluator; treat the measured rates as scoped/auxiliary measurements."
            ),
        }
    )
    return scope


def _not_executed_notes(*, config: Any, task_ids_payload: Mapping[str, Any]) -> list[str]:
    notes = [
        "No claim of reproducing the paper's reported pass rates is made.",
        "Model credentials, endpoints, containers and GPU/deployment parameters are external "
        "configuration and are not verified by this repository.",
    ]
    if config.eval.smoke_test or config.evaluator.kind == "fixture":
        notes.append(
            "This run used fixture/smoke-test agents or evaluators; its rates are NOT benchmark "
            "measurements and must not be reported as SWE-bench/Terminal-Bench results."
        )
    if task_ids_payload.get("benchmark") == "terminal_bench_2":
        notes.append(
            "Terminal-Bench 2.0 execution depends on an external Harbor eval project, task data "
            "and container runtime; a full Clean64 run was not verified by this repository unless "
            "a real run recorded it."
        )
    else:
        notes.append(
            "SWE-bench evaluation depends on the pinned official dataset and the official "
            "`swebench` evaluator with Docker images; if evaluator.kind is not 'official' or the "
            "preflight failed, resolved labels are not official."
        )
    if config.constraints.outbound_bandwidth_mbps is not None:
        notes.append(
            "Outbound bandwidth limiting is recorded per result in config.network_limit; when "
            "`enforced=false` the run was not bandwidth limited."
        )
    return notes


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = [
        f"# Self-Harness Final Report: `{report.get('run_id')}`",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Benchmark: `{report.get('benchmark')}`",
        f"- Dataset revision: `{(report.get('fixed_parameters') or {}).get('dataset', {}).get('revision')}`",
        f"- Dataset manifest SHA-256: `{(report.get('fixed_parameters') or {}).get('dataset', {}).get('manifest_sha256')}`",
        f"- Experiment fingerprint: `{report.get('fixed_parameters', {}).get('experiment_fingerprint')}`",
        f"- Stop reason: `{(report.get('totals') or {}).get('stop_reason')}`",
        "",
        "## Task Manifest (actually evaluated)",
        "",
        f"- Train: `{len(((report.get('task_manifest') or {}).get('splits') or {}).get('train') or [])}` tasks",
        f"- Heldout: `{len(((report.get('task_manifest') or {}).get('splits') or {}).get('heldout') or [])}` tasks",
    ]
    for split, ids in ((report.get("task_manifest") or {}).get("splits") or {}).items():
        lines.append(f"- `{split}` ids: " + (", ".join(f"`{item}`" for item in ids) or "none"))
    lines.extend(
        [
            "",
            "## Fixed Parameters",
            "",
        f"- Model: `{(report.get('fixed_parameters') or {}).get('model', {}).get('model_id')}` "
        f"(provider `{(report.get('fixed_parameters') or {}).get('model', {}).get('provider')}`)",
        f"- Repeats: `{(report.get('fixed_parameters') or {}).get('eval', {}).get('repeats')}`",
        f"- Proposal width: `{(report.get('fixed_parameters') or {}).get('budget', {}).get('proposal_width')}`",
        f"- Max candidate evaluations: `{(report.get('fixed_parameters') or {}).get('budget', {}).get('max_candidate_evaluations')}`",
        "",
            "## Baseline vs Final (measured)",
            "",
            "| Split | Repeat | Baseline | Final |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    baseline_splits = (report.get("baseline") or {}).get("splits") or {}
    final_splits = (report.get("final") or {}).get("splits") or {}
    for split in sorted(set(baseline_splits) | set(final_splits)):
        baseline_repeats = {item["repeat"]: item for item in baseline_splits.get(split, {}).get("repeats", [])}
        final_repeats = {item["repeat"]: item for item in final_splits.get(split, {}).get("repeats", [])}
        for repeat_id in sorted(set(baseline_repeats) | set(final_repeats)):
            left = baseline_repeats.get(repeat_id, {})
            right = final_repeats.get(repeat_id, {})
            lines.append(
                f"| {split} | {repeat_id} | {left.get('passed')}/{left.get('total')} "
                f"({_rate(left.get('pass_rate'))}) | {right.get('passed')}/{right.get('total')} "
                f"({_rate(right.get('pass_rate'))}) |"
            )
    lines.extend(["", "### Split Means And Pooled Statistics", ""])
    for split in sorted(set(baseline_splits) | set(final_splits)):
        left = baseline_splits.get(split, {})
        right = final_splits.get(split, {})
        lines.append(
            f"- `{split}`: baseline mean_repeat_pass_rate="
            f"`{_rate(left.get('mean_repeat_pass_rate'))}`, pooled="
            f"`{left.get('pooled_passed')}/{left.get('pooled_total')}`; final mean_repeat_pass_rate="
            f"`{_rate(right.get('mean_repeat_pass_rate'))}`, pooled="
            f"`{right.get('pooled_passed')}/{right.get('pooled_total')}`"
        )
    lines.extend(["", "## Candidates", ""])
    candidates = report.get("candidates") or []
    if not candidates:
        lines.append("- No candidates were evaluated.")
    for candidate in candidates:
        lines.append(
            f"- `{candidate.get('version_id')}` ({candidate.get('kind')}, parent="
            f"{candidate.get('parent_version_ids')}): train delta="
            f"`{_delta((candidate.get('deltas') or {}).get('train'))}`, heldout delta="
            f"`{_delta((candidate.get('deltas') or {}).get('heldout'))}`, decision="
            f"`{candidate.get('acceptance_decision')}`"
            + (f", reason: {candidate.get('acceptance_reason')}" if candidate.get("acceptance_reason") else "")
        )
    lines.extend(["", "## Repository / Domain Results (final)", ""])
    repository = ((report.get("repository_domain") or {}).get("final") or {})
    for repo, entry in (repository.get("by_repository") or {}).items():
        lines.append(f"- `{repo}`: {entry['passed']}/{entry['total']} ({_rate(entry.get('pass_rate'))})")
    if not repository:
        lines.append("- Not available.")
    lines.extend(["", "## Task Transitions (baseline -> final)", ""])
    for split, counts in (report.get("task_transitions") or {}).items():
        lines.append(f"- `{split}`: {json.dumps(counts, sort_keys=True)}")
    lines.extend(["", "## Versions And Artifacts", ""])
    for version in report.get("versions") or []:
        lines.append(
            f"- `{version.get('version_id')}` kind=`{version.get('kind')}` status="
            f"`{version.get('status')}` snapshot=`{version.get('harness_snapshot_path')}` "
            f"sha256=`{version.get('harness_snapshot_sha256')}` result=`{version.get('eval_result_path')}` "
            f"acceptance=`{version.get('acceptance_path')}`"
        )
    totals = report.get("totals") or {}
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Rounds completed: `{totals.get('rounds_completed')}`",
            f"- Proposals created: `{totals.get('proposals_created')}`",
            f"- Candidate evaluations: `{totals.get('candidate_evaluations')}`",
            f"- Versions: `{totals.get('versions')}`",
            f"- Wall time (s): `{totals.get('wall_time_s')}`",
            f"- Stop reason: `{totals.get('stop_reason')}`",
            "",
            "## Errors",
            "",
            f"```json\n{json.dumps(report.get('errors'), indent=2, sort_keys=True)}\n```",
            "",
            "## Scope / Not Executed",
            "",
            f"```json\n{json.dumps(report.get('scope'), indent=2, sort_keys=True)}\n```",
            "",
        ]
    )
    lines.extend([f"- {note}" for note in report.get("not_executed") or []])
    lines.extend(
        [
            "",
            "## Claims",
            "",
            f"- Scores are computed from recorded results: `{(report.get('claims') or {}).get('scores_are_measured_from_results')}`",
            f"- Reproduces paper results: `{(report.get('claims') or {}).get('reproduces_paper_results')}`",
            f"- {(report.get('claims') or {}).get('note')}",
            "",
        ]
    )
    return "\n".join(lines)


def _rate(value: Any) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "n/a"


def _delta(value: Any) -> str:
    return f"{float(value):+.6f}" if isinstance(value, (int, float)) else "n/a"
