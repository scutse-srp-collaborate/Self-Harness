"""Train-only diagnosis evidence loaders for unified Self-Harness results.

Held-out task-level evidence must never reach diagnosis or the proposer.  This
module reads the *training* split of a unified result, converts failed cases
into a common evidence structure for both benchmarks, clusters failures across
tasks, and renders a diagnosis brief that can be handed to the proposer.

The loader never invents verifier conclusions: when a required artifact is
missing it marks the evidence as ``missing``/``partial`` with a concrete reason.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DIAGNOSIS_SPLIT = "train"
UNSCOREABLE = {"infra_error", "evaluator_error", "not_run"}
SCOREABLE_FAILURES = {"unresolved", "failed", "agent_timeout", "agent_error"}
SUCCESS_STATUSES = {"resolved", "passed"}
EVIDENCE_FORMAT = "self_harness.diagnosis_evidence.v1"

_MECHANISM_PATTERNS = (
    ("missing_dependency", re.compile(r"modulenotfounderror|no module named|importerror", re.I)),
    ("missing_artifact", re.compile(r"filenotfounderror|no such file|does not exist|not found", re.I)),
    ("patch_apply_failure", re.compile(r"patch (?:does not apply|failed)|git apply|conflict", re.I)),
    ("timeout", re.compile(r"timed out|timeout", re.I)),
    ("assertion_failure", re.compile(r"assertionerror|assert|expected .* got", re.I)),
    ("syntax_or_import_error", re.compile(r"syntaxerror|typeerror|valueerror|attributeerror", re.I)),
    ("test_failure", re.compile(r"fail_to_pass|tests? failed|failing test", re.I)),
    ("command_failure", re.compile(r"returncode|exit code|command failed", re.I)),
)


class HeldoutLeakError(RuntimeError):
    """Raised when held-out task-level evidence would reach diagnosis/proposer."""


class DiagnosisEvidenceError(ValueError):
    """Raised when the diagnosis input cannot be interpreted safely."""


@dataclass
class CaseEvidence:
    task_id: str
    split: str
    repository: str | None
    domain: str | None
    repeat: int
    status: str
    failure_stage: str
    evaluator_failure_summary: str | None
    trace_path: str | None
    final_status: str
    agent_status: str
    tool_calls: int | None
    evidence_status: str = "available"
    evidence_missing_reason: str | None = None
    evidence_excerpts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "repeat": self.repeat,
            "repository": self.repository,
            "domain": self.domain,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "evaluator_failure_summary": self.evaluator_failure_summary,
            "trace_path": self.trace_path,
            "final_status": self.final_status,
            "agent_status": self.agent_status,
            "tool_calls": self.tool_calls,
            "evidence_status": self.evidence_status,
            "evidence_missing_reason": self.evidence_missing_reason,
            "evidence_excerpts": list(self.evidence_excerpts),
        }


def load_train_evidence(
    *,
    result: Mapping[str, Any],
    result_path: Path | None,
    run_root: Path | None,
    heldout_task_ids: Sequence[str],
    train_task_ids: Sequence[str] | None = None,
    max_excerpt_chars: int = 600,
) -> dict[str, Any]:
    """Extract train-only evidence from a unified evaluation result."""

    if result.get("schema_version") != "self_harness.eval_result.v1":
        raise DiagnosisEvidenceError(
            "diagnosis requires a self_harness.eval_result.v1 result; got "
            f"{result.get('schema_version')!r}"
        )
    train_split = (result.get("splits") or {}).get(DIAGNOSIS_SPLIT)
    if not isinstance(train_split, list) or not train_split:
        raise DiagnosisEvidenceError("result has no train split; diagnosis requires split=train")

    heldout = {str(item) for item in heldout_task_ids}
    allowed_train = {str(item) for item in train_task_ids} if train_task_ids else None
    leaks: list[str] = []
    case_evidence: list[CaseEvidence] = []
    passing_train: list[str] = []
    unscoreable: list[CaseEvidence] = []
    seen_tasks: set[str] = set()

    for repeat in train_split:
        for case in repeat.get("case_results", []):
            task_id = str(case.get("task_id"))
            if task_id in heldout:
                leaks.append(task_id)
                continue
            if allowed_train is not None and task_id not in allowed_train:
                leaks.append(task_id)
                continue
            seen_tasks.add(task_id)
            status = str(case.get("status"))
            if status in SUCCESS_STATUSES:
                passing_train.append(task_id)
                continue
            if task_id in {item.task_id for item in case_evidence}:
                continue
            evidence = _case_evidence(
                case=case,
                run_root=run_root,
                max_excerpt_chars=max_excerpt_chars,
                benchmark=str((result.get("benchmark") or {}).get("id") or ""),
            )
            if evidence.evidence_status == "missing":
                unscoreable.append(evidence)
                continue
            if status in UNSCOREABLE:
                unscoreable.append(evidence)
            else:
                case_evidence.append(evidence)
    if leaks:
        raise HeldoutLeakError(
            "diagnosis input contains task ids outside the train split: "
            + ", ".join(sorted(set(leaks))[:10])
        )

    clusters = cluster_evidence(case_evidence)
    repo_domain = repo_domain_distribution(case_evidence)
    return {
        "format": EVIDENCE_FORMAT,
        "split": DIAGNOSIS_SPLIT,
        "run_id": result.get("run_id"),
        "result_id": result.get("result_id"),
        "result_path": str(result_path) if result_path else None,
        "benchmark": (result.get("benchmark") or {}).get("id"),
        "case_evidence": [item.to_dict() for item in case_evidence],
        "unscoreable_cases": [item.to_dict() for item in unscoreable],
        "passing_train_task_ids": sorted(set(passing_train)),
        "clusters": clusters,
        "repo_domain_failure_distribution": repo_domain,
        "isolation": {
            "train_only": True,
            "heldout_task_ids_validated": len(heldout),
            "train_task_ids_validated": len(seen_tasks),
            "leaks": [],
        },
        "evidence_gaps": [item.to_dict() for item in unscoreable if item.evidence_status == "missing"],
    }


def _case_evidence(
    *,
    case: Mapping[str, Any],
    run_root: Path | None,
    max_excerpt_chars: int,
    benchmark: str,
) -> CaseEvidence:
    status = str(case.get("status"))
    evaluator = case.get("evaluator") or {}
    agent = case.get("agent") or {}
    summary = evaluator.get("failure_summary")
    excerpts: list[str] = []
    if summary:
        excerpts.append(str(summary)[:max_excerpt_chars])
    artifacts = case.get("artifacts") or {}
    log_path = evaluator.get("log_path")
    log_missing_reason = None
    for candidate in (log_path, artifacts.get("agent_stderr"), artifacts.get("agent_stdout")):
        if not candidate:
            continue
        text = _read_text_relative(candidate, run_root)
        if text:
            excerpts.append(_tail(text, max_excerpt_chars))
        elif log_path == candidate:
            log_missing_reason = f"evaluator log is missing or unreadable: {candidate}"
    benchmark_evidence = _benchmark_specific_evidence(
        case=case, run_root=run_root, benchmark=benchmark
    )
    if benchmark_evidence.get("excerpts"):
        excerpts.extend(benchmark_evidence["excerpts"])
    if not summary and benchmark_evidence.get("failure_summary"):
        summary = benchmark_evidence["failure_summary"]
    trace_path = agent.get("trace_path")
    trace_missing = bool(trace_path) and not _relative_exists(trace_path, run_root)
    evidence_status = "available"
    missing_reason = None
    if status in UNSCOREABLE:
        evidence_status = "missing" if not excerpts else "partial"
        missing_reason = (
            log_missing_reason
            or "case result is unscoreable; no verifier conclusion is available"
        )
    elif not excerpts:
        evidence_status = "partial"
        missing_reason = "no evaluator failure summary or readable log was recorded"
    elif trace_missing:
        evidence_status = "partial"
        missing_reason = f"agent trace is missing: {trace_path}"
    return CaseEvidence(
        task_id=str(case.get("task_id")),
        split=str(case.get("split")),
        repository=case.get("repository"),
        domain=case.get("domain"),
        repeat=int(case.get("repeat", 0)),
        status=status,
        failure_stage=_failure_stage(status),
        evaluator_failure_summary=str(summary)[:max_excerpt_chars] if summary else None,
        trace_path=str(trace_path) if trace_path else None,
        final_status=status,
        agent_status=str(agent.get("status")),
        tool_calls=agent.get("tool_calls"),
        evidence_status=evidence_status,
        evidence_missing_reason=missing_reason,
        evidence_excerpts=excerpts[:4],
    )


def load_swebench_case_evidence(
    *,
    case: Mapping[str, Any],
    run_root: Path | None,
) -> dict[str, Any]:
    """Load SWE-bench evaluator/test-failure evidence for one case.

    Reads the evaluator stdout log (which contains the official harness test
    summary when available) and reports whether the evidence was present.  It
    never reads gold patches or hidden test definitions.
    """

    evaluator = case.get("evaluator") or {}
    excerpts: list[str] = []
    failure_summary = evaluator.get("failure_summary")
    if failure_summary:
        excerpts.append(str(failure_summary))
    log_text = _read_text_relative(evaluator.get("log_path"), run_root)
    if log_text:
        excerpts.append(_tail(log_text, 1200))
    trace_path = (case.get("agent") or {}).get("trace_path")
    trace_text = _read_text_relative(trace_path, run_root)
    if trace_text:
        excerpts.append(_tail(trace_text, 800))
    resolved = evaluator.get("resolved")
    available = bool(excerpts)
    return {
        "benchmark": "swe_bench_verified",
        "case_status": case.get("status"),
        "patch_path": (case.get("patch") or {}).get("path"),
        "resolved": resolved,
        "failure_summary": failure_summary,
        "trace_path": trace_path,
        "evidence_status": "available" if available else "missing",
        "excerpts": excerpts,
    }


def load_tb2_case_evidence(
    *,
    case: Mapping[str, Any],
    run_root: Path | None,
) -> dict[str, Any]:
    """Load Terminal-Bench verifier evidence for one case.

    When the case directory still contains Harbor verifier artifacts
    (``result.json`` + ``verifier/``), reuse the existing TB2 loader; otherwise
    fall back to the evaluator summary/log stored in the unified case result.
    """

    artifacts = case.get("artifacts") or {}
    case_dir = artifacts.get("case_dir")
    resolved_case_dir = _resolve_path(case_dir, run_root)
    excerpts: list[str] = []
    failure_summary = (case.get("evaluator") or {}).get("failure_summary")
    used_loader = False
    if resolved_case_dir is not None:
        try:
            from .tb2 import load_tb2_verifier_evidence

            tb2_evidence = load_tb2_verifier_evidence(job_dir=resolved_case_dir)
        except Exception:
            tb2_evidence = {}
        if tb2_evidence:
            used_loader = True
            terminal = tb2_evidence.get("terminal_summary")
            if terminal:
                excerpts.append(str(terminal))
            if not failure_summary and terminal:
                failure_summary = str(terminal)
    log_text = _read_text_relative((case.get("evaluator") or {}).get("log_path"), run_root)
    if log_text:
        excerpts.append(_tail(log_text, 1000))
    if failure_summary and not excerpts:
        excerpts.append(str(failure_summary))
    return {
        "benchmark": "terminal_bench_2",
        "case_status": case.get("status"),
        "verifier_loader_used": used_loader,
        "failure_summary": failure_summary,
        "evidence_status": "available" if excerpts else "missing",
        "excerpts": excerpts,
    }


def _benchmark_specific_evidence(
    *,
    case: Mapping[str, Any],
    run_root: Path | None,
    benchmark: str,
) -> dict[str, Any]:
    status = str(case.get("status"))
    if benchmark == "swe_bench_verified" or status in {"resolved", "unresolved"}:
        return load_swebench_case_evidence(case=case, run_root=run_root)
    if benchmark == "terminal_bench_2" or status in {"passed", "failed"}:
        return load_tb2_case_evidence(case=case, run_root=run_root)
    return {}


def _resolve_path(value: Any, run_root: Path | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        if run_root is None:
            return None
        path = Path(run_root) / value
    return path


def _failure_stage(status: str) -> str:
    return {
        "unresolved": "evaluator_unresolved",
        "failed": "verifier_failed",
        "agent_timeout": "agent_timeout",
        "agent_error": "agent_error",
        "infra_error": "infrastructure",
        "evaluator_error": "evaluator",
        "not_run": "not_run",
    }.get(status, status)


def _mechanism_bucket(summary: str | None) -> str:
    text = summary or ""
    for name, pattern in _MECHANISM_PATTERNS:
        if pattern.search(text):
            return name
    return "other"


def cluster_evidence(evidence: Sequence[CaseEvidence]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[CaseEvidence]] = {}
    for item in evidence:
        key = (item.failure_stage, _mechanism_bucket(item.evaluator_failure_summary))
        grouped.setdefault(key, []).append(item)
    clusters: list[dict[str, Any]] = []
    for (failure_stage, mechanism), items in grouped.items():
        clusters.append(
            {
                "cluster_id": f"{failure_stage}__{mechanism}",
                "failure_stage": failure_stage,
                "mechanism_bucket": mechanism,
                "case_count": len(items),
                "task_ids": sorted({item.task_id for item in items}),
                "repositories": sorted({item.repository for item in items if item.repository}),
                "domains": sorted({item.domain for item in items if item.domain}),
                "representative_evidence": [
                    {
                        "task_id": item.task_id,
                        "status": item.status,
                        "failure_stage": item.failure_stage,
                        "trace_path": item.trace_path,
                        "evaluator_failure_summary": item.evaluator_failure_summary,
                        "excerpts": item.evidence_excerpts[:3],
                    }
                    for item in items[:5]
                ],
                "provenance_task_ids": sorted({item.task_id for item in items}),
            }
        )
    clusters.sort(key=lambda item: (-item["case_count"], item["cluster_id"]))
    return clusters


def repo_domain_distribution(evidence: Sequence[CaseEvidence]) -> dict[str, Any]:
    repos: dict[str, int] = {}
    domains: dict[str, int] = {}
    for item in evidence:
        repo = item.repository or "unknown"
        domain = item.domain or "unknown"
        repos[repo] = repos.get(repo, 0) + 1
        domains[domain] = domains.get(domain, 0) + 1
    return {
        "by_repository": dict(sorted(repos.items(), key=lambda pair: (-pair[1], pair[0]))),
        "by_domain": dict(sorted(domains.items(), key=lambda pair: (-pair[1], pair[0]))),
        "caveat": (
            "A repository/domain with many failures is not by itself evidence of a reusable "
            "mechanism; inspect the cross-task cluster signatures and representative evidence."
        ),
    }


def render_diagnosis_brief(
    bundle: Mapping[str, Any],
    *,
    heldout_task_ids: Sequence[str],
    max_tasks_per_cluster: int = 5,
) -> str:
    """Render a train-only markdown brief for the proposer."""

    clusters = bundle.get("clusters") or []
    passing = list(bundle.get("passing_train_task_ids") or [])
    lines: list[str] = [
        "# Self-Harness Train Diagnosis Brief",
        "",
        "This brief is built exclusively from `split=train` evaluation evidence.",
        "Held-out task ids, issues, traces, patches, test failures and per-task held-out",
        "metrics are intentionally excluded from diagnosis and proposer context.",
        "",
        "## Run Provenance",
        "",
        f"- run_id: `{bundle.get('run_id')}`",
        f"- result_id: `{bundle.get('result_id')}`",
        f"- benchmark: `{bundle.get('benchmark')}`",
        f"- diagnosis split: `{bundle.get('split')}`",
        f"- failed train cases analyzed: `{len(bundle.get('case_evidence') or [])}`",
        f"- unscoreable train cases: `{len(bundle.get('unscoreable_cases') or [])}`",
        "",
        "## Train Passing Tasks To Preserve",
        "",
    ]
    lines.extend([f"- `{task_id}`" for task_id in passing] or ["- None recorded"])
    lines.extend(
        [
            "",
            "## Cross-Task Failure Clusters",
            "",
            "Cluster signatures are `failure_stage / mechanism_bucket`; mechanism buckets are",
            "heuristic and must be corroborated by the representative evidence below.",
            "",
        ]
    )
    if not clusters:
        lines.append("- No scoreable train failure clusters were found.")
        lines.append("")
    for index, cluster in enumerate(clusters, start=1):
        lines.extend(
            [
                f"### Cluster {index}: `{cluster['failure_stage']} / {cluster['mechanism_bucket']}`",
                "",
                f"- Cases: `{cluster['case_count']}`",
                f"- Train task provenance: {', '.join(f'`{item}`' for item in cluster['provenance_task_ids'])}",
                f"- Repositories: {', '.join(f'`{item}`' for item in cluster['repositories']) or 'unknown'}",
                f"- Domains: {', '.join(f'`{item}`' for item in cluster['domains']) or 'unknown'}",
                "- Representative evidence:",
            ]
        )
        for evidence in cluster["representative_evidence"][:max_tasks_per_cluster]:
            lines.append(
                f"  - `{evidence['task_id']}` status=`{evidence['status']}` "
                f"trace=`{evidence['trace_path']}`"
            )
            if evidence.get("evaluator_failure_summary"):
                lines.append(
                    f"    - Evaluator: {_one_line(evidence['evaluator_failure_summary'])}"
                )
            for excerpt in evidence.get("excerpts") or []:
                lines.append(f"    - Evidence: {_one_line(excerpt)}")
        lines.extend(
            [
                "- Proposer focus: select a bounded, reusable harness mechanism only if this",
                "  signature is corroborated across the listed train tasks; otherwise prefer no-op.",
                "",
            ]
        )
    distribution = bundle.get("repo_domain_failure_distribution") or {}
    lines.extend(["## Failure Distribution (train only)", ""])
    by_repo = distribution.get("by_repository") or {}
    by_domain = distribution.get("by_domain") or {}
    lines.append("- By repository: " + (", ".join(f"`{k}`={v}" for k, v in by_repo.items()) or "none"))
    lines.append("- By domain: " + (", ".join(f"`{k}`={v}" for k, v in by_domain.items()) or "none"))
    lines.extend(["", f"- Caveat: {distribution.get('caveat', '')}", ""])
    gaps = bundle.get("evidence_gaps") or []
    lines.extend(["## Evidence Gaps", ""])
    if gaps:
        for gap in gaps:
            lines.append(
                f"- `{gap.get('task_id')}` ({gap.get('status')}): {gap.get('evidence_missing_reason')}"
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Held-out Isolation Statement",
            "",
            f"- Held-out task ids validated against the input: `{bundle.get('isolation', {}).get('heldout_task_ids_validated')}`",
            "- Held-out task-level evidence is not included in this brief.",
            "",
        ]
    )
    text = "\n".join(lines)
    assert_no_heldout_leak(text, heldout_task_ids)
    return text


def write_diagnosis_brief(
    *,
    result: Mapping[str, Any],
    result_path: Path,
    output_path: Path,
    evidence_path: Path,
    heldout_task_ids: Sequence[str],
    train_task_ids: Sequence[str] | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    bundle = load_train_evidence(
        result=result,
        result_path=result_path,
        run_root=run_root,
        heldout_task_ids=heldout_task_ids,
        train_task_ids=train_task_ids,
    )
    brief = render_diagnosis_brief(bundle, heldout_task_ids=heldout_task_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief, encoding="utf-8")
    evidence_payload = dict(bundle)
    evidence_payload["brief_path"] = str(output_path)
    evidence_payload["brief_sha256"] = hashlib.sha256(brief.encode("utf-8")).hexdigest()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence_payload


# ---------------------------------------------------------------------------
# Isolation helpers
# ---------------------------------------------------------------------------


def assert_no_heldout_leak(text: str, heldout_task_ids: Iterable[str]) -> None:
    if not isinstance(text, str):
        raise HeldoutLeakError("diagnosis text must be a string")
    leaked = sorted({task_id for task_id in heldout_task_ids if task_id and task_id in text})
    if leaked:
        raise HeldoutLeakError(
            "held-out task-level evidence leaked into diagnosis/proposer text: "
            + ", ".join(leaked[:10])
        )


def assert_train_only_task_ids(
    task_ids: Iterable[str],
    *,
    train_task_ids: Iterable[str],
    heldout_task_ids: Iterable[str],
) -> None:
    heldout = set(heldout_task_ids)
    train = set(train_task_ids)
    seen = {str(item) for item in task_ids}
    leaked = sorted(seen & heldout)
    if leaked:
        raise HeldoutLeakError(f"held-out task ids used in train-only context: {leaked[:10]}")
    unknown = sorted(seen - train)
    if unknown:
        raise HeldoutLeakError(
            f"task ids are not part of the train manifest: {unknown[:10]}"
        )


def _read_text_relative(relative: Any, run_root: Path | None) -> str:
    if not isinstance(relative, str) or not relative:
        return ""
    path = Path(relative)
    if not path.is_absolute() and run_root is not None:
        path = Path(run_root) / relative
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _relative_exists(relative: Any, run_root: Path | None) -> bool:
    if not isinstance(relative, str) or not relative:
        return False
    path = Path(relative)
    if not path.is_absolute() and run_root is not None:
        path = Path(run_root) / relative
    return path.exists()


def _tail(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else "..." + text[-max_chars:]


def _one_line(value: str, *, limit: int = 400) -> str:
    compact = " ".join(str(value).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."
