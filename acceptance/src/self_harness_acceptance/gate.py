"""Non-regression Pareto acceptance gate for Self-Harness candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_harness_eval.protocol import (
    SCHEMA_VERSION,
    UNSCOREABLE_STATUSES,
    case_passed,
    recompute_result_metrics,
    utc_now,
    validate_result,
)

RULE_VERSION = "self_harness.acceptance.v1"
FORMAT = "self_harness.acceptance_gate.v1"
DEFAULT_SPLITS = ("train", "heldout")
DEFAULT_EXPECTED_REPEATS = 2
EPSILON = 1e-12

DECISION_ACCEPTED = "accepted"
DECISION_REJECTED = "rejected"
DECISION_INVALID = "invalid"
DECISION_INCONCLUSIVE = "inconclusive"

_REQUIRED_COMPARABILITY_KEYS = (
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


class AcceptanceError(ValueError):
    """Raised when the gate is invoked with unusable inputs."""


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": "pass" if self.ok else "fail", "detail": self.detail}


def run_acceptance_gate(
    *,
    baseline_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    baseline_result_path: Path | None = None,
    candidate_result_path: Path | None = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
    expected_repeats: int | None = None,
) -> dict[str, Any]:
    if expected_repeats is not None and expected_repeats < 1:
        raise AcceptanceError("expected_repeats must be >= 1")
    requested_splits = tuple(dict.fromkeys(splits))
    for split in requested_splits:
        if split not in ("train", "heldout"):
            raise AcceptanceError(f"unknown split {split!r}")

    checks: list[Check] = []
    structural_errors: list[str] = []
    for label, result, path in (
        ("baseline", baseline_result, baseline_result_path),
        ("candidate", candidate_result, candidate_result_path),
    ):
        try:
            validate_result(result)
            checks.append(Check(f"schema:{label}", True))
        except Exception as exc:  # noqa: BLE001 - invalid results are a gate decision
            structural_errors.append(f"{label} result is not valid per {SCHEMA_VERSION}: {exc}")
            checks.append(Check(f"schema:{label}", False, str(exc)))
    if structural_errors:
        return _decision_payload(
            baseline_result=baseline_result,
            candidate_result=candidate_result,
            baseline_result_path=baseline_result_path,
            candidate_result_path=candidate_result_path,
            splits=requested_splits,
            checks=checks,
            decision=DECISION_INVALID,
            reason="; ".join(structural_errors),
            split_blocks={},
            transitions={},
            deltas={},
        )

    compatibility = _compare_structure(
        baseline=baseline_result,
        candidate=candidate_result,
        splits=requested_splits,
        expected_repeats=expected_repeats,
        checks=checks,
    )
    completeness = _completeness(
        baseline=baseline_result,
        candidate=candidate_result,
        splits=requested_splits,
        checks=checks,
    )
    recomputed = {
        "baseline": recompute_result_metrics(baseline_result),
        "candidate": recompute_result_metrics(candidate_result),
    }
    metric_ok = _verify_recomputed_metrics(
        baseline=baseline_result,
        candidate=candidate_result,
        recomputed=recomputed,
        splits=requested_splits,
        checks=checks,
    )
    split_blocks = _split_blocks(
        baseline=baseline_result,
        candidate=candidate_result,
        recomputed=recomputed,
        splits=requested_splits,
    )
    transitions = _task_transitions(
        baseline=baseline_result,
        candidate=candidate_result,
        splits=requested_splits,
    )
    deltas = {split: block["delta"] for split, block in split_blocks.items()}

    if not compatibility["ok"]:
        decision = DECISION_INVALID
        reason = "results are not comparable: " + "; ".join(compatibility["failures"])
    elif not completeness["ok"]:
        decision = DECISION_INCONCLUSIVE
        reason = (
            "results are not scoreable/complete; candidate admission is inconclusive: "
            + "; ".join(completeness["failures"])
        )
    elif not metric_ok:
        decision = DECISION_INVALID
        reason = "stored metrics cannot be reproduced from case_results"
    else:
        accepted, reject_reason = _pareto_decision(deltas=deltas, splits=requested_splits)
        if accepted:
            decision = DECISION_ACCEPTED
            reason = _accept_reason(deltas=deltas, splits=requested_splits)
        else:
            decision = DECISION_REJECTED
            reason = reject_reason

    return _decision_payload(
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        baseline_result_path=baseline_result_path,
        candidate_result_path=candidate_result_path,
        splits=requested_splits,
        checks=checks,
        decision=decision,
        reason=reason,
        split_blocks=split_blocks,
        transitions=transitions,
        deltas=deltas,
        comparability=compatibility,
        completeness=completeness,
    )


def _compare_structure(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    splits: Sequence[str],
    expected_repeats: int | None,
    checks: list[Check],
) -> dict[str, Any]:
    failures: list[str] = []

    def require(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, ok, detail))
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)

    require(
        "run_id",
        baseline.get("run_id") == candidate.get("run_id"),
        f"{baseline.get('run_id')!r} != {candidate.get('run_id')!r}",
    )
    baseline_benchmark = baseline.get("benchmark") or {}
    candidate_benchmark = candidate.get("benchmark") or {}
    for key in ("id", "version", "evaluator", "evaluator_version"):
        require(
            f"benchmark.{key}",
            baseline_benchmark.get(key) == candidate_benchmark.get(key),
            f"{baseline_benchmark.get(key)!r} != {candidate_benchmark.get(key)!r}",
        )
    require("model", baseline.get("model") == candidate.get("model"))
    baseline_dataset = baseline.get("dataset") or {}
    candidate_dataset = candidate.get("dataset") or {}
    for key in ("name", "revision", "manifest_sha256", "task_ids_sha256"):
        require(
            f"dataset.{key}",
            baseline_dataset.get(key) == candidate_dataset.get(key),
            f"{baseline_dataset.get(key)!r} != {candidate_dataset.get(key)!r}",
        )
    baseline_config = baseline.get("config") or {}
    candidate_config = candidate.get("config") or {}
    baseline_comparability = baseline_config.get("comparability") or {}
    candidate_comparability = candidate_config.get("comparability") or {}
    for key in _REQUIRED_COMPARABILITY_KEYS:
        require(
            f"comparability.{key}",
            baseline_comparability.get(key) == candidate_comparability.get(key),
            f"{baseline_comparability.get(key)!r} != {candidate_comparability.get(key)!r}",
        )
    require(
        "config.repeats",
        baseline_config.get("repeats") == candidate_config.get("repeats"),
        f"{baseline_config.get('repeats')!r} != {candidate_config.get('repeats')!r}",
    )
    require(
        "config.task_timeout_s",
        baseline_config.get("task_timeout_s") == candidate_config.get("task_timeout_s"),
    )
    require(
        "config.tool_timeout_s",
        baseline_config.get("tool_timeout_s") == candidate_config.get("tool_timeout_s"),
    )

    baseline_splits = baseline.get("splits") or {}
    candidate_splits = candidate.get("splits") or {}
    require(
        "split sets",
        set(baseline_splits) == set(candidate_splits),
        f"{sorted(baseline_splits)} != {sorted(candidate_splits)}",
    )
    require(
        "required splits",
        all(split in baseline_splits and split in candidate_splits for split in splits),
        f"required={list(splits)} baseline={sorted(baseline_splits)} candidate={sorted(candidate_splits)}",
    )

    for split in splits:
        baseline_repeats = baseline_splits.get(split)
        candidate_repeats = candidate_splits.get(split)
        if not isinstance(baseline_repeats, list) or not isinstance(candidate_repeats, list):
            require(f"repeats:{split}", False, "missing split repeats")
            continue
        baseline_ids = [int(repeat.get("repeat", 0)) for repeat in baseline_repeats]
        candidate_ids = [int(repeat.get("repeat", 0)) for repeat in candidate_repeats]
        require(
            f"repeat ids:{split}",
            baseline_ids == candidate_ids,
            f"{baseline_ids} != {candidate_ids}",
        )
        if expected_repeats is not None:
            require(
                f"expected repeats:{split}",
                len(baseline_ids) == expected_repeats and len(candidate_ids) == expected_repeats,
                f"expected {expected_repeats}, got {len(baseline_ids)} / {len(candidate_ids)}",
            )
        baseline_by_id = {int(repeat.get("repeat", 0)): repeat for repeat in baseline_repeats}
        candidate_by_id = {int(repeat.get("repeat", 0)): repeat for repeat in candidate_repeats}
        for repeat_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
            baseline_repeat = baseline_by_id[repeat_id]
            candidate_repeat = candidate_by_id[repeat_id]
            baseline_task_ids = [str(case.get("task_id")) for case in baseline_repeat.get("case_results", [])]
            candidate_task_ids = [
                str(case.get("task_id")) for case in candidate_repeat.get("case_results", [])
            ]
            require(
                f"task ids:{split}/repeat-{repeat_id}",
                set(baseline_task_ids) == set(candidate_task_ids),
                "case task id sets differ",
            )
            require(
                f"total:{split}/repeat-{repeat_id}",
                int(baseline_repeat.get("total", -1)) == int(candidate_repeat.get("total", -2)),
                f"{baseline_repeat.get('total')} != {candidate_repeat.get('total')}",
            )

    return {"ok": not failures, "failures": failures}


def _completeness(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    splits: Sequence[str],
    checks: list[Check],
) -> dict[str, Any]:
    failures: list[str] = []
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result.get("status") != "complete":
            failures.append(f"{label}.status={result.get('status')!r}")
        for split in splits:
            for repeat in (result.get("splits") or {}).get(split, []):
                if repeat.get("status") != "complete":
                    failures.append(
                        f"{label}.{split}.repeat-{repeat.get('repeat')}={repeat.get('status')!r}"
                    )
                for case in repeat.get("case_results", []):
                    if str(case.get("status")) in UNSCOREABLE_STATUSES:
                        failures.append(
                            f"{label}.{split}.repeat-{repeat.get('repeat')}.{case.get('task_id')}"
                            f"={case.get('status')!r}"
                        )
    checks.append(
        Check(
            "completeness",
            not failures,
            "; ".join(failures[:10]) if failures else "all configured tasks were scored",
        )
    )
    return {"ok": not failures, "failures": failures}


def _verify_recomputed_metrics(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    splits: Sequence[str],
    checks: list[Check],
) -> bool:
    failures: list[str] = []
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        computed_splits = recomputed[label]["splits"]
        for split in splits:
            computed = computed_splits.get(split)
            if computed is None:
                failures.append(f"{label}.{split} missing")
                continue
            stored_summary = (result.get("summary") or {}).get(split) or {}
            for key in ("mean_repeat_pass_rate", "pooled_pass_rate"):
                if abs(float(stored_summary.get(key, -1.0)) - float(computed["summary"][key])) > 1e-9:
                    failures.append(f"{label}.{split}.{key}")
            stored_repeats = {
                int(repeat.get("repeat", 0)): repeat
                for repeat in (result.get("splits") or {}).get(split, [])
            }
            for repeat in computed["repeats"]:
                stored = stored_repeats.get(int(repeat["repeat"]))
                if stored is None:
                    failures.append(f"{label}.{split}.repeat-{repeat['repeat']} missing")
                    continue
                if int(stored.get("passed", -1)) != int(repeat["passed"]) or int(
                    stored.get("total", -1)
                ) != int(repeat["total"]):
                    failures.append(
                        f"{label}.{split}.repeat-{repeat['repeat']} passed/total"
                    )
                if abs(float(stored.get("pass_rate", -1.0)) - float(repeat["pass_rate"])) > 1e-9:
                    failures.append(f"{label}.{split}.repeat-{repeat['repeat']}.pass_rate")
    checks.append(Check("metric recomputation", not failures, "; ".join(failures[:10])))
    return not failures


def _split_blocks(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    splits: Sequence[str],
) -> dict[str, Any]:
    blocks: dict[str, Any] = {}
    for split in splits:
        baseline_block = _metric_block(recomputed["baseline"]["splits"][split])
        candidate_block = _metric_block(recomputed["candidate"]["splits"][split])
        blocks[split] = {
            "baseline": baseline_block,
            "candidate": candidate_block,
            "delta": candidate_block["mean_repeat_pass_rate"]
            - baseline_block["mean_repeat_pass_rate"],
        }
    return blocks


def _metric_block(computed_split: Mapping[str, Any]) -> dict[str, Any]:
    summary = computed_split["summary"]
    return {
        "repeats": [
            {
                "repeat": int(repeat["repeat"]),
                "passed": int(repeat["passed"]),
                "total": int(repeat["total"]),
                "pass_rate": float(repeat["pass_rate"]),
                "status": repeat["status"],
            }
            for repeat in computed_split["repeats"]
        ],
        "mean_repeat_pass_rate": float(summary["mean_repeat_pass_rate"]),
        "pooled_passed": int(summary["pooled_passed"]),
        "pooled_total": int(summary["pooled_total"]),
        "pooled_pass_rate": float(summary["pooled_pass_rate"]),
        "infrastructure_error_count": int(summary["infrastructure_error_count"]),
        "evaluator_error_count": int(summary["evaluator_error_count"]),
    }


def _task_transitions(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    splits: Sequence[str],
) -> dict[str, Any]:
    """Count per-task status transitions.

    A task counts as passing only when every repeat passed; a task counts as
    failing when every scoreable repeat failed; unscoreable repeats make the
    task unscoreable.  The ``per_repeat`` counts preserve raw case-level facts.
    """

    result: dict[str, Any] = {}
    totals = {
        "both_pass": 0,
        "baseline_pass_candidate_fail": 0,
        "baseline_fail_candidate_pass": 0,
        "both_fail": 0,
        "unscoreable": 0,
        "per_repeat": {
            "both_pass": 0,
            "baseline_pass_candidate_fail": 0,
            "baseline_fail_candidate_pass": 0,
            "both_fail": 0,
            "unscoreable": 0,
        },
    }
    for split in splits:
        baseline_tasks = _stable_task_outcomes(baseline, split)
        candidate_tasks = _stable_task_outcomes(candidate, split)
        counts = {
            "both_pass": 0,
            "baseline_pass_candidate_fail": 0,
            "baseline_fail_candidate_pass": 0,
            "both_fail": 0,
            "unscoreable": 0,
        }
        for task_id in sorted(set(baseline_tasks) & set(candidate_tasks)):
            baseline_passed = baseline_tasks[task_id]
            candidate_passed = candidate_tasks[task_id]
            if baseline_passed is None or candidate_passed is None:
                counts["unscoreable"] += 1
            elif baseline_passed and candidate_passed:
                counts["both_pass"] += 1
            elif baseline_passed and not candidate_passed:
                counts["baseline_pass_candidate_fail"] += 1
            elif not baseline_passed and candidate_passed:
                counts["baseline_fail_candidate_pass"] += 1
            else:
                counts["both_fail"] += 1
        repeat_counts = _per_repeat_transitions(baseline, candidate, split)
        counts["per_repeat"] = repeat_counts
        result[split] = counts
        for key in ("both_pass", "baseline_pass_candidate_fail", "baseline_fail_candidate_pass", "both_fail", "unscoreable"):
            totals[key] += counts[key]
            totals["per_repeat"][key] += repeat_counts[key]
    result["total"] = totals
    return result


def _stable_task_outcomes(result: Mapping[str, Any], split: str) -> dict[str, bool | None]:
    outcomes: dict[str, list[bool | None]] = {}
    for repeat in (result.get("splits") or {}).get(split, []):
        for case in repeat.get("case_results", []):
            outcomes.setdefault(str(case.get("task_id")), []).append(
                case_passed(str(case.get("status")))
            )
    stable: dict[str, bool | None] = {}
    for task_id, values in outcomes.items():
        if any(value is None for value in values):
            stable[task_id] = None
        elif values and all(value is True for value in values):
            stable[task_id] = True
        else:
            stable[task_id] = False
    return stable


def _per_repeat_transitions(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    split: str,
) -> dict[str, int]:
    counts = {
        "both_pass": 0,
        "baseline_pass_candidate_fail": 0,
        "baseline_fail_candidate_pass": 0,
        "both_fail": 0,
        "unscoreable": 0,
    }
    baseline_repeats = {
        int(repeat.get("repeat", 0)): repeat
        for repeat in (baseline.get("splits") or {}).get(split, [])
    }
    candidate_repeats = {
        int(repeat.get("repeat", 0)): repeat
        for repeat in (candidate.get("splits") or {}).get(split, [])
    }
    for repeat_id in sorted(set(baseline_repeats) & set(candidate_repeats)):
        baseline_cases = {
            str(case.get("task_id")): case_passed(str(case.get("status")))
            for case in baseline_repeats[repeat_id].get("case_results", [])
        }
        candidate_cases = {
            str(case.get("task_id")): case_passed(str(case.get("status")))
            for case in candidate_repeats[repeat_id].get("case_results", [])
        }
        for task_id in sorted(set(baseline_cases) & set(candidate_cases)):
            baseline_passed = baseline_cases[task_id]
            candidate_passed = candidate_cases[task_id]
            if baseline_passed is None or candidate_passed is None:
                counts["unscoreable"] += 1
            elif baseline_passed and candidate_passed:
                counts["both_pass"] += 1
            elif baseline_passed and not candidate_passed:
                counts["baseline_pass_candidate_fail"] += 1
            elif not baseline_passed and candidate_passed:
                counts["baseline_fail_candidate_pass"] += 1
            else:
                counts["both_fail"] += 1
    return counts


def _pareto_decision(*, deltas: Mapping[str, float], splits: Sequence[str]) -> tuple[bool, str]:
    missing = [split for split in splits if split not in deltas]
    if missing:
        return False, f"rejected: missing split deltas for {missing}"
    dropped = [split for split in splits if deltas[split] < -EPSILON]
    if dropped:
        return False, "rejected: regression on " + ", ".join(
            f"{split} (delta={deltas[split]:+.6f})" for split in dropped
        )
    improved = [split for split in splits if deltas[split] > EPSILON]
    if not improved:
        return False, "rejected: no split improved (deltas are indistinguishable from zero)"
    return True, ""


def _accept_reason(*, deltas: Mapping[str, float], splits: Sequence[str]) -> str:
    parts = [f"{split} delta={deltas[split]:+.6f}" for split in splits]
    return "accepted: no split regressed and at least one improved (" + ", ".join(parts) + ")"


def _decision_payload(
    *,
    baseline_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    baseline_result_path: Path | None,
    candidate_result_path: Path | None,
    splits: Sequence[str],
    checks: list[Check],
    decision: str,
    reason: str,
    split_blocks: Mapping[str, Any],
    transitions: Mapping[str, Any],
    deltas: Mapping[str, float],
    comparability: Mapping[str, Any] | None = None,
    completeness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "eval_schema_version": SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "decision": decision,
        "accepted": decision == DECISION_ACCEPTED,
        "reason": reason,
        "decided_at": utc_now(),
        "run_id": baseline_result.get("run_id"),
        "baseline_result_id": baseline_result.get("result_id"),
        "candidate_result_id": candidate_result.get("result_id"),
        "baseline_version_id": (baseline_result.get("harness") or {}).get("version_id"),
        "candidate_version_id": (candidate_result.get("harness") or {}).get("version_id"),
        "baseline_result_path": str(baseline_result_path) if baseline_result_path else None,
        "candidate_result_path": str(candidate_result_path) if candidate_result_path else None,
        "benchmark": (baseline_result.get("benchmark") or {}).get("id"),
        "splits": split_blocks,
        "deltas": dict(deltas),
        "task_transitions": transitions,
        "checks": [check.to_dict() for check in checks],
        "comparability": {
            "ok": bool((comparability or {}).get("ok")),
            "failures": list((comparability or {}).get("failures", [])),
            "baseline": {
                "model_config_sha256": (baseline_result.get("model") or {}).get("config_sha256"),
                "comparability": (baseline_result.get("config") or {}).get("comparability"),
                "dataset_manifest_sha256": (baseline_result.get("dataset") or {}).get(
                    "manifest_sha256"
                ),
                "task_ids_sha256": (baseline_result.get("dataset") or {}).get("task_ids_sha256"),
                "harness_snapshot_sha256": (baseline_result.get("harness") or {}).get(
                    "snapshot_sha256"
                ),
            },
            "candidate": {
                "model_config_sha256": (candidate_result.get("model") or {}).get("config_sha256"),
                "comparability": (candidate_result.get("config") or {}).get("comparability"),
                "dataset_manifest_sha256": (candidate_result.get("dataset") or {}).get(
                    "manifest_sha256"
                ),
                "task_ids_sha256": (candidate_result.get("dataset") or {}).get("task_ids_sha256"),
                "harness_snapshot_sha256": (candidate_result.get("harness") or {}).get(
                    "snapshot_sha256"
                ),
            },
        },
        "completeness": {
            "ok": bool((completeness or {}).get("ok")),
            "failures": list((completeness or {}).get("failures", [])),
        },
        "rule": {
            "metric": "mean_repeat_pass_rate",
            "comparison": "candidate minus baseline (parent) per split",
            "required_splits": list(splits),
            "accept_if": (
                "delta_train >= 0 and delta_heldout >= 0 and "
                "(delta_train > 0 or delta_heldout > 0) with all completeness and "
                "comparability checks passing"
            ),
            "epsilon": EPSILON,
            "invalid_decision": "non-comparable results are recorded as invalid, not rejected",
            "inconclusive_decision": "unscoreable results are recorded as inconclusive",
        },
    }


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AcceptanceError(f"{path} must contain a JSON object")
    return payload


def write_acceptance(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise AcceptanceError(
            f"refusing to overwrite immutable acceptance decision: {target}"
        ) from exc
    return target


def is_finite_delta(value: float) -> bool:
    return math.isfinite(value)
