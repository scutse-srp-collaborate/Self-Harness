"""Append-only JSONL event log for Self-Harness runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .runstate import RunPaths, utc_now

REQUIRED_EVENT_TYPES = (
    "run_start",
    "baseline_eval_complete",
    "diagnosis_created",
    "proposal_created",
    "candidate_materialized",
    "candidate_eval_started",
    "candidate_eval_complete",
    "candidate_accepted",
    "candidate_rejected",
    "merge_created",
    "merge_eval_complete",
    "branch_promoted",
    "run_stopped",
)

OPTIONAL_EVENT_TYPES = (
    "proposer_failed",
    "materialize_failed",
    "diagnosis_failed",
    "candidate_skipped_budget",
    "merge_failed",
    "run_resumed",
    "eval_reused",
    "report_written",
    "run_finalized",
)

ALL_EVENT_TYPES = REQUIRED_EVENT_TYPES + OPTIONAL_EVENT_TYPES

_HELDOUT_KEYS = {"heldout", "held_out", "heldout_failures", "heldout_case_results"}


def append_event(
    paths: RunPaths,
    event_type: str,
    *,
    version_id: str | None = None,
    candidate_id: str | None = None,
    proposal_id: str | None = None,
    artifacts: Mapping[str, str | None] | None = None,
    status: str | None = None,
    reason: str | None = None,
    run_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if event_type not in ALL_EVENT_TYPES:
        raise ValueError(f"unknown Self-Harness event type: {event_type!r}")
    event: dict[str, Any] = {
        "timestamp": utc_now(),
        "event_type": event_type,
        "run_id": run_id,
        "version_id": version_id,
        "candidate_id": candidate_id,
        "proposal_id": proposal_id,
        "status": status,
        "reason": reason,
        "artifacts": {str(key): value for key, value in sorted((artifacts or {}).items())},
    }
    if payload:
        event["payload"] = _sanitize_payload(payload)
    _reject_heldout_detail(event)
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    with paths.events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def _sanitize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop held-out case-level payload keys from proposer-facing events."""

    result: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).lower() in _HELDOUT_KEYS:
            continue
        result[str(key)] = value
    return result


def _reject_heldout_detail(event: Mapping[str, Any]) -> None:
    text = json.dumps(event, sort_keys=True)
    # Aggregate heldout metrics (deltas, pass rates) are allowed; raw case
    # payloads are not.  Detection here focuses on structural markers.
    for forbidden in ("heldout_case_results", "heldout_failures", "heldout_traces"):
        if forbidden in text:
            raise ValueError(
                f"refusing to write held-out case-level detail into event log: {forbidden}"
            )


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def event_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in read_events(path):
        name = str(event.get("event_type"))
        counts[name] = counts.get(name, 0) + 1
    return counts
