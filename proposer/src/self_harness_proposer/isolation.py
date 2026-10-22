"""Held-out information isolation guards for diagnosis and proposer inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class HeldoutLeakError(RuntimeError):
    """Raised when held-out task-level evidence would reach a proposer."""


def load_task_ids_file(path: Path | str) -> list[str]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, Mapping):
        for key in ("task_ids", "heldout_task_ids", "held_out_task_ids", "train_task_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
        splits = payload.get("splits")
        if isinstance(splits, Mapping):
            result: list[str] = []
            for value in splits.values():
                if isinstance(value, list):
                    result.extend(str(item) for item in value)
            return result
    raise HeldoutLeakError(f"cannot read task id list from {path}")


def find_heldout_leaks(text: str, heldout_task_ids: Iterable[str]) -> list[str]:
    if not isinstance(text, str):
        raise HeldoutLeakError("isolation guard expects text")
    return sorted({str(task_id) for task_id in heldout_task_ids if task_id and str(task_id) in text})


def assert_no_heldout_leak(text: str, heldout_task_ids: Iterable[str], *, label: str = "text") -> None:
    leaks = find_heldout_leaks(text, heldout_task_ids)
    if leaks:
        raise HeldoutLeakError(
            f"{label} contains held-out task-level evidence: " + ", ".join(leaks[:10])
        )


def audit_text(
    text: str,
    *,
    heldout_task_ids: Sequence[str],
    train_task_ids: Sequence[str] | None = None,
    label: str,
) -> dict[str, Any]:
    leaks = find_heldout_leaks(text, heldout_task_ids)
    if leaks:
        raise HeldoutLeakError(
            f"{label} contains held-out task-level evidence: " + ", ".join(leaks[:10])
        )
    mentioned_train = sorted(
        task_id
        for task_id in (train_task_ids or ())
        if task_id and task_id in text
    )
    return {
        "label": label,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "heldout_ids_checked": len(list(heldout_task_ids)),
        "leaks": [],
        "train_task_ids_mentioned": mentioned_train,
    }


def validate_proposal_metadata(
    *,
    proposal: Mapping[str, Any],
    train_task_ids: Iterable[str],
    heldout_task_ids: Iterable[str],
) -> None:
    """Ensure a proposal only cites train task ids as evidence."""

    train = {str(item) for item in train_task_ids}
    heldout = {str(item) for item in heldout_task_ids}
    metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), Mapping) else {}
    referenced: list[str] = []
    for key in ("expected_affected_cases", "protected_passing_cases"):
        value = proposal.get(key, metadata.get(key))
        if isinstance(value, list):
            referenced.extend(str(item) for item in value)
    leaked = sorted(set(referenced) & heldout)
    if leaked:
        raise HeldoutLeakError(
            "proposal references held-out task ids: " + ", ".join(leaked[:10])
        )
    unknown = sorted(set(referenced) - train)
    if unknown:
        raise HeldoutLeakError(
            "proposal references task ids that are not in the train split: "
            + ", ".join(unknown[:10])
        )
    summary = json.dumps(proposal, sort_keys=True)
    assert_no_heldout_leak(summary, heldout, label="proposal")
