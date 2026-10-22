"""Harness version records and lineage for Self-Harness runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .runstate import (
    RunPaths,
    WorkflowError,
    read_json,
    utc_now,
)
from .snapshots import HarnessSnapshot, snapshot_relative_path

VERSION_FORMAT = "self_harness.version.v1"


def allocate_version_id(run_manifest: Mapping[str, Any], *, suffix: str) -> str:
    index = len(run_manifest.get("versions") or [])
    clean_suffix = "".join(char if char.isalnum() or char in "_.-" else "_" for char in suffix).strip("_")
    if not clean_suffix:
        raise WorkflowError("version suffix must be non-empty")
    version_id = f"v{index:03d}_{clean_suffix}"
    if any(item.get("version_id") == version_id for item in run_manifest.get("versions") or []):
        raise WorkflowError(f"version id already exists: {version_id}")
    return version_id


def create_version(
    *,
    paths: RunPaths,
    run_manifest: dict[str, Any],
    version_id: str,
    kind: str,
    snapshot: HarnessSnapshot,
    parent_version_ids: Sequence[str],
    fixed: Mapping[str, Any],
    proposal: Mapping[str, Any] | None = None,
    candidate_id: str | None = None,
    mechanism_family: str | None = None,
    merge_parent_candidate_ids: Sequence[str] = (),
) -> dict[str, Any]:
    directory = paths.version_dir(version_id)
    if (directory / "version.json").exists():
        raise WorkflowError(f"refusing to overwrite existing version record: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    lineage = _build_lineage(run_manifest, parent_version_ids) + [version_id]
    harness_block = snapshot.block(
        run_root=paths.root,
        version_id=version_id,
        parent_version_ids=list(parent_version_ids),
        kind=kind,
    )
    record: dict[str, Any] = {
        "format": VERSION_FORMAT,
        "version_id": version_id,
        "run_id": run_manifest["run_id"],
        "benchmark": run_manifest.get("benchmark"),
        "created_at": utc_now(),
        "kind": kind,
        "harness": harness_block,
        "harness_snapshot_path": snapshot_relative_path(snapshot.directory, paths.root),
        "harness_snapshot_sha256": harness_block["snapshot_sha256"],
        "parent_version_ids": [str(item) for item in parent_version_ids],
        "lineage": lineage,
        "proposal_id": proposal.get("proposal_id") if proposal else None,
        "candidate_id": candidate_id,
        "proposal_path": "proposal.json" if proposal else None,
        "mechanism_family": mechanism_family,
        "status": "materialized",
        "status_history": [{"status": "materialized", "at": utc_now(), "reason": None}],
        "eval": None,
        "acceptance": None,
        "fixed": dict(fixed),
    }
    if merge_parent_candidate_ids:
        record["merge_parent_candidate_ids"] = [str(item) for item in merge_parent_candidate_ids]
    if proposal:
        write_json(directory / "proposal.json", dict(proposal))
    write_json(paths.version_json(version_id), record)
    run_manifest.setdefault("versions", []).append(
        {
            "version_id": version_id,
            "kind": kind,
            "parent_version_ids": list(parent_version_ids),
            "status": "materialized",
            "harness_snapshot_path": record["harness_snapshot_path"],
            "harness_snapshot_sha256": record["harness_snapshot_sha256"],
            "proposal_id": record["proposal_id"],
            "candidate_id": candidate_id,
            "lineage": lineage,
        }
    )
    return record


def update_version(
    *,
    paths: RunPaths,
    run_manifest: dict[str, Any],
    version_id: str,
    status: str,
    reason: str | None = None,
    eval_payload: Mapping[str, Any] | None = None,
    acceptance_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = paths.version_json(version_id)
    record = read_json(path)
    if not isinstance(record, dict):
        raise WorkflowError(f"version record must be an object: {path}")
    record["status"] = status
    history = list(record.get("status_history") or [])
    history.append({"status": status, "at": utc_now(), "reason": reason})
    record["status_history"] = history
    if eval_payload is not None:
        record["eval"] = dict(eval_payload)
    if acceptance_payload is not None:
        record["acceptance"] = dict(acceptance_payload)
    write_json(path, record)
    for item in run_manifest.get("versions") or []:
        if item.get("version_id") == version_id:
            item["status"] = status
            if eval_payload is not None:
                item["eval_result_id"] = eval_payload.get("result_id")
                item["eval_result_path"] = eval_payload.get("result_path")
            if acceptance_payload is not None:
                item["acceptance_decision"] = acceptance_payload.get("decision")
                item["accepted"] = acceptance_payload.get("accepted")
            break
    return record


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_version(paths: RunPaths, version_id: str) -> dict[str, Any]:
    path = paths.version_json(version_id)
    if not path.exists():
        raise WorkflowError(f"version does not exist: {version_id}")
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("format") != VERSION_FORMAT:
        raise WorkflowError(f"invalid version record: {path}")
    return payload


def _build_lineage(run_manifest: Mapping[str, Any], parent_version_ids: Sequence[str]) -> list[str]:
    lineage: list[str] = []
    for raw_parent in parent_version_ids:
        parent_id = str(raw_parent)
        parent_lineage: list[str] = [parent_id]
        for item in run_manifest.get("versions") or []:
            if item.get("version_id") == parent_id:
                parent_lineage = [str(entry) for entry in item.get("lineage") or []] or [parent_id]
                break
        for entry in parent_lineage:
            if entry not in lineage:
                lineage.append(entry)
    return lineage
