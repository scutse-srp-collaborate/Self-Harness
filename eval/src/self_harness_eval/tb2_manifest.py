"""Terminal-Bench 2.0 Clean64 task manifest.

The repository's ``eval/README.md`` documents the fixed 43 train / 21 heldout
Clean64 split.  This module turns that fixed split into a machine-readable
manifest and validates it.  It does not discover tasks dynamically and never
re-randomizes the split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import sha256_file
from .protocol import SPLIT_NAMES, task_ids_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TB2_MANIFEST_PATH = REPO_ROOT / "eval" / "manifests" / "tb2_clean64.json"
TB2_BENCHMARK_ID = "terminal_bench_2"
EXPECTED_TB2_TRAIN = 43
EXPECTED_TB2_HELDOUT = 21
EXPECTED_TB2_TOTAL = 64


class Tb2ManifestError(ValueError):
    """Raised when the Terminal-Bench manifest is malformed."""


@dataclass(frozen=True)
class Tb2Task:
    task_id: str
    split: str
    stratum: str
    selector: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "stratum": self.stratum,
            "selector": self.selector,
        }


@dataclass(frozen=True)
class Tb2Manifest:
    path: Path
    sha256: str
    benchmark: str
    version: str
    name: str
    tasks: tuple[Tb2Task, ...]

    def split_tasks(self, split: str) -> tuple[Tb2Task, ...]:
        if split not in SPLIT_NAMES:
            raise Tb2ManifestError(f"unknown split {split!r}")
        return tuple(task for task in self.tasks if task.split == split)

    def split_task_ids(self, split: str) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.split_tasks(split))

    def task(self, task_id: str) -> Tb2Task:
        matches = [task for task in self.tasks if task.task_id == task_id]
        if len(matches) != 1:
            raise Tb2ManifestError(f"task_id {task_id!r} matched {len(matches)} manifest entries")
        return matches[0]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def task_ids_sha256(self) -> str:
        return task_ids_sha256(self.split_task_ids("train"), self.split_task_ids("heldout"))

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.split_tasks("train")),
            "heldout": len(self.split_tasks("heldout")),
            "total": len(self.tasks),
        }

    def strata(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for task in self.tasks:
            result.setdefault(task.stratum, []).append(task.task_id)
        return {key: sorted(value) for key, value in sorted(result.items())}

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "version": self.version,
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "counts": self.counts(),
            "splits": {
                "train": list(self.split_task_ids("train")),
                "heldout": list(self.split_task_ids("heldout")),
            },
            "task_ids_sha256": self.task_ids_sha256,
            "strata": self.strata(),
        }


def load_tb2_manifest(path: Path | str | None = None) -> Tb2Manifest:
    manifest_path = Path(path).expanduser() if path is not None else DEFAULT_TB2_MANIFEST_PATH
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise Tb2ManifestError(f"Terminal-Bench manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Tb2ManifestError(f"Terminal-Bench manifest is not valid JSON: {exc}") from exc
    return validate_tb2_manifest(payload, path=manifest_path)


def validate_tb2_manifest(payload: Any, *, path: Path) -> Tb2Manifest:
    if not isinstance(payload, Mapping):
        raise Tb2ManifestError("Terminal-Bench manifest must be a JSON object")
    if str(payload.get("benchmark")) != TB2_BENCHMARK_ID:
        raise Tb2ManifestError(
            f"manifest benchmark must be {TB2_BENCHMARK_ID!r}, got {payload.get('benchmark')!r}"
        )
    version = str(payload.get("version") or "").strip()
    if not version:
        raise Tb2ManifestError("manifest version must be a non-empty string")
    name = str(payload.get("name") or "Terminal-Bench 2.0 Clean64")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise Tb2ManifestError("manifest must contain a non-empty tasks list")

    tasks: list[Tb2Task] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, Mapping):
            raise Tb2ManifestError(f"tasks[{index}] must be an object")
        task_id = str(item.get("task_id") or "").strip()
        if not task_id:
            raise Tb2ManifestError(f"tasks[{index}] is missing task_id")
        if task_id in seen:
            raise Tb2ManifestError(f"duplicate Terminal-Bench task id: {task_id!r}")
        seen.add(task_id)
        split = str(item.get("split") or "").strip()
        if split not in SPLIT_NAMES:
            raise Tb2ManifestError(
                f"task {task_id!r} split must be one of {SPLIT_NAMES}, got {split!r}"
            )
        stratum = str(item.get("stratum") or "clean64").strip() or "clean64"
        selector = item.get("selector")
        if selector is not None and (not isinstance(selector, str) or not selector.strip()):
            raise Tb2ManifestError(f"task {task_id!r} selector must be a non-empty string or null")
        tasks.append(
            Tb2Task(
                task_id=task_id,
                split=split,
                stratum=stratum,
                selector=selector.strip() if isinstance(selector, str) else None,
            )
        )

    manifest = Tb2Manifest(
        path=Path(path),
        sha256=sha256_file(Path(path)),
        benchmark=TB2_BENCHMARK_ID,
        version=version,
        name=name,
        tasks=tuple(tasks),
    )
    counts = manifest.counts()
    declared = payload.get("declared_counts")
    if isinstance(declared, Mapping):
        for split in SPLIT_NAMES:
            declared_value = declared.get(split)
            if declared_value is not None and int(declared_value) != counts[split]:
                raise Tb2ManifestError(
                    f"declared_counts.{split}={declared_value} disagrees with tasks ({counts[split]})"
                )
        declared_total = declared.get("total")
        if declared_total is not None and int(declared_total) != counts["total"]:
            raise Tb2ManifestError(
                f"declared_counts.total={declared_total} disagrees with tasks ({counts['total']})"
            )
    if manifest.version.endswith("clean64"):
        expected = {"train": EXPECTED_TB2_TRAIN, "heldout": EXPECTED_TB2_HELDOUT, "total": EXPECTED_TB2_TOTAL}
        if counts != expected:
            raise Tb2ManifestError(
                f"Clean64 manifest must declare {expected['train']} train / "
                f"{expected['heldout']} heldout / {expected['total']} total, got {counts}"
            )
    return manifest


def select_tb2_tasks(
    manifest: Tb2Manifest,
    *,
    splits: Sequence[str],
    task_ids: Sequence[str] | None = None,
) -> dict[str, tuple[Tb2Task, ...]]:
    requested_splits = tuple(dict.fromkeys(splits))
    for split in requested_splits:
        if split not in SPLIT_NAMES:
            raise Tb2ManifestError(f"unknown split {split!r}")
    selected: dict[str, tuple[Tb2Task, ...]] = {}
    if task_ids is None:
        for split in requested_splits:
            selected[split] = manifest.split_tasks(split)
        return selected
    requested = {str(item) for item in task_ids}
    unknown = sorted(requested - set(manifest.task_ids))
    if unknown:
        raise Tb2ManifestError(f"configured task_ids are not in the manifest: {unknown[:10]}")
    for split in requested_splits:
        selected[split] = tuple(
            task for task in manifest.split_tasks(split) if task.task_id in requested
        )
    covered = {task.task_id for tasks in selected.values() for task in tasks}
    missing = sorted(requested - covered)
    if missing:
        raise Tb2ManifestError(
            "some configured task_ids belong to splits not selected by this run: "
            + ", ".join(missing[:10])
        )
    for split, tasks in selected.items():
        if not tasks:
            raise Tb2ManifestError(f"no tasks selected for split {split!r}")
    return selected


def tb2_dataset_manifest_dict(
    manifest: Tb2Manifest,
    *,
    selected: Mapping[str, Sequence[Tb2Task]],
    evaluator: str,
    evaluator_version: str,
) -> dict[str, Any]:
    return {
        "benchmark": TB2_BENCHMARK_ID,
        "version": manifest.version,
        "name": manifest.name,
        "manifest": {"path": str(manifest.path), "sha256": manifest.sha256},
        "evaluator": {"name": evaluator, "version": evaluator_version},
        "selected_splits": {
            split: [task.to_dict() for task in tasks] for split, tasks in selected.items()
        },
        "selected_task_ids_sha256": task_ids_sha256(
            (task.task_id for task in selected.get("train", ())),
            (task.task_id for task in selected.get("heldout", ())),
        ),
        "strata": manifest.strata(),
        "clean64_note": (
            "This manifest is the full Terminal-Bench 2.0 Clean64 split. A single example "
            "task is not a Terminal-Bench evaluation; configs must select the complete fixed split."
        ),
    }
