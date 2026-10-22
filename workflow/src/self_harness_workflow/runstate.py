"""Self-Harness run persistence: paths, run manifest and experiment fingerprints."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from self_harness_eval.config import EvalConfig, config_to_dict, write_resolved_config
from self_harness_eval.hashing import sha256_file, sha256_json, snapshot_tree_hash, tree_file_hashes
from self_harness_eval.protocol import SCHEMA_VERSION

RUN_FORMAT = "self_harness.run_manifest.v1"
RUN_SCHEMA_VERSION = "self_harness.workflow_run.v1"
DEFAULT_VERSION_STATUSES = (
    "materialized",
    "evaluated",
    "accepted",
    "rejected",
    "merged",
    "final",
    "eval_failed",
    "materialize_failed",
    "merge_conflict",
    "not_evaluated_budget",
)


class WorkflowError(RuntimeError):
    """Raised when a Self-Harness run cannot proceed."""


class ResumeMismatch(WorkflowError):
    """Raised when an existing run does not match the current configuration."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def config_resolved(self) -> Path:
        return self.root / "config.resolved.json"

    @property
    def dataset_manifest(self) -> Path:
        return self.root / "dataset_manifest.json"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def task_ids(self) -> Path:
        return self.dataset_dir / "task_ids.json"

    @property
    def train_ids(self) -> Path:
        return self.dataset_dir / "train_task_ids.json"

    @property
    def heldout_ids(self) -> Path:
        return self.dataset_dir / "heldout_task_ids.json"

    @property
    def versions_dir(self) -> Path:
        return self.root / "versions"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def final_report_json(self) -> Path:
        return self.root / "final_report.json"

    @property
    def final_report_md(self) -> Path:
        return self.root / "final_report.md"

    @property
    def state_json(self) -> Path:
        return self.root / "run_state.json"

    def version_dir(self, version_id: str) -> Path:
        return self.versions_dir / version_id

    def version_json(self, version_id: str) -> Path:
        return self.version_dir(version_id) / "version.json"

    def version_harness(self, version_id: str) -> Path:
        return self.version_dir(version_id) / "harness"

    def version_eval_dir(self, version_id: str) -> Path:
        return self.version_dir(version_id) / "eval"

    def version_result(self, version_id: str) -> Path:
        return self.version_eval_dir(version_id) / "result.json"

    def version_acceptance(self, version_id: str) -> Path:
        return self.version_dir(version_id) / "acceptance.json"

    def version_diagnosis_dir(self, version_id: str) -> Path:
        return self.version_dir(version_id) / "diagnosis"


def experiment_fingerprint(config: EvalConfig, *, task_ids_payload: Mapping[str, Any]) -> str:
    """Hash only the variables that must stay fixed across a run.

    Operational toggles (resume, reuse-existing) and output directories are
    intentionally excluded so a run can be resumed with the same experiment
    definition.  Model, decoding, tools, evaluator, dataset revision, task list,
    timeouts, repeats and the baseline harness still participate.
    """

    raw = config_to_dict(config)
    raw.pop("config_path", None)
    raw.pop("env_refs", None)
    run_block = dict(raw.get("run", {}))
    for key in ("resume", "reuse_existing", "run_id", "proposer_response", "proposer_command", "diagnosis_command"):
        run_block.pop(key, None)
    raw["run"] = run_block
    eval_block = dict(raw.get("eval", {}))
    for key in ("reuse_existing", "smoke_test"):
        eval_block.pop(key, None)
    raw["eval"] = eval_block
    dataset_block = dict(raw.get("dataset", {}))
    raw["dataset"] = dataset_block
    return sha256_json({"config": raw, "task_ids": task_ids_payload})


def load_task_ids_payload(config: EvalConfig) -> dict[str, Any]:
    """Resolve the fixed split/task selection for this config."""

    from self_harness_eval.manifest import load_swebench_manifest, select_task_ids
    from self_harness_eval.tb2_manifest import load_tb2_manifest, select_tb2_tasks

    if config.eval.benchmark == "swe_bench_verified":
        manifest = load_swebench_manifest(config.dataset.manifest_path)
        task_selection = config.eval.task_ids
        if task_selection:
            requested = set(task_selection)
            splits = tuple(
                split
                for split in config.eval.splits
                if any(item in requested for item in manifest.split_task_ids(split))
            )
        else:
            splits = tuple(config.eval.splits)
        selected = select_task_ids(manifest, splits=splits, task_ids=task_selection)
        return {
            "benchmark": "swe_bench_verified",
            "dataset": {
                "name": config.dataset.name,
                "revision": config.dataset.revision,
                "manifest_path": str(manifest.path),
                "manifest_sha256": manifest.sha256,
            },
            "splits": {split: list(ids) for split, ids in selected.items()},
            "counts": {split: len(ids) for split, ids in selected.items()},
            "task_ids_sha256": manifest.task_ids_sha256
            if set(selected) == {"train", "heldout"}
            else None,
            "distributions": manifest.distributions(),
            "manifest_metadata": manifest.metadata,
        }

    manifest = load_tb2_manifest(config.dataset.manifest_path)
    task_selection = config.eval.task_ids
    if task_selection:
        requested = set(task_selection)
        splits = tuple(
            split for split in config.eval.splits if any(item in requested for item in manifest.split_task_ids(split))
        )
    else:
        splits = tuple(config.eval.splits)
    selected = select_tb2_tasks(manifest, splits=splits, task_ids=task_selection)
    return {
        "benchmark": "terminal_bench_2",
        "dataset": {
            "name": config.dataset.name,
            "revision": manifest.version,
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.sha256,
        },
        "splits": {
            split: [task.task_id for task in tasks] for split, tasks in selected.items()
        },
        "counts": {split: len(tasks) for split, tasks in selected.items()},
        "task_ids_sha256": manifest.task_ids_sha256
        if set(selected) == {"train", "heldout"}
        else None,
        "distributions": {"strata": manifest.strata()},
        "manifest_metadata": {"name": manifest.name, "version": manifest.version},
    }


def initialize_run(
    *,
    config: EvalConfig,
    run_id: str,
    run_root: Path | None = None,
    reuse_existing: bool = False,
) -> tuple[RunPaths, dict[str, Any], dict[str, Any]]:
    """Create (or validate and load) a run directory.

    Returns ``(paths, run_manifest, task_ids_payload)``.
    """

    root = (
        Path(run_root).expanduser().resolve()
        if run_root is not None
        else (config.run.output_root / run_id).resolve()
    )
    paths = RunPaths(root)
    task_ids_payload = load_task_ids_payload(config)
    fingerprint = experiment_fingerprint(config, task_ids_payload=task_ids_payload)

    if paths.run_manifest.exists():
        run_manifest = load_run_manifest(paths)
        _validate_resume(config=config, run_manifest=run_manifest, fingerprint=fingerprint, paths=paths)
        return paths, run_manifest, task_ids_payload

    root.mkdir(parents=True, exist_ok=True)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.versions_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(paths.config_resolved, config)
    _write_json(paths.task_ids, task_ids_payload)
    _write_json(paths.train_ids, task_ids_payload["splits"].get("train", []))
    _write_json(paths.heldout_ids, task_ids_payload["splits"].get("heldout", []))
    run_manifest = {
        "format": RUN_FORMAT,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "eval_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config_path": str(config.path),
        "config_sha256": sha256_file(config.path),
        "resolved_config_sha256": sha256_file(paths.config_resolved),
        "experiment_fingerprint": fingerprint,
        "benchmark": config.eval.benchmark,
        "dataset": task_ids_payload["dataset"],
        "task_splits": task_ids_payload["splits"],
        "task_counts": task_ids_payload["counts"],
        "model": config.model.non_secret_block(),
        "eval": {
            "name": config.eval.name,
            "repeats": config.eval.repeats,
            "splits": list(config.eval.splits),
            "task_timeout_s": config.eval.task_timeout_s,
            "tool_timeout_s": config.eval.tool_timeout_s,
            "smoke_test": config.eval.smoke_test,
        },
        "harness": {
            "baseline_dir": str(config.harness.baseline_dir) if config.harness.baseline_dir else None,
            "surfaces": {name: str(path) for name, path in config.harness.surfaces.items()},
        },
        "budget": {
            "proposal_width": config.run.proposal_width,
            "max_rounds": config.run.max_rounds,
            "max_candidate_evaluations": config.run.max_candidate_evaluations,
            "max_consecutive_no_improvement": config.run.max_consecutive_no_improvement,
            "max_proposer_failures": config.run.max_proposer_failures,
        },
        "state": {
            "status": "initialized",
            "active_version_id": None,
            "baseline_version_id": None,
            "final_version_id": None,
            "rounds_completed": 0,
            "candidate_evaluations": 0,
            "proposals_created": 0,
            "consecutive_no_improvement": 0,
            "proposer_failures": 0,
            "stop_reason": None,
            "started_at": utc_now(),
            "stopped_at": None,
        },
        "versions": [],
        "environment": {},
    }
    _write_json(paths.run_manifest, run_manifest)
    from .events import append_event

    append_event(
        paths,
        "run_start",
        run_id=run_id,
        artifacts={"run_manifest": "run_manifest.json", "config": "config.resolved.json"},
        payload={"benchmark": config.eval.benchmark, "experiment_fingerprint": fingerprint},
    )
    return paths, run_manifest, task_ids_payload


def load_run_manifest(paths: RunPaths) -> dict[str, Any]:
    if not paths.run_manifest.exists():
        raise WorkflowError(f"run manifest does not exist: {paths.run_manifest}")
    payload = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != RUN_FORMAT:
        raise WorkflowError(f"run manifest format must be {RUN_FORMAT!r}")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise WorkflowError("run manifest state must be an object")
    return payload


def _validate_resume(
    *,
    config: EvalConfig,
    run_manifest: Mapping[str, Any],
    fingerprint: str,
    paths: RunPaths,
) -> None:
    stored = run_manifest.get("experiment_fingerprint")
    if stored != fingerprint:
        raise ResumeMismatch(
            "refusing to resume: experiment configuration changed "
            f"(stored={stored}, current={fingerprint}). Use a new run_id/output root for a "
            "changed model, decoding, evaluator, dataset revision, task list, timeouts or "
            "baseline harness."
        )
    for name in ("config.resolved.json", "dataset/task_ids.json", "dataset/train_task_ids.json"):
        path = paths.root / name
        if not path.exists():
            raise ResumeMismatch(f"refusing to resume: required run artifact is missing: {name}")
    # Verify all recorded snapshots are still intact and immutable.
    for record in run_manifest.get("versions", []):
        if not isinstance(record, Mapping):
            continue
        version_id = record.get("version_id")
        snapshot = record.get("harness_snapshot_path")
        expected = record.get("harness_snapshot_sha256")
        if not version_id or not snapshot or not expected:
            raise ResumeMismatch(f"run version index entry is incomplete: {record}")
        snapshot_path = paths.root / str(snapshot)
        if not snapshot_path.is_dir():
            raise ResumeMismatch(f"refusing to resume: snapshot missing for {version_id}")
        actual = snapshot_tree_hash(tree_file_hashes(snapshot_path))
        if actual != expected:
            raise ResumeMismatch(
                f"refusing to resume: snapshot hash changed for {version_id} "
                f"(stored={expected}, current={actual})"
            )


def save_run_manifest(paths: RunPaths, run_manifest: dict[str, Any]) -> None:
    run_manifest["updated_at"] = utc_now()
    _write_json(paths.run_manifest, run_manifest)


def write_state(paths: RunPaths, run_manifest: dict[str, Any]) -> None:
    _write_json(paths.state_json, run_manifest.get("state", {}))
    save_run_manifest(paths, run_manifest)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_tree(source: Path, target: Path) -> dict[str, str]:
    from self_harness_eval.hashing import IGNORED_DIR_NAMES, IGNORED_FILE_SUFFIXES

    if target.exists():
        raise WorkflowError(f"refusing to overwrite existing snapshot directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(source).rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if any(part in IGNORED_DIR_NAMES for part in relative.parts[:-1]):
            continue
        if relative.suffix in IGNORED_FILE_SUFFIXES:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return tree_file_hashes(target)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_file(Path(path))


def relative(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
