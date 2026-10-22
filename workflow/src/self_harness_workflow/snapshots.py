"""Harness snapshot materialization and verification.

Snapshots are written once per version and never edited afterwards.  A
candidate snapshot is the parent snapshot with the candidate's edited surface
files overlaid at their manifest-relative paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_harness_eval.hashing import sha256_text, snapshot_tree_hash, tree_file_hashes
from self_harness_eval.protocol import build_harness_block

from .runstate import WorkflowError, copy_tree, read_json, relative


@dataclass(frozen=True)
class HarnessSnapshot:
    directory: Path
    files: Mapping[str, str]
    surfaces: Mapping[str, str] = field(default_factory=dict)

    @property
    def tree_sha256(self) -> str:
        return snapshot_tree_hash(self.files)

    def block(
        self,
        *,
        run_root: Path,
        version_id: str,
        parent_version_ids: Sequence[str],
        kind: str,
    ) -> dict[str, Any]:
        return build_harness_block(
            run_root=run_root,
            snapshot_dir=self.directory,
            version_id=version_id,
            parent_version_ids=list(parent_version_ids),
            kind=kind,
        )


def surface_relative_paths(config: Any) -> dict[str, str]:
    """Map editable surface names to paths relative to the baseline harness."""

    baseline = config.harness.baseline_dir
    if baseline is None:
        raise WorkflowError("[harness].baseline_dir is required to materialize harness snapshots")
    baseline = Path(baseline).resolve()
    mapping: dict[str, str] = {}
    for name, raw_path in config.harness.surfaces.items():
        path = Path(raw_path).resolve()
        try:
            relative_path = path.relative_to(baseline).as_posix()
        except ValueError:
            relative_path = path.name
        if relative_path in mapping.values():
            raise WorkflowError(
                f"harness surfaces {name!r} and another surface map to the same snapshot path "
                f"{relative_path!r}; rename the surfaces"
            )
        mapping[str(name)] = relative_path
    return mapping


def create_baseline_snapshot(*, config: Any, target: Path) -> HarnessSnapshot:
    baseline = config.harness.baseline_dir
    if baseline is None:
        raise WorkflowError("[harness].baseline_dir is required to materialize the baseline harness")
    if not Path(baseline).is_dir():
        raise WorkflowError(f"baseline harness directory does not exist: {baseline}")
    files = copy_tree(Path(baseline), target)
    return HarnessSnapshot(directory=target, files=files, surfaces=surface_relative_paths(config))


def create_candidate_snapshot(
    *,
    parent: HarnessSnapshot,
    target: Path,
    changed_files: Mapping[str, str],
    surfaces: Mapping[str, str] | None = None,
) -> HarnessSnapshot:
    if not changed_files:
        raise WorkflowError("candidate snapshot requires at least one changed harness file")
    files = copy_tree(parent.directory, target)
    for relative_path, content in changed_files.items():
        if relative_path.startswith("/") or ".." in Path(relative_path).parts:
            raise WorkflowError(f"unsafe candidate surface path: {relative_path!r}")
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        files[relative_path] = sha256_text(content)
    if snapshot_tree_hash(files) == parent.tree_sha256:
        raise WorkflowError("candidate snapshot is identical to its parent harness snapshot")
    return HarnessSnapshot(
        directory=target,
        files=files,
        surfaces=dict(surfaces or parent.surfaces),
    )


def create_merged_snapshot(
    *,
    parent: HarnessSnapshot,
    candidates: Sequence[HarnessSnapshot],
    target: Path,
) -> HarnessSnapshot:
    """Merge accepted candidate edits onto the parent snapshot.

    Candidates that edit disjoint files merge directly.  When several accepted
    candidates edit different top-level functions of the same Python surface,
    the function-level merge used by the existing Self-Harness loop is reused;
    conflicting edits to the same function are reported as merge conflicts.
    """

    files = copy_tree(parent.directory, target)
    by_file: dict[str, list[HarnessSnapshot]] = {}
    for candidate in candidates:
        for relative_path in snapshot_diff(parent, candidate):
            by_file.setdefault(relative_path, []).append(candidate)
    if not by_file:
        raise WorkflowError("merged candidate has no harness file changes")
    for relative_path, editing_candidates in sorted(by_file.items()):
        parent_text = (parent.directory / relative_path).read_text(encoding="utf-8")
        candidate_texts = [
            (candidate.directory / relative_path).read_text(encoding="utf-8")
            for candidate in editing_candidates
        ]
        if len(candidate_texts) == 1:
            content = candidate_texts[0]
        else:
            if not relative_path.endswith(".py"):
                raise WorkflowError(
                    "merge conflict: multiple accepted candidates modify the non-Python harness "
                    f"file {relative_path!r}"
                )
            content = parent_text
            for index, candidate_text in enumerate(candidate_texts):
                try:
                    content = _merge_python_surface(
                        parent_source=parent_text,
                        current_source=content,
                        candidate_source=candidate_text,
                        candidate_id=f"candidate-{index + 1}",
                        surface_name=relative_path,
                    )
                except Exception as exc:  # noqa: BLE001 - convert to merge conflict
                    raise WorkflowError(
                        f"merge conflict on {relative_path!r}: {exc}"
                    ) from exc
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        files[relative_path] = sha256_text(content)
    return HarnessSnapshot(directory=target, files=files, surfaces=dict(parent.surfaces))


def _merge_python_surface(
    *,
    parent_source: str,
    current_source: str,
    candidate_source: str,
    candidate_id: str,
    surface_name: str,
) -> str:
    """Reuse the existing Self-Harness function-level merge implementation."""

    module = _legacy_merge_module()
    return module.merge_candidate_changes(
        parent_source=parent_source,
        current_source=current_source,
        candidate_source=candidate_source,
        candidate_id=candidate_id,
        surface_name=surface_name,
    )


def _legacy_merge_module() -> Any:
    global _LEGACY_MERGE_MODULE
    if _LEGACY_MERGE_MODULE is not None:
        return _LEGACY_MERGE_MODULE
    import importlib.util

    script = Path(__file__).resolve().parents[3] / "workflow" / "scripts" / "run_self_harness_loop.py"
    if not script.exists():
        raise WorkflowError(f"function-level harness merge helper is missing: {script}")
    spec = importlib.util.spec_from_file_location("self_harness_legacy_merge", script)
    if spec is None or spec.loader is None:
        raise WorkflowError(f"cannot load function-level harness merge helper: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LEGACY_MERGE_MODULE = module
    return module


_LEGACY_MERGE_MODULE: Any = None


def snapshot_diff(left: HarnessSnapshot, right: HarnessSnapshot) -> list[str]:
    changed: list[str] = []
    for relative_path in sorted(set(left.files) | set(right.files)):
        if left.files.get(relative_path) != right.files.get(relative_path):
            changed.append(relative_path)
    return changed


def load_snapshot(directory: Path) -> HarnessSnapshot:
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise WorkflowError(f"snapshot directory does not exist: {directory}")
    files = tree_file_hashes(directory)
    return HarnessSnapshot(directory=directory, files=files)


def verify_snapshot(directory: Path, *, expected_sha256: str) -> HarnessSnapshot:
    snapshot = load_snapshot(directory)
    if snapshot.tree_sha256 != expected_sha256:
        raise WorkflowError(
            f"harness snapshot hash mismatch for {directory}: "
            f"expected {expected_sha256}, got {snapshot.tree_sha256}"
        )
    return snapshot


def snapshot_relative_path(directory: Path, run_root: Path) -> str:
    return relative(directory, run_root)


def read_candidate_manifest(candidate_dir: Path) -> dict[str, Any]:
    manifest_path = Path(candidate_dir) / "manifest.json"
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise WorkflowError(f"candidate manifest must be an object: {manifest_path}")
    return payload


def changed_surface_contents(
    *,
    candidate_dir: Path,
    manifest: Mapping[str, Any],
    surfaces: Mapping[str, str],
) -> dict[str, tuple[str, str]]:
    """Return ``{surface_name: (relative_path, content)}`` for the candidate."""

    surface_files = manifest.get("surface_files")
    if not isinstance(surface_files, Mapping):
        raise WorkflowError(f"candidate manifest is missing surface_files: {candidate_dir}")
    result: dict[str, tuple[str, str]] = {}
    for name in manifest.get("changed_surfaces") or []:
        relative_path = surface_files.get(name)
        if not isinstance(relative_path, str):
            raise WorkflowError(f"candidate manifest has no surface file for {name!r}")
        snapshot_relative = surfaces.get(str(name))
        if snapshot_relative is None:
            raise WorkflowError(f"surface {name!r} is not part of the configured harness")
        path = Path(candidate_dir) / relative_path
        if not path.exists():
            raise WorkflowError(f"candidate surface file is missing: {path}")
        result[str(name)] = (snapshot_relative, path.read_text(encoding="utf-8"))
    return result
