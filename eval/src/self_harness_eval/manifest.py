"""SWE-bench Verified 100-task manifest loading and validation.

The repository ships ``swe_bench_verified_stratified_100.json`` at its root.
The manifest was produced by a *five-domain stratified* sample, not by equal
sampling per repository; this module validates the fixed split and exposes the
actual repository/domain distribution for reports.  It never re-randomizes the
split at runtime and never exposes gold patches or hidden test metadata.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import sha256_file, sha256_json
from .protocol import SPLIT_NAMES, task_ids_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "swe_bench_verified_stratified_100.json"
EXPECTED_TOTAL = 100
EXPECTED_TRAIN = 67
EXPECTED_HELDOUT = 33
SWEBENCH_BENCHMARK = "SWE-bench_Verified"


class ManifestError(ValueError):
    """Raised when the SWE-bench split manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class SwebenchManifest:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    held_in_task_ids: tuple[str, ...]
    held_out_task_ids: tuple[str, ...]
    repo_distribution_in_100: dict[str, int]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.held_in_task_ids + self.held_out_task_ids

    @property
    def split_by_task(self) -> dict[str, str]:
        mapping = {task_id: "train" for task_id in self.held_in_task_ids}
        mapping.update({task_id: "heldout" for task_id in self.held_out_task_ids})
        return mapping

    @property
    def task_ids_sha256(self) -> str:
        return task_ids_sha256(self.held_in_task_ids, self.held_out_task_ids)

    def split_task_ids(self, split: str) -> tuple[str, ...]:
        if split == "train":
            return self.held_in_task_ids
        if split == "heldout":
            return self.held_out_task_ids
        raise ManifestError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")

    def domain_definitions(self) -> dict[str, str]:
        raw = self.metadata.get("domain_definitions")
        return {str(key): str(value) for key, value in raw.items()} if isinstance(raw, Mapping) else {}

    def repository_of(self, task_id: str) -> str | None:
        for repo in self.domain_definitions():
            if task_id.startswith(repo.replace("/", "__") + "-"):
                return repo
        prefix = task_id.split("__", 1)[0] if "__" in task_id else ""
        if not prefix:
            return None
        owner = prefix
        remainder = task_id.split("__", 1)[1]
        repo_name = remainder.rsplit("-", 1)[0]
        return f"{owner}/{repo_name}"

    def domain_of(self, task_id: str) -> str | None:
        repo = self.repository_of(task_id)
        if repo is None:
            return None
        return self.domain_definitions().get(repo)

    def count_by_split(self) -> dict[str, int]:
        return {
            "train": len(self.held_in_task_ids),
            "heldout": len(self.held_out_task_ids),
            "total": len(self.task_ids),
        }

    def distributions(self) -> dict[str, Any]:
        """Actual repository/domain distributions for reports."""

        repo_counts: Counter[str] = Counter()
        domain_counts: Counter[str] = Counter()
        split_repo: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_NAMES}
        for task_id, split in self.split_by_task.items():
            repo = self.repository_of(task_id) or "unknown"
            domain = self.domain_of(task_id) or "unknown"
            repo_counts[repo] += 1
            domain_counts[domain] += 1
            split_repo[split][repo] += 1
        return {
            "sampling_note": (
                "Manifest metadata states this is a five-domain stratified sample, not a "
                "repository-balanced sample. Repository frequencies are reported as observed "
                "and are not rebalanced by the harness."
            ),
            "total": dict(sorted(repo_counts.items())),
            "by_domain_total": dict(sorted(domain_counts.items())),
            "train": dict(sorted(split_repo["train"].items())),
            "heldout": dict(sorted(split_repo["heldout"].items())),
        }

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "benchmark": self.metadata.get("benchmark"),
            "splits": {
                "train": list(self.held_in_task_ids),
                "heldout": list(self.held_out_task_ids),
            },
            "counts": self.count_by_split(),
            "task_ids_sha256": self.task_ids_sha256,
            "metadata": self.metadata,
            "repo_distribution_in_100": self.repo_distribution_in_100,
            "distributions": self.distributions(),
        }


def load_swebench_manifest(path: Path | str | None = None) -> SwebenchManifest:
    manifest_path = Path(path).expanduser() if path is not None else DEFAULT_MANIFEST_PATH
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise ManifestError(f"SWE-bench manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"SWE-bench manifest is not valid JSON: {exc}") from exc
    return validate_swebench_manifest(payload, path=manifest_path)


def validate_swebench_manifest(payload: Any, *, path: Path) -> SwebenchManifest:
    if not isinstance(payload, Mapping):
        raise ManifestError("SWE-bench manifest must be a JSON object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ManifestError("SWE-bench manifest must contain a metadata object")
    if str(metadata.get("benchmark", "")) != SWEBENCH_BENCHMARK:
        raise ManifestError(
            f"manifest metadata.benchmark must be {SWEBENCH_BENCHMARK!r}, "
            f"got {metadata.get('benchmark')!r}"
        )
    domain_definitions = metadata.get("domain_definitions")
    if not isinstance(domain_definitions, Mapping) or not domain_definitions:
        raise ManifestError("manifest metadata.domain_definitions must be a non-empty object")
    for repo, domain in domain_definitions.items():
        if not isinstance(repo, str) or "/" not in repo:
            raise ManifestError(f"invalid repository in domain_definitions: {repo!r}")
        if not isinstance(domain, str) or not domain.strip():
            raise ManifestError(f"invalid domain for repository {repo!r}")

    train = _require_task_id_list(payload, "held_in_task_ids")
    heldout = _require_task_id_list(payload, "held_out_task_ids")

    duplicates_train = _duplicates(train)
    duplicates_heldout = _duplicates(heldout)
    if duplicates_train:
        raise ManifestError(f"held_in_task_ids contains duplicate ids: {duplicates_train[:5]}")
    if duplicates_heldout:
        raise ManifestError(f"held_out_task_ids contains duplicate ids: {duplicates_heldout[:5]}")
    overlap = sorted(set(train) & set(heldout))
    if overlap:
        raise ManifestError(f"train and heldout splits overlap: {overlap[:5]}")
    if len(train) != EXPECTED_TRAIN:
        raise ManifestError(f"held_in_task_ids must contain {EXPECTED_TRAIN} ids, got {len(train)}")
    if len(heldout) != EXPECTED_HELDOUT:
        raise ManifestError(
            f"held_out_task_ids must contain {EXPECTED_HELDOUT} ids, got {len(heldout)}"
        )
    union = train + heldout
    if len(union) != EXPECTED_TOTAL:
        raise ManifestError(
            f"train+heldout union must contain {EXPECTED_TOTAL} ids, got {len(union)}"
        )
    if len(set(union)) != EXPECTED_TOTAL:
        raise ManifestError("train+heldout union must contain 100 unique ids")

    declared_total = metadata.get("total_sampled")
    if declared_total is not None and int(declared_total) != EXPECTED_TOTAL:
        raise ManifestError(
            f"metadata.total_sampled must equal {EXPECTED_TOTAL}, got {declared_total}"
        )
    declared_train = metadata.get("held_in_count")
    declared_heldout = metadata.get("held_out_count")
    if declared_train is not None and int(declared_train) != len(train):
        raise ManifestError(
            f"metadata.held_in_count={declared_train} disagrees with held_in_task_ids ({len(train)})"
        )
    if declared_heldout is not None and int(declared_heldout) != len(heldout):
        raise ManifestError(
            f"metadata.held_out_count={declared_heldout} disagrees with held_out_task_ids "
            f"({len(heldout)})"
        )

    repo_distribution = payload.get("repo_distribution_in_100")
    if not isinstance(repo_distribution, Mapping) or not repo_distribution:
        raise ManifestError("manifest must contain repo_distribution_in_100")
    declared_repo = {str(key): int(value) for key, value in repo_distribution.items()}
    if sum(declared_repo.values()) != EXPECTED_TOTAL:
        raise ManifestError(
            f"repo_distribution_in_100 must sum to {EXPECTED_TOTAL}, got {sum(declared_repo.values())}"
        )

    manifest = SwebenchManifest(
        path=Path(path),
        sha256=sha256_file(Path(path)),
        metadata={str(key): value for key, value in metadata.items()},
        held_in_task_ids=train,
        held_out_task_ids=heldout,
        repo_distribution_in_100=declared_repo,
    )

    computed_repo = Counter(manifest.repository_of(task_id) or "unknown" for task_id in manifest.task_ids)
    unknown_repos = sorted(repo for repo in computed_repo if repo not in domain_definitions)
    if unknown_repos:
        raise ManifestError(
            "manifest tasks use repositories missing from metadata.domain_definitions: "
            + ", ".join(unknown_repos)
        )
    if dict(computed_repo) != declared_repo:
        mismatch = {
            repo: (declared_repo.get(repo, 0), computed_repo.get(repo, 0))
            for repo in sorted(set(declared_repo) | set(computed_repo))
            if declared_repo.get(repo, 0) != computed_repo.get(repo, 0)
        }
        raise ManifestError(
            "repo_distribution_in_100 disagrees with the actual task ids "
            f"(repo: declared, actual): {mismatch}"
        )

    quotas = metadata.get("domain_quotas")
    if isinstance(quotas, Mapping):
        declared_quotas = {str(key): int(value) for key, value in quotas.items()}
        if sum(declared_quotas.values()) != EXPECTED_TOTAL:
            raise ManifestError(
                f"metadata.domain_quotas must sum to {EXPECTED_TOTAL}, "
                f"got {sum(declared_quotas.values())}"
            )
        computed_domains = Counter(manifest.domain_of(task_id) or "unknown" for task_id in manifest.task_ids)
        if dict(computed_domains) != declared_quotas:
            raise ManifestError(
                "metadata.domain_quotas disagrees with the actual task ids: "
                f"declared={dict(sorted(declared_quotas.items()))} "
                f"actual={dict(sorted(computed_domains.items()))}"
            )
    return manifest


def _require_task_id_list(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise ManifestError(f"manifest must contain a {key} list")
    if not raw:
        raise ManifestError(f"{key} must not be empty")
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise ManifestError(f"{key}[{index}] must be a non-empty string task id")
        if item != item.strip():
            raise ManifestError(f"{key}[{index}] must not contain surrounding whitespace")
        result.append(item)
    return tuple(result)


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def select_task_ids(
    manifest: SwebenchManifest,
    *,
    splits: Sequence[str],
    task_ids: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return the configured task ids per split, preserving manifest order."""

    requested_splits = tuple(dict.fromkeys(splits))
    for split in requested_splits:
        if split not in SPLIT_NAMES:
            raise ManifestError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if task_ids is None:
        return {split: manifest.split_task_ids(split) for split in requested_splits}

    requested = {str(item) for item in task_ids}
    all_ids = set(manifest.task_ids)
    unknown = sorted(requested - all_ids)
    if unknown:
        raise ManifestError(f"configured task_ids are not in the manifest: {unknown[:10]}")
    selected: dict[str, tuple[str, ...]] = {}
    for split in requested_splits:
        selected[split] = tuple(
            task_id for task_id in manifest.split_task_ids(split) if task_id in requested
        )
    covered = {task_id for ids in selected.values() for task_id in ids}
    missing = sorted(requested - covered)
    if missing:
        raise ManifestError(
            "some configured task_ids belong to splits not selected by this run: "
            + ", ".join(missing[:10])
        )
    for split, ids in selected.items():
        if not ids:
            raise ManifestError(f"no tasks selected for split {split!r}")
    return selected


def dataset_manifest_dict(
    manifest: SwebenchManifest,
    *,
    selected: Mapping[str, Sequence[str]],
    dataset_name: str,
    dataset_revision: str,
    source: str | None,
    source_sha256: str | None,
) -> dict[str, Any]:
    """Audit payload written as ``dataset_manifest.json`` inside a run."""

    splits = {split: [str(item) for item in ids] for split, ids in selected.items()}
    return {
        "benchmark": "swe_bench_verified",
        "dataset": {
            "name": dataset_name,
            "revision": dataset_revision,
            "source": source,
            "source_sha256": source_sha256,
        },
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
            "metadata": manifest.metadata,
            "repo_distribution_in_100": manifest.repo_distribution_in_100,
        },
        "selected_splits": splits,
        "selected_task_ids_sha256": task_ids_sha256(splits.get("train", ()), splits.get("heldout", ())),
        "distributions": manifest.distributions(),
        "selection_note": (
            "Held-out task ids are retained for acceptance gating and final reporting only. "
            "Diagnosis and proposer inputs are restricted to split=train."
        ),
    }


def manifest_fingerprint(manifest: SwebenchManifest) -> str:
    return sha256_json({"sha256": manifest.sha256, "task_ids_sha256": manifest.task_ids_sha256})
