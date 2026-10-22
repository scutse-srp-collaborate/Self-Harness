"""SWE-bench Verified dataset access with pinned revision and explicit cache.

Only the agent-visible subset of a task instance (repository, base commit and
issue statement) is ever handed to an agent.  Gold patches, test patches and
FAIL_TO_PASS/PASS_TO_PASS test metadata stay in the evaluator path.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import sha256_file, sha256_json

DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
AGENT_VISIBLE_FIELDS = (
    "task_id",
    "repo",
    "base_commit",
    "version",
    "problem_statement",
    "hints_text",
)
#: Fields that must never be handed to an agent or proposer.
PROTECTED_FIELDS = (
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
    "resolved",
    "gold_patch",
)


class DatasetUnavailable(RuntimeError):
    """Raised when the pinned SWE-bench dataset cannot be resolved locally."""


@dataclass(frozen=True)
class SwebenchInstance:
    task_id: str
    repo: str
    base_commit: str
    problem_statement: str
    version: str | None = None
    hints_text: str | None = None
    patch: str | None = None
    test_patch: str | None = None
    fail_to_pass: tuple[str, ...] = field(default_factory=tuple)
    pass_to_pass: tuple[str, ...] = field(default_factory=tuple)
    environment_setup_commit: str | None = None
    created_at: str | None = None
    difficulty: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def agent_view(self) -> dict[str, Any]:
        """Return only information an agent may see for this task."""

        view: dict[str, Any] = {
            "task_id": self.task_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "version": self.version,
            "problem_statement": self.problem_statement,
        }
        if self.hints_text:
            view["hints_text"] = self.hints_text
        return view

    def evaluator_fields(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "version": self.version,
            "environment_setup_commit": self.environment_setup_commit,
        }

    def instruction_text(self) -> str:
        """Issue-only instruction handed to the agent."""

        lines = [
            f"Repository: {self.repo}",
            f"Base commit: {self.base_commit}",
            "",
            self.problem_statement.strip(),
        ]
        if self.hints_text:
            lines.extend(["", "Hints:", self.hints_text.strip()])
        return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class SwebenchDataset:
    name: str
    revision: str
    split: str
    source: str
    source_sha256: str
    instances: Mapping[str, SwebenchInstance]

    def instance(self, task_id: str) -> SwebenchInstance | None:
        return self.instances.get(task_id)

    def require(self, task_ids: Sequence[str]) -> tuple[list[SwebenchInstance], list[str]]:
        found: list[SwebenchInstance] = []
        missing: list[str] = []
        for task_id in task_ids:
            instance = self.instances.get(task_id)
            if instance is None:
                missing.append(task_id)
            else:
                found.append(instance)
        return found, missing

    def select(self, task_ids: Sequence[str]) -> list[SwebenchInstance]:
        found, missing = self.require(task_ids)
        if missing:
            raise DatasetUnavailable(
                "pinned SWE-bench dataset is missing manifest task ids: " + ", ".join(missing[:10])
            )
        return found

    def audit_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "split": self.split,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "instance_count": len(self.instances),
            "task_ids_sha256": sha256_json(sorted(self.instances)),
        }


def load_swebench_dataset(
    *,
    name: str = DEFAULT_DATASET_NAME,
    revision: str,
    source: str | Path | None = None,
    cache_dir: str | Path | None = None,
    split: str = "test",
    allow_hf_download: bool = False,
) -> SwebenchDataset:
    """Load the pinned SWE-bench Verified records.

    Resolution order: explicit ``source`` file, then a file in ``cache_dir``,
    then (only when explicitly enabled) HuggingFace ``datasets`` with a pinned
    revision.  Credentials for a private endpoint are never read here.
    """

    if not isinstance(revision, str) or not revision.strip():
        raise DatasetUnavailable(
            "dataset.revision must be a pinned revision (for example a dataset commit hash)"
        )
    if not isinstance(split, str) or not split.strip():
        raise DatasetUnavailable("dataset.split must be a non-empty string")
    name = name or DEFAULT_DATASET_NAME

    source_path = _resolve_local_source(source=source, cache_dir=cache_dir, name=name)
    if source_path is not None:
        records = _load_records(source_path)
        dataset = _dataset_from_records(
            records=records,
            name=name,
            revision=revision,
            split=split,
            source=str(source_path),
            source_sha256=sha256_file(source_path),
        )
        return dataset

    if allow_hf_download:
        return _load_from_huggingface(name=name, revision=revision, split=split)

    expected = _expected_cache_paths(cache_dir=cache_dir, name=name)
    raise DatasetUnavailable(
        "pinned SWE-bench Verified dataset is not available locally. Provide "
        "[dataset].source (a local JSON/JSONL export of the pinned split) or place one of "
        f"{[str(path) for path in expected]} in the cache directory, or set "
        "[dataset].allow_hf_download=true with network access and a pinned revision. "
        "The dataset is never fabricated or substituted at runtime."
    )


def _resolve_local_source(
    *,
    source: str | Path | None,
    cache_dir: str | Path | None,
    name: str,
) -> Path | None:
    if source is not None:
        path = Path(os.path.expandvars(os.path.expanduser(str(source))))
        if not path.exists():
            raise DatasetUnavailable(f"configured dataset source does not exist: {path}")
        return path.resolve()
    for candidate in _expected_cache_paths(cache_dir=cache_dir, name=name):
        if candidate.exists():
            return candidate.resolve()
    return None


def _expected_cache_paths(*, cache_dir: str | Path | None, name: str) -> list[Path]:
    if cache_dir is None:
        return []
    root = Path(os.path.expandvars(os.path.expanduser(str(cache_dir))))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return [
        root / f"{safe}.jsonl",
        root / f"{safe}.json",
        root / "swe_bench_verified.jsonl",
        root / "swe_bench_verified.json",
        root / "verified.jsonl",
        root / "verified.json",
    ]


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetUnavailable(f"cannot read dataset source {path}: {exc}") from exc
    stripped = text.lstrip()
    records: list[Any]
    if stripped.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetUnavailable(f"dataset source {path} is not valid JSON: {exc}") from exc
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetUnavailable(
                    f"dataset source {path} line {line_number} is not valid JSON: {exc}"
                ) from exc
    if not isinstance(records, list) or not records:
        raise DatasetUnavailable(f"dataset source {path} contains no records")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DatasetUnavailable(f"dataset source {path} record {index} is not an object")
    return records


def _dataset_from_records(
    *,
    records: Iterable[Mapping[str, Any]],
    name: str,
    revision: str,
    split: str,
    source: str,
    source_sha256: str,
) -> SwebenchDataset:
    instances: dict[str, SwebenchInstance] = {}
    for index, record in enumerate(records):
        instance = _instance_from_record(record, index=index)
        if instance.task_id in instances:
            raise DatasetUnavailable(f"duplicate SWE-bench instance id: {instance.task_id!r}")
        instances[instance.task_id] = instance
    return SwebenchDataset(
        name=name,
        revision=revision,
        split=split,
        source=source,
        source_sha256=source_sha256,
        instances=instances,
    )


def _instance_from_record(record: Mapping[str, Any], *, index: int) -> SwebenchInstance:
    task_id = str(record.get("instance_id") or record.get("task_id") or "").strip()
    if not task_id:
        raise DatasetUnavailable(f"dataset record {index} is missing instance_id")
    repo = str(record.get("repo") or "").strip()
    base_commit = str(record.get("base_commit") or "").strip()
    problem = str(record.get("problem_statement") or "").strip()
    if not repo or "/" not in repo:
        raise DatasetUnavailable(f"dataset record {task_id!r} has an invalid repo")
    if not base_commit:
        raise DatasetUnavailable(f"dataset record {task_id!r} has no base_commit")
    if not problem:
        raise DatasetUnavailable(f"dataset record {task_id!r} has an empty problem_statement")
    return SwebenchInstance(
        task_id=task_id,
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem,
        version=_optional_text(record.get("version")),
        hints_text=_optional_text(record.get("hints_text")),
        patch=_optional_text(record.get("patch")),
        test_patch=_optional_text(record.get("test_patch")),
        fail_to_pass=_test_id_tuple(record.get("FAIL_TO_PASS")),
        pass_to_pass=_test_id_tuple(record.get("PASS_TO_PASS")),
        environment_setup_commit=_optional_text(record.get("environment_setup_commit")),
        created_at=_optional_text(record.get("created_at")),
        difficulty=_optional_text(record.get("difficulty")),
        extra={
            key: value
            for key, value in record.items()
            if key
            not in {
                "instance_id",
                "task_id",
                "repo",
                "base_commit",
                "problem_statement",
                "version",
                "hints_text",
                "patch",
                "test_patch",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
                "environment_setup_commit",
                "created_at",
                "difficulty",
            }
        },
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _test_id_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return (stripped,)
        return _test_id_tuple(parsed)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _load_from_huggingface(*, name: str, revision: str, split: str) -> SwebenchDataset:
    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DatasetUnavailable(
            "allow_hf_download=true requires the 'datasets' package to be installed"
        ) from exc
    try:  # pragma: no cover - requires network
        dataset = datasets.load_dataset(name, revision=revision, split=split)
        records = [dict(row) for row in dataset]
    except Exception as exc:  # pragma: no cover - requires network
        raise DatasetUnavailable(
            f"cannot load {name}@{revision} split {split!r} from HuggingFace: {exc}"
        ) from exc
    return _dataset_from_records(
        records=records,
        name=name,
        revision=revision,
        split=split,
        source=f"huggingface:{name}@{revision}:{split}",
        source_sha256=sha256_json({"name": name, "revision": revision, "split": split}),
    )


def assert_no_protected_fields(payload: Mapping[str, Any]) -> None:
    """Raise if an agent-visible payload leaks evaluator-only fields."""

    leaked = sorted(key for key in PROTECTED_FIELDS if key in payload and payload.get(key))
    if leaked:
        raise ValueError(f"agent-visible payload leaks evaluator-only fields: {leaked}")
