"""Deterministic hashing helpers used for run provenance and reuse checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

IGNORED_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
    }
)
IGNORED_FILE_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd"})


class HashingError(ValueError):
    """Raised when a path cannot be turned into a stable provenance hash."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise HashingError(f"cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Return a stable JSON encoding suitable for hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def secure_relative_path(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` as a POSIX string, or raise."""

    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise HashingError(f"path {path} is outside run root {root}") from exc
    return relative.as_posix() or "."


def _is_ignored(relative: Path) -> bool:
    if any(part in IGNORED_DIR_NAMES for part in relative.parts[:-1]):
        return True
    if relative.parts and relative.parts[0] in IGNORED_DIR_NAMES:
        return True
    return relative.suffix in IGNORED_FILE_SUFFIXES


def iter_tree_files(root: Path, *, ignore_ignored: bool = True) -> list[str]:
    """List regular files under ``root`` as sorted POSIX relative paths."""

    root = Path(root)
    if not root.exists():
        raise HashingError(f"snapshot root does not exist: {root}")
    if not root.is_dir():
        raise HashingError(f"snapshot root is not a directory: {root}")
    results: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() and not path.exists():
            continue
        relative = path.relative_to(root)
        if ignore_ignored and _is_ignored(relative):
            continue
        results.append(relative.as_posix())
    return sorted(results)


def tree_file_hashes(root: Path, *, ignore_ignored: bool = True) -> dict[str, str]:
    root = Path(root)
    return {
        relative: sha256_file(root / relative)
        for relative in iter_tree_files(root, ignore_ignored=ignore_ignored)
    }


def snapshot_tree_hash(files: Mapping[str, str]) -> str:
    """Hash a ``{relative_path: sha256}`` map deterministically."""

    lines = [f"{digest}  {path}" for path, digest in sorted(files.items())]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    return sha256_text(payload)


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    """Hash every path, keyed by its basename (used for small artifact sets)."""

    result: dict[str, str] = {}
    for path in paths:
        candidate = Path(path)
        result[candidate.name] = sha256_file(candidate)
    return result


def hash_tree_or_file(path: Path) -> str:
    """Hash a directory tree (``snapshot_tree_hash``) or a single file."""

    candidate = Path(path)
    if candidate.is_dir():
        return snapshot_tree_hash(tree_file_hashes(candidate))
    return sha256_file(candidate)
