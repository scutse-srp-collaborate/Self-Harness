#!/usr/bin/env python3
"""Prepare pinned SWE-bench Verified data and pilot repository clones.

Default data root: ``../Self-Harness-data`` (relative to this repository),
overridable with the non-sensitive ``SELF_HARNESS_DATA_ROOT`` environment
variable.  Nothing large is written into the Git working tree.

The dataset revision is pinned explicitly.  The script refuses to proceed if the
remote dataset HEAD no longer matches the pinned revision, and it validates that
the export covers exactly the frozen 100-task manifest (67 train / 33 heldout,
unique, disjoint, manifest order preserved).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("eval/src", "eval/fixtures"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from self_harness_eval.manifest import (  # noqa: E402
    EXPECTED_HELDOUT,
    EXPECTED_TOTAL,
    EXPECTED_TRAIN,
    ManifestError,
    load_swebench_manifest,
)
from self_harness_eval.pilot import (  # noqa: E402
    PILOT_MAX_TASKS,
    PILOT_SELECTION_RULE,
    select_pilot_tasks,
)

DATASET_ID = "princeton-nlp/SWE-bench_Verified"
# Pinned from the HuggingFace dataset repository (last modified 2025-02-18).
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
HF_API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
PARQUET_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/"
    "data/test-00000-of-00001.parquet"
)
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "Self-Harness-data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_root_from_environment(override: Path | None = None) -> Path:
    raw = override or os.environ.get("SELF_HARNESS_DATA_ROOT")
    root = Path(raw).expanduser() if raw else DEFAULT_DATA_ROOT
    return root.resolve()


def dataset_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "datasets": root / "datasets",
        "cache": root / "cache",
        "huggingface": root / "cache" / "huggingface",
        "swebench_cache": root / "cache" / "swebench",
        "repos": root / "repos",
        "runs": root / "runs",
        "logs": root / "logs",
        "json": root / "datasets" / "swe_bench_verified_100.json",
        "jsonl": root / "datasets" / "swe_bench_verified_100.jsonl",
        "meta": root / "datasets" / "swe_bench_verified_100.meta.json",
        "pilot_ids": root / "datasets" / "pilot_5_task_ids.json",
    }


def ensure_directories(root: Path) -> dict[str, Path]:
    paths = dataset_paths(root)
    for key in ("datasets", "cache", "huggingface", "swebench_cache", "repos", "runs", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def fetch_remote_revision() -> str:
    request = urllib.request.Request(HF_API_URL, headers={"User-Agent": "self-harness"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    revision = str(payload.get("sha") or "").strip()
    if not revision:
        raise RuntimeError("HuggingFace dataset API response has no revision sha")
    return revision


def download_parquet(destination: Path, *, revision: str = DATASET_REVISION) -> Path:
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/"
        "data/test-00000-of-00001.parquet"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "self-harness"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return destination


def read_records(parquet_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    if not records:
        raise RuntimeError(f"dataset parquet is empty: {parquet_path}")
    return records


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = json.dumps(value)
    if "instance_id" not in normalized and "task_id" in normalized:
        normalized["instance_id"] = normalized["task_id"]
    return normalized


def export_manifest_dataset(
    *,
    root: Path,
    manifest_path: Path | None = None,
    revision: str = DATASET_REVISION,
    force: bool = False,
    verify_remote: bool = True,
) -> dict[str, Any]:
    manifest = load_swebench_manifest(manifest_path)
    paths = ensure_directories(root)
    if verify_remote:
        remote_revision = fetch_remote_revision()
        if remote_revision != revision:
            raise RuntimeError(
                "pinned SWE-bench Verified revision no longer matches HuggingFace: "
                f"pinned={revision} remote={remote_revision}"
            )
    parquet_path = paths["huggingface"] / f"{DATASET_ID.replace('/', '__')}-{revision}.parquet"
    if force and parquet_path.exists():
        parquet_path.unlink()
    if force or not paths["json"].exists():
        download_parquet(parquet_path, revision=revision)
        records = [normalize_record(record) for record in read_records(parquet_path)]
        by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            task_id = str(record.get("instance_id") or "")
            if not task_id:
                raise RuntimeError("dataset record without instance_id")
            if task_id in by_id:
                raise RuntimeError(f"duplicate dataset record: {task_id}")
            by_id[task_id] = record
        manifest_ids = list(manifest.task_ids)
        missing = [task_id for task_id in manifest_ids if task_id not in by_id]
        extra = sorted(set(by_id) - set(manifest_ids))
        if missing:
            raise RuntimeError(
                f"pinned dataset is missing {len(missing)} manifest task ids: {missing[:10]}"
            )
        # The pinned Verified test split contains more tasks than the stratified
        # 100-task manifest; the export deliberately covers exactly the manifest.
        ordered = [by_id[task_id] for task_id in manifest_ids]
        payload = json.dumps(ordered, indent=2) + "\n"
        paths["json"].write_text(payload, encoding="utf-8")
        paths["jsonl"].write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in ordered) + "\n",
            encoding="utf-8",
        )
        counts = {
            "total": len(ordered),
            "train": sum(1 for task_id in manifest_ids if task_id in manifest.held_in_task_ids),
            "heldout": sum(1 for task_id in manifest_ids if task_id in manifest.held_out_task_ids),
        }
        if counts != {"total": EXPECTED_TOTAL, "train": EXPECTED_TRAIN, "heldout": EXPECTED_HELDOUT}:
            raise RuntimeError(f"exported split counts are wrong: {counts}")
        meta = {
            "format": "self_harness.swebench_dataset_export.v1",
            "dataset_id": DATASET_ID,
            "revision": revision,
            "revision_source": "HuggingFace dataset repository commit SHA (pinned in this repo)",
            "paper_original_revision": None,
            "paper_original_revision_note": (
                "The paper does not publish a dataset revision SHA for this experiment; this "
                "pinned commit is the reproduction configuration, not a claimed paper revision."
            ),
            "source_parquet": str(parquet_path),
            "source_parquet_sha256": sha256_file(parquet_path),
            "export_json": str(paths["json"]),
            "export_json_sha256": sha256_file(paths["json"]),
            "export_jsonl": str(paths["jsonl"]),
            "export_jsonl_sha256": sha256_file(paths["jsonl"]),
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.sha256,
            "manifest_task_ids_sha256": manifest.task_ids_sha256,
            "counts": counts,
            "source_split_total": len(by_id),
            "source_task_ids_outside_manifest": len(extra),
            "export_covers_exactly_manifest": True,
            "record_order": "manifest held_in_task_ids followed by held_out_task_ids",
            "repo_distribution_in_100": manifest.repo_distribution_in_100,
            "domain_distribution_in_100": manifest.distributions()["by_domain_total"],
        }
        paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if meta.get("revision") != revision:
        raise RuntimeError(
            f"existing dataset export has revision {meta.get('revision')!r}, expected {revision!r}; "
            "rerun with --force-dataset"
        )
    return {"meta": meta, "paths": {key: str(value) for key, value in paths.items()}}


def write_pilot_task_list(root: Path, *, max_tasks: int = PILOT_MAX_TASKS) -> dict[str, Any]:
    manifest = load_swebench_manifest()
    tasks = select_pilot_tasks(manifest, max_tasks=max_tasks)
    payload = {
        "format": "self_harness.swebench_pilot_tasks.v1",
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "selection_rule": PILOT_SELECTION_RULE,
        "max_tasks": max_tasks,
        "task_ids": [task["task_id"] for task in tasks],
        "tasks": tasks,
        "heldout_task_ids_used": [],
    }
    paths = ensure_directories(root)
    paths["pilot_ids"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _run_git(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def ensure_repositories(
    root: Path,
    manifest,
    tasks: list[dict[str, Any]],
    *,
    derive_from_list: bool = True,
) -> list[dict[str, Any]]:
    del manifest, derive_from_list  # task selection is already train-only
    repos = sorted({str(task["repository"]) for task in tasks})
    results: list[dict[str, Any]] = []
    for repo in repos:
        target = root / "repos" / repo
        entry: dict[str, Any] = {"repository": repo, "path": str(target), "status": "missing"}
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            clone = _run_git(
                ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(target)],
                cwd=root / "repos",
            )
            if clone.returncode != 0:
                entry["error"] = f"git clone failed: {clone.stderr[-800:]}"
                results.append(entry)
                continue
        if not (target / ".git").is_dir():
            entry["error"] = "not a git repository"
            results.append(entry)
            continue
        entry["status"] = "cloned"
        base_commits = sorted(
            {
                str(task.get("base_commit") or "")
                for task in tasks
                if task["repository"] == repo
            }
        )
        missing: list[str] = []
        for commit in base_commits:
            check = _run_git(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=target)
            if check.returncode != 0:
                fetch = _run_git(["git", "fetch", "--quiet", "origin", commit], cwd=target)
                check = _run_git(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=target)
                if check.returncode != 0:
                    missing.append(commit)
                    entry["fetch_error"] = (fetch.stderr or "")[-800:]
        entry["base_commits"] = base_commits
        entry["missing_base_commits"] = missing
        entry["status"] = "ready" if not missing else "missing_commits"
        results.append(entry)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare pinned SWE-bench Verified data and pilot repos.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--pilot-tasks", type=int, default=PILOT_MAX_TASKS)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-repos", action="store_true")
    parser.add_argument("--skip-remote-verify", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = data_root_from_environment(args.data_root)
    try:
        if args.skip_download and not dataset_paths(root)["json"].exists():
            raise RuntimeError("--skip-download requested but no local dataset export exists")
        dataset_info = export_manifest_dataset(
            root=root,
            manifest_path=args.manifest,
            revision=args.revision,
            force=args.force_dataset,
            verify_remote=not args.skip_remote_verify,
        )
        pilot = write_pilot_task_list(root, max_tasks=args.pilot_tasks)
        manifest = load_swebench_manifest(args.manifest)
        records = json.loads(dataset_paths(root)["json"].read_text(encoding="utf-8"))
        by_id = {str(record["instance_id"]): record for record in records}
        selected = []
        for task in pilot["tasks"]:
            record = by_id[task["task_id"]]
            selected.append(
                {
                    "task_id": task["task_id"],
                    "repository": task["repository"],
                    "domain": task["domain"],
                    "base_commit": record.get("base_commit"),
                }
            )
        repo_results = []
        if not args.skip_repos:
            repo_results = ensure_repositories(root, manifest, selected)
    except (RuntimeError, ManifestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "data_root": str(root),
        "dataset": dataset_info["meta"],
        "pilot_tasks": pilot["task_ids"],
        "pilot_task_details": pilot["tasks"],
        "repositories": repo_results,
        "status": (
            "ready"
            if all(item.get("status") == "ready" for item in repo_results) or not repo_results
            else "blocked"
        ),
    }
    (root / "logs").mkdir(parents=True, exist_ok=True)
    output = root / "logs" / "prepare_swebench_data.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"data_root: {root}")
        print(f"dataset revision: {summary['dataset']['revision']}")
        print(f"dataset export sha256: {summary['dataset']['export_jsonl_sha256']}")
        print("pilot tasks:")
        for task in pilot["tasks"]:
            print(f"  {task['task_id']}  repo={task['repository']} domain={task['domain']}")
        for repo in repo_results:
            print(f"repo {repo['repository']}: {repo['status']}")
        print(f"wrote {output}")
    return 0 if summary["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
