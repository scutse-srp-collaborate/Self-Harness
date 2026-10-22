#!/usr/bin/env python3
"""Read-only preflight for a real SWE-bench pilot.

Checks (without ever printing or persisting a credential value):
API key presence, OpenAI-compatible endpoint reachability, model configuration,
DeepAgents/LangChain/SWE-bench/Docker availability, dataset export integrity,
pilot task selection against the frozen train split, repository base commits and
disk space.  The result is written to ``<data-root>/logs/preflight.json``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("eval/src", "eval/scripts", "eval/fixtures"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from self_harness_eval.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

from self_harness_eval.manifest import load_swebench_manifest  # noqa: E402
from self_harness_eval.pilot import (  # noqa: E402
    PILOT_MAX_TASKS,
    select_pilot_tasks,
    validate_pilot_task_ids,
)
from prepare_swebench_data import (  # noqa: E402
    DATASET_REVISION,
    data_root_from_environment,
    dataset_paths,
    sha256_file,
)

API_KEY_ENV = "SELF_HARNESS_MODEL_API_KEY"
MODEL_ENV = "SELF_HARNESS_MODEL"
BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.qadg.xyz/v1"


def _check(name: str, ok: bool, detail: Any = None, *, blocking: bool = False) -> dict[str, Any]:
    return {
        "check": name,
        "ok": bool(ok),
        "blocking": bool(blocking) and not bool(ok),
        "detail": detail,
    }


def _endpoint_check(
    *, base_url: str, api_key_present: bool, model: str, api_key: str | None
) -> dict[str, Any]:
    models_url = base_url.rstrip("/") + "/models"
    started = time.time()
    headers = {"User-Agent": "self-harness"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    model_available: bool | None = None
    try:
        request = urllib.request.Request(models_url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
                body = response.read(500_000)
                try:
                    listing = json.loads(body)
                    entries = listing.get("data") if isinstance(listing, dict) else None
                    if isinstance(entries, list):
                        model_available = any(
                            str(entry.get("id") or "") == model
                            for entry in entries
                            if isinstance(entry, dict)
                        )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    model_available = None
        except urllib.error.HTTPError as exc:
            status = exc.code
            if exc.fp:
                exc.read(200_000)
        return {
            "check": "endpoint_reachability",
            "ok": True,
            "blocking": False,
            "detail": {
                "url": models_url,
                "http_status": status,
                "latency_s": round(time.time() - started, 3),
                "api_key_present": api_key_present,
                "auth_checked": bool(api_key),
                "model": model,
                "model_available": model_available,
                "note": (
                    "Endpoint is reachable. Without an API key only an unauthorized response can "
                    "be observed; authenticated model availability is not verified."
                    if not api_key_present
                    else "Endpoint reachable and authenticated model listing was checked; "
                    "the actual chat completion is made by the pilot."
                ),
            },
        }
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return {
            "check": "endpoint_reachability",
            "ok": False,
            "blocking": True,
            "detail": f"{exc.__class__.__name__}: {exc}",
        }


def _docker_check(image: str) -> dict[str, Any]:
    if shutil.which("docker") is None:
        return _check("docker_runtime", False, "docker CLI not found", blocking=True)
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            return _check("docker_runtime", False, completed.stderr[-500:], blocking=True)
        image_check = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return _check(
            "docker_runtime",
            image_check.returncode == 0,
            {
                "server_version": completed.stdout.strip(),
                "agent_image": image,
                "agent_image_id": image_check.stdout.strip() or None,
            },
            blocking=image_check.returncode != 0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _check("docker_runtime", False, str(exc), blocking=True)


def _package_check(name: str) -> dict[str, Any]:
    found = importlib.util.find_spec(name) is not None
    version = None
    if found:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return _check(f"package:{name}", found, version, blocking=not found)


def _dataset_check(root: Path) -> dict[str, Any]:
    paths = dataset_paths(root)
    if not paths["jsonl"].exists() or not paths["meta"].exists():
        return _check("dataset_export", False, "dataset export/meta is missing", blocking=True)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    actual = sha256_file(paths["jsonl"])
    ok = (
        meta.get("revision") == DATASET_REVISION
        and meta.get("export_jsonl_sha256") == actual
        and meta.get("counts") == {"total": 100, "train": 67, "heldout": 33}
    )
    return _check(
        "dataset_export",
        ok,
        {
            "revision": meta.get("revision"),
            "expected_revision": DATASET_REVISION,
            "export_sha256": actual,
            "recorded_export_sha256": meta.get("export_jsonl_sha256"),
            "counts": meta.get("counts"),
        },
        blocking=not ok,
    )


def _pilot_check(root: Path, *, max_tasks: int) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    manifest = load_swebench_manifest()
    pilot_payload_path = dataset_paths(root)["pilot_ids"]
    if pilot_payload_path.exists():
        payload = json.loads(pilot_payload_path.read_text(encoding="utf-8"))
        task_ids = [str(item) for item in payload.get("task_ids") or []]
        tasks = validate_pilot_task_ids(manifest, task_ids, max_tasks=max_tasks)
        source = "pilot_5_task_ids.json"
    else:
        tasks = select_pilot_tasks(manifest, max_tasks=max_tasks)
        source = "deterministic_selection"
    heldout = set(manifest.held_out_task_ids)
    ok = (
        bool(tasks)
        and len(tasks) <= max_tasks
        and len({task["task_id"] for task in tasks}) == len(tasks)
        and not ({task["task_id"] for task in tasks} & heldout)
        and all(task["split"] == "train" for task in tasks)
    )
    return (
        _check(
            "pilot_tasks",
            ok,
            {
                "source": source,
                "count": len(tasks),
                "max_tasks": max_tasks,
                "task_ids": [task["task_id"] for task in tasks],
                "repositories": [task["repository"] for task in tasks],
                "domains": [task["domain"] for task in tasks],
                "heldout_used": [],
            },
            blocking=not ok,
        ),
        tasks,
        manifest,
    )


def _repository_check(root: Path, manifest: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    blocking = False
    for task in tasks:
        repo = task["repository"]
        target = root / "repos" / repo
        commit = None
        records_jsonl = dataset_paths(root)["jsonl"]
        # Base commit comes from the exported record.
        if records_jsonl.exists():
            import json as _json

            for line in records_jsonl.read_text(encoding="utf-8").splitlines():
                record = _json.loads(line)
                if record.get("instance_id") == task["task_id"]:
                    commit = str(record.get("base_commit") or "")
                    break
        exists = (target / ".git").is_dir()
        commit_ok = False
        if exists and commit:
            check = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=target,
                capture_output=True,
                text=True,
                check=False,
            )
            commit_ok = check.returncode == 0
        if not (exists and commit_ok):
            blocking = True
        records.append(
            {
                "task_id": task["task_id"],
                "repository": repo,
                "clone_path": str(target),
                "clone_present": exists,
                "base_commit": commit,
                "base_commit_present": commit_ok,
            }
        )
    return _check(
        "pilot_repositories",
        not blocking,
        {"tasks": records},
        blocking=blocking,
    )


def _disk_check(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    return _check(
        "disk_space",
        free_gb >= 20,
        {"free_gb": round(free_gb, 1), "minimum_gb": 20},
        blocking=free_gb < 20,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only SWE-bench pilot preflight.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--max-tasks", type=int, default=PILOT_MAX_TASKS)
    parser.add_argument("--agent-image", default=os.environ.get("SELF_HARNESS_AGENT_IMAGE", "python:3.12"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = data_root_from_environment(args.data_root)
    model = os.environ.get(MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_url = os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    api_key = os.environ.get(API_KEY_ENV, "").strip() or None
    api_key_present = bool(api_key)
    checks: list[dict[str, Any]] = [
        _check(
            "api_key_present",
            api_key_present,
            {
                "environment_variable": API_KEY_ENV,
                "value_logged": False,
                "note": (
                    "present; value is never read into this report"
                    if api_key_present
                    else "missing; real model calls must not run"
                ),
            },
            blocking=not api_key_present,
        ),
        _check(
            "model_configuration",
            bool(model) and bool(base_url),
            {"model": model, "base_url": base_url, "provider": "openai_compatible"},
            blocking=False,
        ),
        _endpoint_check(
            base_url=base_url,
            api_key_present=api_key_present,
            model=model,
            api_key=api_key,
        ),
        _check(
            "python_executable",
            True,
            {
                "path": sys.executable,
                "version": sys.version.split()[0],
                "recommended_adapter_python": str(REPO_ROOT / ".venv" / "bin" / "python"),
            },
        ),
    ]
    for package in (
        "deepagents",
        "langchain",
        "langchain_openai",
        "openai",
        "swebench",
        "docker",
        "pyarrow",
    ):
        checks.append(_package_check(package))
    checks.append(_docker_check(args.agent_image))
    checks.append(_dataset_check(root))
    pilot_check, tasks, manifest = _pilot_check(root, max_tasks=args.max_tasks)
    checks.append(pilot_check)
    checks.append(_repository_check(root, manifest, tasks))
    checks.append(_disk_check(root))

    blockers = [check["check"] for check in checks if check["blocking"]]
    report = {
        "format": "self_harness.swebench_preflight.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root": str(root),
        "ready_for_real_pilot": not blockers,
        "blockers": blockers,
        "checks": checks,
        "scope": {
            "benchmark": "SWE-bench_Verified",
            "pilot_max_tasks": args.max_tasks,
            "heldout_run": False,
            "full_benchmark": False,
        },
        "limitations": [
            "Preflight is read-only and does not make a real model call.",
            "Official SWE-bench evaluator images are built/validated per instance by the evaluator "
            "during the pilot; this preflight only checks package/Docker availability.",
        ],
    }
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = logs / "preflight.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"preflight written: {output}")
        print(f"ready_for_real_pilot: {report['ready_for_real_pilot']}")
        for blocker in blockers:
            print(f"BLOCKER: {blocker}")
        for check in checks:
            status = "OK" if check["ok"] else ("BLOCKED" if check["blocking"] else "WARN")
            print(f"  [{status}] {check['check']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
