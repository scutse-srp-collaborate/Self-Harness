#!/usr/bin/env python3
"""Preflight and task-image preparation for the environment-fixed SWE-bench pilot.

This script is intentionally read-only with respect to the reference run.  It
validates that the new run keeps every experimental condition fixed, resolves
the official SWE-bench task images from the pinned evaluator (not guessed), pulls
or verifies them, probes the image's dependency environment, and writes
``task_images.json`` on which the DeepAgents adapter depends.

It never reads or writes the API key value into any artifact.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("eval/src", "eval/scripts"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from self_harness_eval.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

from self_harness_eval.config import load_eval_config  # noqa: E402
from self_harness_eval.hashing import hash_tree_or_file, sha256_file  # noqa: E402

API_KEY_ENV = "SELF_HARNESS_MODEL_API_KEY"
EXPECTED_MODEL = "deepseek-flash"
EXPECTED_BASE_URL = "https://api.qadg.xyz/v1"
EXPECTED_DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
EXPECTED_HARNESS_SHA = "dbe6e1a76efdaa7799f8db75bbc2473f960662da864552709b352156cdf66abe"
TASK_REQUIRED_MODULES = {
    "astropy__astropy-13398": ("numpy", "erfa"),
    "django__django-14017": ("django",),
    "matplotlib__matplotlib-24627": ("matplotlib", "numpy"),
    "pydata__xarray-6461": ("xarray", "numpy"),
    "sphinx-doc__sphinx-8548": ("docutils",),
}
EXPECTED_TASK_IDS = (
    "astropy__astropy-13398",
    "django__django-14017",
    "matplotlib__matplotlib-24627",
    "pydata__xarray-6461",
    "sphinx-doc__sphinx-8548",
)
EXPECTED_MANIFEST_SHA = "e8358cc6d30986a646cdb06d4532c7bfe7e39a1495540974f5d8d65c1ca4b2ed"
EXPECTED_DATASET_JSONL_SHA = "140def9433d4586ef25281480730b3eb0e36bce152bb10f38b871ecfadc88c6b"
DEFAULT_REFERENCE_RUN = Path("/harddisk/codehd/Self-Harness-data/runs/swebench-pilot5-002")


def _run(argv: list[str], *, timeout: float = 120, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
        env=env,
    )


def _docker_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    for key in ("HOME", "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _docker(argv: list[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return _run(["docker", *argv], timeout=timeout, env=_docker_env())


def _image_inspect(image: str) -> dict[str, Any] | None:
    completed = _docker(["image", "inspect", image], timeout=60)
    if completed.returncode != 0:
        return None
    try:
        items = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or not items:
        return None
    item = items[0]
    return item if isinstance(item, dict) else None


def _pull(image: str) -> tuple[bool, str]:
    completed = _docker(["pull", image], timeout=1800)
    output = ((completed.stdout or "") + (completed.stderr or ""))[-4000:]
    (Path("/tmp") / f"pull-{abs(hash(image))}.log").write_text(output, encoding="utf-8")
    return completed.returncode == 0, output


def _probe_image(
    image: str, base_commit: str, required_modules: tuple[str, ...] = ()
) -> dict[str, Any]:
    script = r'''
import importlib.util
import json
import os
import shutil
import sys

def spec(name):
    return importlib.util.find_spec(name) is not None

payload = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "cwd": os.getcwd(),
    "testbed_exists": os.path.isdir("/testbed"),
    "testbed_git": os.path.isdir("/testbed/.git"),
    "path": os.environ.get("PATH", ""),
    "conda_envs": sorted(os.listdir("/opt/miniconda3/envs")) if os.path.isdir("/opt/miniconda3/envs") else [],
    "has_conda": os.path.isfile("/opt/miniconda3/bin/conda"),
    "modules": {name: spec(name) for name in ("pytest", "numpy", "pyerfa", "docutils", "django", "matplotlib", "xarray")},
    "pytest": shutil.which("pytest"),
    "testbed_pytest_exists": os.path.isfile("/opt/miniconda3/envs/testbed/bin/pytest"),
    "testbed_python_exists": os.path.isfile("/opt/miniconda3/envs/testbed/bin/python"),
}
print(json.dumps(payload))
'''
    wrapped_script = "python - <<'PY'\n" + script + "\nPY"
    completed = _docker(
        ["run", "--rm", "--network", "none", "--entrypoint", "/bin/bash", image, "-lc", wrapped_script],
        timeout=180,
    )
    payload: dict[str, Any] = {
        "returncode": completed.returncode,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-1000:],
    }
    if completed.returncode == 0 and completed.stdout.strip():
        try:
            payload.update(json.loads(completed.stdout.strip().splitlines()[-1]))
        except json.JSONDecodeError:
            pass
    payload["testbed_head"] = None
    if payload.get("testbed_exists"):
        head = _docker(
            ["run", "--rm", "--network", "none", "--entrypoint", "/bin/bash", image, "-lc", "git -C /testbed rev-parse HEAD"],
            timeout=120,
        )
        if head.returncode == 0:
            payload["testbed_head"] = head.stdout.strip().splitlines()[-1] if head.stdout.strip() else None
        payload["base_commit_match"] = payload.get("testbed_head") == base_commit
    # Derive a dependency PATH rather than relying on a login shell to activate conda.
    path = str(payload.get("path") or "")
    testbed_bin = "/opt/miniconda3/envs/testbed/bin"
    container_env: dict[str, str] = {"HOME": "/root"}
    if payload.get("testbed_python_exists"):
        container_env["PATH"] = f"{testbed_bin}:/opt/miniconda3/bin:{path}"
        container_env["CONDA_PREFIX"] = "/opt/miniconda3/envs/testbed"
        container_env["CONDA_DEFAULT_ENV"] = "testbed"
        payload["python"] = f"{testbed_bin}/python"
        if payload.get("testbed_pytest_exists"):
            payload["pytest"] = f"{testbed_bin}/pytest"
    else:
        container_env["PATH"] = path
    payload["container_env"] = container_env
    env_args: list[str] = []
    for key, value in sorted(container_env.items()):
        env_args.extend(["--env", f"{key}={value}"])
    # Check whether the stricter read-only root + tmpfs layout keeps the image usable.
    ro = _docker(
        [
            "run", "--rm", "--network", "none", "--read-only",
            *env_args,
            "--tmpfs", "/tmp:rw,exec,size=1g", "--tmpfs", "/root:rw,size=256m",
            "--tmpfs", "/run:rw,size=128m", "--entrypoint", "/bin/bash", image, "-lc",
            'cd /testbed && python -c "import sys; print(sys.version.split()[0])"',
        ],
        timeout=180,
    )
    payload["read_only_root_ok"] = ro.returncode == 0
    if ro.returncode != 0:
        payload["read_only_root_error"] = ((ro.stdout or "") + (ro.stderr or ""))[-1000:]
    # Verify the dependency environment with the exact env used by the agent.
    dep = _docker(
        [
            "run", "--rm", "--network", "none", *env_args,
            "--entrypoint", "/bin/bash", image, "-lc",
            'python -c "import sys, json; print(json.dumps({sys.executable: sys.version.split()[0]}))"',
        ],
        timeout=180,
    )
    payload["dependency_probe_ok"] = dep.returncode == 0
    payload["dependency_probe_output"] = ((dep.stdout or "") + (dep.stderr or ""))[-1000:]
    modules: dict[str, bool] = {}
    for module in required_modules:
        check = _docker(
            [
                "run", "--rm", "--network", "none", *env_args,
                "--entrypoint", "/bin/bash", image, "-lc",
                f"python -c 'import {module}'",
            ],
            timeout=120,
        )
        modules[module] = check.returncode == 0
    payload["required_modules"] = modules
    payload["required_modules_ok"] = all(modules.values()) if modules else True
    return payload


def _manifest_checks(config: Any, reference: Any, blockers: list[str], report: dict[str, Any]) -> None:
    manifest_path = Path(config.dataset.manifest_path)
    if not manifest_path.is_file():
        blockers.append(f"manifest missing: {manifest_path}")
        return
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = sha256_file(manifest_path)
    report["manifest_sha256_matches_reference"] = report["manifest_sha256"] == EXPECTED_MANIFEST_SHA
    if report["manifest_sha256"] != EXPECTED_MANIFEST_SHA:
        blockers.append("current manifest sha256 does not match the reference pilot")
    reference_manifest = str((reference.get("pilot") or {}).get("manifest_sha256") or "")
    report["reference_manifest_sha256"] = reference_manifest
    if reference_manifest and reference_manifest != report["manifest_sha256"]:
        blockers.append("reference run manifest sha256 changed")

    source = Path(config.dataset.source)
    report["dataset_source_path"] = str(source)
    if source.is_file():
        report["dataset_source_sha256"] = sha256_file(source)
    else:
        blockers.append(f"dataset export missing: {source}")


def _endpoint_check(base_url: str, model: str, blockers: list[str], report: dict[str, Any]) -> None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    report["api_key_present"] = bool(key)
    if not key:
        blockers.append(f"{API_KEY_ENV} is not set")
        return
    headers = {"User-Agent": "self-harness-preflight"}
    headers["Authorization"] = f"Bearer {key}"
    started = time.time()
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/models", headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
            body = response.read(500_000)
        available: bool | None = None
        try:
            listing = json.loads(body)
            data = listing.get("data") if isinstance(listing, dict) else None
            if isinstance(data, list):
                available = any(
                    isinstance(item, dict) and str(item.get("id") or "") == model for item in data
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            available = None
        report.update(
            {
                "endpoint_url": base_url.rstrip("/") + "/models",
                "http_status": status,
                "latency_s": round(time.time() - started, 3),
                "model": model,
                "model_available": available,
            }
        )
        if not available:
            blockers.append(f"model {model!r} is not listed by the authenticated endpoint")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        report["endpoint_error"] = f"{exc.__class__.__name__}: {str(exc)[:300]}"
        blockers.append("OpenAI-compatible endpoint is not reachable/authenticated")


def _docker_checks(blockers: list[str], report: dict[str, Any]) -> None:
    version = _docker(["version", "--format", "client={{.Client.Version}} server={{.Server.Version}}"], timeout=30)
    report["docker_version_returncode"] = version.returncode
    report["docker_version"] = (version.stdout or version.stderr or "").strip()[:500]
    if version.returncode != 0:
        blockers.append("docker version failed")
        return
    smoke = _docker(
        ["run", "--rm", "python:3.12", "python", "-c", "print('docker-smoke-ok')"], timeout=120
    )
    report["minimal_container_ok"] = smoke.returncode == 0 and "docker-smoke-ok" in (smoke.stdout or "")
    report["minimal_container_output"] = ((smoke.stdout or "") + (smoke.stderr or "")).strip()[-500:]
    if not report["minimal_container_ok"]:
        blockers.append("minimal Docker container failed")


def _evaluator_check(blockers: list[str], report: dict[str, Any]) -> None:
    try:
        version = importlib.metadata.version("swebench")
    except importlib.metadata.PackageNotFoundError:
        version = None
    report["swebench_package_version"] = version
    if version != "4.0.5":
        blockers.append(f"swebench package version is {version!r}, expected '4.0.5'")


def _task_image_entry(task_id: str, record: dict[str, Any]) -> dict[str, Any]:
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(record, namespace="swebench", instance_image_tag="latest")
    return {
        "task_id": task_id,
        "repository": record["repo"],
        "base_commit": record["base_commit"],
        "image": spec.instance_image_key,
        "official_env_image_key": spec.env_image_key,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--cache-dir", type=Path, default=Path("/harddisk/codehd/Self-Harness-data/cache"))
    parser.add_argument("--no-pull", action="store_true", help="Only verify local images; do not pull.")
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    config = load_eval_config(args.config)
    data_root = Path(os.environ.get("SELF_HARNESS_DATA_ROOT", "/harddisk/codehd/Self-Harness-data"))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (data_root / "runs" / args.run_id).resolve()
    )
    reference_path = args.reference_run / "result.json"
    blockers: list[str] = []
    warnings: list[str] = []
    preflight: dict[str, Any] = {
        "format": "self_harness.envfixed_preflight.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "reference_run": str(args.reference_run),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(Path(args.config)),
        "api_key_value_logged": False,
        "reference_run_modified": False,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"error: output directory already exists and is not empty: {output_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reference_path.is_file():
        blockers.append(f"reference result missing: {reference_path}")
        reference: dict[str, Any] = {}
    else:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        preflight["reference_status"] = reference.get("status")
        preflight["reference_result_sha256"] = sha256_file(reference_path)

    # ---- fixed experimental conditions ---------------------------------
    eval_ = config.eval
    model = config.model
    fixed = {
        "model_id": model.model_id,
        "base_url": model.base_url,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "max_tokens": model.max_tokens,
        "repeats": eval_.repeats,
        "task_timeout_s": eval_.task_timeout_s,
        "tool_timeout_s": eval_.tool_timeout_s,
        "case_concurrency": eval_.case_concurrency,
        "splits": list(eval_.splits),
        "task_ids": list(eval_.task_ids or ()),
        "pilot": eval_.pilot,
        "heldout_run": bool((reference.get("pilot") or {}).get("heldout_run")),
        "candidate_iteration": bool((reference.get("pilot") or {}).get("candidate_iteration")),
        "evaluator_kind": config.evaluator.kind,
        "evaluator_name": config.evaluator.name,
        "evaluator_version": config.evaluator.version,
    }
    preflight["fixed_conditions"] = fixed
    if fixed["model_id"] != EXPECTED_MODEL:
        blockers.append(f"model {fixed['model_id']!r} != {EXPECTED_MODEL!r}")
    if fixed["base_url"].rstrip("/") != EXPECTED_BASE_URL:
        blockers.append(f"base_url {fixed['base_url']!r} != {EXPECTED_BASE_URL!r}")
    if (fixed["temperature"], fixed["top_p"], fixed["max_tokens"]) != (0.0, 1.0, 8192):
        blockers.append("decoding parameters changed")
    if fixed["repeats"] != 2 or fixed["task_timeout_s"] != 3600 or fixed["tool_timeout_s"] != 120:
        blockers.append("timeout/repeats changed")
    if fixed["case_concurrency"] != 1:
        blockers.append("case concurrency changed")
    if fixed["splits"] != ["train"]:
        blockers.append("split list must be exactly ['train']")
    if tuple(fixed["task_ids"]) != EXPECTED_TASK_IDS:
        blockers.append("task list changed")
    if not fixed["pilot"]:
        blockers.append("pilot flag is not set")
    if fixed["heldout_run"] or fixed["candidate_iteration"]:
        blockers.append("reference run was not a heldout-free/no-candidate pilot")
    if fixed["evaluator_name"] != "swebench-official-harness" or fixed["evaluator_version"] != "swebench-4.0.5":
        blockers.append("official evaluator identity/version changed")
    if config.dataset.revision != EXPECTED_DATASET_REVISION:
        blockers.append("dataset revision changed")
    if config.constraints.docker_image == "python:3.12":
        # The adapter must select per-task images from the map; a generic fallback
        # is still recorded but is explicitly forbidden by this pilot contract.
        preflight["generic_image_in_config"] = True

    # compare reference provenance
    ref_harness = ((reference.get("harness") or {}).get("snapshot_sha256"))
    current_harness = hash_tree_or_file(REPO_ROOT / "eval" / "harness_workspace")
    preflight["harness"] = {
        "source": str(REPO_ROOT / "eval" / "harness_workspace"),
        "current_sha256": current_harness,
        "reference_sha256": ref_harness,
        "matches_reference": current_harness == ref_harness == EXPECTED_HARNESS_SHA,
    }
    if current_harness != EXPECTED_HARNESS_SHA or ref_harness != EXPECTED_HARNESS_SHA:
        blockers.append("Harness snapshot does not match the reference pilot")
    ref_dataset = reference.get("dataset") or {}
    preflight["dataset_revision"] = {
        "current": config.dataset.revision,
        "reference": ref_dataset.get("revision"),
        "matches": config.dataset.revision == ref_dataset.get("revision") == EXPECTED_DATASET_REVISION,
    }
    if preflight["dataset_revision"]["matches"] is not True:
        blockers.append("dataset revision does not match the reference pilot")
    dataset_report: dict[str, Any] = {}
    _manifest_checks(config, reference, blockers, dataset_report)
    preflight["dataset"] = dataset_report
    if dataset_report.get("dataset_source_sha256") not in (None, EXPECTED_DATASET_JSONL_SHA):
        blockers.append("dataset export sha256 does not match the pinned export")
    if dataset_report.get("dataset_source_sha256") is None:
        blockers.append("dataset export hash unavailable")

    # task metadata from dataset + manifest
    manifest = json.loads(Path(config.dataset.manifest_path).read_text(encoding="utf-8"))
    train_ids = [str(item) for item in manifest.get("held_in_task_ids", [])]
    heldout_ids = [str(item) for item in manifest.get("held_out_task_ids", [])]
    preflight["split_membership"] = {
        "all_pilot_tasks_in_train": all(tid in train_ids for tid in EXPECTED_TASK_IDS),
        "pilot_tasks_in_heldout": [tid for tid in EXPECTED_TASK_IDS if tid in heldout_ids],
        "heldout_count": len(heldout_ids),
        "train_count": len(train_ids),
    }
    if not preflight["split_membership"]["all_pilot_tasks_in_train"]:
        blockers.append("pilot task is not in the train split")
    if preflight["split_membership"]["pilot_tasks_in_heldout"]:
        blockers.append("pilot includes a heldout task")

    source_path = Path(config.dataset.source)
    if source_path.suffix == ".jsonl":
        source_records = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        source_records = json.loads(source_path.read_text(encoding="utf-8"))
    records = {str(item["instance_id"]): item for item in source_records}
    repositories = []
    for tid in EXPECTED_TASK_IDS:
        record = records.get(tid)
        if record is None:
            blockers.append(f"task record missing from dataset: {tid}")
            continue
        repo_path = Path(config.dataset.repos_dir) / str(record["repo"])
        exists = (repo_path / ".git").is_dir() if repo_path.exists() else False
        commit_ok = False
        if exists:
            commit = _run(["git", "-C", str(repo_path), "cat-file", "-e", f"{record['base_commit']}^{{commit}}"], timeout=30)
            commit_ok = commit.returncode == 0
        repositories.append(
            {
                "task_id": tid,
                "repository": record["repo"],
                "base_commit": record["base_commit"],
                "path": str(repo_path),
                "clone_exists": exists,
                "base_commit_exists": commit_ok,
            }
        )
        if not exists:
            blockers.append(f"repo clone missing: {repo_path}")
        elif not commit_ok:
            blockers.append(f"base commit missing for {tid}: {record['base_commit']}")
    preflight["repositories"] = repositories

    # credentials/endpoint (value never logged)
    endpoint_report: dict[str, Any] = {}
    _endpoint_check(config.model.base_url, config.model.model_id, blockers, endpoint_report)
    preflight["endpoint"] = endpoint_report

    docker_report: dict[str, Any] = {}
    _docker_checks(blockers, docker_report)
    preflight["docker"] = docker_report
    evaluator_report: dict[str, Any] = {}
    _evaluator_check(blockers, evaluator_report)
    preflight["evaluator"] = evaluator_report
    try:
        usage = shutil.disk_usage(data_root)
        preflight["disk"] = {
            "path": str(data_root),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "free_gib": round(usage.free / (1024**3), 2),
        }
        if usage.free < 30 * 1024**3:
            blockers.append("less than 30 GiB free under the data root")
    except OSError as exc:
        blockers.append(f"cannot stat data root: {exc}")

    # ---- official task images -----------------------------------------
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, Any] = {
        "format": "self_harness.task_images.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "namespace": "swebench",
        "evaluator": {"name": "swebench-official-harness", "version": "swebench-4.0.5"},
        "image_cache_dir": str(cache_dir),
        "tasks": {},
        "blockers": [],
    }
    for tid in EXPECTED_TASK_IDS:
        record = records.get(tid)
        if record is None:
            continue
        try:
            entry = _task_image_entry(tid, record)
        except Exception as exc:  # noqa: BLE001
            image_map["blockers"].append(f"{tid}: cannot derive official image: {exc.__class__.__name__}: {exc}")
            continue
        image = entry["image"]
        inspect = _image_inspect(image)
        source = "local"
        if inspect is None and not args.no_pull:
            ok, output = _pull(image)
            entry["pull_ok"] = ok
            entry["pull_output_tail"] = output[-2000:]
            source = "pull"
            if not ok:
                image_map["blockers"].append(f"{tid}: docker pull failed for {image}: {output[-500:]}")
                entry["image_missing"] = True
                image_map["tasks"][tid] = entry
                continue
            inspect = _image_inspect(image)
        if inspect is None:
            image_map["blockers"].append(f"{tid}: image is not available locally: {image}")
            entry["image_missing"] = True
            image_map["tasks"][tid] = entry
            continue
        entry.update(
            {
                "image_digest": inspect.get("Id"),
                "image_source": source,
                "image_size_bytes": inspect.get("Size"),
                "repo_digests": inspect.get("RepoDigests") or [],
            }
        )
        if not args.no_probe:
            probe = _probe_image(
                image,
                str(record["base_commit"]),
                TASK_REQUIRED_MODULES.get(tid, ()),
            )
            entry["probe"] = probe
            if probe.get("returncode") != 0:
                image_map["blockers"].append(f"{tid}: image probe failed: {probe.get('stderr_tail')}")
            if probe.get("testbed_exists") is False:
                image_map["blockers"].append(f"{tid}: image has no /testbed")
            if probe.get("read_only_root_ok") is False:
                warnings.append(
                    f"{tid}: read-only-root layout failed; adapter will run without --read-only "
                    "(the container root is still ephemeral, but this weakens write isolation)"
                )
            if probe.get("dependency_probe_ok") is not True:
                image_map["blockers"].append(
                    f"{tid}: dependency probe failed with the derived task PATH: "
                    f"{probe.get('dependency_probe_output')}"
                )
            if probe.get("testbed_exists") is not True:
                image_map["blockers"].append(f"{tid}: official task image has no /testbed")
            if probe.get("required_modules_ok") is not True:
                image_map["blockers"].append(
                    f"{tid}: required module probe failed: {probe.get('required_modules')}"
                )
            if probe.get("base_commit_match") is False:
                warnings.append(
                    f"{tid}: image /testbed HEAD differs from the base commit; the runner mounts the "
                    "pinned host workspace at /testbed and verifies its HEAD anyway"
                )
            entry["container_env"] = probe.get("container_env") or {"HOME": "/root"}
            entry["read_only_root"] = bool(probe.get("read_only_root_ok"))
            entry["bootstrap_workspace_from_image"] = True
            entry["image_repo_path"] = "/testbed"
        else:
            entry["container_env"] = {"HOME": "/root"}
            entry["read_only_root"] = False
            entry["bootstrap_workspace_from_image"] = True
            entry["image_repo_path"] = "/testbed"
        image_map["tasks"][tid] = entry

    if len(image_map["tasks"]) != len(EXPECTED_TASK_IDS) or image_map["blockers"]:
        blockers.extend(image_map["blockers"])
    preflight["task_images"] = {
        "all_tasks_resolved": len(image_map["tasks"]) == len(EXPECTED_TASK_IDS) and not image_map["blockers"],
        "tasks": {
            tid: {
                "image": entry.get("image"),
                "image_digest": entry.get("image_digest"),
                "image_source": entry.get("image_source"),
                "pytest": (entry.get("probe") or {}).get("pytest"),
                "read_only_root": entry.get("read_only_root"),
                "base_commit_match": (entry.get("probe") or {}).get("base_commit_match"),
            }
            for tid, entry in image_map["tasks"].items()
        },
    }

    preflight["warnings"] = warnings
    preflight["blockers"] = blockers
    preflight["ready"] = not blockers
    preflight["duration_s"] = round(time.time() - started, 3)
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "task_images.json").write_text(json.dumps(image_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if blockers:
        print("preflight blocked:", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        print(f"wrote {output_dir / 'preflight.json'}", file=sys.stderr)
        return 2
    print(f"preflight OK; wrote {output_dir / 'preflight.json'}")
    print(f"task image map: {output_dir / 'task_images.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
