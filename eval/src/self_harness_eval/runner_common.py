"""Shared run-context plumbing for the Terminal-Bench and SWE-bench runners."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bandwidth import BandwidthPlan
from .config import EvalConfig, write_resolved_config
from .hashing import (
    IGNORED_DIR_NAMES,
    IGNORED_FILE_SUFFIXES,
    sha256_file,
    sha256_json,
    sha256_text,
    snapshot_tree_hash,
    tree_file_hashes,
)
from .protocol import (
    SCHEMA_VERSION,
    build_benchmark_block,
    build_config_block,
    build_dataset_block,
    build_eval_result,
    build_harness_block,
    build_model_block,
    task_ids_sha256,
    utc_now,
    validate_result,
    write_result_immutable,
)


class RunnerError(RuntimeError):
    """Raised when an evaluation cannot be prepared or a reuse check fails."""


class ReuseMismatch(RunnerError):
    """Raised when reuse-existing is requested with incompatible provenance."""


@dataclass
class RunContext:
    config: EvalConfig
    output_dir: Path
    run_root: Path
    run_id: str
    result_id: str
    benchmark: dict[str, Any]
    model: dict[str, Any]
    harness: dict[str, Any]
    dataset: dict[str, Any]
    config_block: dict[str, Any]
    environment: dict[str, Any]
    started_at: str
    bandwidth: BandwidthPlan | None = None
    dataset_manifest_path: Path | None = None
    resolved_config_path: Path | None = None

    def rel(self, path: Path | str) -> str:
        return relative_to_run(Path(path), self.run_root)

    def case_dir_relative(self, path: Path) -> str:
        return self.rel(path)


def relative_to_run(path: Path, run_root: Path) -> str:
    resolved = Path(path).resolve()
    root = Path(run_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise RunnerError(
            f"artifact path must live inside the run root {root}: got {resolved}"
        ) from exc


def prepare_run_context(
    *,
    config: EvalConfig,
    output_dir: Path,
    run_root: Path | None,
    run_id: str,
    result_id: str,
    benchmark_id: str,
    benchmark_version: str,
    evaluator_name: str,
    evaluator_version: str,
    evaluator_config: Mapping[str, Any] | None = None,
    harness_snapshot: Path | None,
    harness_version_id: str,
    harness_kind: str,
    parent_version_ids: Sequence[str] = (),
    dataset_manifest_payload: Mapping[str, Any],
    split_task_ids: Mapping[str, Sequence[str]],
    dataset_source_sha256: str | None = None,
    environment_extra: Mapping[str, Any] | None = None,
    bandwidth: BandwidthPlan | None = None,
    reuse_existing: bool = False,
) -> RunContext:
    output_dir = Path(output_dir).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve() if run_root is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.relative_to(run_root)
    except ValueError as exc:
        raise RunnerError(f"output_dir {output_dir} must be inside run_root {run_root}") from exc

    resolved_config_path = run_root / "config.resolved.json"
    write_resolved_config(resolved_config_path, config)

    harness_snapshot = _prepare_harness_snapshot(
        config=config,
        snapshot=harness_snapshot,
        output_dir=output_dir,
        run_root=run_root,
        reuse_existing=reuse_existing,
    )

    dataset_manifest_path = run_root / "dataset_manifest.json"
    _write_json_idempotent(dataset_manifest_path, dict(dataset_manifest_payload))

    train_ids = [str(item) for item in split_task_ids.get("train", ())]
    heldout_ids = [str(item) for item in split_task_ids.get("heldout", ())]

    dataset_block = build_dataset_block(
        run_root=run_root,
        name=config.dataset.name,
        revision=config.dataset.revision,
        manifest_path=dataset_manifest_path,
        train_task_ids=train_ids,
        heldout_task_ids=heldout_ids,
        source_sha256=dataset_source_sha256,
    )
    benchmark_block = build_benchmark_block(
        benchmark_id=benchmark_id,
        version=benchmark_version,
        evaluator=evaluator_name,
        evaluator_version=evaluator_version,
        evaluator_config=evaluator_config,
    )
    model_block = build_model_block(
        provider=config.model.provider,
        model_id=config.model.model_id,
        endpoint_type=config.model.endpoint_type,
        decoding=config.model.decoding(),
        extra={
            "base_url": config.model.base_url,
            "max_retries": config.model.max_retries,
            **(
                {"reasoning_budget": config.model.reasoning_budget}
                if config.model.reasoning_budget is not None
                else {}
            ),
            **({"deployment": dict(config.model.deployment)} if config.model.deployment else {}),
        },
    )
    harness_block = build_harness_block(
        run_root=run_root,
        snapshot_dir=harness_snapshot,
        version_id=harness_version_id,
        parent_version_ids=list(parent_version_ids),
        kind=harness_kind,
    )
    environment = collect_environment(config, extra=environment_extra)
    comparability = build_comparability(
        config=config,
        model_block=model_block,
        benchmark_block=benchmark_block,
        dataset_block=dataset_block,
        environment=environment,
    )
    config_block = build_config_block(
        run_root=run_root,
        resolved_config_path=resolved_config_path,
        repeats=config.eval.repeats,
        task_timeout_s=config.eval.task_timeout_s,
        tool_timeout_s=config.eval.tool_timeout_s,
        comparability=comparability,
        network_limit=bandwidth.to_dict() if bandwidth is not None else None,
    )
    return RunContext(
        config=config,
        output_dir=output_dir,
        run_root=run_root,
        run_id=run_id,
        result_id=result_id,
        benchmark=benchmark_block,
        model=model_block,
        harness=harness_block,
        dataset=dataset_block,
        config_block=config_block,
        environment=environment,
        started_at=utc_now(),
        bandwidth=bandwidth,
        dataset_manifest_path=dataset_manifest_path,
        resolved_config_path=resolved_config_path,
    )


def _prepare_harness_snapshot(
    *,
    config: EvalConfig,
    snapshot: Path | None,
    output_dir: Path,
    run_root: Path,
    reuse_existing: bool,
) -> Path:
    source = Path(snapshot).expanduser().resolve() if snapshot is not None else None
    if source is None:
        candidate = config.harness.baseline_dir
        if candidate is None:
            legacy_workspace = config.raw.get("eval", {}).get("harness_workspace")
            if isinstance(legacy_workspace, str):
                candidate = Path(legacy_workspace).expanduser().resolve()
        if candidate is None:
            raise RunnerError(
                "no harness snapshot configured: pass --harness-snapshot or set "
                "[harness].baseline_dir"
            )
        source = Path(candidate).resolve()
    if not source.is_dir():
        raise RunnerError(f"harness snapshot source is not a directory: {source}")
    try:
        source.relative_to(run_root)
        inside_run_root = True
    except ValueError:
        inside_run_root = False
    if inside_run_root:
        return source

    target = output_dir / "harness_snapshot"
    if target.exists():
        expected = tree_file_hashes(source)
        actual = tree_file_hashes(target)
        if expected != actual:
            raise RunnerError(
                "existing harness snapshot in the output directory does not match the current "
                f"harness source; refusing to overwrite an immutable snapshot: {target}"
            )
        return target
    _copy_tree(source, target)
    return target


def _copy_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
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


def hash_for(path: Path) -> str:
    if path.is_dir():
        return snapshot_tree_hash(tree_file_hashes(path))
    return sha256_file(path)


def collect_environment(
    config: EvalConfig,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the fixed runtime configuration hash inputs (non-secret only)."""

    environment: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "tool_timeout_s": config.eval.tool_timeout_s,
        "task_timeout_s": config.eval.task_timeout_s,
        "dependency_pins": dict(sorted(config.constraints.dependency_pins.items())),
        "docker_image": config.constraints.docker_image,
        "docker_image_digest": config.constraints.docker_image_digest,
        "bandwidth_limit_requested_mbps": config.constraints.outbound_bandwidth_mbps,
        "bandwidth_enforcement": config.constraints.bandwidth_enforcement,
    }
    if extra:
        environment.update({str(key): value for key, value in extra.items()})
    environment["version"] = "self_harness.environment.v1"
    return environment


def build_comparability(
    *,
    config: EvalConfig,
    model_block: Mapping[str, Any],
    benchmark_block: Mapping[str, Any],
    dataset_block: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, str]:
    tools = {
        "agent_kind": config.agent.kind,
        "command_template": _template_hash(config.agent.command),
        "harbor_command_template": _template_hash(config.agent.harbor_command),
        "env_passthrough": sorted(config.agent.env_passthrough),
        "extra_env_keys": sorted(config.agent.extra_env),
        "max_retries": config.agent.max_retries,
        "tool_timeout_s": config.eval.tool_timeout_s,
    }
    evaluator = {
        "kind": config.evaluator.kind,
        "name": config.evaluator.name,
        "version": config.evaluator.version,
        "command_template": _template_hash(config.evaluator.command),
        "dataset": config.evaluator.dataset,
        "split": config.evaluator.split,
        "max_workers": config.evaluator.max_workers,
        "evaluator_config_sha256": benchmark_block.get("evaluator_config_sha256"),
    }
    dataset = {
        "name": dataset_block.get("name"),
        "revision": dataset_block.get("revision"),
        "source_sha256": dataset_block.get("source_sha256"),
    }
    return {
        "benchmark": sha256_json(
            {"id": benchmark_block.get("id"), "version": benchmark_block.get("version")}
        ),
        "dataset": sha256_json(dataset),
        "decoding": sha256_json(dict(model_block.get("decoding", {}))),
        "environment": sha256_json(dict(environment)),
        "evaluator": sha256_json(evaluator),
        "manifest": str(dataset_block.get("manifest_sha256")),
        "model": sha256_json(dict(model_block)),
        "task_ids": str(dataset_block.get("task_ids_sha256")),
        "tools": sha256_json(tools),
    }


def _template_hash(template: str | Sequence[str] | None) -> str | None:
    if template is None:
        return None
    if isinstance(template, str):
        return sha256_text(template)
    return sha256_text("\x00".join(str(item) for item in template))


def finalize_result(
    *,
    context: RunContext,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    errors: Sequence[Mapping[str, Any]] = (),
    artifacts: Mapping[str, str | None] | None = None,
    extra: Mapping[str, Any] | None = None,
    status: str | None = None,
    write: bool = True,
) -> tuple[dict[str, Any], Path]:
    finished = utc_now()
    artifact_block = dict(artifacts or {})
    artifact_block.setdefault("config_resolved", context.rel(context.resolved_config_path))
    if context.dataset_manifest_path is not None:
        artifact_block.setdefault("dataset_manifest", context.rel(context.dataset_manifest_path))
    result = build_eval_result(
        run_id=context.run_id,
        result_id=context.result_id,
        benchmark=context.benchmark,
        model=context.model,
        harness=context.harness,
        dataset=context.dataset,
        config=context.config_block,
        started_at=context.started_at,
        finished_at=finished,
        splits=splits,
        artifacts=artifact_block,
        errors=errors,
        status=status,
        expected_repeats=context.config.eval.repeats,
        extra=extra,
    )
    target = context.output_dir / "result.json"
    if write:
        write_result_immutable(target, result)
    return result, target


def expected_reuse_fingerprint(result: Mapping[str, Any]) -> dict[str, Any]:
    dataset = result.get("dataset") or {}
    return {
        "schema_version": result.get("schema_version"),
        "benchmark_id": (result.get("benchmark") or {}).get("id"),
        "benchmark_version": (result.get("benchmark") or {}).get("version"),
        "evaluator": (result.get("benchmark") or {}).get("evaluator"),
        "evaluator_version": (result.get("benchmark") or {}).get("evaluator_version"),
        "dataset_revision": dataset.get("revision"),
        "dataset_manifest_sha256": dataset.get("manifest_sha256"),
        "task_ids_sha256": dataset.get("task_ids_sha256"),
        "model_config_sha256": (result.get("model") or {}).get("config_sha256"),
        "resolved_config_sha256": (result.get("config") or {}).get("resolved_config_sha256"),
        "comparability_sha256": (result.get("config") or {}).get("comparability_sha256"),
        "harness_snapshot_sha256": (result.get("harness") or {}).get("snapshot_sha256"),
        "repeats": (result.get("config") or {}).get("repeats"),
        "splits": {
            split: sorted(
                [
                    [int(repeat.get("repeat", 0)), int(repeat.get("total", 0))]
                    for repeat in repeat_list
                ]
            )
            for split, repeat_list in (result.get("splits") or {}).items()
        },
    }


def assert_reuse_compatible(
    *,
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
    existing_path: Path,
) -> None:
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise ReuseMismatch(
            f"existing result {existing_path} was written with schema "
            f"{existing.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )
    diffs: list[str] = []
    for key in (
        "benchmark_id",
        "benchmark_version",
        "evaluator",
        "evaluator_version",
        "dataset_revision",
        "dataset_manifest_sha256",
        "task_ids_sha256",
        "model_config_sha256",
        "resolved_config_sha256",
        "comparability_sha256",
        "harness_snapshot_sha256",
        "repeats",
    ):
        if existing.get(key) != expected.get(key):
            diffs.append(f"{key}: existing={existing.get(key)!r} expected={expected.get(key)!r}")
    if existing.get("splits") != expected.get("splits"):
        diffs.append(
            f"splits: existing={existing.get('splits')!r} expected={expected.get('splits')!r}"
        )
    if diffs:
        raise ReuseMismatch(
            f"refusing to reuse {existing_path}: provenance mismatch ("
            + "; ".join(diffs)
            + ")"
        )


def load_existing_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RunnerError(f"no existing result to reuse: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunnerError(f"existing result is not a JSON object: {path}")
    validate_result(payload)
    return payload


def _write_json_idempotent(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise RunnerError(
                f"refusing to overwrite immutable dataset manifest with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_task_id_map(
    *,
    splits: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    return {split: [str(item) for item in ids] for split, ids in splits.items()}


def split_ids_hash(splits: Mapping[str, Sequence[str]]) -> str:
    return task_ids_sha256(splits.get("train", ()), splits.get("heldout", ()))


def expected_fingerprint_from_context(
    context: RunContext,
    *,
    split_task_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Fingerprint a fresh context would produce, for reuse-existing checks."""

    repeats = context.config.eval.repeats
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": context.benchmark.get("id"),
        "benchmark_version": context.benchmark.get("version"),
        "evaluator": context.benchmark.get("evaluator"),
        "evaluator_version": context.benchmark.get("evaluator_version"),
        "dataset_revision": context.dataset.get("revision"),
        "dataset_manifest_sha256": context.dataset.get("manifest_sha256"),
        "task_ids_sha256": context.dataset.get("task_ids_sha256"),
        "model_config_sha256": context.model.get("config_sha256"),
        "resolved_config_sha256": context.config_block.get("resolved_config_sha256"),
        "comparability_sha256": context.config_block.get("comparability_sha256"),
        "harness_snapshot_sha256": context.harness.get("snapshot_sha256"),
        "repeats": repeats,
        "splits": {
            split: [[repeat, len(list(ids))] for repeat in range(1, repeats + 1)]
            for split, ids in split_task_ids.items()
        },
    }
