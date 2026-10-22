"""Shared helpers for building synthetic unified evaluation results in tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_harness_eval.hashing import sha256_file, sha256_json
from self_harness_eval.protocol import (
    BENCHMARK_SWEBENCH,
    BENCHMARK_TB2,
    build_benchmark_block,
    build_config_block,
    build_dataset_block,
    build_eval_result,
    build_harness_block,
    build_model_block,
    make_agent_block,
    make_case_result,
    make_evaluator_block,
    make_repeat_result,
    utc_now,
)

SUCCESS = {"swe_bench_verified": "resolved", "terminal_bench_2": "passed"}
FAILURE = {"swe_bench_verified": "unresolved", "terminal_bench_2": "failed"}


def make_result(
    tmp: Path,
    *,
    benchmark: str = BENCHMARK_SWEBENCH,
    splits: Mapping[str, Sequence[Sequence[tuple[str, str]]]],
    run_id: str = "run-test",
    result_id: str = "result-a",
    model_id: str = "model-a",
    provider: str = "provider-a",
    decoding: Mapping[str, Any] | None = None,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    dataset_revision: str = "rev-1",
    benchmark_version: str = "benchmark-v1",
    evaluator_name: str = "test-evaluator",
    evaluator_version: str = "1.0",
    environment_variant: str = "environment-a",
    tools_variant: str = "tools-a",
    harness_version_id: str = "v000_baseline",
    harness_kind: str = "baseline",
    parent_version_ids: Sequence[str] = (),
    harness_content: str | None = None,
    failure_summary: str = "verifier failure",
    tag: str = "a",
    as_eval_harness: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Build a schema-valid result plus its on-disk run root."""

    run_root = Path(tmp) / f"run-{tag}"
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "config.resolved.json"
    config_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "decoding": dict(decoding or {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8}),
                "environment": environment_variant,
                "tools": tools_variant,
                "benchmark": benchmark,
                "dataset": f"{dataset_name}@{dataset_revision}",
                "evaluator": f"{evaluator_name}@{evaluator_version}",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = run_root / "dataset_manifest.json"
    manifest_payload = {
        "benchmark": benchmark,
        "dataset": {"name": dataset_name, "revision": dataset_revision},
        "splits": {
            split: [task_id for task_id, _status in repeats[0]]
            for split, repeats in splits.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    snapshot_dir = run_root / f"harness-{harness_version_id}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "repo_baseline.py").write_text(
        harness_content
        if harness_content is not None
        else f"# harness {harness_version_id}\nBASELINE = '{result_id}'\n",
        encoding="utf-8",
    )

    repeat_counts = {split: len(repeats) for split, repeats in splits.items()}
    if len(set(repeat_counts.values())) != 1:
        raise ValueError("all splits must have the same repeat count")
    repeats_total = next(iter(repeat_counts.values()))
    split_task_ids = {
        split: [task_id for task_id, _status in repeats[0]]
        for split, repeats in splits.items()
    }
    for split, repeats in splits.items():
        for repeat in repeats:
            if [task_id for task_id, _ in repeat] != split_task_ids[split]:
                raise ValueError("task id order must be stable across repeats")

    train_ids = split_task_ids.get("train", [])
    heldout_ids = split_task_ids.get("heldout", [])
    dataset_block = build_dataset_block(
        run_root=run_root,
        name=dataset_name,
        revision=dataset_revision,
        manifest_path=manifest_path,
        train_task_ids=train_ids,
        heldout_task_ids=heldout_ids,
    )
    model_block = build_model_block(
        provider=provider,
        model_id=model_id,
        endpoint_type="hosted_api",
        decoding=decoding or {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8},
    )
    benchmark_block = build_benchmark_block(
        benchmark_id=benchmark,
        version=benchmark_version,
        evaluator=evaluator_name,
        evaluator_version=evaluator_version,
    )
    harness_block = build_harness_block(
        run_root=run_root,
        snapshot_dir=snapshot_dir,
        version_id=harness_version_id,
        parent_version_ids=list(parent_version_ids),
        kind=harness_kind,
    )
    comparability = {
        "benchmark": sha256_json({"id": benchmark, "version": benchmark_version}),
        "dataset": sha256_json({"name": dataset_name, "revision": dataset_revision}),
        "decoding": sha256_json(dict(model_block["decoding"])),
        "environment": sha256_json({"environment": environment_variant}),
        "evaluator": sha256_json({"name": evaluator_name, "version": evaluator_version}),
        "manifest": dataset_block["manifest_sha256"],
        "model": sha256_json(dict(model_block)),
        "task_ids": dataset_block["task_ids_sha256"],
        "tools": sha256_json({"tools": tools_variant}),
    }
    config_block = build_config_block(
        run_root=run_root,
        resolved_config_path=config_path,
        repeats=repeats_total,
        task_timeout_s=60.0,
        tool_timeout_s=10.0,
        comparability=comparability,
        network_limit={"requested_mbps": 2.0, "enforced": False, "method": "none",
                       "detail": "test fixture"},
    )

    split_records: dict[str, list[dict[str, Any]]] = {}
    patch_dir = run_root / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    for split, repeats in splits.items():
        split_records[split] = []
        for repeat_index, repeat in enumerate(repeats, start=1):
            case_results = []
            for task_id, status in repeat:
                case_results.append(
                    _case(
                        benchmark=benchmark,
                        task_id=task_id,
                        split=split,
                        repeat=repeat_index,
                        status=status,
                        patch_dir=patch_dir,
                        run_root=run_root,
                        failure_summary=failure_summary,
                    )
                )
            split_records[split].append(
                make_repeat_result(
                    split=split,
                    repeat=repeat_index,
                    expected_task_ids=split_task_ids[split],
                    case_results=case_results,
                )
            )
    started = utc_now()
    finished = utc_now()
    result = build_eval_result(
        run_id=run_id,
        result_id=result_id,
        benchmark=benchmark_block,
        model=model_block,
        harness=harness_block,
        dataset=dataset_block,
        config=config_block,
        started_at=started,
        finished_at=finished,
        splits=split_records,
        errors=[],
        expected_repeats=repeats_total,
    )
    if as_eval_harness:
        result["harness"]["kind"] = harness_kind
    return result, run_root


def _case(
    *,
    benchmark: str,
    task_id: str,
    split: str,
    repeat: int,
    status: str,
    patch_dir: Path,
    run_root: Path,
    failure_summary: str,
) -> dict[str, Any]:
    if benchmark == BENCHMARK_TB2:
        patch = None
    elif status in (SUCCESS[benchmark], FAILURE[benchmark]):
        patch_path = patch_dir / f"{task_id}.diff"
        patch_path.write_text(f"diff --git a/{task_id} b/{task_id}\n", encoding="utf-8")
        patch = {
            "path": patch_path.relative_to(run_root).as_posix(),
            "sha256": sha256_file(patch_path),
        }
    else:
        patch = None
    error = None
    if status in {"resolved", "passed"}:
        agent = make_agent_block(status="complete", timed_out=False, tool_calls=2, trace_path=None)
        evaluator = make_evaluator_block(
            status="complete",
            name="test-evaluator",
            resolved=True,
            failure_summary=None,
            log_path=None,
        )
    elif status in {"unresolved", "failed"}:
        agent = make_agent_block(status="complete", timed_out=False, tool_calls=2, trace_path=None)
        evaluator = make_evaluator_block(
            status="complete",
            name="test-evaluator",
            resolved=False,
            failure_summary=failure_summary,
            log_path=None,
        )
    elif status == "agent_timeout":
        agent = make_agent_block(status="timeout", timed_out=True, tool_calls=None, trace_path=None)
        evaluator = make_evaluator_block(
            status="not_run",
            name="test-evaluator",
            resolved=None,
            failure_summary=None,
            log_path=None,
        )
        error = "agent timed out"
    elif status == "agent_error":
        agent = make_agent_block(status="error", timed_out=False, tool_calls=None, trace_path=None)
        evaluator = make_evaluator_block(
            status="not_run",
            name="test-evaluator",
            resolved=None,
            failure_summary=None,
            log_path=None,
        )
        error = "agent errored"
    elif status in {"infra_error", "evaluator_error", "not_run"}:
        if status == "evaluator_error":
            agent = make_agent_block(status="complete", timed_out=False, tool_calls=1, trace_path=None)
            evaluator = make_evaluator_block(
                status="error",
                name="test-evaluator",
                resolved=None,
                failure_summary=failure_summary,
                log_path=None,
            )
        else:
            agent = make_agent_block(status="not_run", timed_out=False, tool_calls=None, trace_path=None)
            evaluator = make_evaluator_block(
                status="not_run",
                name="test-evaluator",
                resolved=None,
                failure_summary=failure_summary,
                log_path=None,
            )
        error = f"{status}: {failure_summary}"
    else:
        raise ValueError(f"unknown synthetic case status: {status!r}")
    return make_case_result(
        task_id=task_id,
        split=split,
        repeat=repeat,
        repository="repo/repo",
        domain="Domain",
        status=status,
        duration_s=1.0,
        agent=agent,
        evaluator=evaluator,
        patch=patch,
        artifacts={},
        error=error,
    )


def statuses(*, train: Sequence[str], heldout: Sequence[str]) -> dict[str, list[list[tuple[str, str]]]]:
    return {
        "train": [[(f"train-{index}", status) for index, status in enumerate(train, start=1)]],
        "heldout": [
            [(f"heldout-{index}", status) for index, status in enumerate(heldout, start=1)]
        ],
    }


def clone(result: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(result))


def case_outcome(result: Mapping[str, Any], split: str, task_id: str) -> bool | None:
    for repeat in result["splits"][split]:
        for case in repeat["case_results"]:
            if case["task_id"] == task_id:
                return case["passed"]
    raise KeyError(task_id)


def dump_result(result: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
