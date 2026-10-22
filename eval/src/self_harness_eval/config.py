"""Configuration loading for the Self-Harness evaluators and workflow.

A single TOML file drives both benchmarks.  The loader resolves paths, records
environment-variable *references* (never their values), and produces a
sanitized ``config.resolved.json`` whose SHA-256 becomes part of the run
provenance.  Credentials are always injected through the environment.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .protocol import BENCHMARKS, SPLIT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_NAME_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_CONFIG_KEY_RE = re.compile(
    r"(api[_-]?key|apikey|secret|password|passwd|authorization|credential|"
    r"access[_-]?token|refresh[_-]?token|private[_-]?key)",
    re.IGNORECASE,
)


class ConfigError(ValueError):
    """Raised when a Self-Harness config file is incomplete or inconsistent."""


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model_id: str
    endpoint_type: str
    base_url: str
    temperature: float
    top_p: float
    max_tokens: int
    reasoning_budget: int | None = None
    max_retries: int = 2
    deployment: Mapping[str, Any] = field(default_factory=dict)

    def decoding(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def non_secret_block(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "model_id": self.model_id,
            "endpoint_type": self.endpoint_type,
            "base_url": self.base_url,
            "decoding": self.decoding(),
            "max_retries": self.max_retries,
        }
        if self.reasoning_budget is not None:
            payload["reasoning_budget"] = self.reasoning_budget
        if self.deployment:
            payload["deployment"] = dict(self.deployment)
        return payload


@dataclass(frozen=True)
class DatasetSettings:
    name: str
    revision: str
    manifest_path: Path
    split: str
    source: str | None
    cache_dir: Path | None
    repos_dir: Path | None
    allow_hf_download: bool
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorSettings:
    kind: str
    name: str | None
    version: str | None
    command: str | tuple[str, ...] | None
    dataset: str | None
    split: str | None
    max_workers: int
    report_dir: Path | None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSettings:
    kind: str
    command: str | tuple[str, ...] | None
    harbor_command: str | tuple[str, ...] | None
    extra_env: Mapping[str, str]
    env_passthrough: tuple[str, ...]
    max_retries: int
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintSettings:
    outbound_bandwidth_mbps: float | None
    bandwidth_enforcement: str
    cache_dir: Path | None
    docker_image: str | None
    docker_image_digest: str | None
    dependency_pins: Mapping[str, str]
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessSettings:
    baseline_dir: Path | None
    kind: str
    surfaces: Mapping[str, Path]
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSettings:
    output_root: Path
    run_id: str | None
    max_rounds: int
    proposal_width: int
    max_candidate_evaluations: int
    max_consecutive_no_improvement: int
    max_proposer_failures: int
    resume: bool
    reuse_existing: bool
    proposer_command: str | None
    proposer_response: Path | None
    diagnosis_command: str | None
    proposer_max_slot_attempts: int
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSettings:
    benchmark: str
    name: str
    repeats: int
    splits: tuple[str, ...]
    task_ids: tuple[str, ...] | None
    task_timeout_s: float
    tool_timeout_s: float
    case_concurrency: int
    max_retries: int
    smoke_test: bool
    reuse_existing: bool
    pilot: bool
    pilot_max_tasks: int
    result_dir: Path | None
    legacy: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    path: Path
    sha256: str
    eval: EvalSettings
    dataset: DatasetSettings
    model: ModelSettings
    agent: AgentSettings
    evaluator: EvaluatorSettings
    constraints: ConstraintSettings
    harness: HarnessSettings
    run: RunSettings
    raw: Mapping[str, Any]
    env_refs: Mapping[str, str]
    legacy: Mapping[str, Any]

    def resolved_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    def harness_source_dir(self) -> Path | None:
        return self.harness.baseline_dir


def load_eval_config(path: Path | str) -> EvalConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"config file does not exist: {config_path}")
    tomllib = _toml_reader()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"cannot parse TOML config {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("config must be a TOML table")

    env_refs: dict[str, str] = {}

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            def replace(match: re.Match[str]) -> str:
                name = match.group(1)
                default = match.group(2)
                if name in os.environ and os.environ[name] != "":
                    env_refs[name] = "set"
                    return os.environ[name]
                if default is not None:
                    env_refs[name] = "unset(default)"
                    return default
                env_refs[name] = "unset"
                raise ConfigError(f"environment variable {name!r} is required by config")

            return ENV_PATTERN.sub(replace, value)
        if isinstance(value, Mapping):
            return {str(key): expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    expanded = expand(dict(raw))
    base_dir = config_path.parent

    legacy = {
        key: expanded.get(key)
        for key in ("env",)
        if key in expanded
    }
    eval_raw = _mapping(expanded.get("eval"), "[eval]")
    benchmark = str(eval_raw.get("benchmark") or "terminal_bench_2").strip()
    if benchmark not in BENCHMARKS:
        raise ConfigError(f"[eval].benchmark must be one of {BENCHMARKS}, got {benchmark!r}")

    dataset_raw = _mapping(expanded.get("dataset"), "[dataset]", required=False)
    model_raw = _mapping(expanded.get("model"), "[model]", required=False)
    agent_raw = _mapping(expanded.get("agent"), "[agent]", required=False)
    evaluator_raw = _mapping(expanded.get("evaluator"), "[evaluator]", required=False)
    constraints_raw = _mapping(expanded.get("constraints"), "[constraints]", required=False)
    harness_raw = _mapping(expanded.get("harness"), "[harness]", required=False)
    run_raw = _mapping(expanded.get("run"), "[run]", required=False)

    model = _load_model(model_raw=model_raw, eval_raw=eval_raw)
    dataset = _load_dataset(dataset_raw=dataset_raw, benchmark=benchmark, base_dir=base_dir)
    evaluator = _load_evaluator(evaluator_raw=evaluator_raw, benchmark=benchmark, base_dir=base_dir)
    agent = _load_agent(agent_raw=agent_raw, base_dir=base_dir, benchmark=benchmark)
    constraints = _load_constraints(constraints_raw=constraints_raw, base_dir=base_dir)
    harness = _load_harness(harness_raw=harness_raw, base_dir=base_dir)
    run = _load_run(run_raw=run_raw, base_dir=base_dir, expanded=expanded)
    evals = _load_eval(
        eval_raw=eval_raw,
        benchmark=benchmark,
        base_dir=base_dir,
        expanded=expanded,
    )

    if evals.smoke_test and evaluator.kind == "fixture" and agent.kind == "command":
        pass  # fixture evaluator requires smoke_test; already validated in loader
    if evaluator.kind == "fixture" and not evals.smoke_test:
        raise ConfigError(
            "[evaluator].kind='fixture' is only allowed for smoke tests; set [eval].smoke_test=true "
            "and never report fixture results as an official SWE-bench measurement"
        )
    _reject_secret_config_keys(expanded)

    raw_sha = _sha256_json(expanded)
    config = EvalConfig(
        path=config_path,
        sha256=raw_sha,
        eval=evals,
        dataset=dataset,
        model=model,
        agent=agent,
        evaluator=evaluator,
        constraints=constraints,
        harness=harness,
        run=run,
        raw=expanded,
        env_refs=env_refs,
        legacy=legacy,
    )
    return config


def _load_model(*, model_raw: Mapping[str, Any], eval_raw: Mapping[str, Any]) -> ModelSettings:
    model_id = str(model_raw.get("model_id") or eval_raw.get("model") or "").strip()
    env_model = os.environ.get("SELF_HARNESS_MODEL", "").strip()
    if env_model:
        model_id = env_model
    if not model_id:
        raise ConfigError("[model].model_id or [eval].model is required")
    base_url = str(
        model_raw.get("base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.qadg.xyz/v1"
    ).strip()
    if not base_url:
        raise ConfigError("[model].base_url must not be empty")
    decoding_raw = model_raw.get("decoding") if isinstance(model_raw.get("decoding"), Mapping) else {}
    temperature = float(model_raw.get("temperature", decoding_raw.get("temperature", 0.0)))
    top_p = float(model_raw.get("top_p", decoding_raw.get("top_p", 1.0)))
    max_tokens = int(model_raw.get("max_tokens", decoding_raw.get("max_tokens", 8192)))
    if max_tokens < 1:
        raise ConfigError("model max_tokens must be >= 1")
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("model temperature must be between 0 and 2")
    if not 0.0 < top_p <= 1.0:
        raise ConfigError("model top_p must be in (0, 1]")
    reasoning_raw = model_raw.get("reasoning_budget")
    reasoning_budget = int(reasoning_raw) if reasoning_raw is not None else None
    retries = int(model_raw.get("max_retries", 2))
    if retries < 0 or retries > 10:
        raise ConfigError("model max_retries must be between 0 and 10")
    deployment = model_raw.get("deployment")
    return ModelSettings(
        provider=str(model_raw.get("provider") or "unspecified").strip() or "unspecified",
        model_id=model_id,
        endpoint_type=str(model_raw.get("endpoint_type") or "unspecified").strip() or "unspecified",
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_budget=reasoning_budget,
        max_retries=retries,
        deployment=dict(deployment) if isinstance(deployment, Mapping) else {},
    )


def _load_dataset(
    *,
    dataset_raw: Mapping[str, Any],
    benchmark: str,
    base_dir: Path,
) -> DatasetSettings:
    if benchmark == "swe_bench_verified":
        name = str(dataset_raw.get("name") or "princeton-nlp/SWE-bench_Verified").strip()
        revision = str(dataset_raw.get("revision") or "").strip()
        if not revision:
            raise ConfigError(
                "[dataset].revision must pin the SWE-bench Verified data source "
                "(a dataset commit hash or an explicit local export version)"
            )
        manifest_raw = str(
            dataset_raw.get("manifest_path") or REPO_ROOT / "swe_bench_verified_stratified_100.json"
        )
        manifest_path = _resolve_path(base_dir, manifest_raw)
        split = str(dataset_raw.get("split") or "test").strip()
        source_raw = dataset_raw.get("source")
        source = _resolve_path(base_dir, str(source_raw)) if source_raw else None
        cache_dir = dataset_raw.get("cache_dir")
        repos_dir = dataset_raw.get("repos_dir")
        allow_hf = bool(dataset_raw.get("allow_hf_download", False))
        return DatasetSettings(
            name=name,
            revision=revision,
            manifest_path=manifest_path,
            split=split,
            source=str(source) if source else None,
            cache_dir=_resolve_path(base_dir, str(cache_dir)) if cache_dir else None,
            repos_dir=_resolve_path(base_dir, str(repos_dir)) if repos_dir else None,
            allow_hf_download=allow_hf,
            raw=dict(dataset_raw),
        )
    manifest_raw = dataset_raw.get("manifest_path") if dataset_raw else None
    manifest_path = _resolve_path(
        base_dir,
        str(manifest_raw or REPO_ROOT / "eval" / "manifests" / "tb2_clean64.json"),
    )
    return DatasetSettings(
        name=str(dataset_raw.get("name") or "Terminal-Bench 2.0 Clean64"),
        revision=str(dataset_raw.get("revision") or "2.0-clean64"),
        manifest_path=manifest_path,
        split="test",
        source=None,
        cache_dir=_resolve_path(base_dir, str(dataset_raw["cache_dir"])) if dataset_raw.get("cache_dir") else None,
        repos_dir=None,
        allow_hf_download=False,
        raw=dict(dataset_raw),
    )


def _load_evaluator(
    *,
    evaluator_raw: Mapping[str, Any],
    benchmark: str,
    base_dir: Path,
) -> EvaluatorSettings:
    kind = str(evaluator_raw.get("kind") or ("official" if benchmark == "swe_bench_verified" else "harbor")).strip()
    allowed = {"official", "command", "fixture"} if benchmark == "swe_bench_verified" else {"harbor"}
    if kind not in allowed:
        raise ConfigError(f"[evaluator].kind must be one of {sorted(allowed)}, got {kind!r}")
    command = evaluator_raw.get("command")
    if command is not None:
        command = _command_value(command, field="[evaluator].command")
    report_dir = evaluator_raw.get("report_dir")
    return EvaluatorSettings(
        kind=kind,
        name=str(evaluator_raw["name"]) if evaluator_raw.get("name") else None,
        version=str(evaluator_raw["version"]) if evaluator_raw.get("version") else None,
        command=command,
        dataset=str(evaluator_raw["dataset"]) if evaluator_raw.get("dataset") else None,
        split=str(evaluator_raw["split"]) if evaluator_raw.get("split") else None,
        max_workers=int(evaluator_raw.get("max_workers", 1) or 1),
        report_dir=_resolve_path(base_dir, str(report_dir)) if report_dir else None,
        raw=dict(evaluator_raw),
    )


def _load_agent(
    *,
    agent_raw: Mapping[str, Any],
    base_dir: Path,
    benchmark: str = "swe_bench_verified",
) -> AgentSettings:
    default_kind = "harbor" if benchmark == "terminal_bench_2" else "command"
    allowed_kinds = {"harbor"} if benchmark == "terminal_bench_2" else {"command", "harbor", "harbor_exec"}
    kind = str(agent_raw.get("kind") or default_kind).strip()
    if kind not in allowed_kinds:
        raise ConfigError(
            f"[agent].kind for {benchmark} must be one of {sorted(allowed_kinds)}, got {kind!r}"
        )
    command = agent_raw.get("command")
    if command is not None:
        command = _command_value(command, field="[agent].command")
    harbor_command = agent_raw.get("harbor_command")
    if harbor_command is not None:
        harbor_command = _command_value(harbor_command, field="[agent].harbor_command")
    if benchmark != "terminal_bench_2":
        if kind == "command" and not command:
            raise ConfigError("[agent].command is required when [agent].kind='command'")
        if kind in {"harbor", "harbor_exec"} and not harbor_command:
            raise ConfigError("[agent].harbor_command is required when [agent].kind is Harbor-based")
    extra_env_raw = agent_raw.get("extra_env")
    if extra_env_raw is not None and not isinstance(extra_env_raw, Mapping):
        raise ConfigError("[agent].extra_env must be an object")
    extra_env = {str(key): str(value) for key, value in (extra_env_raw or {}).items()}
    for key in extra_env:
        if _SECRET_NAME_RE.search(key):
            raise ConfigError(
                f"[agent].extra_env must not carry secret-looking key {key!r}; use "
                "[agent].env_passthrough with process environment variables instead"
            )
    passthrough_raw = agent_raw.get("env_passthrough") or []
    if not isinstance(passthrough_raw, list):
        raise ConfigError("[agent].env_passthrough must be a list of environment variable names")
    passthrough = tuple(str(item) for item in passthrough_raw)
    retries = int(agent_raw.get("max_retries", 1))
    if retries < 0 or retries > 10:
        raise ConfigError("[agent].max_retries must be between 0 and 10")
    return AgentSettings(
        kind=kind,
        command=command,
        harbor_command=harbor_command,
        extra_env=extra_env,
        env_passthrough=passthrough,
        max_retries=retries,
        raw=dict(agent_raw),
    )


def _load_constraints(*, constraints_raw: Mapping[str, Any], base_dir: Path) -> ConstraintSettings:
    bandwidth_raw = constraints_raw.get("outbound_bandwidth_mbps")
    bandwidth = float(bandwidth_raw) if bandwidth_raw is not None else None
    if bandwidth is not None and bandwidth <= 0:
        raise ConfigError("[constraints].outbound_bandwidth_mbps must be positive")
    enforcement = str(constraints_raw.get("bandwidth_enforcement") or "auto").strip().lower()
    if enforcement not in {"auto", "tc", "trickle", "none"}:
        raise ConfigError(
            "[constraints].bandwidth_enforcement must be auto|tc|trickle|none"
        )
    cache_raw = constraints_raw.get("cache_dir")
    pins_raw = constraints_raw.get("dependency_pins")
    if pins_raw is not None and not isinstance(pins_raw, Mapping):
        raise ConfigError("[constraints].dependency_pins must be an object")
    return ConstraintSettings(
        outbound_bandwidth_mbps=bandwidth,
        bandwidth_enforcement=enforcement,
        cache_dir=_resolve_path(base_dir, str(cache_raw)) if cache_raw else None,
        docker_image=str(constraints_raw["docker_image"]) if constraints_raw.get("docker_image") else None,
        docker_image_digest=str(constraints_raw["docker_image_digest"])
        if constraints_raw.get("docker_image_digest")
        else None,
        dependency_pins={str(key): str(value) for key, value in (pins_raw or {}).items()},
        raw=dict(constraints_raw),
    )


def _load_harness(*, harness_raw: Mapping[str, Any], base_dir: Path) -> HarnessSettings:
    baseline_raw = harness_raw.get("baseline_dir")
    baseline_dir = _resolve_path(base_dir, str(baseline_raw)) if baseline_raw else None
    kind = str(harness_raw.get("kind") or "baseline").strip()
    if kind not in {"baseline", "candidate", "merged", "final"}:
        raise ConfigError("[harness].kind must be baseline|candidate|merged|final")
    surfaces_raw = harness_raw.get("surfaces")
    if surfaces_raw is not None and not isinstance(surfaces_raw, Mapping):
        raise ConfigError("[harness].surfaces must be an object of name -> path")
    surfaces = {
        str(name): _resolve_path(base_dir, str(value))
        for name, value in (surfaces_raw or {}).items()
    }
    return HarnessSettings(
        baseline_dir=baseline_dir,
        kind=kind,
        surfaces=surfaces,
        raw=dict(harness_raw),
    )


def _load_run(
    *,
    run_raw: Mapping[str, Any],
    base_dir: Path,
    expanded: Mapping[str, Any],
) -> RunSettings:
    output_root = _resolve_path(base_dir, str(run_raw.get("output_root") or "runs"))
    run_id_raw = run_raw.get("run_id") or (expanded.get("run_id") if isinstance(expanded, Mapping) else None)
    proposal_width = int(run_raw.get("proposal_width", 4))
    max_rounds = int(run_raw.get("max_rounds", 5))
    max_evals = int(run_raw.get("max_candidate_evaluations", 20))
    no_improve = int(run_raw.get("max_consecutive_no_improvement", 2))
    max_proposer_failures = int(run_raw.get("max_proposer_failures", 3))
    if proposal_width < 1:
        raise ConfigError("[run].proposal_width must be >= 1")
    if max_rounds < 1:
        raise ConfigError("[run].max_rounds must be >= 1")
    if max_evals < 0:
        raise ConfigError("[run].max_candidate_evaluations must be >= 0")
    if no_improve < 1:
        raise ConfigError("[run].max_consecutive_no_improvement must be >= 1")
    proposer_raw = _mapping(expanded.get("proposer"), "[proposer]", required=False)
    diagnosis_raw = _mapping(expanded.get("diagnosis"), "[diagnosis]", required=False)
    response_raw = run_raw.get("proposer_response")
    builder_raw = proposer_raw.get("command") or run_raw.get("proposer_command")
    diagnosis_command_raw = diagnosis_raw.get("command") or run_raw.get("diagnosis_command")
    return RunSettings(
        output_root=output_root,
        run_id=str(run_id_raw).strip() if run_id_raw else None,
        max_rounds=max_rounds,
        proposal_width=proposal_width,
        max_candidate_evaluations=max_evals,
        max_consecutive_no_improvement=no_improve,
        max_proposer_failures=max_proposer_failures,
        resume=bool(run_raw.get("resume", False)),
        reuse_existing=bool(run_raw.get("reuse_existing", False)),
        proposer_command=str(builder_raw) if builder_raw else None,
        proposer_response=_resolve_path(base_dir, str(response_raw)) if response_raw else None,
        diagnosis_command=str(diagnosis_command_raw) if diagnosis_command_raw else None,
        proposer_max_slot_attempts=int(proposer_raw.get("max_slot_attempts", 1) or 1),
        raw=dict(run_raw),
    )


def _load_eval(
    *,
    eval_raw: Mapping[str, Any],
    benchmark: str,
    base_dir: Path,
    expanded: Mapping[str, Any],
) -> EvalSettings:
    default_task_timeout = 3600.0 if benchmark == "swe_bench_verified" else 900.0
    task_timeout_raw = eval_raw.get("task_timeout_s", eval_raw.get("timeout_s"))
    task_timeout = float(task_timeout_raw) if task_timeout_raw is not None else default_task_timeout
    tool_timeout = float(eval_raw.get("tool_timeout_s", 120))
    if task_timeout <= 0:
        raise ConfigError("[eval].task_timeout_s must be positive")
    if tool_timeout <= 0:
        raise ConfigError("[eval].tool_timeout_s must be positive")
    repeats = int(eval_raw.get("repeats", 2))
    if repeats < 1:
        raise ConfigError("[eval].repeats must be >= 1")
    concurrency = int(eval_raw.get("case_concurrency", 1) or 1)
    if concurrency < 1:
        raise ConfigError("[eval].case_concurrency must be >= 1")
    retries = int(eval_raw.get("max_retries", 1))
    if retries < 0 or retries > 10:
        raise ConfigError("[eval].max_retries must be between 0 and 10")
    split_raw = eval_raw.get("split") or eval_raw.get("splits") or ["train", "heldout"]
    if isinstance(split_raw, str):
        split_raw = [split_raw]
    if not isinstance(split_raw, list) or not split_raw:
        raise ConfigError("[eval].split must be a non-empty list or string")
    splits: list[str] = []
    for split in split_raw:
        name = str(split).strip()
        if name not in SPLIT_NAMES:
            raise ConfigError(f"[eval].split contains unknown split {name!r}")
        if name not in splits:
            splits.append(name)
    task_ids_raw = eval_raw.get("task_ids")
    task_ids: tuple[str, ...] | None = None
    if task_ids_raw is not None:
        if not isinstance(task_ids_raw, list) or not task_ids_raw:
            raise ConfigError("[eval].task_ids must be a non-empty list when provided")
        task_ids = tuple(str(item) for item in task_ids_raw)
    result_dir_raw = eval_raw.get("result_dir")
    legacy_env_raw = expanded.get("env")
    legacy_env = (
        {str(key): str(value) for key, value in legacy_env_raw.items()}
        if isinstance(legacy_env_raw, Mapping)
        else {}
    )
    legacy: dict[str, Any] = {
        "evals_project": _resolve_path(base_dir, str(eval_raw["evals_project"]))
        if eval_raw.get("evals_project")
        else None,
        "harness_workspace": _resolve_path(base_dir, str(eval_raw["harness_workspace"]))
        if eval_raw.get("harness_workspace")
        else None,
        "agent_import_path": eval_raw.get("agent_import_path"),
        "model_flag": str(eval_raw.get("model_flag", "--model")),
        "summary_flag": str(eval_raw.get("summary_flag", "--evals-report-file")),
        "pytest_args": [str(item) for item in eval_raw.get("pytest_args", ["-q"])],
        "pytest_command": eval_raw.get("pytest_command"),
        "timeout_s": eval_raw.get("timeout_s"),
        "env": legacy_env,
        "cases": list(expanded.get("cases") or []),
    }
    return EvalSettings(
        benchmark=benchmark,
        name=str(eval_raw.get("name") or benchmark),
        repeats=repeats,
        splits=tuple(splits),
        task_ids=task_ids,
        task_timeout_s=task_timeout,
        tool_timeout_s=tool_timeout,
        case_concurrency=concurrency,
        max_retries=retries,
        smoke_test=bool(eval_raw.get("smoke_test", False)),
        reuse_existing=bool(eval_raw.get("reuse_existing", False)),
        pilot=bool(eval_raw.get("pilot", False)),
        pilot_max_tasks=int(eval_raw.get("pilot_max_tasks", 5) or 5),
        result_dir=_resolve_path(base_dir, str(result_dir_raw)) if result_dir_raw else None,
        legacy=legacy,
        raw=dict(eval_raw),
    )


# ---------------------------------------------------------------------------
# Serialization / sanitization
# ---------------------------------------------------------------------------


def config_to_dict(config: EvalConfig) -> dict[str, Any]:
    return {
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "benchmark": config.eval.benchmark,
        "eval": {
            "name": config.eval.name,
            "repeats": config.eval.repeats,
            "splits": list(config.eval.splits),
            "task_ids": list(config.eval.task_ids) if config.eval.task_ids else None,
            "task_timeout_s": config.eval.task_timeout_s,
            "tool_timeout_s": config.eval.tool_timeout_s,
            "case_concurrency": config.eval.case_concurrency,
            "max_retries": config.eval.max_retries,
            "smoke_test": config.eval.smoke_test,
            "reuse_existing": config.eval.reuse_existing,
            "pilot": config.eval.pilot,
            "pilot_max_tasks": config.eval.pilot_max_tasks,
            "legacy": {
                "evals_project": str(config.eval.legacy.get("evals_project"))
                if config.eval.legacy.get("evals_project")
                else None,
                "harness_workspace": str(config.eval.legacy.get("harness_workspace"))
                if config.eval.legacy.get("harness_workspace")
                else None,
                "agent_import_path": config.eval.legacy.get("agent_import_path"),
                "model_flag": config.eval.legacy.get("model_flag"),
                "summary_flag": config.eval.legacy.get("summary_flag"),
                "pytest_args": config.eval.legacy.get("pytest_args"),
                "pytest_command": config.eval.legacy.get("pytest_command"),
                "timeout_s": config.eval.legacy.get("timeout_s"),
                "env": config.eval.legacy.get("env"),
            },
        },
        "dataset": {
            "name": config.dataset.name,
            "revision": config.dataset.revision,
            "manifest_path": str(config.dataset.manifest_path),
            "split": config.dataset.split,
            "source": config.dataset.source,
            "cache_dir": str(config.dataset.cache_dir) if config.dataset.cache_dir else None,
            "repos_dir": str(config.dataset.repos_dir) if config.dataset.repos_dir else None,
            "allow_hf_download": config.dataset.allow_hf_download,
        },
        "model": dict(config.model.non_secret_block()),
        "agent": {
            "kind": config.agent.kind,
            "command": _command_to_json(config.agent.command),
            "harbor_command": _command_to_json(config.agent.harbor_command),
            "extra_env": dict(sorted(config.agent.extra_env.items())),
            "env_passthrough": list(config.agent.env_passthrough),
            "max_retries": config.agent.max_retries,
        },
        "evaluator": {
            "kind": config.evaluator.kind,
            "name": config.evaluator.name,
            "version": config.evaluator.version,
            "command": _command_to_json(config.evaluator.command),
            "dataset": config.evaluator.dataset,
            "split": config.evaluator.split,
            "max_workers": config.evaluator.max_workers,
        },
        "constraints": {
            "outbound_bandwidth_mbps": config.constraints.outbound_bandwidth_mbps,
            "bandwidth_enforcement": config.constraints.bandwidth_enforcement,
            "cache_dir": str(config.constraints.cache_dir) if config.constraints.cache_dir else None,
            "docker_image": config.constraints.docker_image,
            "docker_image_digest": config.constraints.docker_image_digest,
            "dependency_pins": dict(sorted(config.constraints.dependency_pins.items())),
        },
        "harness": {
            "baseline_dir": str(config.harness.baseline_dir) if config.harness.baseline_dir else None,
            "kind": config.harness.kind,
            "surfaces": {name: str(path) for name, path in sorted(config.harness.surfaces.items())},
        },
        "run": {
            "output_root": str(config.run.output_root),
            "run_id": config.run.run_id,
            "max_rounds": config.run.max_rounds,
            "proposal_width": config.run.proposal_width,
            "max_candidate_evaluations": config.run.max_candidate_evaluations,
            "max_consecutive_no_improvement": config.run.max_consecutive_no_improvement,
            "max_proposer_failures": config.run.max_proposer_failures,
            "resume": config.run.resume,
            "reuse_existing": config.run.reuse_existing,
            "proposer_command": config.run.proposer_command,
            "proposer_response": str(config.run.proposer_response) if config.run.proposer_response else None,
            "diagnosis_command": config.run.diagnosis_command,
            "proposer_max_slot_attempts": config.run.proposer_max_slot_attempts,
        },
        "env_refs": dict(sorted(config.env_refs.items())),
    }


def sanitize_resolved_config(resolved: Mapping[str, Any], env_refs: Mapping[str, str]) -> dict[str, Any]:
    """Redact any expanded credential values from the persisted config."""

    secret_values = {
        name: os.environ.get(name, "")
        for name in env_refs
        if name in os.environ and _SECRET_NAME_RE.search(name)
    }
    return _redact_values(dict(resolved), secret_values)


def _redact_values(value: Any, secret_values: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for name, secret in secret_values.items():
            if secret and len(secret) >= 4 and secret in result:
                result = result.replace(secret, f"<redacted:{name}>")
        return result
    if isinstance(value, Mapping):
        return {str(key): _redact_values(item, secret_values) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_values(item, secret_values) for item in value]
    return value


def write_resolved_config(path: Path, config: EvalConfig) -> Path:
    resolved = sanitize_resolved_config(config_to_dict(config), config.env_refs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _reject_secret_config_keys(value: Any, *, path: str = "$") -> None:
    """Reject credential-bearing config *keys* (values must come from env)."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_CONFIG_KEY_RE.search(key_text) and key_text.lower() not in {
                "max_tokens",
                "reasoning_budget_tokens",
            }:
                raise ConfigError(
                    f"config key {path}.{key_text} must not carry credentials; inject them via "
                    "environment variables instead (for example [agent].env_passthrough)"
                )
            _reject_secret_config_keys(item, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_config_keys(item, path=f"{path}[{index}]")


def _mapping(value: Any, field_name: str, *, required: bool = True) -> Mapping[str, Any]:
    if value is None:
        if required:
            raise ConfigError(f"config must contain a {field_name} table")
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be a TOML table")
    return value


def _command_value(value: Any, *, field: str) -> str | tuple[str, ...]:
    if isinstance(value, str):
        if not value.strip():
            raise ConfigError(f"{field} must not be empty")
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return tuple(str(item) for item in value)
    raise ConfigError(f"{field} must be a string or a list of strings")


def _command_to_json(value: str | tuple[str, ...] | None) -> Any:
    if value is None:
        return None
    return list(value) if isinstance(value, tuple) else value


def _resolve_path(base_dir: Path, raw: str) -> Path:
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _sha256_json(value: Any) -> str:
    from .hashing import sha256_json

    return sha256_json(value)


def _toml_reader() -> Any:
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        try:
            import tomli

            return tomli
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ConfigError(
                "TOML config parsing requires Python 3.11+ or the 'tomli' package"
            ) from exc
