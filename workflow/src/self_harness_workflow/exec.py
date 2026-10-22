"""Subprocess execution helpers for the Self-Harness workflow.

The workflow always drives the public scripts (evaluation dispatcher, acceptance
gate, diagnosis brief builder, proposer prompt builder/parser) as subprocesses.
That keeps each stage reproducible from the command line and prevents hidden
in-process state from making a run impossible to audit.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_harness_eval.hashing import sha256_file

from .runstate import RunPaths, WorkflowError, utc_now

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CommandOutcome:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": self.argv,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
        }


def _scripts() -> dict[str, Path]:
    return {
        "eval": REPO_ROOT / "eval" / "scripts" / "run_eval.py",
        "gate": REPO_ROOT / "acceptance" / "scripts" / "run_acceptance_gate.py",
        "diagnosis": REPO_ROOT / "diagnosis" / "scripts" / "build_diagnosis_brief.py",
        "proposer": REPO_ROOT / "proposer" / "scripts" / "run_multi_proposer.py",
        "materialize": REPO_ROOT / "proposer" / "scripts" / "materialize_candidate.py",
    }


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path | str,
    env: Mapping[str, str] | None = None,
    timeout_s: float | None = None,
) -> CommandOutcome:
    started = __import__("time").time()
    try:
        completed = subprocess.run(
            [str(item) for item in argv],
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=timeout_s,
        )
        return CommandOutcome(
            argv=[str(item) for item in argv],
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_s=__import__("time").time() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandOutcome(
            argv=[str(item) for item in argv],
            returncode=124,
            stdout=_normalize(exc.stdout),
            stderr=_normalize(exc.stderr),
            duration_s=__import__("time").time() - started,
            timed_out=True,
        )
    except OSError as exc:
        return CommandOutcome(
            argv=[str(item) for item in argv],
            returncode=127,
            stdout="",
            stderr=f"{exc.__class__.__name__}: {exc}",
            duration_s=__import__("time").time() - started,
        )


def run_eval_stage(
    *,
    paths: RunPaths,
    config_path: Path,
    version_id: str,
    harness_snapshot: Path,
    output_dir: Path,
    kind: str,
    parent_version_ids: Sequence[str],
    run_id: str,
    candidate_workspace: Path | None,
    reuse_existing: bool,
    log_path: Path,
) -> tuple[dict[str, Any], CommandOutcome]:
    argv = [
        sys.executable,
        str(_scripts()["eval"]),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-root",
        str(paths.root),
        "--run-id",
        run_id,
        "--result-id",
        version_id,
        "--harness-snapshot",
        str(harness_snapshot),
        "--harness-version-id",
        version_id,
        "--harness-kind",
        kind,
    ]
    for parent in parent_version_ids:
        argv.extend(["--parent-version-id", str(parent)])
    if reuse_existing:
        argv.append("--reuse-existing")
    env = os.environ.copy()
    env["SELF_HARNESS_RUN_ID"] = run_id
    if candidate_workspace is not None:
        env["SELF_HARNESS_CANDIDATE_WORKSPACE"] = str(candidate_workspace)
    outcome = run_command(argv, cwd=REPO_ROOT, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {shlex.join(argv)}\n\n--- stdout ---\n{outcome.stdout}\n--- stderr ---\n{outcome.stderr}\n",
        encoding="utf-8",
    )
    result_path = output_dir / "result.json"
    if not result_path.exists():
        raise WorkflowError(
            f"evaluation produced no result for {version_id}: "
            f"exit={outcome.returncode} stderr={outcome.stderr[-1200:]}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return result, outcome


def run_acceptance_stage(
    *,
    paths: RunPaths,
    baseline_result: Path,
    candidate_result: Path,
    output_path: Path,
    expected_repeats: int,
    reuse_existing: bool,
    log_path: Path,
) -> tuple[dict[str, Any], CommandOutcome]:
    argv = [
        sys.executable,
        str(_scripts()["gate"]),
        "--baseline-result",
        str(baseline_result),
        "--candidate-result",
        str(candidate_result),
        "--output",
        str(output_path),
        "--expected-repeats",
        str(expected_repeats),
    ]
    if reuse_existing:
        argv.append("--reuse-existing")
    outcome = run_command(argv, cwd=REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {shlex.join(argv)}\n\n--- stdout ---\n{outcome.stdout}\n--- stderr ---\n{outcome.stderr}\n",
        encoding="utf-8",
    )
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return payload, outcome
    if not outcome.ok:
        raise WorkflowError(
            f"acceptance gate failed for {candidate_result}: exit={outcome.returncode} "
            f"stderr={outcome.stderr[-1200:]}"
        )
    raise WorkflowError(f"acceptance gate wrote no decision file: {output_path}")


def build_proposer_prompt_stage(
    *,
    paths: RunPaths,
    diagnosis_path: Path,
    surfaces: Sequence[tuple[str, Path]],
    route_count: int,
    output_path: Path,
    heldout_ids_path: Path,
    train_ids_path: Path | None,
    isolation_path: Path,
    log_path: Path,
) -> CommandOutcome:
    argv = [
        sys.executable,
        str(_scripts()["proposer"]),
        "--diagnosis",
        str(diagnosis_path),
        "--route-count",
        str(route_count),
        "--output",
        str(output_path),
        "--heldout-ids-file",
        str(heldout_ids_path),
        "--isolation-output",
        str(isolation_path),
        "--strict-isolation",
    ]
    if train_ids_path is not None:
        argv.extend(["--train-ids-file", str(train_ids_path)])
    for name, path in surfaces:
        argv.extend(["--surface", f"{name}={path}"])
    outcome = run_command(argv, cwd=REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {shlex.join(argv)}\n\n--- stdout ---\n{outcome.stdout}\n--- stderr ---\n{outcome.stderr}\n",
        encoding="utf-8",
    )
    return outcome


def parse_proposer_response_stage(
    *,
    diagnosis_path: Path,
    surfaces: Sequence[tuple[str, Path]],
    response_path: Path,
    output_path: Path,
    heldout_ids_path: Path,
    train_ids_path: Path | None,
    isolation_path: Path,
    log_path: Path,
) -> CommandOutcome:
    argv = [
        sys.executable,
        str(_scripts()["proposer"]),
        "--diagnosis",
        str(diagnosis_path),
        "--response",
        str(response_path),
        "--output",
        str(output_path),
        "--heldout-ids-file",
        str(heldout_ids_path),
        "--isolation-output",
        str(isolation_path),
        "--strict-isolation",
    ]
    if train_ids_path is not None:
        argv.extend(["--train-ids-file", str(train_ids_path)])
    for name, path in surfaces:
        argv.extend(["--surface", f"{name}={path}"])
    outcome = run_command(argv, cwd=REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {shlex.join(argv)}\n\n--- stdout ---\n{outcome.stdout}\n--- stderr ---\n{outcome.stderr}\n",
        encoding="utf-8",
    )
    return outcome


def materialize_candidate_stage(
    *,
    proposal_bundle: Path,
    proposal_id: str,
    candidate_id: str,
    surfaces: Sequence[tuple[str, Path]],
    surface_filenames: Mapping[str, str],
    output_dir: Path,
    log_path: Path,
) -> CommandOutcome:
    argv = [
        sys.executable,
        str(_scripts()["materialize"]),
        "--bundle",
        str(proposal_bundle),
        "--proposal-id",
        proposal_id,
        "--candidate-id",
        candidate_id,
        "--output-dir",
        str(output_dir),
    ]
    for name, path in surfaces:
        argv.extend(["--surface", f"{name}={path}"])
        relative_path = surface_filenames.get(name)
        if relative_path:
            argv.extend(["--surface-filename", f"{name}={relative_path}"])
    outcome = run_command(argv, cwd=REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {shlex.join(argv)}\n\n--- stdout ---\n{outcome.stdout}\n--- stderr ---\n{outcome.stderr}\n",
        encoding="utf-8",
    )
    return outcome


def build_diagnosis_stage(
    *,
    result_path: Path,
    output_path: Path,
    evidence_path: Path,
    heldout_ids_path: Path,
    train_ids_path: Path | None,
    run_root: Path,
    log_path: Path,
) -> CommandOutcome:
    argv = [
        sys.executable,
        str(_scripts()["diagnosis"]),
        "--result",
        str(result_path),
        "--output",
        str(output_path),
        "--evidence-output",
        str(evidence_path),
        "--heldout-ids-file",
        str(heldout_ids_path),
        "--run-root",
        str(run_root),
    ]
    if train_ids_path is not None:
        argv.extend(["--train-ids-file", str(train_ids_path)])
    outcome = run_command(argv, cwd=REPO_ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {shlex.join(argv)}\n\n--- stdout ---\n{outcome.stdout}\n--- stderr ---\n{outcome.stderr}\n",
        encoding="utf-8",
    )
    return outcome


def run_external_template(
    template: str,
    *,
    placeholders: Mapping[str, Any],
    cwd: Path,
    timeout_s: float | None,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> CommandOutcome:
    try:
        tokens = shlex.split(template)
    except ValueError as exc:
        raise WorkflowError(f"cannot parse external command template: {exc}") from exc
    mapping = {key: str(value) for key, value in placeholders.items()}
    try:
        argv = [token.format(**mapping) for token in tokens]
    except KeyError as exc:
        raise WorkflowError(f"unknown placeholder in external command: {exc}") from exc
    outcome = run_command(argv, cwd=cwd, timeout_s=timeout_s, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {shlex.join(argv)}\n\n--- stdout ---\n{outcome.stdout}\n--- stderr ---\n{outcome.stderr}\n",
        encoding="utf-8",
    )
    return outcome


def copy_or_link(source: Path, destination: Path) -> str:
    """Copy a small artifact when it must live under the run root."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(Path(source).read_bytes())
    return sha256_file(destination)


def _normalize(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def timestamp() -> str:
    return utc_now()
