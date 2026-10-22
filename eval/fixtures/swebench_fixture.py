#!/usr/bin/env python3
"""Fixture environment builder for the SWE-bench runner smoke tests.

Creates local git repositories, a pinned-format dataset export covering three
manifest tasks, plus the fake agent/evaluator commands.  Everything is local,
deterministic and disposable so CI never downloads containers or datasets.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).resolve().parent
FAKE_AGENT = FIXTURE_ROOT / "fake_agent.py"
FAKE_COMMAND_EVALUATOR = FIXTURE_ROOT / "fake_official_evaluator.py"
FAKE_SWEBENCH_PKG = FIXTURE_ROOT / "fake_swebench_pkg"
FAKE_PROPOSER = FIXTURE_ROOT / "fake_proposer.py"

#: Three real manifest task ids (2 train + 1 heldout) used by the fixture only.
FIXTURE_TASKS = {
    "astropy__astropy-13398": {"split": "train", "repo": "astropy/astropy"},
    "sympy__sympy-21930": {"split": "train", "repo": "sympy/sympy"},
    "astropy__astropy-7336": {"split": "heldout", "repo": "astropy/astropy"},
}
GOLD_PATCH_MARKER = "GOLD_PATCH_MARKER_9f3a_do_not_leak"
GOLD_TEST_MARKER = "GOLD_TEST_PATCH_MARKER_9f3a_do_not_leak"

FIXTURE_HARNESS_SOURCE = """from __future__ import annotations


def build_system_prompt() -> str:
    return "You are a fixture SWE-bench agent."


BASELINE_SYSTEM_PROMPT = build_system_prompt()


def build_bootstrap_instruction() -> str:
    return "Inspect the repository before editing."


def build_execution_instruction() -> str:
    return "Make a bounded code change."


def build_verification_instruction() -> str:
    return "Verify the change before finishing."


def build_failure_recovery_instruction() -> str:
    return "Read tool errors and adapt."


def build_multimodal_instruction() -> str:
    return "Ground image claims in visible evidence."


def build_fixed_harness_agent(model, backend=None):
    raise NotImplementedError("fixture harness is never executed")
"""



def build_environment(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    repos_dir = root / "repos"
    data_dir = root / "data"
    repos_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    commits: dict[str, str] = {}
    for repo in sorted({task["repo"] for task in FIXTURE_TASKS.values()}):
        commits[repo] = _build_repo(repos_dir / repo)

    records = []
    for index, (task_id, spec) in enumerate(FIXTURE_TASKS.items(), start=1):
        repo = spec["repo"]
        records.append(
            {
                "instance_id": task_id,
                "repo": repo,
                "base_commit": commits[repo],
                "version": "0",
                "problem_statement": (
                    f"Fixture issue for {task_id}: make the repository return a solved result. "
                    "Edit the code in the working copy."
                ),
                "hints_text": "The fixture agent writes solution.py into the workspace.",
                "patch": f"diff --git a/fix.py b/fix.py\n+{GOLD_PATCH_MARKER}\n",
                "test_patch": f"diff --git a/tests/test_fix.py b/tests/test_fix.py\n+{GOLD_TEST_MARKER}\n",
                "FAIL_TO_PASS": json.dumps([f"tests/test_fix.py::test_{index}"]),
                "PASS_TO_PASS": json.dumps([]),
                "environment_setup_commit": commits[repo],
                "created_at": "2024-01-01T00:00:00Z",
                "difficulty": "fixture",
            }
        )
    dataset_path = data_dir / "swe_bench_verified_fixture.json"
    dataset_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    resolved_path = root / "resolved_task_ids.json"
    resolved_path.write_text(json.dumps(list(FIXTURE_TASKS), indent=2) + "\n", encoding="utf-8")

    harness_dir = root / "harness"
    harness_dir.mkdir()
    (harness_dir / "repo_baseline.py").write_text(FIXTURE_HARNESS_SOURCE, encoding="utf-8")
    return {
        "root": root,
        "repos_dir": repos_dir,
        "dataset_path": dataset_path,
        "resolved_path": resolved_path,
        "harness_dir": harness_dir,
        "task_ids": list(FIXTURE_TASKS),
        "commits": commits,
        "fake_agent": FAKE_AGENT,
        "fake_command_evaluator": FAKE_COMMAND_EVALUATOR,
        "fake_swebench_pkg": FAKE_SWEBENCH_PKG,
        "fake_proposer": FAKE_PROPOSER,
    }


def _build_repo(path: Path) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "fixture@example.invalid")
    _git(path, "config", "user.name", "Fixture")
    (path / "README.md").write_text("fixture repository\n", encoding="utf-8")
    (path / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")
    (path / "module.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "task base commit")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)


def render_swebench_config(
    *,
    env: Mapping[str, Any],
    config_path: Path,
    repeats: int = 2,
    splits: tuple[str, ...] = ("train", "heldout"),
    task_ids: tuple[str, ...] | None = None,
    evaluator_kind: str = "command",
    smoke_test: bool = True,
    case_concurrency: int = 2,
    task_timeout_s: float = 120,
    tool_timeout_s: float = 10,
    extra_eval: str = "",
) -> Path:
    fixture_ids = task_ids if task_ids is not None else tuple(env["task_ids"])
    task_ids_toml = "[" + ", ".join(f'"{item}"' for item in fixture_ids) + "]"
    split_toml = "[" + ", ".join(f'"{item}"' for item in splits) + "]"
    if evaluator_kind == "command":
        evaluator_toml = f"""
[evaluator]
kind = "command"
name = "fake-command-evaluator"
version = "fixture"
command = "{sys.executable} {env['fake_command_evaluator']} --predictions {{predictions}} --output {{evaluation_output}}"
"""
    elif evaluator_kind == "fixture":
        evaluator_toml = f"""
[evaluator]
kind = "fixture"
name = "fixture-fake-swebench-evaluator"
version = "fixture"
resolved_task_ids = [{", ".join(f'"{item}"' for item in FIXTURE_TASKS)}]
"""
    else:
        evaluator_toml = """
[evaluator]
kind = "official"
split = "test"
max_workers = 1
"""
    text = f"""
[eval]
benchmark = "swe_bench_verified"
name = "swebench-fixture-smoke"
repeats = {repeats}
split = {split_toml}
task_ids = {task_ids_toml}
task_timeout_s = {task_timeout_s}
tool_timeout_s = {tool_timeout_s}
case_concurrency = {case_concurrency}
max_retries = 1
smoke_test = {str(smoke_test).lower()}
{extra_eval}

[dataset]
name = "princeton-nlp/SWE-bench_Verified"
revision = "fixture-revision-2024-01-01"
manifest_path = "{REPO_ROOT / 'swe_bench_verified_stratified_100.json'}"
source = "{env['dataset_path']}"
repos_dir = "{env['repos_dir']}"
allow_hf_download = false

[agent]
kind = "command"
command = "{sys.executable} {env['fake_agent']} --workspace {{workspace}} --case-dir {{case_dir}} --instruction {{instruction_file}}"
env_passthrough = ["SELF_HARNESS_FAKE_AGENT_BEHAVIOR", "SELF_HARNESS_FAKE_AGENT_SLEEP", "SELF_HARNESS_FAKE_RESOLVED_TASKS"]

[model]
provider = "fixture"
model_id = "fixture-model"
endpoint_type = "local"
temperature = 0.0
top_p = 1.0
max_tokens = 256
{evaluator_toml}
[constraints]
outbound_bandwidth_mbps = 2.0
bandwidth_enforcement = "none"

[harness]
baseline_dir = "{env['harness_dir']}"

[harness.surfaces]
baseline = "{Path(env['harness_dir']) / 'repo_baseline.py'}"

[run]
output_root = "{Path(env['root']) / 'runs'}"
proposal_width = 2
max_rounds = 2
max_candidate_evaluations = 4
max_consecutive_no_improvement = 1
proposer_command = "{sys.executable} {env['fake_proposer']} --prompt {{prompt}} --response {{response}} --width {{width}} --train-ids-file {{train_ids_file}}"
"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    return config_path


def fake_official_pythonpath(env: Mapping[str, Any]) -> str:
    return str(env["fake_swebench_pkg"])


def set_fake_official_environment(env: Mapping[str, Any]) -> None:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(env["fake_swebench_pkg"]), os.environ.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    sys.path.insert(0, str(env["fake_swebench_pkg"]))


def clear_fake_official_environment(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = previous
    if str(FAKE_SWEBENCH_PKG) in sys.path:
        sys.path.remove(str(FAKE_SWEBENCH_PKG))
