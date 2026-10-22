#!/usr/bin/env python3
"""Fixture environment builder for the Terminal-Bench runner smoke tests."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).resolve().parent
FAKE_PYTEST = FIXTURE_ROOT / "fake_pytest.py"

#: Three real Clean64 task ids (2 train + 1 heldout) used by the fixture only.
FIXTURE_TASKS = {
    "extract_elf": "train",
    "regex_chess": "train",
    "headless_terminal": "heldout",
}


def build_environment(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    evals_project = root / "evals_project"
    evals_project.mkdir(parents=True)
    (evals_project / "pyproject.toml").write_text(
        '[project]\nname = "fixture-evals"\nversion = "0.0.0"\n', encoding="utf-8"
    )
    harness_dir = root / "harness_workspace"
    harness_dir.mkdir()
    (harness_dir / "repo_baseline.py").write_text(
        "def build_fixed_harness_agent(model, backend=None):\n"
        "    raise NotImplementedError('fixture harness is never executed')\n",
        encoding="utf-8",
    )
    (harness_dir / "harbor_wrapper.py").write_text(
        "class AgentWrapper:\n    pass\n", encoding="utf-8"
    )
    return {
        "root": root,
        "evals_project": evals_project,
        "harness_dir": harness_dir,
        "fake_pytest": FAKE_PYTEST,
        "task_ids": list(FIXTURE_TASKS),
        "splits": FIXTURE_TASKS,
    }


def render_tb2_config(
    *,
    env: Mapping[str, Any],
    config_path: Path,
    repeats: int = 2,
    splits: tuple[str, ...] = ("train", "heldout"),
    task_ids: tuple[str, ...] | None = None,
    case_concurrency: int = 2,
    task_timeout_s: float = 60,
    tool_timeout_s: float = 15,
    bandwidth_mbps: float | None = 2.0,
    bandwidth_enforcement: str = "none",
) -> Path:
    fixture_ids = task_ids if task_ids is not None else tuple(env["task_ids"])
    task_ids_toml = "[" + ", ".join(f'"{item}"' for item in fixture_ids) + "]"
    split_toml = "[" + ", ".join(f'"{item}"' for item in splits) + "]"
    bandwidth_line = (
        f"outbound_bandwidth_mbps = {bandwidth_mbps}" if bandwidth_mbps is not None else ""
    )
    text = f"""
[eval]
benchmark = "terminal_bench_2"
name = "tb2-fixture-smoke"
repeats = {repeats}
split = {split_toml}
task_ids = {task_ids_toml}
task_timeout_s = {task_timeout_s}
tool_timeout_s = {tool_timeout_s}
case_concurrency = {case_concurrency}
evals_project = "{env['evals_project']}"
harness_workspace = "{env['harness_dir']}"
agent_import_path = "self_harness_harbor.harbor_wrapper:AgentWrapper"
model_flag = "--model"
summary_flag = "--evals-report-file"
pytest_args = ["-q"]
selector_template = "tests/evals/{{task_id}}.py"
pytest_command = ["{sys.executable}", "{env['fake_pytest']}"]

[dataset]
manifest_path = "{REPO_ROOT / 'eval' / 'manifests' / 'tb2_clean64.json'}"

[model]
provider = "fixture"
model_id = "fixture-model"
endpoint_type = "local"
temperature = 0.0
top_p = 1.0
max_tokens = 256

[constraints]
{bandwidth_line}
bandwidth_enforcement = "{bandwidth_enforcement}"

[harness]
baseline_dir = "{env['harness_dir']}"

[run]
output_root = "{Path(env['root']) / 'runs'}"
proposal_width = 2
max_rounds = 2
max_candidate_evaluations = 4
"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    return config_path


def write_legacy_config(env: Mapping[str, Any], config_path: Path) -> Path:
    text = f"""
[eval]
name = "tb2-legacy"
model = "fixture-model"
repeats = 2
evals_project = "{env['evals_project']}"
harness_workspace = "{env['harness_dir']}"
agent_import_path = "self_harness_harbor.harbor_wrapper:AgentWrapper"
model_flag = "--model"
summary_flag = "--evals-report-file"
pytest_args = ["-q"]
timeout_s = 60
pytest_command = ["{sys.executable}", "{env['fake_pytest']}"]

[[cases]]
case_id = "tests/evals/extract_elf.py"
task_id = "extract_elf"
split = "train"
stratum = "clean64"

[[cases]]
case_id = "tests/evals/headless_terminal.py"
task_id = "headless_terminal"
split = "heldout"
stratum = "clean64"
"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    return config_path


def fake_manifest_path() -> Path:
    return REPO_ROOT / "eval" / "manifests" / "tb2_clean64.json"


def manifest_dict() -> dict[str, Any]:
    return json.loads(fake_manifest_path().read_text(encoding="utf-8"))
