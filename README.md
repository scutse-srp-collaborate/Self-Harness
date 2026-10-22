# Self-Harness: Harnesses That Improve Themselves

<p align="center">
  <a href="https://arxiv.org/abs/2606.09498">
    <img src="https://img.shields.io/badge/arXiv-2606.09498-b31b1b.svg" alt="arXiv:2606.09498">
  </a>
</p>

<p align="center">
  <img src="assets/figure1.png" alt="Figure 1: Three paradigms of harness improvement" width="900">
</p>

## Overview

Self-Harness keeps the model weights and evaluator fixed while improving the
surrounding harness. Each round evaluates the current harness on tasks, mines
failure patterns from execution traces, asks the same model to propose bounded
harness edits, and promotes a candidate only when held-in and held-out
regression checks support the change.

<p align="center">
  <img src="assets/overview.png" alt="Overview of the Self-Harness optimization loop" width="900">
</p>

## Runnable experiment harness

This repository now contains a runnable, reproducible and auditable
Self-Harness experiment pipeline for two benchmarks:

- **Terminal-Bench 2.0 Clean64** (43 train / 21 heldout), executed through the
  existing Harbor + pytest runner. A single example case is **not** a
  Terminal-Bench evaluation; the shipped config selects the complete fixed
  Clean64 split.
- **SWE-bench Verified stratified 100** (`swe_bench_verified_stratified_100.json`,
  67 train / 33 heldout), executed by a dedicated SWE-bench runner that prepares
  the correct repository/base-commit workspace, collects the agent patch and
  delegates the resolved decision to the SWE-bench evaluator.

Both runners emit the same immutable `self_harness.eval_result.v1` result, and
candidate admission is a separate non-regression Pareto `acceptance.json`.

### Repository layout

```text
eval/src/self_harness_eval/        unified protocol, configs, manifests, dataset,
                                   Terminal-Bench and SWE-bench runners
eval/scripts/run_eval.py           unified evaluation dispatcher
eval/scripts/run_harbor_eval.py    Terminal-Bench 2.0 (Harbor + pytest) runner
eval/scripts/run_swebench_eval.py  SWE-bench Verified runner
eval/scripts/run_smoke_swebench.py offline fixture smoke test
eval/configs/                      example configs for both benchmarks
eval/manifests/tb2_clean64.json    fixed Terminal-Bench Clean64 split
eval/fixtures/                     fake agent/evaluator fixtures (tests/smoke only)
diagnosis/                         train-only evidence loaders + diagnosis brief
proposer/                          mechanism-diverse proposer + held-out isolation guard
acceptance/                        non-regression Pareto acceptance gate
workflow/src/self_harness_workflow run state, snapshots, versions, events, merge, reports
workflow/scripts/run_self_harness.py  experiment CLI (init/baseline/round/run/resume/report/eval)
tests/                             protocol, manifest, runner, gate, isolation, workflow tests
```

### Quick start

```bash
# Offline smoke test (no network, Docker image, model endpoint or dataset download):
python3 eval/scripts/run_smoke_swebench.py            # evaluation runner only
python3 eval/scripts/run_smoke_swebench.py --mode workflow

# Unit/integration test suite (adapter venv: includes DeepAgents/Docker integration tests):
.venv/bin/pytest -q tests
# Standard-library subset (DeepAgents adapter tests are skipped when deps are absent):
python3 -m unittest discover -s tests -v
```

A real experiment is a single command driven by a TOML config:

```bash
# 1) terminal-bench 2.0 clean64 or swe-bench verified 100
python3 workflow/scripts/run_self_harness.py run \
  --config eval/configs/swebench_verified_100.example.toml \
  --run-id swebench-100-run01 \
  --max-rounds 5

# Inspect the run:
#   runs/<run_id>/run_manifest.json   run state and experiment fingerprint
#   runs/<run_id>/config.resolved.json, dataset_manifest.json
#   runs/<run_id>/versions/vNNN_*/{version.json,harness/,proposal.json,eval/result.json,acceptance.json}
#   runs/<run_id>/rounds/round-NNN/   prompt, response, proposal bundle, isolation audit
#   runs/<run_id>/events.jsonl        append-only event log
#   runs/<run_id>/final_report.json|.md
#
# The final version is the last accepted/active harness version; its snapshot is
# the same immutable directory, marked with status "final" in version.json.

# Other commands:
python3 workflow/scripts/run_self_harness.py init     --config ...
python3 workflow/scripts/run_self_harness.py baseline --config ...
python3 workflow/scripts/run_self_harness.py round    --config ...
python3 workflow/scripts/run_self_harness.py resume   --config ... --max-rounds 2
python3 workflow/scripts/run_self_harness.py report   --config ...
python3 workflow/scripts/run_self_harness.py eval     --config ... --version-id v003_round001_merged
```

Candidate evaluation can also be run directly with the unified dispatcher:

```bash
python3 eval/scripts/run_eval.py --config <config.toml> --output-dir <dir> \
  --run-root <run> --run-id <run> --result-id <version> \
  --harness-snapshot <run>/versions/<version>/harness \
  --harness-version-id <version> --harness-kind candidate \
  --parent-version-id <parent> [--split train] [--task-id ID] [--repeats 2] [--reuse-existing]
```

### Real SWE-bench pilot with DeepAgents + OpenAI-compatible models

The SWE-bench runner can execute real tasks with a DeepAgents adapter and an
OpenAI-compatible hosted model:

```bash
# One-time: pinned data + pilot repo clones (outside the Git tree)
.venv/bin/python eval/scripts/prepare_swebench_data.py

# Credentials are injected only through the environment (never TOML/argv/logs).
# A gitignored, chmod-600 repo-local .env is also loaded automatically if present:
export SELF_HARNESS_MODEL_API_KEY=...          # required for real calls
export SELF_HARNESS_MODEL=deepseek-flash       # default
export OPENAI_BASE_URL=https://api.qadg.xyz/v1 # default

# Read-only preflight, then the bounded pilot (max 5 fixed train tasks, 2 repeats,
# no heldout, no candidate iteration):
.venv/bin/python eval/scripts/swebench_preflight.py
.venv/bin/python eval/scripts/run_swebench_pilot.py   --config eval/configs/swebench_pilot_5.toml --run-id swebench-pilot5-001
```

- Execution adapter: [eval/scripts/run_deepagent_swebench.py](eval/scripts/run_deepagent_swebench.py)
  builds `ChatOpenAI`, dynamically loads the current Harness snapshot's
  `repo_baseline.py` (`build_fixed_harness_agent`), and runs DeepAgents tools in
  a disposable task-scoped Docker container that mounts only the task workspace.
- Proposer adapter: [proposer/scripts/run_openai_compatible_proposer.py](proposer/scripts/run_openai_compatible_proposer.py)
  uses the same model/endpoint/key/decoding and writes JSON consumable by the
  existing parser/validation.
- Dependencies are pinned in [eval/requirements-deepagent.txt](eval/requirements-deepagent.txt)
  (`deepagents==0.7.29`, `langchain-openai==1.7.2`, `swebench==4.0.5`, ...).
- Data/repo/cache root defaults to `../Self-Harness-data/`, overridable with
  `SELF_HARNESS_DATA_ROOT`.

The pilot is deliberately limited to five fixed train tasks: it is **not** the
100-task SWE-bench Verified benchmark and it never uses heldout results or
candidate acceptance. Because this run uses DeepSeek Flash while the paper uses
MiniMax M2.5 / GLM-5 / Qwen3.5-35B-A3B, any pilot result is described as
"evaluating Self-Harness on DeepSeek Flash", not a model-for-model reproduction.
See [eval/data/README.md](eval/data/README.md) for preparation, evaluator and
pilot details.

### Prerequisites that live outside this repository

- **Terminal-Bench 2.0**: a Harbor eval project that exposes the fixed Clean64
  tasks to pytest, the Terminal-Bench 2.0 task data and a working container
  runtime. `eval/configs/tb2_clean64.example.toml` documents every required
  value. The runner keeps the existing Harbor + pytest/JUnit behaviour and adds
  the unified result protocol, task/tool timeouts, a 2 MB/s outbound bandwidth
  capability check and a legacy aggregate (`result.legacy.json`).
- **SWE-bench Verified**: a pinned dataset revision (local export or HuggingFace
  with a fixed revision), local repository clones containing the base commits,
  and the official `swebench` evaluator with Docker (or a configured
  containerized official-harness command). See `eval/data/README.md`.
- **Model**: provider/model/endpoint are configuration; credentials are injected
  through environment variables. No key is written to any result, config
  snapshot, log index or event.

`final_report.json` explicitly separates measured results from not-executed
items and sets `claims.reproduces_paper_results=false`. A fixture/smoke run is
labeled as such and must not be reported as a benchmark result.

### Unified result protocol (summary)

Every evaluation writes one immutable `result.json`:

```text
schema_version, run_id, result_id, status, benchmark, model, harness, dataset,
config, started_at, finished_at, duration_s,
splits: {train: [repeat...], heldout: [repeat...]}, summary, artifacts, errors
```

- `run_id` is constant across the baseline and all candidates of one experiment;
  `result_id` is unique per run and is the harness version id.
- `status=complete` only when every configured task has a scoreable evaluator
  result; `partial` when some tasks are unscoreable; `failed` when the run could
  not execute (all `not_run`) or preflight infrastructure failed.
- Case statuses: `resolved` (SWE-bench success, official evaluator
  `resolved=true`), `passed` (Terminal-Bench verifier success), `unresolved` /
  `failed` / `agent_timeout` / `agent_error` (scoreable failures, `passed=false`),
  `infra_error` / `evaluator_error` / `not_run` (unscoreable, `passed=null`).
- `total` always equals the fixed configured task count; `passed`, `pass_rate`,
  repeat/split/summary metrics are recomputed from `case_results` and validated.
- Artifact paths are relative to the run root; result files are never
  overwritten. Corrections produce a new `result_id`/directory.

### Acceptance gate

`acceptance/scripts/run_acceptance_gate.py` compares a candidate to its parent
using `mean_repeat_pass_rate` per split. It accepts only when:

```text
delta_train >= 0 and delta_heldout >= 0 and (delta_train > 0 or delta_heldout > 0)
```

It additionally requires identical benchmark/evaluator, model/decoding/tools/
environment/data/manifest/task-list hashes, identical split/repeat ids and
task-id sets, identical denominators, complete scoreable results, and metrics
that can be reproduced from `case_results`. Non-comparable results are recorded
as `invalid`; unscoreable results as `inconclusive` — never as ordinary rejects.

### Held-out isolation

Diagnosis and proposer inputs are built only from `split=train`; the evidence
loader rejects held-out task ids and marks missing evidence instead of inventing
conclusions. The proposer prompt builder audits prompt/response text against the
held-out id list and the workflow records that audit in
`rounds/round-NNN/proposer_isolation.json`. Held-out artifacts stay in
`eval/result.json` for acceptance gating and reporting only.


## Results

Terminal-Bench-2.0 pass rates (%). More results are coming soon.

| Model | Initial | Self-Harness |
| --- | ---: | ---: |
| MiniMax M2.5 | 42.2 | 53.9 |
| Qwen3.5-35B-A3B | 18.0 | 36.7 |
| GLM-5 | 46.1 | 57.0 |

## Citation

```bibtex
@misc{zhang2026selfharnessharnessesimprove,
      title={Self-Harness: Harnesses That Improve Themselves},
      author={Hangfan Zhang and Shao Zhang and Kangcong Li and Chen Zhang and Yang Chen and Yiqun Zhang and Lei Bai and Shuyue Hu},
      year={2026},
      eprint={2606.09498},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.09498},
}
```
