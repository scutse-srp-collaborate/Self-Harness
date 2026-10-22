# SWE-bench Verified data, model endpoint and evaluator setup

This document describes the **pinned** SWE-bench Verified data preparation, the
OpenAI-compatible model configuration, the official evaluator and the pilot /
full-run entry points. No benchmark score is recorded here.

## 0. Fixed experiment configuration

| Item | Value |
| --- | --- |
| Benchmark | SWE-bench Verified |
| Manifest | repository root `swe_bench_verified_stratified_100.json` (67 train / 33 heldout, five-domain stratified sample; **not** repository-balanced) |
| Held-in / train | `held_in_task_ids` (67) |
| Held-out | `held_out_task_ids` (33) |
| Model | `deepseek-flash` (OpenAI-compatible hosted API) |
| Base URL | `https://api.qadg.xyz/v1` |
| API key source | environment variable `SELF_HARNESS_MODEL_API_KEY` |
| Decoding | `temperature=0.0`, `top_p=1.0`, `max_tokens=8192` |
| Repeats | 2 |
| Case timeout / tool timeout | 3600 s / 120 s |
| Proposal width / iterations | K=4, max_rounds=5, max_candidate_evaluations=8 |
| Acceptance | train and heldout must not regress, at least one must improve |
| Data root | `../Self-Harness-data/` (override with `SELF_HARNESS_DATA_ROOT`) |

> **Model limitation:** the paper's results use MiniMax M2.5, GLM-5 and
> Qwen3.5-35B-A3B. This repository evaluates the Self-Harness method on
> DeepSeek Flash; it is **not** a model-for-model reproduction of the paper.

## 1. Data directory layout

Everything large lives outside the Git working tree by default:

```text
../Self-Harness-data/
  datasets/
    swe_bench_verified_100.jsonl        # 100 manifest records, manifest order
    swe_bench_verified_100.json         # same records as a JSON array (official evaluator input)
    swe_bench_verified_100.meta.json    # revision + hashes + counts
    pilot_5_task_ids.json               # deterministic 5-task train pilot list
  repos/
    astropy/astropy/  django/django/  matplotlib/matplotlib/
    pydata/xarray/    sphinx-doc/sphinx/  ... (each contains base commits)
  cache/
    swebench/  huggingface/
  runs/        # pilot / full-run artifacts (not committed)
  logs/        # prepare + preflight reports
```

Prepare the pinned data and the pilot repositories (network required):

```bash
.venv/bin/python eval/scripts/prepare_swebench_data.py
```

The script pins the HuggingFace dataset repository commit
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a` (Verified test split). The export
produced for this environment has:

- source parquet SHA-256 `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd`
- JSONL export SHA-256 `140def9433d4586ef25281480730b3eb0e36bce152bb10f38b871ecfadc88c6b`
- manifest SHA-256 `e8358cc6d30986a646cdb06d4532c7bfe7e39a1495540974f5d8d65c1ca4b2ed`
- 100/67/33 records selected from the 500-instance Verified test split.

It writes:

- source parquet SHA-256 and export JSON/JSONL SHA-256,
- revision, manifest SHA-256, manifest task-id hash,
- 67/33/100 counts, and the note that the paper does not publish a dataset
  revision (this pinned commit is the **reproduction configuration**, not a
  claimed paper revision),
- the deterministic five-task train pilot list (one task per domain and five
  distinct repositories).

If the remote dataset HEAD no longer equals the pinned revision, the script
refuses to proceed. The main 100-task manifest is never modified.

## 2. Model endpoint and API key

```bash
# Required for real model calls. Do NOT paste the key into chat/config/argv.
export SELF_HARNESS_MODEL_API_KEY=...        # value is read only by the adapter processes
export SELF_HARNESS_MODEL=deepseek-flash     # optional override (default deepseek-flash)
export OPENAI_BASE_URL=https://api.qadg.xyz/v1  # optional override (this is the default)
```

For local runs the entry points also read the repository-local `.env` when the
variable is not already in the process environment. That file is gitignored and
should be `chmod 600`; never commit it or paste its contents. Tests force
`SELF_HARNESS_DISABLE_DOTENV=1` so a developer's local key can never enter a
test run. The loader returns/logs only variable names, never values.

Both the execution agent and the proposer read the same variables and use the
same decoding configuration. The key is never written to TOML, `result.json`,
`resolved config`, traces, run manifests or reports; it is also not passed into
the task Docker container and is stripped from evaluator subprocess environments.

## 3. Adapter virtualenv and dependencies

The current repository has no root dependency manager, so the validated pins are
in `eval/requirements-deepagent.txt`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r eval/requirements-deepagent.txt
```

Validated versions include `deepagents==0.7.29`, `langchain-openai==1.7.2`,
`langchain==1.4.6`, `openai==3.35.1`, `swebench==4.0.5`,
`docker==7.2.0`, `datasets==5.1.0`.

Interface self-checks (no real model call, no API key needed for the interface
check):

```bash
.venv/bin/python eval/scripts/run_deepagent_swebench.py --self-check \
  --instruction /dev/null --workspace /tmp --case-dir /tmp/sh-selfcheck \
  --task-id check --repo a/b --base-commit x --harness-snapshot eval/harness_workspace
.venv/bin/python proposer/scripts/run_openai_compatible_proposer.py --self-check \
  --prompt /dev/null --response /tmp/proposer-selfcheck.json
```

## 4. Execution adapter (DeepAgents)

`eval/scripts/run_deepagent_swebench.py` is the real agent command used by
`[agent].command`:

- builds `ChatOpenAI` from the environment (model/base URL/key),
- dynamically imports `repo_baseline.py` from the current
  `SELF_HARNESS_CANDIDATE_WORKSPACE`/Harness snapshot (unique module per run;
  import caches are purged so a candidate cannot reuse a previous version),
- calls the Harness `build_fixed_harness_agent(model, backend=...)`,
- runs DeepAgents tools inside a **task-scoped Docker container** that mounts
  only the task workspace (`/testbed`); harness snapshot, datasets, repository
  cache, credentials and the rest of the host are not mounted,
- enforces the 120 s tool timeout in the container executor and the 3600 s case
  timeout around the agent invocation,
- writes a JSONL trace with model/tool/command/timeout/error events (secrets
  redacted) and an `agent_result.json` with container/image metadata.

The runner (not the agent's final text) collects the workspace patch and the
official SWE-bench evaluator decides `resolved`.

Recorded agent sandbox image: `python:3.12`, image ID
`sha256:68f411ef38fc2d8176cef518099aa88dfb8fbfd7a4c50587c60e1c8d1fdc843e`
also recorded in the config as `docker_image_digest`.

## 5. Proposer adapter

`proposer/scripts/run_openai_compatible_proposer.py` consumes the workflow's
train-only proposer prompt and uses the same model/endpoint/key/decoding. It:

- validates exactly `--width` proposals, unique ids/signatures and required
  fields before writing a response consumable by the existing parser,
- rejects invalid JSON/counts/duplicates after a bounded retry,
- audits the response against held-out task ids when `--heldout-ids-file` is
  provided,
- writes no credentials to the response, trace or logs.

The workflow runs it from the repository root with
`[run].proposer_command` (see `eval/configs/swebench_verified_100.example.toml`).

## 6. Official evaluator

`[evaluator].kind="official"` invokes the pinned classic evaluator
`swebench==4.0.5` (`python -m swebench.harness.run_evaluation`) with the local
JSON export, `--instance_ids`, `--report_dir` and the configured timeout.
Resolved labels come only from its report. Docker is required; the evaluator
builds/uses the per-instance SWE-bench images. Evaluator subprocesses never
receive the model API key.

Version note: `swebench==4.0.5` is the validated classic harness for the pinned
`SWE-bench_Verified` parquet export. Newer 5.x harnesses expect an enriched
"task repo" instance format (image/eval_script/log_parser fields) that the
classic HuggingFace Verified export does not contain; they fail with
`KeyError: 'image'` when pointed at the classic export. The pinned version and
local dataset were validated by building/running the evaluator on one pilot
task with a dummy non-empty patch (resolved=false, no errors); see
`../Self-Harness-data/logs/evaluator-preflight-405-nonempty/` and
`../Self-Harness-data/logs/swebench-evaluator-preflight-logs/`.

If the evaluator cannot run (missing package, Docker, image build failure,
malformed report), the result records `evaluator_error`/`infra_error` and the
task is unscoreable (`passed=null`); it is never counted as an agent failure or
a resolved label.

## 7. Running the bounded pilot (max 5 train tasks)

```bash
.venv/bin/python eval/scripts/swebench_preflight.py            # read-only checks
.venv/bin/python eval/scripts/run_swebench_pilot.py \
  --config eval/configs/swebench_pilot_5.toml \
  --run-id swebench-pilot5-001
```

Pilot properties:

- tasks: 5 fixed train ids, one per domain / distinct repository
  (`astropy__astropy-13398`, `django__django-14017`,
  `matplotlib__matplotlib-24627`, `pydata__xarray-6461`,
  `sphinx-doc__sphinx-8548`);
- repeats: 2, case concurrency 1, no heldout, no proposer/candidates/acceptance;
- output: `../Self-Harness-data/runs/<run_id>/result.json` (unified schema,
  marked with a `pilot` block), `pilot_summary.json`, `pilot_summary.md`;
- if the API key or required data/repositories are missing, the pilot writes
  `pilot_blocked.json` and exits **before any model call**; no fake evaluator or
  synthetic resolved label is substituted.

## 8. Full 100-task experiment and fixture smoke

```bash
# Full baseline + candidate loop (requires all prerequisites; not run here).
.venv/bin/python workflow/scripts/run_self_harness.py run \
  --config eval/configs/swebench_verified_100.example.toml \
  --run-id swebench-100-run01 --max-rounds 5

# Offline fixture smoke (fake agent/evaluator; explicitly NOT a benchmark).
.venv/bin/python eval/scripts/run_smoke_swebench.py
.venv/bin/python eval/scripts/run_smoke_swebench.py --mode workflow
```

A fixture smoke result is labelled with fake evaluator/agent names and must not
be reported as SWE-bench benchmark output. A pilot is not the 100-task
benchmark either. The full 100-task run additionally requires all 100 repo
clones/base commits, official evaluator images and the hosted API endpoint.

### Environment-fixed pilot (official per-task container)

The environment-fixed rerun keeps every model/dataset/harness parameter fixed
and replaces only the agent container image with the official SWE-bench task
image derived from the pinned evaluator:

```bash
.venv/bin/python eval/scripts/prepare_envfixed_pilot.py \
  --config eval/configs/swebench_pilot5_envfixed.toml \
  --run-id swebench-pilot5-envfixed-001
.venv/bin/python eval/scripts/run_swebench_pilot.py \
  --config eval/configs/swebench_pilot5_envfixed.toml \
  --run-id swebench-pilot5-envfixed-001
```

`prepare_envfixed_pilot.py` validates the fixed conditions against the original
pilot run, pulls/checks the official image for each task, probes its conda test
environment, and writes `preflight.json` plus `task_images.json`. The adapter
mounts the host workspace at `/testbed`, mounts the Harness snapshot read-only
at `/harness`, keeps the network disabled, and copies only git-ignored build
artifacts from the image repo into the host workspace before the agent starts
(so compiled extensions remain available without overwriting tracked source).
Comparison artifacts are `pilot_comparison.json` / `pilot_comparison.md`; the
reference run is never modified.
