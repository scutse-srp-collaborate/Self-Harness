# Evaluation Runners: Terminal-Bench 2.0 Clean64 and SWE-bench Verified

This directory contains the evaluation layer for the Self-Harness experiment.
Both benchmarks write the unified, immutable `self_harness.eval_result.v1`
result and are dispatched by `eval/scripts/run_eval.py` based on
`[eval].benchmark` in the config.

## Entry points

| Script | Purpose |
| --- | --- |
| `eval/scripts/run_eval.py` | Unified dispatcher (`terminal_bench_2` / `swe_bench_verified`) |
| `eval/scripts/run_harbor_eval.py` | Terminal-Bench 2.0 Harbor + pytest/JUnit runner (legacy CLI flags preserved) |
| `eval/scripts/run_swebench_eval.py` | SWE-bench Verified runner (repo/base-commit workspace, patch, evaluator resolved) |
| `eval/scripts/run_smoke_swebench.py` | Offline fixture smoke test (no network/Docker/model) |
| `eval/configs/tb2_clean64.example.toml` | Terminal-Bench Clean64 config template |
| `eval/configs/swebench_verified_100.example.toml` | SWE-bench Verified 100 config template |
| `eval/manifests/tb2_clean64.json` | Fixed Clean64 task ids and train/heldout split |
| `eval/data/README.md` | How to pin the SWE-bench dataset, repositories and official evaluator |

The evaluation model, runner and acceptance contract are documented in the
repository root `README.md`.

## Real SWE-bench execution adapters

- Execution: `eval/scripts/run_deepagent_swebench.py` + `eval/src/self_harness_eval/deepagent_adapter.py`
  build `ChatOpenAI` from `SELF_HARNESS_MODEL` / `OPENAI_BASE_URL` /
  `SELF_HARNESS_MODEL_API_KEY`, dynamically import the active candidate
  Harness snapshot's `repo_baseline.py`, and run DeepAgents tools inside a
  task-scoped Docker container (`python:3.12`) that mounts only the task
  workspace.
- Proposer: `proposer/scripts/run_openai_compatible_proposer.py` uses the same
  model/endpoint/key/decoding and writes JSON for the existing parser and
  held-out isolation audit.
- Pinned dependencies: `eval/requirements-deepagent.txt`
  (`deepagents==0.7.29`, `langchain-openai==1.7.2`, `langchain==1.4.6`,
  `swebench==4.0.5`, ...).
- Bounded pilot (≤5 fixed train tasks, 2 repeats, no heldout/candidates):
  `eval/scripts/run_swebench_pilot.py --config eval/configs/swebench_pilot_5.toml`.
- Full data/evaluator/model setup and limitations are documented in
  `eval/data/README.md`. DeepSeek Flash is not one of the paper's models, so
  results are an evaluation of the method on that model, not a paper-model
  reproduction.

## Terminal-Bench 2.0 Clean64 behaviour

- The runner keeps the Harbor + pytest execution used by the existing workflow:
  it invokes the configured pytest eval project with `--junitxml` and the
  Harbor report flag, and archives stdout/stderr, JUnit XML, command metadata and
  the Harbor agent trace per case.
- Task-level timeout defaults to **900 s**, tool/command timeout defaults to
  **120 s**. Both are passed to the execution layer through
  `SELF_HARNESS_AGENT_TIMEOUT_SEC` and `SELF_HARNESS_TOOL_TIMEOUT_S`;
  `eval/harness_workspace/self_harness_harbor/backend_bridge.py` reads the tool
  timeout for every sandbox command.
- The outbound bandwidth cap is configured as **2 MB/s**. Enforcement is applied
  with `trickle` or a host `tc` qdisc when available; the result records
  `config.network_limit {requested_mbps, method, enforced, detail}`. When no
  limiter is available the result explicitly says `enforced=false`.
- JUnit status mapping: clean pass -> `passed`; test failure -> `failed`; skipped,
  zero-test or missing JUnit -> `evaluator_error` (unscoreable, `passed=null`);
  task timeout -> `agent_timeout`. Missing logs or evaluator exceptions are never
  silently converted into ordinary failures.
- `result.json` is the unified result; `result.legacy.json` keeps the historical
  aggregate (`name/model/repeats/passed/total/correctness/splits`) for older
  tooling.
- A single example case is **not** a Terminal-Bench evaluation. The shipped
  Clean64 config selects the complete fixed 43 train / 21 heldout split.
- Required external prerequisites: a Harbor eval project exposing the fixed task
  ids to pytest, the Terminal-Bench 2.0 task data and a working container
  runtime. `eval/configs/tb2_clean64.example.toml` documents every value
  (`evals_project`, `harness_workspace`, `agent_import_path`, selector template).

## SWE-bench Verified behaviour

- The runner is a dedicated adapter, **not** a pytest node wrapper. For each task
  it checks out the manifest base commit from a local clone, verifies
  `git rev-parse HEAD`, writes an issue-only instruction, runs the configured
  agent adapter, collects the model patch and calls the SWE-bench evaluator.
- `passed=true` is only emitted when the evaluator reports
  `resolved=true`; unresolved, patch-apply failures and test failures are
  `unresolved` (`passed=false`); agent/model failures are `agent_timeout` /
  `agent_error`; missing evaluator output, missing workspace or missing dataset
  metadata are `infra_error` / `evaluator_error` (`passed=null`).
- Agent inputs are filtered to `task_id, repo, base_commit, version,
  problem_statement, hints_text`; gold patch, test patch,
  `FAIL_TO_PASS`/`PASS_TO_PASS` and the resolved label never reach the agent.
- Task timeout defaults to **3600 s**, tool timeout to **120 s**; retries are
  bounded by `[eval].max_retries` / `[agent].max_retries` and every attempt with
  its reason is recorded in the case result.
- Supported agent adapters: `[agent].kind="command"` (external command template)
  and `[agent].kind="harbor"`/`"harbor_exec"` (requires a
  `harbor_workspace_verification.json` with matching base/head commit; the final
  resolved decision still comes from the evaluator).
- Evaluators: `official` (pinned `swebench` package + Docker), `command`
  (containerized official harness wrapper), and `fixture` (smoke tests only,
  rejected unless `[eval].smoke_test=true`, labelled
  `fixture-fake-swebench-evaluator` in the result).
- Required external prerequisites and the pinned data/evaluator setup are
  documented in `eval/data/README.md`.

## Splits, repeats and reuse

- Split names are `train` and `heldout`; repeats default to 2 and start at 1.
- `[eval].split` can restrict a run to `train` or `heldout`; `[eval].task_ids`
  can restrict to specific fixed task ids (used by smoke tests). The manifest
  split is never re-randomized.
- Reuse-existing validates the resolved config hash, model config hash, harness
  snapshot hash, dataset revision/version, dataset manifest hash and task-id
  list hash before reusing a result. The SWE-bench runner additionally supports
  case-level resume (`case_provenance.json`), and repeat directories never
  overwrite each other.
- Results are immutable: writing to an existing `result.json` fails unless the
  reuse validation passed; corrected runs produce new `result_id`s/directories.

## Not executed by this repository

A real benchmark run still requires the external prerequisites listed above
(Harbor eval project/task data/containers for Terminal-Bench; pinned dataset,
repository clones, official evaluator and Docker for SWE-bench; plus a model
endpoint). The fixture smoke test intentionally uses a fake agent and fake
evaluator, and its output is labelled as a smoke test rather than a benchmark
measurement. No benchmark score is claimed here.

# TB2 Clean64 Split

## Train (43)

```text
bn_fit_modify
build_cython_ext
build_pmars
build_pov_ray
caffe_cifar_10
cancel_async_tasks
cobol_modernization
constraints_scheduling
custom_memory_heap_crash
db_wal_recovery
dna_insert
extract_elf
feal_differential_cryptanalysis
fix_git
gcode_to_text
git_leak_recovery
llm_inference_batching_scheduler
log_summary_date_ranges
mailman
make_doom_for_mips
mcmc_sampling_stan
merge_diff_arc_agi_task
model_extraction_relu_logits
multi_source_data_merger
nginx_request_logging
overfull_hbox
password_recovery
path_tracing
polyglot_c_py
polyglot_rust_c
protein_assembly
prove_plus_comm
qemu_alpine_ssh
qemu_startup
raman_fitting
regex_chess
regex_log
sparql_university
sqlite_with_gcov
torch_pipeline_parallelism
torch_tensor_parallelism
vulnerable_secret
write_compressor
```

## Heldout (21)

```text
break_filter_js_from_html
count_dataset_tokens
distribution_search
dna_assembly
extract_moves_from_video
feal_linear_cryptanalysis
filter_js_from_html
git_multibranch
headless_terminal
kv_store_grpc
large_scale_text_editing
largest_eigenval
make_mips_interpreter
modernize_scientific_stack
openssl_selfsigned_cert
path_tracing_reverse
pypi_server
pytorch_model_cli
rstan_to_pystan
sqlite_db_truncate
tune_mjcf
```
