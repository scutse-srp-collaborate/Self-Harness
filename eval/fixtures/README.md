# Evaluation fixtures (tests and smoke only)

These files exist only to exercise the real runners and workflow without a
network, a Docker image, a pinned SWE-bench dataset, a model endpoint or a
Harbor eval project:

- `fake_agent.py` — deterministic SWE-bench agent command. It writes a patch
  marker and never reads the pinned dataset/gold data.
- `fake_official_evaluator.py` — deterministic SWE-bench *command* evaluator.
- `fake_swebench_pkg/` — fake `swebench.harness.run_evaluation` CLI used to test
  the official-evaluator adapter contract.
- `fake_pytest.py` — fake Harbor/pytest command used to test the
  Terminal-Bench runner.
- `fake_proposer.py` — deterministic proposer response for workflow tests.
- `swebench_fixture.py` / `tb2_fixture.py` — builders for local git repositories,
  fixture datasets, harnesses and configs.

Fixture results are labelled with fake evaluator/agent names and with
`claims.reproduces_paper_results=false`; they are **not** benchmark
measurements.
