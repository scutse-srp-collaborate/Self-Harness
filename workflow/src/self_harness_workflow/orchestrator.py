"""Self-Harness propose -> materialize -> eval -> gate -> promote orchestrator."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from self_harness_eval.config import EvalConfig
from self_harness_eval.hashing import sha256_file, sha256_json
from self_harness_eval.protocol import build_model_block, task_ids_sha256

from . import exec as exec_stage
from .events import append_event
from .report import build_final_report, write_final_report
from .runstate import (
    WorkflowError,
    initialize_run,
    read_json,
    utc_now,
    write_state,
)
from .snapshots import (
    HarnessSnapshot,
    changed_surface_contents,
    create_baseline_snapshot,
    create_candidate_snapshot,
    create_merged_snapshot,
    surface_relative_paths,
    verify_snapshot,
)
from .versions import allocate_version_id, create_version, load_version, update_version


class Orchestrator:
    def __init__(
        self,
        *,
        config: EvalConfig,
        run_dir: Path | None = None,
        run_id: str | None = None,
        reuse_existing: bool = False,
    ) -> None:
        self.config = config
        self.reuse_existing = bool(reuse_existing) or config.run.reuse_existing
        resolved_run_id = run_id or config.run.run_id or self._default_run_id()
        self.paths, self.run_manifest, self.task_ids_payload = initialize_run(
            config=config,
            run_id=resolved_run_id,
            run_root=run_dir,
            reuse_existing=self.reuse_existing,
        )
        self._ensure_surface_mapping()
        self.run_id = str(self.run_manifest["run_id"])

    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------

    def cmd_init(self) -> dict[str, Any]:
        self._save()
        return {"run_id": self.run_id, "run_root": str(self.paths.root), "status": "initialized"}

    def cmd_baseline(self) -> dict[str, Any]:
        baseline_id = self.ensure_baseline()
        state = self._state()
        return {
            "run_id": self.run_id,
            "baseline_version_id": baseline_id,
            "status": state.get("status"),
            "stop_reason": state.get("stop_reason"),
        }

    def cmd_round(self) -> dict[str, Any]:
        self.ensure_baseline()
        if self._is_stopped():
            return self._summary()
        result = self.run_round()
        if result.get("stop_reason"):
            self.finalize_run(reason=str(result["stop_reason"]))
        self._save()
        return self._summary()

    def cmd_run(self, *, max_rounds: int | None = None) -> dict[str, Any]:
        self.ensure_baseline()
        if self._is_stopped():
            return self._summary()
        limit = int(max_rounds if max_rounds is not None else self.config.run.max_rounds)
        rounds_this_call = 0
        while not self._is_stopped() and rounds_this_call < limit:
            stop_reason = self._check_stop_before_round()
            if stop_reason:
                self._stop(stop_reason)
                break
            outcome = self.run_round()
            rounds_this_call += 1
            stop_reason = outcome.get("stop_reason") or self._check_stop_after_round()
            if stop_reason:
                self._stop(str(stop_reason))
                break
        if not self._is_stopped():
            self._stop("max_rounds_reached")
        self.finalize_run(reason=str(self._state().get("stop_reason") or "max_rounds_reached"))
        self._save()
        return self._summary()

    def cmd_resume(self, *, max_rounds: int | None = None) -> dict[str, Any]:
        if self._is_stopped():
            raise WorkflowError(
                f"run {self.run_id} is already stopped "
                f"(reason={self._state().get('stop_reason')}); start a new run to continue"
            )
        append_event(self.paths, "run_resumed", run_id=self.run_id,
                     artifacts={"config": "config.resolved.json"})
        return self.cmd_run(max_rounds=max_rounds)

    def cmd_eval_version(self, version_id: str) -> dict[str, Any]:
        record = load_version(self.paths, version_id)
        result_path = self.paths.version_result(version_id)
        if result_path.exists():
            return {"version_id": version_id, "result_path": self._rel(result_path), "reused": True}
        result = self._eval_version(record)
        self._save()
        return {
            "version_id": version_id,
            "result_id": result.get("result_id"),
            "status": result.get("status"),
            "result_path": self._rel(result_path),
        }

    def cmd_report(self) -> dict[str, Any]:
        report = self._write_report()
        append_event(self.paths, "report_written", run_id=self.run_id,
                     artifacts={"final_report_json": "final_report.json",
                                "final_report_md": "final_report.md"})
        return {
            "run_id": self.run_id,
            "final_report_json": str(self.paths.final_report_json),
            "final_report_md": str(self.paths.final_report_md),
            "final_version_id": report["final"]["version_id"],
        }

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def ensure_baseline(self) -> str:
        state = self._state()
        baseline_id = state.get("baseline_version_id")
        if baseline_id:
            self._verify_version_snapshot(baseline_id)
            if not self.paths.version_result(baseline_id).exists():
                record = load_version(self.paths, baseline_id)
                self._eval_version(record)
                self._save()
            return str(baseline_id)

        version_id = allocate_version_id(self.run_manifest, suffix="baseline")
        snapshot_dir = self.paths.version_harness(version_id)
        snapshot = create_baseline_snapshot(config=self.config, target=snapshot_dir)
        fixed = self._config_fixed_payload()
        record = create_version(
            paths=self.paths,
            run_manifest=self.run_manifest,
            version_id=version_id,
            kind="baseline",
            snapshot=snapshot,
            parent_version_ids=[],
            fixed=fixed,
        )
        state = self._state()
        state["baseline_version_id"] = version_id
        state["active_version_id"] = version_id
        state["status"] = "baseline_pending"
        self._save()

        result = self._eval_version(record)
        state["status"] = "baseline_ready" if result.get("status") == "complete" else "baseline_failed"
        self._record_run_provenance(result)
        self._save()
        append_event(
            self.paths,
            "baseline_eval_complete",
            version_id=version_id,
            status=str(result.get("status")),
            artifacts={
                "result": self._rel(self.paths.version_result(version_id)),
                "snapshot": self._rel(snapshot_dir),
            },
            payload={"result_id": result.get("result_id")},
        )
        if result.get("status") != "complete":
            self._stop("baseline_not_complete")
            self.finalize_run(reason="baseline_not_complete")
            return version_id
        self._build_diagnosis(version_id)
        return version_id

    # ------------------------------------------------------------------
    # One round
    # ------------------------------------------------------------------

    def run_round(self) -> dict[str, Any]:
        state = self._state()
        active_id = str(state.get("active_version_id") or state.get("baseline_version_id"))
        if not active_id or active_id == "None":
            raise WorkflowError("cannot run a round without a baseline version")
        round_index = int(state.get("rounds_completed", 0)) + 1
        round_dir = self._round_dir(round_index)
        round_dir.mkdir(parents=True, exist_ok=True)
        active_snapshot = self._load_version_snapshot(active_id)
        active_result_path = self.paths.version_result(active_id)
        if not active_result_path.exists():
            raise WorkflowError(f"active version {active_id} has no evaluation result")
        active_result = read_json(active_result_path)
        if active_result.get("status") != "complete":
            self._stop("baseline_not_complete" if active_id == state.get("baseline_version_id") else "active_eval_not_complete")
            return self._round_outcome(round_index, "active_eval_not_complete")

        if not self.config.harness.surfaces:
            self._stop("no_editable_surfaces")
            return self._round_outcome(round_index, "no_editable_surfaces")

        diagnosis_path, evidence_path = self._build_diagnosis(active_id)
        proposals = self._propose(round_index, round_dir, active_snapshot, diagnosis_path)
        if proposals is None:
            append_event(self.paths, "proposer_failed", run_id=self.run_id,
                         artifacts={"round_dir": self._rel(round_dir)},
                         reason="proposer did not produce a valid proposal bundle")
            state["proposer_failures"] = int(state.get("proposer_failures", 0)) + 1
            self._save()
            return self._round_outcome(round_index, None)
        if not proposals:
            state["rounds_completed"] = round_index
            self._save()
            return self._round_outcome(round_index, "no_candidates")

        candidates: list[dict[str, Any]] = []
        for slot, proposal in enumerate(proposals, start=1):
            candidate = self._materialize_candidate(
                round_index=round_index,
                slot=slot,
                proposal=proposal,
                proposal_bundle=round_dir / "proposal_bundle.json",
                parent_snapshot=active_snapshot,
                parent_version_id=active_id,
                round_dir=round_dir,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            state["rounds_completed"] = round_index
            self._save()
            return self._round_outcome(round_index, "no_candidates")

        accepted: list[dict[str, Any]] = []
        for candidate in candidates:
            if int(state.get("candidate_evaluations", 0)) >= self.config.run.max_candidate_evaluations:
                self._record_version_state(
                    candidate["version_id"], "not_evaluated_budget", reason="candidate evaluation budget exhausted"
                )
                append_event(
                    self.paths,
                    "candidate_skipped_budget",
                    version_id=candidate["version_id"],
                    candidate_id=candidate["candidate_id"],
                    reason="candidate evaluation budget exhausted",
                )
                continue
            result = self._evaluate_candidate(candidate["version_id"], active_id)
            if result is None:
                continue
            if result.get("status") != "complete":
                self._record_acceptance_unavailable(
                    candidate["version_id"], active_id, reason="candidate evaluation is not complete"
                )
                continue
            acceptance = self._gate_candidate(candidate["version_id"], active_id)
            if acceptance is None:
                continue
            if bool(acceptance.get("accepted")):
                accepted.append(candidate)
            else:
                self._record_version_state(
                    candidate["version_id"],
                    "rejected",
                    reason=str(acceptance.get("reason")),
                    acceptance=acceptance,
                )

        promoted = False
        if len(accepted) == 1:
            candidate = accepted[0]
            self._record_version_state(candidate["version_id"], "accepted", reason="accepted by gate")
            self._promote(version_id=candidate["version_id"], round_index=round_index,
                          reason="single accepted candidate")
            promoted = True
        elif len(accepted) > 1:
            promoted = self._merge_accepted(
                accepted=accepted,
                round_index=round_index,
                round_dir=round_dir,
                active_id=active_id,
            )

        state = self._state()
        state["rounds_completed"] = round_index
        if promoted:
            state["consecutive_no_improvement"] = 0
        else:
            state["consecutive_no_improvement"] = int(state.get("consecutive_no_improvement", 0)) + 1
        self._save()
        return self._round_outcome(round_index, None)

    # ------------------------------------------------------------------
    # Proposal / materialization
    # ------------------------------------------------------------------

    def _propose(
        self,
        round_index: int,
        round_dir: Path,
        active_snapshot: HarnessSnapshot,
        diagnosis_path: Path,
    ) -> list[dict[str, Any]] | None:
        surfaces = self._active_surfaces(active_snapshot)
        prompt_path = round_dir / "proposer_prompt.txt"
        response_path = round_dir / "proposer_response.json"
        prompt_log = round_dir / "logs" / "proposer_prompt.log"
        response_log = round_dir / "logs" / "proposer_response.log"
        parse_log = round_dir / "logs" / "proposer_parse.log"
        bundle_path = round_dir / "proposal_bundle.json"
        isolation_path = round_dir / "proposer_isolation.json"
        try:
            prompt_outcome = exec_stage.build_proposer_prompt_stage(
                paths=self.paths,
                diagnosis_path=diagnosis_path,
                surfaces=surfaces,
                route_count=self.config.run.proposal_width,
                output_path=prompt_path,
                heldout_ids_path=self.paths.heldout_ids,
                train_ids_path=self.paths.train_ids,
                isolation_path=isolation_path,
                log_path=prompt_log,
            )
            if not prompt_outcome.ok or not prompt_path.exists():
                raise WorkflowError(
                    f"proposer prompt build failed: exit={prompt_outcome.returncode} "
                    f"{prompt_outcome.stderr[-1200:]}"
                )

            proposer_command = self.config.run.proposer_command
            if self.config.run.proposer_response is not None:
                shutil.copyfile(self.config.run.proposer_response, response_path)
            elif proposer_command:
                proposer_env = os.environ.copy()
                proposer_env["OPENAI_BASE_URL"] = self.config.model.base_url
                proposer_env["SELF_HARNESS_MODEL"] = self.config.model.model_id
                response_outcome = exec_stage.run_external_template(
                    proposer_command,
                    placeholders={
                        "prompt": prompt_path,
                        "response": response_path,
                        "run_dir": self.paths.root,
                        "run_root": self.paths.root,
                        "repo_root": exec_stage.REPO_ROOT,
                        "round": round_index,
                        "width": self.config.run.proposal_width,
                        "diagnosis": diagnosis_path,
                        "heldout_ids_file": self.paths.heldout_ids,
                        "train_ids_file": self.paths.train_ids,
                        "run_id": self.run_id,
                    },
                    cwd=exec_stage.REPO_ROOT,
                    timeout_s=1800.0,
                    log_path=response_log,
                    env=proposer_env,
                )
                if not response_outcome.ok:
                    raise WorkflowError(
                        f"proposer command failed: exit={response_outcome.returncode} "
                        f"{response_outcome.stderr[-1200:]}"
                    )
            else:
                raise WorkflowError(
                    "no proposer configured: set [run].proposer_command, [run].proposer_response, "
                    "or --proposer-command"
                )
            if not response_path.exists():
                raise WorkflowError(f"proposer did not write a response: {response_path}")

            parse_outcome = exec_stage.parse_proposer_response_stage(
                diagnosis_path=diagnosis_path,
                surfaces=surfaces,
                response_path=response_path,
                output_path=bundle_path,
                heldout_ids_path=self.paths.heldout_ids,
                train_ids_path=self.paths.train_ids,
                isolation_path=isolation_path,
                log_path=parse_log,
            )
            if not parse_outcome.ok or not bundle_path.exists():
                raise WorkflowError(
                    f"proposal parsing failed: exit={parse_outcome.returncode} "
                    f"{parse_outcome.stderr[-2000:]}"
                )
            bundle = read_json(bundle_path)
            raw_proposals = bundle.get("proposals") if isinstance(bundle, dict) else None
            if not isinstance(raw_proposals, list) or not raw_proposals:
                raise WorkflowError("proposal bundle contains no proposals")
            evaluated: list[dict[str, Any]] = []
            declined: list[dict[str, Any]] = []
            for proposal in raw_proposals:
                if not isinstance(proposal, dict):
                    raise WorkflowError("proposal bundle entries must be objects")
                metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), dict) else {}
                if str(metadata.get("selection_decision") or "").strip().lower() == "decline":
                    declined.append(proposal)
                else:
                    evaluated.append(proposal)
            self._validate_proposal_width(evaluated=evaluated, declined=declined)
            self._validate_distinctness(evaluated)
            state = self._state()
            state["proposals_created"] = int(state.get("proposals_created", 0)) + len(evaluated)
            self._save()
            for proposal in evaluated:
                append_event(
                    self.paths,
                    "proposal_created",
                    proposal_id=str(proposal.get("proposal_id")),
                    artifacts={"bundle": self._rel(bundle_path)},
                    payload={
                        "selection_decision": "eval_candidate",
                        "mechanism_family": (proposal.get("metadata") or {}).get("mechanism_family"),
                        "exact_hook": (proposal.get("metadata") or {}).get("exact_hook"),
                        "selected_cluster": proposal.get("selected_cluster"),
                        "expected_affected_cases": (proposal.get("metadata") or {}).get(
                            "expected_affected_cases"
                        ),
                    },
                )
            if not evaluated:
                return []
            return evaluated
        except Exception as exc:  # noqa: BLE001 - record every failure
            failure_path = round_dir / "proposer_failure.json"
            failure_path.write_text(
                json.dumps(
                    {"failure": str(exc), "at": utc_now()},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            append_event(
                self.paths,
                "proposer_failed",
                run_id=self.run_id,
                artifacts={"round_dir": self._rel(round_dir), "failure": self._rel(failure_path)},
                reason=str(exc)[:1500],
            )
            state = self._state()
            state["proposer_failures"] = int(state.get("proposer_failures", 0)) + 1
            self._save()
            return None

    def _validate_proposal_width(
        self,
        *,
        evaluated: Sequence[Mapping[str, Any]],
        declined: Sequence[Mapping[str, Any]],
    ) -> None:
        configured = self.config.run.proposal_width
        total = len(evaluated) + len(declined)
        if total != configured:
            raise WorkflowError(
                f"proposer returned {total} proposals but proposal_width={configured}"
            )
        if not evaluated:
            return
        if len(evaluated) != configured and len(declined) == 0:
            raise WorkflowError("proposer returned fewer eval candidates than proposal_width")

    def _validate_distinctness(self, proposals: Sequence[Mapping[str, Any]]) -> None:
        signatures: dict[str, str] = {}
        proposal_ids: set[str] = set()
        for proposal in proposals:
            proposal_id = str(proposal.get("proposal_id") or "").strip()
            if not proposal_id:
                raise WorkflowError("proposal is missing proposal_id")
            if proposal_id in proposal_ids:
                raise WorkflowError(f"duplicate proposal_id in bundle: {proposal_id!r}")
            proposal_ids.add(proposal_id)
            metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), dict) else {}
            signature = sha256_json(
                {
                    "selected_cluster": proposal.get("selected_cluster"),
                    "selected_cluster_id": metadata.get("selected_cluster_id"),
                    "mechanism_family": metadata.get("mechanism_family"),
                    "exact_hook": metadata.get("exact_hook"),
                    "values": proposal.get("values"),
                }
            )
            if signature in signatures:
                raise WorkflowError(
                    "proposal distinctness violation: "
                    f"{proposal_id!r} duplicates {signatures[signature]!r}"
                )
            signatures[signature] = proposal_id

    def _materialize_candidate(
        self,
        *,
        round_index: int,
        slot: int,
        proposal: Mapping[str, Any],
        proposal_bundle: Path,
        parent_snapshot: HarnessSnapshot,
        parent_version_id: str,
        round_dir: Path,
    ) -> dict[str, Any] | None:
        version_id = allocate_version_id(
            self.run_manifest, suffix=f"round{round_index:03d}_candidate{slot:02d}"
        )
        proposal_id = str(proposal.get("proposal_id"))
        materialized_dir = round_dir / "materialized" / f"candidate-{slot:02d}"
        materialize_log = round_dir / "logs" / f"materialize-candidate-{slot:02d}.log"
        surface_filenames = self._surface_mapping()
        surfaces = self._snapshot_surfaces(parent_snapshot)
        try:
            outcome = exec_stage.materialize_candidate_stage(
                proposal_bundle=proposal_bundle,
                proposal_id=proposal_id,
                candidate_id=proposal_id,
                surfaces=surfaces,
                surface_filenames=surface_filenames,
                output_dir=materialized_dir,
                log_path=materialize_log,
            )
            if not outcome.ok or not (materialized_dir / "manifest.json").exists():
                raise WorkflowError(
                    f"candidate materialization failed: exit={outcome.returncode} "
                    f"{outcome.stderr[-1500:]}"
                )
            manifest = read_json(materialized_dir / "manifest.json")
            changed = changed_surface_contents(
                candidate_dir=materialized_dir,
                manifest=manifest,
                surfaces=surface_filenames,
            )
            overrides = {relative_path: content for relative_path, content in changed.values()}
            snapshot_dir = self.paths.version_harness(version_id)
            snapshot = create_candidate_snapshot(
                parent=parent_snapshot,
                target=snapshot_dir,
                changed_files=overrides,
                surfaces=dict(surface_filenames),
            )
            materialized_proposal = read_json(materialized_dir / "proposal.json")
            fixed = self._parent_fixed_payload(parent_version_id)
            record = create_version(
                paths=self.paths,
                run_manifest=self.run_manifest,
                version_id=version_id,
                kind="candidate",
                snapshot=snapshot,
                parent_version_ids=[parent_version_id],
                fixed=fixed,
                proposal=materialized_proposal,
                candidate_id=proposal_id,
                mechanism_family=(proposal.get("metadata") or {}).get("mechanism_family"),
            )
            shutil.copytree(materialized_dir, self.paths.version_dir(version_id) / "candidate")
            append_event(
                self.paths,
                "candidate_materialized",
                version_id=version_id,
                candidate_id=proposal_id,
                proposal_id=proposal_id,
                artifacts={
                    "snapshot": self._rel(snapshot_dir),
                    "proposal": self._rel(self.paths.version_dir(version_id) / "proposal.json"),
                },
                payload={"mechanism_family": record.get("mechanism_family")},
            )
            self._save()
            return {"version_id": version_id, "candidate_id": proposal_id, "proposal": proposal}
        except Exception as exc:  # noqa: BLE001 - keep artifacts and continue
            failure_dir = round_dir / "materialize_failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_path = failure_dir / f"candidate-{slot:02d}.json"
            failure_path.write_text(
                json.dumps({"failure": str(exc), "proposal_id": proposal_id}, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            append_event(
                self.paths,
                "materialize_failed",
                version_id=version_id,
                candidate_id=proposal_id,
                proposal_id=proposal_id,
                artifacts={"failure": self._rel(failure_path)},
                reason=str(exc)[:1500],
            )
            state = self._state()
            state["proposer_failures"] = int(state.get("proposer_failures", 0)) + 1
            self._save()
            return None

    # ------------------------------------------------------------------
    # Evaluation / gate
    # ------------------------------------------------------------------

    def _eval_version(self, record: Mapping[str, Any]) -> dict[str, Any]:
        version_id = str(record["version_id"])
        snapshot_dir = self.paths.version_harness(version_id)
        eval_dir = self.paths.version_eval_dir(version_id)
        result_path = eval_dir / "result.json"
        reuse = bool(self.reuse_existing and result_path.exists())
        state = self._state()
        if not reuse:
            state["candidate_evaluations"] = int(state.get("candidate_evaluations", 0)) + (
                0 if record.get("kind") == "baseline" else 1
            )
            self._save()
            append_event(
                self.paths,
                "candidate_eval_started",
                version_id=version_id,
                candidate_id=record.get("candidate_id"),
                proposal_id=record.get("proposal_id"),
                artifacts={"snapshot": self._rel(snapshot_dir)},
            )
        try:
            result, outcome = exec_stage.run_eval_stage(
                paths=self.paths,
                config_path=self.config.path,
                version_id=version_id,
                harness_snapshot=snapshot_dir,
                output_dir=eval_dir,
                kind=str(record.get("kind")),
                parent_version_ids=list(record.get("parent_version_ids") or []),
                run_id=self.run_id,
                candidate_workspace=snapshot_dir,
                reuse_existing=self.reuse_existing or bool(record.get("kind") == "baseline"),
                log_path=self.paths.version_dir(version_id) / "eval" / "eval_stdout.log",
            )
        except Exception as exc:  # noqa: BLE001 - record failed evaluation
            failure_path = self.paths.version_dir(version_id) / "eval" / "eval_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps({"failure": str(exc), "at": utc_now()}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._record_version_state(
                version_id, "eval_failed", reason=str(exc)[:1500],
                eval_payload={
                    "result_id": None,
                    "result_path": self._rel(failure_path),
                    "status": "failed",
                },
            )
            append_event(
                self.paths,
                "candidate_eval_complete",
                version_id=version_id,
                candidate_id=record.get("candidate_id"),
                proposal_id=record.get("proposal_id"),
                status="failed",
                reason=str(exc)[:1500],
                artifacts={"failure": self._rel(failure_path)},
            )
            self._save()
            return {"status": "failed", "result_id": None}
        eval_payload = {
            "result_id": result.get("result_id"),
            "result_path": self._rel(result_path),
            "status": result.get("status"),
            "harness_snapshot_sha256": (result.get("harness") or {}).get("snapshot_sha256"),
        }
        self._record_version_state(version_id, "evaluated", eval_payload=eval_payload)
        append_event(
            self.paths,
            "candidate_eval_complete",
            version_id=version_id,
            candidate_id=record.get("candidate_id"),
            proposal_id=record.get("proposal_id"),
            status=str(result.get("status")),
            artifacts={"result": self._rel(result_path)},
            payload={
                "result_id": result.get("result_id"),
                "splits": {
                    split: {
                        "mean_repeat_pass_rate": block.get("mean_repeat_pass_rate"),
                        "pooled_passed": block.get("pooled_passed"),
                        "pooled_total": block.get("pooled_total"),
                    }
                    for split, block in (result.get("summary") or {}).items()
                },
            },
        )
        self._save()
        return result

    def _evaluate_candidate(self, version_id: str, parent_version_id: str) -> dict[str, Any] | None:
        record = load_version(self.paths, version_id)
        try:
            result = self._eval_version(record)
        except Exception as exc:  # noqa: BLE001
            self._record_version_state(version_id, "eval_failed", reason=str(exc)[:1500])
            return None
        if result.get("status") == "failed":
            return None
        return result

    def _gate_candidate(self, version_id: str, parent_version_id: str) -> dict[str, Any] | None:
        candidate_result = self.paths.version_result(version_id)
        baseline_result = self.paths.version_result(parent_version_id)
        acceptance_path = self.paths.version_acceptance(version_id)
        log_path = self.paths.version_dir(version_id) / "acceptance.log"
        try:
            acceptance, _outcome = exec_stage.run_acceptance_stage(
                paths=self.paths,
                baseline_result=baseline_result,
                candidate_result=candidate_result,
                output_path=acceptance_path,
                expected_repeats=self.config.eval.repeats,
                reuse_existing=self.reuse_existing,
                log_path=log_path,
            )
        except Exception as exc:  # noqa: BLE001 - never silently drop a gate failure
            acceptance = {
                "format": "self_harness.acceptance_gate.v1",
                "decision": "invalid",
                "accepted": False,
                "reason": f"acceptance gate could not run: {exc}",
                "decided_at": utc_now(),
                "rule_version": "self_harness.acceptance.v1",
                "deltas": {},
            }
            self._write_json(acceptance_path, acceptance)
        self._record_version_state(
            version_id,
            "rejected" if not acceptance.get("accepted") else "accepted",
            reason=str(acceptance.get("reason")),
            acceptance={
                "path": self._rel(acceptance_path),
                "decision": acceptance.get("decision"),
                "accepted": bool(acceptance.get("accepted")),
                "reason": acceptance.get("reason"),
                "deltas": acceptance.get("deltas"),
            },
        )
        event_type = "candidate_accepted" if acceptance.get("accepted") else "candidate_rejected"
        append_event(
            self.paths,
            event_type,
            version_id=version_id,
            candidate_id=(load_version(self.paths, version_id) or {}).get("candidate_id"),
            status=str(acceptance.get("decision")),
            reason=str(acceptance.get("reason"))[:1500],
            artifacts={"acceptance": self._rel(acceptance_path)},
            payload={
                "deltas": acceptance.get("deltas"),
                "task_transitions": acceptance.get("task_transitions"),
                "completeness_ok": (acceptance.get("completeness") or {}).get("ok"),
                "comparability_ok": (acceptance.get("comparability") or {}).get("ok"),
            },
        )
        self._save()
        return acceptance

    def _record_acceptance_unavailable(
        self, version_id: str, parent_version_id: str, *, reason: str
    ) -> None:
        acceptance = {
            "format": "self_harness.acceptance_gate.v1",
            "decision": "inconclusive",
            "accepted": False,
            "reason": reason,
            "decided_at": utc_now(),
            "rule_version": "self_harness.acceptance.v1",
            "deltas": {},
        }
        acceptance_path = self.paths.version_acceptance(version_id)
        self._write_json(acceptance_path, acceptance)
        self._record_version_state(
            version_id,
            "rejected",
            reason=reason,
            acceptance={
                "path": self._rel(acceptance_path),
                "decision": "inconclusive",
                "accepted": False,
                "reason": reason,
                "deltas": {},
            },
        )
        append_event(
            self.paths,
            "candidate_rejected",
            version_id=version_id,
            status="inconclusive",
            reason=reason,
            artifacts={"acceptance": self._rel(acceptance_path)},
        )
        self._save()

    # ------------------------------------------------------------------
    # Merge / promotion
    # ------------------------------------------------------------------

    def _merge_accepted(
        self,
        *,
        accepted: Sequence[Mapping[str, Any]],
        round_index: int,
        round_dir: Path,
        active_id: str,
    ) -> bool:
        version_id = allocate_version_id(self.run_manifest, suffix=f"round{round_index:03d}_merged")
        snapshot_dir = self.paths.version_harness(version_id)
        parent_snapshot = self._load_version_snapshot(active_id)
        candidate_snapshots = [
            self._load_version_snapshot(str(candidate["version_id"])) for candidate in accepted
        ]
        try:
            merged_snapshot = create_merged_snapshot(
                parent=parent_snapshot,
                candidates=candidate_snapshots,
                target=snapshot_dir,
            )
        except WorkflowError as exc:
            for candidate in accepted:
                self._record_version_state(
                    str(candidate["version_id"]), "merge_conflict", reason=str(exc)
                )
            append_event(
                self.paths,
                "merge_failed",
                version_id=None,
                reason=str(exc)[:1500],
                artifacts={"round_dir": self._rel(round_dir)},
            )
            self._save()
            return False
        parent_ids = [active_id] + [str(candidate["version_id"]) for candidate in accepted]
        create_version(
            paths=self.paths,
            run_manifest=self.run_manifest,
            version_id=version_id,
            kind="merged",
            snapshot=merged_snapshot,
            parent_version_ids=parent_ids,
            fixed=self._parent_fixed_payload(active_id),
            merge_parent_candidate_ids=[str(candidate["version_id"]) for candidate in accepted],
        )
        append_event(
            self.paths,
            "merge_created",
            version_id=version_id,
            artifacts={"snapshot": self._rel(snapshot_dir)},
            payload={
                "parent_version_ids": parent_ids,
                "merged_candidate_ids": [str(candidate["candidate_id"]) for candidate in accepted],
            },
        )
        self._save()
        if int(self._state().get("candidate_evaluations", 0)) >= self.config.run.max_candidate_evaluations:
            self._record_version_state(version_id, "not_evaluated_budget",
                                       reason="candidate evaluation budget exhausted before merge eval")
            for candidate in accepted:
                self._record_version_state(
                    str(candidate["version_id"]), "accepted",
                    reason="accepted but merge evaluation skipped by budget",
                )
            self._save()
            return False
        result = self._evaluate_candidate(version_id, active_id)
        if result is None or result.get("status") != "complete":
            self._record_acceptance_unavailable(
                version_id, active_id, reason="merged candidate evaluation is not complete"
            )
            append_event(
                self.paths,
                "merge_eval_complete",
                version_id=version_id,
                status=str((result or {}).get("status")),
                reason="merged evaluation incomplete; not promoted",
                artifacts={"result": self._rel(self.paths.version_result(version_id))}
                if self.paths.version_result(version_id).exists()
                else {},
            )
            return False
        acceptance = self._gate_candidate(version_id, active_id)
        append_event(
            self.paths,
            "merge_eval_complete",
            version_id=version_id,
            status=str(result.get("status")),
            reason=str((acceptance or {}).get("reason")),
            artifacts={"result": self._rel(self.paths.version_result(version_id))},
            payload={"deltas": (acceptance or {}).get("deltas")},
        )
        if acceptance is not None and acceptance.get("accepted"):
            self._record_version_state(version_id, "accepted", reason=str(acceptance.get("reason")))
            for candidate in accepted:
                self._record_version_state(
                    str(candidate["version_id"]),
                    "accepted",
                    reason=f"merged into {version_id}",
                )
            self._promote(version_id=version_id, round_index=round_index,
                          reason="merged candidate accepted by gate")
            return True
        for candidate in accepted:
            self._record_version_state(
                str(candidate["version_id"]),
                "rejected",
                reason=f"merged candidate {version_id} rejected; individual acceptance did not transfer",
            )
        self._save()
        return False

    def _promote(self, *, version_id: str, round_index: int, reason: str) -> None:
        state = self._state()
        previous = state.get("active_version_id")
        state["active_version_id"] = version_id
        state["status"] = "running"
        self._record_version_state(version_id, "accepted", reason=reason)
        self._save()
        append_event(
            self.paths,
            "branch_promoted",
            version_id=version_id,
            status="accepted",
            reason=reason,
            artifacts={"snapshot": self._rel(self.paths.version_harness(version_id))},
            payload={"previous_active_version_id": previous, "round": round_index},
        )

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def _build_diagnosis(self, version_id: str) -> tuple[Path, Path]:
        diagnosis_dir = self.paths.version_diagnosis_dir(version_id)
        brief_path = diagnosis_dir / "diagnosis.md"
        evidence_path = diagnosis_dir / "evidence.json"
        if brief_path.exists() and evidence_path.exists():
            return brief_path, evidence_path
        diagnosis_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.paths.version_result(version_id)
        if not result_path.exists():
            raise WorkflowError(f"cannot build diagnosis without a result for {version_id}")
        try:
            outcome = exec_stage.build_diagnosis_stage(
                result_path=result_path,
                output_path=brief_path,
                evidence_path=evidence_path,
                heldout_ids_path=self.paths.heldout_ids,
                train_ids_path=self.paths.train_ids,
                run_root=self.paths.root,
                log_path=diagnosis_dir / "diagnosis.log",
            )
            if not outcome.ok or not brief_path.exists():
                raise WorkflowError(
                    f"diagnosis builder failed: exit={outcome.returncode} "
                    f"{outcome.stderr[-1500:]}"
                )
        except Exception as exc:  # noqa: BLE001
            append_event(
                self.paths,
                "diagnosis_failed",
                version_id=version_id,
                reason=str(exc)[:1500],
                artifacts={"result": self._rel(result_path)},
            )
            raise
        append_event(
            self.paths,
            "diagnosis_created",
            version_id=version_id,
            artifacts={
                "diagnosis": self._rel(brief_path),
                "evidence": self._rel(evidence_path),
            },
        )
        return brief_path, evidence_path

    # ------------------------------------------------------------------
    # Stops / finalize / helpers
    # ------------------------------------------------------------------

    def _check_stop_before_round(self) -> str | None:
        state = self._state()
        if int(state.get("candidate_evaluations", 0)) >= self.config.run.max_candidate_evaluations:
            return "candidate_budget_exhausted"
        if int(state.get("rounds_completed", 0)) >= self.config.run.max_rounds:
            return "max_rounds_reached"
        if int(state.get("proposer_failures", 0)) >= self.config.run.max_proposer_failures:
            return "proposer_failures"
        return None

    def _check_stop_after_round(self) -> str | None:
        state = self._state()
        if int(state.get("candidate_evaluations", 0)) >= self.config.run.max_candidate_evaluations:
            return "candidate_budget_exhausted"
        if int(state.get("rounds_completed", 0)) >= self.config.run.max_rounds:
            return "max_rounds_reached"
        if int(state.get("consecutive_no_improvement", 0)) >= self.config.run.max_consecutive_no_improvement:
            return "no_improvement"
        if int(state.get("proposer_failures", 0)) >= self.config.run.max_proposer_failures:
            return "proposer_failures"
        return None

    def _stop(self, reason: str) -> None:
        state = self._state()
        if state.get("status") == "stopped":
            return
        state["status"] = "stopped"
        state["stop_reason"] = reason
        state["stopped_at"] = utc_now()
        self._save()
        append_event(
            self.paths,
            "run_stopped",
            run_id=self.run_id,
            status="stopped",
            reason=reason,
            payload={"rounds_completed": state.get("rounds_completed")},
        )

    def finalize_run(self, *, reason: str | None = None) -> dict[str, Any]:
        state = self._state()
        active_id = state.get("active_version_id")
        if active_id:
            state["final_version_id"] = active_id
            record = load_version(self.paths, str(active_id))
            if record.get("status") != "final":
                self._record_version_state(str(active_id), "final", reason="run finalized")
        if state.get("status") != "stopped":
            state["status"] = "stopped"
            state["stop_reason"] = reason or state.get("stop_reason") or "stopped"
            state["stopped_at"] = utc_now()
        self._save()
        report = self._write_report()
        append_event(
            self.paths,
            "run_finalized",
            run_id=self.run_id,
            status="final",
            reason=state.get("stop_reason"),
            artifacts={"final_report_json": "final_report.json", "final_report_md": "final_report.md"},
            payload={"final_version_id": state.get("final_version_id")},
        )
        return report

    def _write_report(self) -> dict[str, Any]:
        report = build_final_report(
            paths=self.paths,
            run_manifest=self.run_manifest,
            task_ids_payload=self.task_ids_payload,
            config=self.config,
        )
        write_final_report(paths=self.paths, report=report)
        return report

    def _record_version_state(
        self,
        version_id: str,
        status: str,
        *,
        reason: str | None = None,
        eval_payload: Mapping[str, Any] | None = None,
        acceptance: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            update_version(
                paths=self.paths,
                run_manifest=self.run_manifest,
                version_id=version_id,
                status=status,
                reason=reason,
                eval_payload=eval_payload,
                acceptance_payload=acceptance,
            )
        except WorkflowError:
            return

    def _load_version_snapshot(self, version_id: str) -> HarnessSnapshot:
        directory = self.paths.version_harness(version_id)
        record = load_version(self.paths, version_id)
        expected = str((record.get("harness") or {}).get("snapshot_sha256"))
        return verify_snapshot(directory, expected_sha256=expected)

    def _verify_version_snapshot(self, version_id: str) -> None:
        self._load_version_snapshot(version_id)

    def _active_surfaces(self, snapshot: HarnessSnapshot) -> list[tuple[str, Path]]:
        mapping = self._surface_mapping()
        return [(name, snapshot.directory / relative_path) for name, relative_path in mapping.items()]

    def _snapshot_surfaces(self, snapshot: HarnessSnapshot) -> list[tuple[str, Path]]:
        mapping = self._surface_mapping()
        return [(name, snapshot.directory / relative_path) for name, relative_path in mapping.items()]

    def _surface_mapping(self) -> dict[str, str]:
        harness = self.run_manifest.get("harness") or {}
        mapping = harness.get("surface_relative_paths")
        if isinstance(mapping, dict) and mapping:
            return {str(name): str(value) for name, value in mapping.items()}
        mapping = surface_relative_paths(self.config)
        harness["surface_relative_paths"] = mapping
        self.run_manifest["harness"] = harness
        self._save()
        return mapping

    def _ensure_surface_mapping(self) -> None:
        harness = self.run_manifest.setdefault("harness", {})
        existing = harness.get("surface_relative_paths")
        computed = surface_relative_paths(self.config) if self.config.harness.surfaces else {}
        if existing and computed and {str(k): str(v) for k, v in existing.items()} != computed:
            raise WorkflowError(
                "refusing to resume: editable harness surface mapping changed "
                f"(stored={existing}, current={computed})"
            )
        if computed and not existing:
            harness["surface_relative_paths"] = computed
            self._save()

    def _record_run_provenance(self, result: Mapping[str, Any]) -> None:
        """Persist dataset source/cache provenance and environment hash in the run manifest."""

        dataset_block = result.get("dataset") or {}
        dataset = self.run_manifest.setdefault("dataset", {})
        dataset.update(
            {
                "name": dataset_block.get("name") or self.config.dataset.name,
                "revision": dataset_block.get("revision") or self.config.dataset.revision,
                "manifest_sha256": dataset_block.get("manifest_sha256"),
                "source_sha256": dataset_block.get("source_sha256"),
                "task_ids_sha256": dataset_block.get("task_ids_sha256"),
                "source": self.config.dataset.source,
                "cache_dir": str(self.config.dataset.cache_dir)
                if self.config.dataset.cache_dir
                else None,
                "repos_dir": str(self.config.dataset.repos_dir)
                if self.config.dataset.repos_dir
                else None,
                "data_split": self.config.dataset.split,
            }
        )
        comparability = (result.get("config") or {}).get("comparability") or {}
        self.run_manifest["environment"] = {
            "comparability": comparability,
            "environment_sha256": comparability.get("environment"),
            "tools_sha256": comparability.get("tools"),
        }
        self.run_manifest["resolved_config"] = {
            "path": "config.resolved.json",
            "sha256": sha256_file(self.paths.config_resolved)
            if self.paths.config_resolved.exists()
            else None,
        }

    def _config_fixed_payload(self) -> dict[str, Any]:
        model_block = build_model_block(
            provider=self.config.model.provider,
            model_id=self.config.model.model_id,
            endpoint_type=self.config.model.endpoint_type,
            decoding=self.config.model.decoding(),
            extra={
                "base_url": self.config.model.base_url,
                "max_retries": self.config.model.max_retries,
            },
        )
        splits = self.task_ids_payload.get("splits") or {}
        return {
            "model_config_sha256": model_block["config_sha256"],
            "dataset_revision": self.config.dataset.revision,
            "dataset_manifest_sha256": (self.task_ids_payload.get("dataset") or {}).get(
                "manifest_sha256"
            ),
            "task_ids_sha256": task_ids_sha256(
                splits.get("train", []), splits.get("heldout", [])
            ),
        }

    def _parent_fixed_payload(self, version_id: str) -> dict[str, Any]:
        result_path = self.paths.version_result(version_id)
        if not result_path.exists():
            return self._config_fixed_payload()
        result = read_json(result_path)
        return {
            "model_config_sha256": (result.get("model") or {}).get("config_sha256"),
            "comparability": (result.get("config") or {}).get("comparability"),
            "dataset_revision": (result.get("dataset") or {}).get("revision"),
            "dataset_manifest_sha256": (result.get("dataset") or {}).get("manifest_sha256"),
            "task_ids_sha256": (result.get("dataset") or {}).get("task_ids_sha256"),
        }

    def _round_dir(self, round_index: int) -> Path:
        return self.paths.root / "rounds" / f"round-{round_index:03d}"

    def _round_outcome(self, round_index: int, stop_reason: str | None) -> dict[str, Any]:
        state = self._state()
        outcome = {
            "round": round_index,
            "active_version_id": state.get("active_version_id"),
            "rounds_completed": state.get("rounds_completed"),
            "candidate_evaluations": state.get("candidate_evaluations"),
            "stop_reason": stop_reason,
        }
        if stop_reason:
            self._stop(stop_reason)
        self._save()
        return outcome

    def _summary(self) -> dict[str, Any]:
        state = self._state()
        return {
            "run_id": self.run_id,
            "run_root": str(self.paths.root),
            "status": state.get("status"),
            "active_version_id": state.get("active_version_id"),
            "baseline_version_id": state.get("baseline_version_id"),
            "final_version_id": state.get("final_version_id"),
            "rounds_completed": state.get("rounds_completed"),
            "candidate_evaluations": state.get("candidate_evaluations"),
            "proposals_created": state.get("proposals_created"),
            "consecutive_no_improvement": state.get("consecutive_no_improvement"),
            "stop_reason": state.get("stop_reason"),
            "events": self._rel(self.paths.events),
            "final_report_json": self._rel(self.paths.final_report_json)
            if self.paths.final_report_json.exists()
            else None,
            "final_report_md": self._rel(self.paths.final_report_md)
            if self.paths.final_report_md.exists()
            else None,
        }

    def _state(self) -> dict[str, Any]:
        state = self.run_manifest.setdefault("state", {})
        assert isinstance(state, dict)
        return state

    def _is_stopped(self) -> bool:
        return self._state().get("status") in {"stopped", "failed"}

    def _save(self) -> None:
        write_state(self.paths, self.run_manifest)

    def _rel(self, path: Path | str | None) -> str | None:
        if path is None:
            return None
        try:
            return Path(path).resolve().relative_to(self.paths.root).as_posix()
        except ValueError:
            return None

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _default_run_id() -> str:
        from datetime import datetime, timezone

        return "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
