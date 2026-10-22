"""OpenAI-compatible proposer adapter and validation tests."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

try:
    import langchain_openai  # noqa: F401

    OPENAI_DEPS_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENAI_DEPS_AVAILABLE = False

OPENAI_SKIP_REASON = "OpenAI-compatible adapter dependencies are not installed; run .venv/bin/pytest"


def openai_test(cls):
    return unittest.skipUnless(OPENAI_DEPS_AVAILABLE, OPENAI_SKIP_REASON)(cls)


from self_harness_eval.openai_compat import (
    API_KEY_ENV,
    MissingApiKeyError,
    build_chat_model,
    extract_json_object,
    settings_from_environment,
    validate_proposals_payload,
)
from self_harness_proposer import parse_multi_proposer_response

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "fake-proposer-key-not-real"
HELDOUT_ID = "astropy__astropy-7336"
TRAIN_ID = "astropy__astropy-13398"

SURFACE_SOURCE = (
    "def build_system_prompt() -> str:\n"
    "    return 'baseline prompt'\n\n"
    "def build_verification_instruction() -> str:\n"
    "    return 'verify'\n"
)


def _proposal(index: int, *, hook: str = "build_system_prompt") -> dict:
    return {
        "proposal_id": f"p{index}",
        "title": f"proposal {index}",
        "selection_decision": "eval_candidate",
        "selected_cluster": f"cluster-{index}",
        "selected_surface": hook,
        "mechanism": f"mechanism {index}",
        "mechanism_family": "prompt_instruction",
        "exact_hook": hook,
        "mechanism_to_hook_rationale": "maps guidance to the hook",
        "why_not_noop": "train failure evidence",
        "decline_reason": "",
        "skipped_clusters": [],
        "expected_affected_cases": [TRAIN_ID],
        "protected_passing_cases": [],
        "why_distinct": f"distinct {index}",
        "net_gain_hypothesis": "gain",
        "regression_guard": "guard",
        "summary": f"summary {index}",
        "final_message": "impl",
        "candidate_values": {hook: f"value-{index}"},
    }


@openai_test
class ProposalValidatorTest(unittest.TestCase):
    def test_valid_proposals(self) -> None:
        proposals = [_proposal(1), _proposal(2, hook="build_verification_instruction")]
        validated = validate_proposals_payload({"proposals": proposals}, expected_width=2)
        self.assertEqual(len(validated), 2)

    def test_rejects_wrong_width(self) -> None:
        with self.assertRaises(ValueError):
            validate_proposals_payload({"proposals": [_proposal(1)]}, expected_width=2)

    def test_rejects_duplicates(self) -> None:
        proposal = _proposal(1)
        with self.assertRaises(ValueError):
            validate_proposals_payload({"proposals": [proposal, dict(proposal)]}, expected_width=2)

    def test_rejects_missing_fields_and_decline_with_values(self) -> None:
        broken = _proposal(1)
        for key in ("mechanism_family", "exact_hook", "mechanism_to_hook_rationale", "why_not_noop"):
            broken.pop(key, None)
        with self.assertRaises(ValueError):
            validate_proposals_payload({"proposals": [broken, _proposal(2)]}, expected_width=2)
        decline = _proposal(1)
        decline["selection_decision"] = "decline"
        with self.assertRaises(ValueError):
            validate_proposals_payload({"proposals": [decline]}, expected_width=1)

    def test_extract_json_from_fences(self) -> None:
        payload = {"proposals": [_proposal(1)]}
        text = "Here is JSON:\n```json\n" + json.dumps(payload) + "\n```\n"
        self.assertEqual(extract_json_object(text)["proposals"][0]["proposal_id"], "p1")

    def test_chat_model_uses_shared_settings_and_hides_key(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                API_KEY_ENV: FAKE_KEY,
                "SELF_HARNESS_MODEL": "deepseek-flash",
                "OPENAI_BASE_URL": "https://api.qadg.xyz/v1",
            },
            clear=False,
        ):
            settings = settings_from_environment()
            model = build_chat_model(settings)
        self.assertEqual(model.model_name, "deepseek-flash")
        self.assertEqual(model.openai_api_base, "https://api.qadg.xyz/v1")
        self.assertFalse(FAKE_KEY in repr(model))

    def test_missing_key_clear_error(self) -> None:
        with mock.patch.dict(os.environ, {API_KEY_ENV: ""}, clear=False):
            with self.assertRaises(MissingApiKeyError) as caught:
                build_chat_model()
        self.assertIn(API_KEY_ENV, str(caught.exception))
        self.assertNotIn(FAKE_KEY, str(caught.exception))


@openai_test
class ProposerCliTest(unittest.TestCase):
    def _run_script(self, args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(REPO_ROOT / "proposer" / "scripts" / "run_openai_compatible_proposer.py"),
                *args,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_self_check_without_key(self) -> None:
        env = dict(os.environ)
        env.pop(API_KEY_ENV, None)
        completed = self._run_script(["--self-check", "--prompt", "/dev/null", "--response", "/tmp/x.json"], env=env)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["api_key_present"])
        self.assertNotIn(FAKE_KEY, completed.stdout)

    def test_missing_key_exits_without_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_text("proposal prompt", encoding="utf-8")
            response = root / "response.json"
            env = dict(os.environ)
            env.pop(API_KEY_ENV, None)
            completed = self._run_script(
                [
                    "--prompt",
                    str(prompt),
                    "--response",
                    str(response),
                    "--width",
                    "2",
                ],
                env=env,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(API_KEY_ENV, completed.stderr)
            self.assertFalse(response.exists())

    def _stub_server(self, payloads: list[dict]) -> tuple[ThreadingHTTPServer, threading.Thread]:
        responses = list(payloads)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: ANN002
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                self.rfile.read(length)
                content = responses[min(len(responses) - 1, int(getattr(self, "count", 0) or 0))]
                self.count = int(getattr(self, "count", 0) or 0) + 1
                payload = {
                    "id": "stub",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(content)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_cli_writes_parseable_response_with_heldout_audit(self) -> None:
        server, _thread = self._stub_server(
            [{"proposals": [_proposal(1), _proposal(2, hook="build_verification_instruction")]}]
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prompt = root / "prompt.txt"
                prompt.write_text("train-only proposal prompt", encoding="utf-8")
                heldout = root / "heldout.json"
                heldout.write_text(json.dumps([HELDOUT_ID]), encoding="utf-8")
                train = root / "train.json"
                train.write_text(json.dumps([TRAIN_ID]), encoding="utf-8")
                response = root / "response.json"
                trace = root / "proposer.trace.jsonl"
                env = dict(os.environ)
                env.update(
                    {
                        API_KEY_ENV: FAKE_KEY,
                        "SELF_HARNESS_MODEL": "deepseek-flash",
                        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    }
                )
                completed = self._run_script(
                    [
                        "--prompt",
                        str(prompt),
                        "--response",
                        str(response),
                        "--width",
                        "2",
                        "--heldout-ids-file",
                        str(heldout),
                        "--train-ids-file",
                        str(train),
                        "--trace",
                        str(trace),
                    ],
                    env=env,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
                payload = json.loads(response.read_text(encoding="utf-8"))
                bundles = parse_multi_proposer_response(
                    response_text=json.dumps(payload),
                    current_values={"baseline": SURFACE_SOURCE},
                    require_one=False,
                )
                self.assertEqual(len(bundles), 2)
                self.assertNotEqual(bundles[0].proposal_id, bundles[1].proposal_id)
                trace_text = trace.read_text(encoding="utf-8")
                self.assertNotIn(FAKE_KEY, trace_text)
                self.assertNotIn(HELDOUT_ID, response.read_text(encoding="utf-8"))
        finally:
            server.shutdown()

    def test_cli_rejects_wrong_width_after_retry(self) -> None:
        server, _thread = self._stub_server(
            [{"proposals": [_proposal(1), _proposal(2), _proposal(3)]}]
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prompt = root / "prompt.txt"
                prompt.write_text("prompt", encoding="utf-8")
                response = root / "response.json"
                env = dict(os.environ)
                env.update(
                    {
                        API_KEY_ENV: FAKE_KEY,
                        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    }
                )
                completed = self._run_script(
                    [
                        "--prompt",
                        str(prompt),
                        "--response",
                        str(response),
                        "--width",
                        "2",
                        "--max-attempts",
                        "1",
                    ],
                    env=env,
                )
                self.assertEqual(completed.returncode, 4)
                self.assertFalse(response.exists())
                self.assertNotIn(FAKE_KEY, completed.stderr)
        finally:
            server.shutdown()

    def test_cli_rejects_heldout_ids_in_response(self) -> None:
        leaky = _proposal(1)
        leaky["expected_affected_cases"] = [HELDOUT_ID]
        server, _thread = self._stub_server([{"proposals": [leaky, _proposal(2)]}])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prompt = root / "prompt.txt"
                prompt.write_text("prompt", encoding="utf-8")
                heldout = root / "heldout.json"
                heldout.write_text(json.dumps([HELDOUT_ID]), encoding="utf-8")
                train = root / "train.json"
                train.write_text(json.dumps([TRAIN_ID]), encoding="utf-8")
                response = root / "response.json"
                env = dict(os.environ)
                env.update(
                    {
                        API_KEY_ENV: FAKE_KEY,
                        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    }
                )
                completed = self._run_script(
                    [
                        "--prompt",
                        str(prompt),
                        "--response",
                        str(response),
                        "--width",
                        "2",
                        "--max-attempts",
                        "1",
                        "--heldout-ids-file",
                        str(heldout),
                        "--train-ids-file",
                        str(train),
                    ],
                    env=env,
                )
                self.assertEqual(completed.returncode, 4)
                self.assertFalse(response.exists())
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
