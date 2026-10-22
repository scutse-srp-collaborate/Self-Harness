#!/usr/bin/env python3
"""OpenAI-compatible proposer entry point for the Self-Harness workflow.

Reads the train-only proposer prompt produced by the workflow, calls the same
model/endpoint/key/decoding as the execution agent, validates the JSON response
against the existing proposer contract (width, distinctness, required fields,
train-only task references) and writes a response file consumable by
``proposer/scripts/run_multi_proposer.py``.

The script never reads held-out evidence and never writes credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for relative in ("proposer/src", "eval/src"):
    candidate = REPO_ROOT / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from self_harness_eval.dotenv import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()

from self_harness_eval.openai_compat import (  # noqa: E402
    API_KEY_ENV,
    MissingApiKeyError,
    build_chat_model,
    extract_json_object,
    redact_secrets,
    settings_from_environment,
    validate_proposals_payload,
)
from self_harness_proposer.isolation import (  # noqa: E402
    assert_no_heldout_leak,
    load_task_ids_file,
    validate_proposal_metadata,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_INVALID = 4

SYSTEM_PROMPT = (
    "You are the proposer stage of the Self-Harness loop. Follow the user prompt exactly. "
    "Return only the requested JSON object; no prose, no markdown fences, no logs."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible Self-Harness proposer.")
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--heldout-ids-file", type=Path)
    parser.add_argument("--train-ids-file", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    settings = settings_from_environment()
    trace_path = (
        args.trace.expanduser().resolve()
        if args.trace
        else args.response.expanduser().resolve().with_suffix(args.response.suffix + ".trace.jsonl")
    )
    if args.self_check:
        return _self_check(settings=settings)
    if not os.environ.get(API_KEY_ENV, "").strip():
        print(
            f"error: {API_KEY_ENV} is not set in the environment. Refusing to make real model "
            "calls. Set the variable in the runtime environment; do not paste the key into "
            "chat, config files or command arguments.",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    prompt_path = args.prompt.expanduser().resolve()
    if not prompt_path.exists():
        print(f"error: proposer prompt does not exist: {prompt_path}", file=sys.stderr)
        return EXIT_CONFIG
    prompt = prompt_path.read_text(encoding="utf-8")
    heldout_ids = _read_ids(args.heldout_ids_file) if args.heldout_ids_file else None
    train_ids = _read_ids(args.train_ids_file) if args.train_ids_file else None
    if args.width < 1:
        print("error: --width must be >= 1", file=sys.stderr)
        return EXIT_CONFIG

    try:
        model = build_chat_model(settings)
    except MissingApiKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001
        print(
            f"error: cannot create OpenAI-compatible model: {exc.__class__.__name__}: "
            f"{_safe(str(exc))}",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    trace_lines: list[dict[str, object]] = []
    last_error = "unknown error"
    for attempt in range(1, max(1, args.max_attempts) + 1):
        started = time.time()
        try:
            reply = model.invoke(messages)
            text = _message_text(reply)
            trace_lines.append(
                {
                    "attempt": attempt,
                    "duration_s": round(time.time() - started, 3),
                    "response_preview": _safe(text)[:4000],
                }
            )
            payload = extract_json_object(text)
            proposals = validate_proposals_payload(payload, expected_width=args.width)
            _validate_proposal_train_only(
                proposals=proposals,
                train_ids=train_ids,
                heldout_ids=heldout_ids,
            )
            response_payload = {"proposals": proposals}
            args.response.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            args.response.expanduser().resolve().write_text(
                json.dumps(response_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _write_trace(trace_path, settings.non_secret_dict(), trace_lines, status="complete")
            print(f"proposer wrote {len(proposals)} proposals to {args.response}")
            return EXIT_OK
        except Exception as exc:  # noqa: BLE001 - retry on malformed output
            last_error = f"{exc.__class__.__name__}: {_safe(str(exc))}"
            trace_lines.append({"attempt": attempt, "error": last_error[:2000]})
            if attempt < args.max_attempts:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid: "
                            f"{last_error}. Return exactly one JSON object with `proposals` "
                            f"containing exactly {args.width} distinct valid proposals."
                        ),
                    },
                ]
    _write_trace(trace_path, settings.non_secret_dict(), trace_lines, status="invalid")
    print(f"error: proposer output invalid after retries: {last_error}", file=sys.stderr)
    return EXIT_INVALID


def _read_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return load_task_ids_file(path.expanduser().resolve())


def _validate_proposal_train_only(
    *,
    proposals: list[dict[str, object]],
    train_ids: list[str] | None,
    heldout_ids: list[str] | None,
) -> None:
    if heldout_ids:
        text = json.dumps({"proposals": proposals}, sort_keys=True)
        assert_no_heldout_leak(text, heldout_ids, label="proposer_response")
        for proposal in proposals:
            validate_proposal_metadata(
                proposal=proposal,
                train_task_ids=train_ids or [],
                heldout_task_ids=heldout_ids,
            )


def _message_text(reply: object) -> str:
    if isinstance(reply, str):
        return reply
    content = getattr(reply, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        return "\n".join(parts)
    return str(reply)


def _write_trace(
    path: Path,
    settings: dict[str, object],
    attempts: list[dict[str, object]],
    *,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "format": "self_harness.proposer_trace.v1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "proposer_start",
            "settings": settings,
        },
        *[
            {
                "format": "self_harness.proposer_trace.v1",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event_type": "proposer_attempt",
                **attempt,
            }
            for attempt in attempts
        ],
        {
            "format": "self_harness.proposer_trace.v1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "proposer_finish",
            "status": status,
        },
    ]
    key = os.environ.get(API_KEY_ENV, "")
    path.write_text(
        "\n".join(redact_secrets(json.dumps(item, sort_keys=True), secrets=(key,)) for item in records)
        + "\n",
        encoding="utf-8",
    )


def _self_check(*, settings: object) -> int:
    from langchain_openai import ChatOpenAI

    problems: list[str] = []
    api_key_present = bool(os.environ.get(API_KEY_ENV, "").strip())
    try:
        model = build_chat_model(
            settings,  # type: ignore[arg-type]
            api_key_override=None if api_key_present else "self-check-placeholder-not-a-key",
        )
        if not isinstance(model, ChatOpenAI):
            problems.append("chat_model: unexpected type")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"chat_model: {exc.__class__.__name__}: {_safe(str(exc))}")
    sample = {
        "proposals": [
            {
                "proposal_id": f"p{index}",
                "selection_decision": "eval_candidate",
                "selected_cluster": "cluster",
                "candidate_values": {"build_system_prompt": f"value {index}"},
                "metadata": {
                    "mechanism_family": "prompt_instruction",
                    "exact_hook": "build_system_prompt",
                    "mechanism_to_hook_rationale": "self-check",
                    "why_not_noop": "self-check",
                },
            }
            for index in range(1, 5)
        ]
    }
    try:
        validate_proposals_payload(sample, expected_width=4)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"validator: {exc}")
    summary = {
        "check": "openai_compatible_proposer_self_check",
        "ok": not problems,
        "problems": problems,
        "provider": "openai_compatible",
        "model": settings.model,  # type: ignore[attr-defined]
        "base_url": settings.base_url,  # type: ignore[attr-defined]
        "api_key_present": api_key_present,
        "note": "Interface validation only; no real model call is made.",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_OK if not problems else EXIT_INVALID


def _safe(text: str) -> str:
    return redact_secrets(str(text), secrets=(os.environ.get(API_KEY_ENV, ""),))[:2000]


if __name__ == "__main__":
    raise SystemExit(main())
