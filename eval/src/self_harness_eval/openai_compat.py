"""Shared OpenAI-compatible model settings for the agent and proposer.

The agent adapter and the proposer adapter must use exactly the same model,
endpoint, key source and decoding configuration.  Keeping this in one module
prevents silent drift between the two adapters.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.qadg.xyz/v1"
MODEL_ENV = "SELF_HARNESS_MODEL"
BASE_URL_ENV = "OPENAI_BASE_URL"
API_KEY_ENV = "SELF_HARNESS_MODEL_API_KEY"

MISSING_API_KEY_MESSAGE = (
    "SELF_HARNESS_MODEL_API_KEY is not set in the environment. Refusing to make real "
    "model calls. Set it in the process environment before running the pilot; do not "
    "paste the value into chat, config files or command arguments."
)

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


class MissingApiKeyError(RuntimeError):
    """Raised when a real model call is attempted without an injected API key."""


@dataclass(frozen=True)
class OpenAIModelSettings:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    request_timeout_s: float = 120.0
    max_retries: int = 2

    def non_secret_dict(self) -> dict[str, Any]:
        return {
            "provider": "openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "endpoint_type": "hosted_api",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "request_timeout_s": self.request_timeout_s,
            "max_retries": self.max_retries,
        }


def settings_from_environment() -> OpenAIModelSettings:
    return OpenAIModelSettings(
        model=os.environ.get(MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        base_url=os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
    )


def api_key_from_environment() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise MissingApiKeyError(MISSING_API_KEY_MESSAGE)
    return key


def build_chat_model(
    settings: OpenAIModelSettings | None = None,
    *,
    api_key_override: str | None = None,
) -> Any:
    """Create a ChatOpenAI client.  No network request is made here."""

    from pydantic import SecretStr
    from langchain_openai import ChatOpenAI

    settings = settings or settings_from_environment()
    key = api_key_override or api_key_from_environment()
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=SecretStr(key),
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        timeout=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from a model response (possibly fenced)."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("model response is empty")
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, Mapping):
            return dict(payload)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fenced:
        try:
            payload = json.loads(fenced.group(1).strip())
            if isinstance(payload, Mapping):
                return dict(payload)
        except json.JSONDecodeError:
            pass
    match = _JSON_OBJECT_RE.search(stripped)
    if match is None:
        raise ValueError("model response did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, Mapping):
        raise ValueError("model response JSON must be an object")
    return dict(payload)


def redact_secrets(text: str, *, secrets: Sequence[str] = ()) -> str:
    result = str(text)
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result


def validate_proposals_payload(
    payload: Mapping[str, Any],
    *,
    expected_width: int,
) -> list[dict[str, Any]]:
    """Validate proposer JSON before it reaches the existing workflow parser."""

    if expected_width < 1:
        raise ValueError("expected_width must be >= 1")
    proposals = payload.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError("proposal response must contain a proposals list")
    if len(proposals) != expected_width:
        raise ValueError(
            f"proposal response must contain exactly {expected_width} proposals, "
            f"got {len(proposals)}"
        )
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_signatures: set[str] = set()
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise ValueError(f"proposal {index} is not an object")
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        if not proposal_id:
            raise ValueError(f"proposal {index} is missing proposal_id")
        if proposal_id in seen_ids:
            raise ValueError(f"duplicate proposal_id: {proposal_id!r}")
        seen_ids.add(proposal_id)
        decision = str(proposal.get("selection_decision") or "").strip().lower()
        if decision not in {"eval_candidate", "decline"}:
            raise ValueError(
                f"proposal {proposal_id!r} selection_decision must be eval_candidate or decline"
            )
        candidate_values = proposal.get("candidate_values")
        if candidate_values is None:
            candidate_values = {}
        if not isinstance(candidate_values, Mapping):
            raise ValueError(f"proposal {proposal_id!r} candidate_values must be an object")
        metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), Mapping) else {}
        for key in ("mechanism_family", "exact_hook", "mechanism_to_hook_rationale", "why_not_noop"):
            value = proposal.get(key, metadata.get(key))
            if decision == "eval_candidate" and not str(value or "").strip():
                raise ValueError(f"proposal {proposal_id!r} is missing {key}")
        if decision == "eval_candidate":
            if len(candidate_values) != 1:
                raise ValueError(
                    f"proposal {proposal_id!r} must change exactly one candidate_values hook"
                )
            hook = str(metadata.get("exact_hook") or proposal.get("exact_hook") or "")
            changed = str(next(iter(candidate_values)))
            if hook and changed != hook:
                raise ValueError(
                    f"proposal {proposal_id!r} exact_hook {hook!r} does not match "
                    f"candidate_values key {changed!r}"
                )
        elif candidate_values:
            raise ValueError(f"decline proposal {proposal_id!r} must not change candidate_values")
        signature = json.dumps(
            {
                "selected_cluster": proposal.get("selected_cluster"),
                "mechanism_family": metadata.get("mechanism_family"),
                "exact_hook": metadata.get("exact_hook"),
                "candidate_values": candidate_values,
            },
            sort_keys=True,
            default=str,
        )
        if signature in seen_signatures:
            raise ValueError(f"duplicate proposal signature for {proposal_id!r}")
        seen_signatures.add(signature)
        result.append(dict(proposal))
    return result
