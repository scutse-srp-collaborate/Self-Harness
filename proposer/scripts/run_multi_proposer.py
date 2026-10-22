#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from self_harness_proposer import (  # noqa: E402
    EditableSurface,
    MultiProposerRequest,
    audit_text,
    build_multi_proposer_prompt,
    load_task_ids_file,
    parse_multi_proposer_response,
    validate_proposal_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or parse a mechanism-diverse multi-proposer prompt.")
    parser.add_argument("--diagnosis", required=True, help="Canonical train diagnosis text file.")
    parser.add_argument("--surface", action="append", default=[], help="Surface spec as name=path. Repeatable.")
    parser.add_argument("--route-count", type=int, default=4)
    parser.add_argument("--harness-overview", default="")
    parser.add_argument("--surface-hints", default="")
    parser.add_argument("--current-eval-observations", default="")
    parser.add_argument("--response", help="Optional proposer response JSON to parse and materialize.")
    parser.add_argument("--output", required=True, help="Prompt or parsed bundle output JSON path.")
    parser.add_argument("--heldout-ids-file", type=Path, help="Held-out task ids to keep out of proposer context.")
    parser.add_argument("--train-ids-file", type=Path, help="Optional train task ids for provenance validation.")
    parser.add_argument(
        "--isolation-output",
        type=Path,
        help="Audit JSON path (defaults to <output>.isolation.json).",
    )
    parser.add_argument(
        "--strict-isolation",
        action="store_true",
        help="Fail when held-out ids are not supplied for the isolation audit.",
    )
    args = parser.parse_args()

    surfaces = []
    for item in args.surface:
        name, sep, raw_path = item.partition("=")
        if not sep or not name.strip() or not raw_path.strip():
            raise RuntimeError(f"surface must be name=path, got: {item!r}")
        path = Path(raw_path).expanduser().resolve()
        surfaces.append(
            EditableSurface(
                name=name.strip(),
                kind="workspace_file",
                target=path.name,
                filename=path.name,
                current_value=path.read_text(encoding="utf-8"),
            )
        )
    if not surfaces:
        raise RuntimeError("at least one --surface name=path is required")

    request = MultiProposerRequest(
        diagnosis=Path(args.diagnosis).read_text(encoding="utf-8"),
        surfaces=tuple(surfaces),
        route_count=args.route_count,
        harness_overview=_read_optional_text(args.harness_overview),
        surface_hints=_read_optional_text(args.surface_hints),
        current_eval_observations=_read_optional_text(args.current_eval_observations),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    heldout_ids = _read_ids(args.heldout_ids_file) if args.heldout_ids_file else None
    train_ids = _read_ids(args.train_ids_file) if args.train_ids_file else None
    if heldout_ids is None and args.strict_isolation:
        raise RuntimeError("--strict-isolation requires --heldout-ids-file")
    audit: dict[str, object] = {"enforced": heldout_ids is not None}

    if args.response:
        response_text = Path(args.response).read_text(encoding="utf-8")
        if heldout_ids is not None:
            audit["response"] = audit_text(
                response_text,
                heldout_task_ids=heldout_ids,
                train_task_ids=train_ids,
                label="proposer_response",
            )
        bundles = parse_multi_proposer_response(
            response_text=response_text,
            current_values={surface.name: surface.current_value for surface in surfaces},
            require_one=False,
        )
        if heldout_ids is not None:
            for bundle in bundles:
                validate_proposal_metadata(
                    proposal=bundle.to_dict(),
                    train_task_ids=train_ids or [],
                    heldout_task_ids=heldout_ids,
                )
        output.write_text(json.dumps({"proposals": [bundle.to_dict() for bundle in bundles]}, indent=2) + "\n")
        audit["proposal_count"] = len(bundles)
    else:
        prompt = build_multi_proposer_prompt(request=request)
        if heldout_ids is not None:
            audit["prompt"] = audit_text(
                prompt,
                heldout_task_ids=heldout_ids,
                train_task_ids=train_ids,
                label="proposer_prompt",
            )
        output.write_text(prompt + "\n", encoding="utf-8")
    isolation_path = args.isolation_output or output.parent / f"{output.name}.isolation.json"
    isolation_path.parent.mkdir(parents=True, exist_ok=True)
    isolation_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _read_ids(path: Path) -> list[str]:
    return load_task_ids_file(path)


def _read_optional_text(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
