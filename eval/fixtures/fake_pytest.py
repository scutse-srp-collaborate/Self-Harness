#!/usr/bin/env python3
"""Fake pytest used by the Terminal-Bench runner smoke tests.

It writes a JUnit XML file and a Harbor-style eval summary with trace metadata
so the runner can be exercised without a real Harbor project or container.  The
behaviour is selected with SELF_HARNESS_FAKE_PYTEST_BEHAVIOR.
"""

from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main(argv: list[str]) -> int:
    junit_path = None
    summary_path = None
    selector = ""
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--junitxml" and index + 1 < len(argv):
            junit_path = Path(argv[index + 1])
            index += 2
            continue
        if token in {"--evals-report-file"} and index + 1 < len(argv):
            summary_path = Path(argv[index + 1])
            index += 2
            continue
        if token == "--model" and index + 1 < len(argv):
            index += 2
            continue
        if not token.startswith("-"):
            selector = token
        index += 1
    if junit_path is None:
        print("fake pytest: no --junitxml supplied", file=sys.stderr)
        return 4
    case_dir = junit_path.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    behavior = os.environ.get("SELF_HARNESS_FAKE_PYTEST_BEHAVIOR", "pass").strip().lower()
    (case_dir / "env_capture.json").write_text(
        json.dumps(
            {
                "SELF_HARNESS_TOOL_TIMEOUT_S": os.environ.get("SELF_HARNESS_TOOL_TIMEOUT_S"),
                "SELF_HARNESS_AGENT_TIMEOUT_SEC": os.environ.get("SELF_HARNESS_AGENT_TIMEOUT_SEC"),
                "SELF_HARNESS_CANDIDATE_WORKSPACE": os.environ.get(
                    "SELF_HARNESS_CANDIDATE_WORKSPACE"
                ),
                "SELF_HARNESS_EVAL_SPLIT": os.environ.get("SELF_HARNESS_EVAL_SPLIT"),
                "SELF_HARNESS_EVAL_REPEAT": os.environ.get("SELF_HARNESS_EVAL_REPEAT"),
                "selector": selector,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    messages_path = case_dir / "messages.json"
    events_path = case_dir / "events.jsonl"
    messages_path.write_text(
        json.dumps(
            [
                {"type": "ai", "content": "inspect", "tool_calls": [{"name": "read_file"}]},
                {"type": "tool", "content": "ok", "name": "read_file"},
                {"type": "ai", "content": "edit", "tool_calls": [{"name": "write_file"}]},
            ]
        ),
        encoding="utf-8",
    )
    events_path.write_text("{}\n", encoding="utf-8")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "agent_result": {
                        "metadata": {
                            "messages_path": str(messages_path),
                            "events_path": str(events_path),
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    if behavior == "timeout":
        time.sleep(float(os.environ.get("SELF_HARNESS_FAKE_PYTEST_TIMEOUT_SLEEP", "3600")))
        return 0
    if behavior == "no_junit":
        print("fake pytest: crashing before writing junit", file=sys.stderr)
        return 9
    testsuite = ET.Element("testsuite", {"name": "fake", "tests": "1", "failures": "0", "errors": "0"})
    testcase = ET.SubElement(testsuite, "testcase", {"classname": "fake", "name": selector or "case"})
    if behavior == "fail":
        testsuite.set("failures", "1")
        failure = ET.SubElement(testcase, "failure", {"message": "fake verifier assertion failed"})
        failure.text = "AssertionError: expected solved artifact\n"
    elif behavior == "skip":
        testsuite.set("skipped", "1")
        ET.SubElement(testcase, "skipped", {"message": "fake skip"})
    tree = ET.ElementTree(testsuite)
    tree.write(junit_path, encoding="utf-8", xml_declaration=True)
    return 0 if behavior in {"pass", "skip", "fail"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
