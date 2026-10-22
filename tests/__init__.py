"""Test package bootstrap.

Importing ``tests`` (including via unittest discovery) puts every repository
package root on ``sys.path`` so tests can import the runners, diagnosis,
proposer, acceptance and workflow packages without installation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Never let a developer's local .env key leak into test runs or subprocesses.
os.environ.setdefault("SELF_HARNESS_DISABLE_DOTENV", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "eval/src",
    "eval/fixtures",
    "diagnosis/src",
    "proposer/src",
    "acceptance/src",
    "workflow/src",
):
    candidate = str(REPO_ROOT / relative)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
