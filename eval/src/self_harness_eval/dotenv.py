"""Minimal, non-logging ``.env`` loader for local Self-Harness runs.

The repository-local ``.env`` file is gitignored and may contain the model API
key.  Entry points call :func:`load_dotenv` so a user can keep credentials out
of shell history/CLI arguments.  The loader:

* never prints or returns values (only the names it applied),
* does not override variables already present in the process environment,
* can be disabled with ``SELF_HARNESS_DISABLE_DOTENV=1`` (used by tests/CI so a
  developer's local key can never leak into a test run),
* skips malformed/comment lines instead of executing anything.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DISABLE_ENV_VAR = "SELF_HARNESS_DISABLE_DOTENV"


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> list[str]:
    """Load ``KEY=VALUE`` pairs from the local env file.

    Returns the list of variable *names* applied; no value is ever logged or
    returned.  Missing file or disabled loading is a no-op.
    """

    if os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}:
        return []
    env_path = Path(path).expanduser() if path is not None else DEFAULT_ENV_FILE
    try:
        if not env_path.is_file():
            return []
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    applied: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, sep, value = stripped.partition("=")
        key = key.strip()
        if not sep or not key or not key.replace("_", "").isalnum():
            continue
        if key in os.environ and not override:
            continue
        raw_value = value.strip()
        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '"'}:
            raw_value = raw_value[1:-1]
        if not raw_value:
            continue
        os.environ[key] = raw_value
        applied.append(key)
    return applied
