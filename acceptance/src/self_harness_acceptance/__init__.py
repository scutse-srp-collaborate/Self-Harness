"""Non-regression Pareto acceptance gate for Self-Harness candidates."""

from .gate import (
    AcceptanceError,
    DEFAULT_EXPECTED_REPEATS,
    DEFAULT_SPLITS,
    DECISION_ACCEPTED,
    DECISION_INCONCLUSIVE,
    DECISION_INVALID,
    DECISION_REJECTED,
    RULE_VERSION,
    load_result,
    run_acceptance_gate,
    write_acceptance,
)

__all__ = [
    "AcceptanceError",
    "DEFAULT_EXPECTED_REPEATS",
    "DEFAULT_SPLITS",
    "DECISION_ACCEPTED",
    "DECISION_INCONCLUSIVE",
    "DECISION_INVALID",
    "DECISION_REJECTED",
    "RULE_VERSION",
    "load_result",
    "run_acceptance_gate",
    "write_acceptance",
]
