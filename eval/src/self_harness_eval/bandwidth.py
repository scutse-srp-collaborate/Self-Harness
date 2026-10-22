"""Outbound bandwidth limiting that is actually applied and honestly recorded.

The experiment protocol requires a 2 MB/s outbound cap for Terminal-Bench runs
and requires the result to state whether the limit was really enforced.  This
module wraps the executed command with ``trickle`` when available, optionally
attempts a host-level ``tc`` qdisc add on an explicit interface, and otherwise
records ``enforced=false`` with a concrete reason instead of pretending the
limit is active.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass
class BandwidthPlan:
    requested_mbps: float | None
    enforcement: str
    method: str
    enforced: bool
    detail: str
    command_prefix: tuple[str, ...] = ()
    interface: str | None = None
    applied_commands: tuple[tuple[str, ...], ...] = ()

    @property
    def active(self) -> bool:
        return self.enforced

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mbps": self.requested_mbps,
            "enforcement_requested": self.enforcement,
            "method": self.method,
            "enforced": self.enforced,
            "detail": self.detail,
            "interface": self.interface,
            "applied_commands": [list(item) for item in self.applied_commands],
        }

    def wrap(self, argv: Sequence[str]) -> list[str]:
        return [*self.command_prefix, *(str(item) for item in argv)]


def plan_bandwidth_limit(
    *,
    requested_mbps: float | None,
    enforcement: str = "auto",
    interface: str | None = None,
) -> BandwidthPlan:
    enforcement = (enforcement or "auto").strip().lower()
    if requested_mbps is None:
        return BandwidthPlan(
            requested_mbps=None,
            enforcement=enforcement,
            method="none",
            enforced=False,
            detail="no outbound bandwidth limit requested",
        )
    if requested_mbps <= 0:
        raise ValueError("requested_mbps must be positive")
    if enforcement == "none":
        return BandwidthPlan(
            requested_mbps=float(requested_mbps),
            enforcement=enforcement,
            method="none",
            enforced=False,
            detail="bandwidth limit explicitly disabled by config",
        )
    if enforcement in {"auto", "trickle"}:
        trickle = shutil.which("trickle")
        if trickle:
            kbps = max(1, int(round(requested_mbps * 1024)))
            return BandwidthPlan(
                requested_mbps=float(requested_mbps),
                enforcement=enforcement,
                method="trickle",
                enforced=True,
                detail=(
                    f"wrapped subprocess with `trickle -d {kbps}` (KB/s). This caps the "
                    "agent subprocess network socket usage; it does not cap other processes."
                ),
                command_prefix=(trickle, "-d", str(kbps)),
            )
        if enforcement == "trickle":
            return BandwidthPlan(
                requested_mbps=float(requested_mbps),
                enforcement=enforcement,
                method="trickle",
                enforced=False,
                detail="requested enforcement=trickle but the `trickle` binary is not installed",
            )
    if enforcement in {"auto", "tc"}:
        plan = _try_tc_limit(requested_mbps=float(requested_mbps), interface=interface)
        if plan is not None:
            return plan
        if enforcement == "tc":
            return BandwidthPlan(
                requested_mbps=float(requested_mbps),
                enforcement=enforcement,
                method="tc",
                enforced=False,
                detail=_tc_unavailable_reason(interface=interface),
                interface=interface,
            )
    return BandwidthPlan(
        requested_mbps=float(requested_mbps),
        enforcement=enforcement,
        method="none",
        enforced=False,
        detail=(
            "no supported bandwidth limiter is available on this host "
            "(`trickle` and a usable `tc` root qdisc were both unavailable)"
        ),
    )


def _try_tc_limit(*, requested_mbps: float, interface: str | None) -> BandwidthPlan | None:
    tc = shutil.which("tc")
    if not tc:
        return None
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        return None
    iface = interface or _default_interface()
    if not iface:
        return None
    rate = f"{requested_mbps:g}mbit"
    command = (tc, "qdisc", "add", "dev", iface, "root", "tbf", "rate", rate, "burst", "32kbit", "latency", "50ms")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    return BandwidthPlan(
        requested_mbps=requested_mbps,
        enforcement="tc",
        method="tc",
        enforced=True,
        detail=(
            f"installed a host-level tbf qdisc on {iface} at {rate}. This caps the whole "
            "interface and requires cleanup after the run."
        ),
        interface=iface,
        applied_commands=(command,),
    )


def cleanup_bandwidth(plan: BandwidthPlan, *, tc_binary: str | None = None) -> list[tuple[str, ...]]:
    """Remove host-level qdiscs installed by :func:`plan_bandwidth_limit`."""

    if plan.method != "tc" or not plan.interface:
        return []
    tc = tc_binary or shutil.which("tc")
    if not tc:
        return []
    command = (tc, "qdisc", "del", "dev", plan.interface, "root")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return [command] if completed.returncode == 0 else []


def _tc_unavailable_reason(*, interface: str | None) -> str:
    if not shutil.which("tc"):
        return "requested enforcement=tc but the `tc` binary is not installed"
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        return "requested enforcement=tc but the runner is not running as root"
    if not (interface or _default_interface()):
        return "requested enforcement=tc but no network interface was configured"
    return "requested enforcement=tc but installing the qdisc failed"


def _default_interface() -> str | None:
    for candidate in ("eth0", "ens3", "enp0s3", "wlan0"):
        if os.path.exists(f"/sys/class/net/{candidate}"):
            return candidate
    return None


def bandwidth_plan_from_settings(settings: Any, *, raw: Mapping[str, Any] | None = None) -> BandwidthPlan:
    """Build a plan from ``[constraints]`` settings."""

    raw = raw or {}
    interface = raw.get("bandwidth_interface")
    return plan_bandwidth_limit(
        requested_mbps=getattr(settings, "outbound_bandwidth_mbps", None),
        enforcement=getattr(settings, "bandwidth_enforcement", "auto"),
        interface=str(interface) if interface else None,
    )
