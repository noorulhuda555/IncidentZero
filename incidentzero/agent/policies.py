from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from incidentzero.domain.models import RiskLevel


class RiskPolicy:
    def __init__(self, config_path: str | Path = "configs/risk_policy.json") -> None:
        self.mapping = json.loads(Path(config_path).read_text(encoding="utf-8"))

    def risk(self, tool_name: str) -> RiskLevel:
        return RiskLevel(self.mapping.get(tool_name, "critical"))

    def requires_human_approval(self, tool_name: str) -> bool:
        return self.risk(tool_name) in {RiskLevel.HIGH, RiskLevel.CRITICAL}


_REMEDIATION_TOOLS = frozenset({
    "restart_service",
    "scale_service",
    "clear_cache",
    "rollback_deployment",
    "failover_database",
    "shift_traffic",
})


class ReplanPolicy:
    """Decide when the current plan is no longer safe to follow blindly."""

    def should_replan(self, tool_result: dict[str, Any]) -> bool:
        status = tool_result.get("status")
        if status in {
            "stale_precondition",
            "approval_denied",
            "contradictory_evidence",
            "budget_insufficient",
        }:
            return True
        if status == "ok" and tool_result.get("tool") == "verify_recovery":
            data = tool_result.get("data") or {}
            if data.get("criteria_met") is False:
                return True
        if status == "error" and not tool_result.get("retryable", False):
            return True
        if status == "ok" and tool_result.get("tool") in _REMEDIATION_TOOLS:
            data = tool_result.get("data") or {}
            effect = str(data.get("effect", "")).lower()
            if any(
                phrase in effect
                for phrase in (
                    "may remain",
                    "no confirmed",
                    "without evidence",
                    "no evidence",
                    "symptoms remain",
                )
            ):
                return True
        return False

    def replan_reason(self, tool_result: dict[str, Any]) -> str:
        status = tool_result.get("status")
        tool = tool_result.get("tool", "unknown")
        if status == "stale_precondition":
            return "world_version_changed"
        if status == "approval_denied":
            return "approval_denied"
        if status == "contradictory_evidence":
            return "contradictory_evidence"
        if status == "budget_insufficient":
            return "budget_too_low"
        if status == "ok" and tool == "verify_recovery":
            return "recovery_not_verified"
        if status == "error":
            return "non_retryable_failure"
        if status == "ok" and tool in _REMEDIATION_TOOLS:
            return "action_ok_but_unverified"
        return "replan_required"


class PostToolAction(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    RE_OBSERVE = "re_observe"
    REPLAN = "replan"
    ABORT = "abort"


class ToolOutcomeRouter:
    """Map a tool result to one explicit controller path (not a generic try-again)."""

    def __init__(self, replan_policy: ReplanPolicy | None = None) -> None:
        self.replan_policy = replan_policy or ReplanPolicy()

    def decide(
        self,
        tool_name: str,
        result: dict[str, Any],
        *,
        budget_low: bool = False,
    ) -> PostToolAction:
        if budget_low:
            trigger = {"status": "budget_insufficient", "tool": tool_name, "retryable": False}
            if self.replan_policy.should_replan(trigger):
                return PostToolAction.REPLAN
            return PostToolAction.ABORT

        status = result.get("status")
        enriched = {**result, "tool": tool_name}

        if status == "stale_precondition":
            return PostToolAction.RE_OBSERVE
        if status == "transient_error" and result.get("retryable"):
            return PostToolAction.RETRY
        if status == "validation_error":
            return PostToolAction.CONTINUE
        if self.replan_policy.should_replan(enriched):
            return PostToolAction.REPLAN
        if status == "error" and not result.get("retryable", False):
            return PostToolAction.ABORT
        return PostToolAction.CONTINUE


class LoopGuard:
    def __init__(self, max_same_action_repeats: int = 2) -> None:
        self.max_same_action_repeats = max_same_action_repeats
        self._counts: dict[str, int] = {}

    def record(self, action_name: str, arguments: dict[str, Any]) -> bool:
        """Return True when the exact same action has repeated too often."""
        # TODO(A1): create a stable action fingerprint and detect looping.
        return False
