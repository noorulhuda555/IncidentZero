from __future__ import annotations

from typing import Any

from incidentzero.agent.policies import LoopGuard, RiskPolicy
from incidentzero.agent.state import AgentState
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.tools.consequential import tools_requiring_world_version
from incidentzero.tools.registry import ToolRegistry

_WORLD_VERSION_TOOLS = tools_requiring_world_version()
_NEAR_BUDGET_ALLOWED = frozenset({
    "get_incident",
    "get_service_health",
    "get_metrics",
    "get_logs",
    "get_deployments",
    "get_dependencies",
    "get_runbook",
    "verify_recovery",
    "close_incident",
    "escalate_incident",
})


class ToolExecutionGate:
    """Pre-execution checks — invalid or stale calls must not reach the simulator."""

    def __init__(
        self,
        registry: ToolRegistry,
        risk: RiskPolicy,
        loop_guard: LoopGuard,
    ) -> None:
        self.registry = registry
        self.risk = risk
        self.loop_guard = loop_guard

    def _reject(
        self,
        tool: str,
        status: str,
        message: str,
        *,
        retryable: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": status,
            "tool": tool,
            "world_version": self.registry.environment.world_version,
            "evidence_id": None,
            "data": None,
            "retryable": retryable,
            "message": message,
        }
        out.update(extra)
        return out

    def check(
        self,
        tool: str,
        arguments: dict[str, Any],
        state: AgentState,
        budget: BudgetManager,
        *,
        skip_loop_check: bool = False,
    ) -> dict[str, Any] | None:
        """Return a tool result dict when blocked; None when the simulator may be called."""
        ok, error = self.registry.validate(tool, arguments)
        if not ok:
            return self._reject(tool, "validation_error", error or "Invalid tool call.", retryable=False)

        _ = self.risk.risk(tool)

        if budget.remaining_tools <= 0:
            return self._reject(tool, "validation_error", "Tool-call budget exhausted.", retryable=False)

        if budget.runtime_exceeded():
            return self._reject(
                tool,
                "validation_error",
                f"Wall-clock budget exhausted ({budget.max_runtime_seconds:.0f}s from configs/limits.json).",
                retryable=False,
            )

        if budget.critical_exhaustion() and tool not in _NEAR_BUDGET_ALLOWED:
            return self._reject(
                tool,
                "budget_insufficient",
                "Near budget exhaustion: prioritize verify_recovery or escalate_incident.",
                retryable=False,
            )

        if not skip_loop_check and self.loop_guard.record(tool, arguments):
            return self._reject(
                tool,
                "validation_error",
                "Repeated identical action blocked; change approach or re-plan.",
                retryable=False,
            )

        if self.risk.requires_human_approval(tool) and not state.evidence_ids:
            return self._reject(
                tool,
                "validation_error",
                "High/critical actions require at least one evidence id already collected in state.",
                retryable=False,
            )

        if tool == "close_incident":
            cited = arguments.get("evidence_ids") or []
            if not state.verification_criteria_met or not state.last_verify_recovery_evidence_id:
                return self._reject(
                    tool,
                    "validation_error",
                    "Run verify_recovery with criteria_met=true before close_incident.",
                    retryable=False,
                )
            if state.last_verify_recovery_evidence_id not in cited:
                return self._reject(
                    tool,
                    "validation_error",
                    "close_incident must cite the latest verify_recovery evidence id.",
                    retryable=False,
                )

        if tool == "escalate_incident" and not state.evidence_ids:
            return self._reject(
                tool,
                "validation_error",
                "Escalation requires evidence ids gathered during the investigation.",
                retryable=False,
            )

        if tool in _WORLD_VERSION_TOOLS:
            latest = state.latest_world_version
            expected = arguments.get("expected_world_version")
            if latest is None:
                return self._reject(
                    tool,
                    "validation_error",
                    "Observe the environment first to learn world_version before consequential actions.",
                    retryable=False,
                )
            if expected != latest:
                return self._reject(
                    tool,
                    "stale_precondition",
                    "expected_world_version does not match latest observed world_version; re-observe, do not patch the version.",
                    retryable=True,
                    expected=expected,
                    actual=latest,
                )

        return None
