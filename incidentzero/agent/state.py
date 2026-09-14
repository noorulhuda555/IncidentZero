from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from incidentzero.domain.models import AgentPlan
from incidentzero.telemetry.budget import BudgetManager


class TerminalStatus(str, Enum):
    RUNNING = "running"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ABORTED = "aborted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


@dataclass(slots=True)
class ToolOutcomeRecord:
    tool: str
    status: str
    world_version: int | None
    evidence_id: str | None
    message: str | None = None


def action_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    """Stable fingerprint for loop detection (tool + normalized arguments)."""
    normalized = json.dumps(
        {"tool": tool_name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return normalized


@dataclass
class AgentState:
    """Explicit controller memory — reconstruct a run without reading chat messages."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    plan: AgentPlan | None = None
    evidence_ids: list[str] = field(default_factory=list)
    latest_world_version: int | None = None
    recent_tool_outcomes: list[ToolOutcomeRecord] = field(default_factory=list)
    llm_call_count: int = 0
    tool_call_count: int = 0
    action_fingerprint_history: list[str] = field(default_factory=list)
    fingerprint_counts: dict[str, int] = field(default_factory=dict)
    terminal_status: TerminalStatus = TerminalStatus.RUNNING
    plan_history: list[dict[str, Any]] = field(default_factory=list)
    last_verify_recovery_evidence_id: str | None = None
    verification_criteria_met: bool = False

    _MAX_RECENT_OUTCOMES: int = field(default=8, init=False, repr=False)

    @property
    def plan_revision(self) -> int:
        return self.plan.revision if self.plan is not None else 0

    @property
    def status(self) -> str:
        """String status for backward compatibility with baseline code."""
        return self.terminal_status.value

    def set_terminal_status(self, status: TerminalStatus | str) -> None:
        if isinstance(status, str):
            status = TerminalStatus(status)
        self.terminal_status = status

    def reset_run(self, system_prompt: str, user_goal: str) -> None:
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_goal},
        ]
        self.plan = None
        self.evidence_ids.clear()
        self.latest_world_version = None
        self.recent_tool_outcomes.clear()
        self.llm_call_count = 0
        self.tool_call_count = 0
        self.action_fingerprint_history.clear()
        self.fingerprint_counts.clear()
        self.plan_history.clear()
        self.last_verify_recovery_evidence_id = None
        self.verification_criteria_met = False
        self.terminal_status = TerminalStatus.RUNNING

    def _plan_snapshot(self, plan: AgentPlan) -> dict[str, Any]:
        return {
            "hypothesis": plan.hypothesis,
            "revision": plan.revision,
            "rationale_summary": plan.rationale_summary,
            "steps": [
                {
                    "step_id": step.step_id,
                    "objective": step.objective,
                    "success_signal": step.success_signal,
                    "status": step.status,
                }
                for step in plan.steps
            ],
        }

    def set_plan(self, plan: AgentPlan) -> None:
        self.plan = plan

    def apply_revised_plan(self, plan: AgentPlan) -> None:
        if self.plan is not None:
            self.plan_history.append(self._plan_snapshot(self.plan))
        self.plan = plan

    def consume_llm(self, budget: BudgetManager) -> None:
        budget.consume_llm()
        self.llm_call_count = budget.llm_calls

    def consume_tool(self, budget: BudgetManager) -> None:
        budget.consume_tool()
        self.tool_call_count = budget.tool_calls

    def has_budget(self, budget: BudgetManager) -> bool:
        return budget.remaining_llm > 0 and budget.remaining_tools > 0

    def reset_action_fingerprints(self) -> None:
        self.fingerprint_counts.clear()
        self.action_fingerprint_history.clear()

    def record_action_fingerprint(self, tool_name: str, arguments: dict[str, Any]) -> str:
        fp = action_fingerprint(tool_name, arguments)
        self.fingerprint_counts[fp] = self.fingerprint_counts.get(fp, 0) + 1
        self.action_fingerprint_history.append(fp)
        if len(self.action_fingerprint_history) > 64:
            self.action_fingerprint_history = self.action_fingerprint_history[-64:]
        return fp

    def observe_result(self, tool_name: str, result: dict[str, Any]) -> None:
        evidence = result.get("evidence_id")
        if evidence:
            self.evidence_ids.append(evidence)
        version = result.get("world_version")
        if isinstance(version, int):
            if self.latest_world_version is not None and version > self.latest_world_version:
                self.reset_action_fingerprints()
                self.last_verify_recovery_evidence_id = None
                self.verification_criteria_met = False
            self.latest_world_version = version
        if tool_name == "verify_recovery" and result.get("status") == "ok":
            data = result.get("data") or {}
            if data.get("criteria_met") and isinstance(evidence, str):
                self.last_verify_recovery_evidence_id = evidence
                self.verification_criteria_met = True
            else:
                self.verification_criteria_met = False
        record = ToolOutcomeRecord(
            tool=tool_name,
            status=str(result.get("status", "unknown")),
            world_version=result.get("world_version") if isinstance(result.get("world_version"), int) else None,
            evidence_id=evidence if isinstance(evidence, str) else None,
            message=result.get("message") if isinstance(result.get("message"), str) else None,
        )
        self.recent_tool_outcomes.append(record)
        if len(self.recent_tool_outcomes) > self._MAX_RECENT_OUTCOMES:
            self.recent_tool_outcomes = self.recent_tool_outcomes[-self._MAX_RECENT_OUTCOMES :]

    def append_system_note(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})

    def to_snapshot(self) -> dict[str, Any]:
        """Serializable view of everything the controller knows (excluding chat)."""
        plan_block: dict[str, Any] | None = None
        if self.plan is not None:
            plan_block = {
                "hypothesis": self.plan.hypothesis,
                "revision": self.plan.revision,
                "rationale_summary": self.plan.rationale_summary,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "objective": step.objective,
                        "success_signal": step.success_signal,
                        "status": step.status,
                    }
                    for step in self.plan.steps
                ],
            }
        return {
            "terminal_status": self.terminal_status.value,
            "plan": plan_block,
            "plan_revision": self.plan_revision,
            "evidence_ids": list(self.evidence_ids),
            "latest_world_version": self.latest_world_version,
            "recent_tool_outcomes": [
                {
                    "tool": row.tool,
                    "status": row.status,
                    "world_version": row.world_version,
                    "evidence_id": row.evidence_id,
                    "message": row.message,
                }
                for row in self.recent_tool_outcomes
            ],
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "action_fingerprint_history": list(self.action_fingerprint_history),
            "fingerprint_counts": dict(self.fingerprint_counts),
            "plan_history": list(self.plan_history),
            "last_verify_recovery_evidence_id": self.last_verify_recovery_evidence_id,
            "verification_criteria_met": self.verification_criteria_met,
        }

    def format_snapshot(self) -> str:
        return json.dumps(self.to_snapshot(), indent=2, ensure_ascii=False)

    def run_summary(self) -> str:
        """Human-readable reconstruction of progress without chat messages."""
        snap = self.to_snapshot()
        lines = [
            f"status={snap['terminal_status']}",
            f"world_version={snap['latest_world_version']}",
            f"plan_revision={snap['plan_revision']}",
            f"llm_calls={snap['llm_call_count']} tool_calls={snap['tool_call_count']}",
            f"evidence_ids={snap['evidence_ids']}",
        ]
        if snap["plan"]:
            lines.append(f"hypothesis={snap['plan']['hypothesis']!r}")
        for row in snap["recent_tool_outcomes"]:
            lines.append(
                f"  tool {row['tool']}: {row['status']} "
                f"(ev={row['evidence_id']}, wv={row['world_version']})"
            )
        return "\n".join(lines)
