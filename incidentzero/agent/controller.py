from __future__ import annotations

import json
from typing import Any

from incidentzero.approval.gateway import ApprovalGateway
from incidentzero.domain.models import AgentOutcome, ModelReply, ToolCall
from incidentzero.model.base import ModelClient
from incidentzero.telemetry.budget import BudgetExceeded, BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

from .planner import Planner
from .policies import LoopGuard, ReplanPolicy, RiskPolicy
from .prompts import SYSTEM_PROMPT
from .recovery import RetryPolicy
from .state import AgentState, TerminalStatus


class AgentController:
    """Baseline controller derived from the class 'first agent' loop.

    It intentionally lacks production reliability. Your assignment is to evolve this
    controller rather than replacing it with an agent framework.
    """

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        approval: ApprovalGateway,
        budget: BudgetManager,
        trace: TraceRecorder,
    ) -> None:
        self.model = model
        self.tools = tools
        self.approval = approval
        self.budget = budget
        self.trace = trace
        self.state = AgentState()
        self.planner = Planner(model)
        self.risk = RiskPolicy()
        self.replan_policy = ReplanPolicy()
        self.loop_guard = LoopGuard()
        self.retry_policy = RetryPolicy()

    def _record_state(self, label: str) -> None:
        self.trace.record(label, self.state.to_snapshot())

    def _outcome(self, status: TerminalStatus, summary: str) -> AgentOutcome:
        self.state.set_terminal_status(status)
        self._record_state("terminal_state")
        return AgentOutcome(
            status=status.value,
            summary=summary,
            llm_calls=self.state.llm_call_count,
            tool_calls=self.state.tool_call_count,
            final_world_version=self.state.latest_world_version,
            evidence_ids=list(self.state.evidence_ids),
            trace_path=str(self.trace.path),
        )

    def _model_decide(self) -> ModelReply:
        # TODO(A1): integrate bounded retries, budget accounting and trace metrics.
        self.state.consume_llm(self.budget)
        return self.model.decide(self.state.messages, self.tools.groq_tools)

    def _execute_tool_call(self, call: ToolCall) -> dict[str, Any]:
        """Validate, approve if needed, execute, trace, and return one observation.

        The baseline only validates and executes. Students must add:
        - malformed-call recovery,
        - approval for high/critical actions,
        - loop detection,
        - budget-aware behavior,
        - stale-precondition handling,
        - retryable tool-error handling where appropriate.
        """
        self.state.consume_tool(self.budget)
        self.state.record_action_fingerprint(call.name, call.arguments)
        ok, error = self.tools.validate(call.name, call.arguments)
        if not ok:
            return {
                "status": "validation_error", "tool": call.name,
                "world_version": self.tools.environment.world_version,
                "evidence_id": None, "data": None,
                "retryable": False, "message": error,
            }
        # TODO(A1): ask self.approval before high/critical actions. The LLM cannot approve itself.
        result = self.tools.execute(call.name, call.arguments)
        self.trace.record("tool_result", {"call": {"name": call.name, "arguments": call.arguments}, "result": result})
        return result

    def _append_assistant(self, reply: ModelReply) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.content}
        if reply.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in reply.tool_calls
            ]
        self.state.messages.append(msg)

    def _append_tool_result(self, call: ToolCall, result: dict[str, Any]) -> None:
        self.state.messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    def run(self) -> AgentOutcome:
        self.state.reset_run(
            SYSTEM_PROMPT,
            "Investigate the active production incident, mitigate it safely, verify recovery, "
            "then close it; otherwise escalate with evidence.",
        )
        try:
            self.state.consume_tool(self.budget)
            incident = self.tools.execute("get_incident", {})
            self.state.observe_result("get_incident", incident)
            self.trace.record("bootstrap_incident", incident)
            self.state.append_system_note(f"Current incident evidence: {json.dumps(incident)}")
            self._record_state("state_after_bootstrap")

            self.state.consume_llm(self.budget)
            self.state.set_plan(self.planner.create(incident))
            self.trace.record("plan_created", {"plan": str(self.state.plan)})
            self._record_state("state_after_plan")

            while self.state.has_budget(self.budget):
                reply = self._model_decide()
                self._append_assistant(reply)
                self.trace.record("model_reply", {"content": reply.content, "tool_calls": [c.__dict__ if hasattr(c, "__dict__") else {"name": c.name, "arguments": c.arguments} for c in reply.tool_calls]})

                if not reply.tool_calls:
                    return self._outcome(
                        TerminalStatus.FAILED,
                        "Model stopped without a tool call; baseline controller cannot prove resolution.",
                    )

                call = reply.tool_calls[0]
                result = self._execute_tool_call(call)
                self.state.observe_result(call.name, result)
                self._append_tool_result(call, result)
                self._record_state("state_after_tool")

                if call.name == "close_incident" and result.get("status") == "ok":
                    return self._outcome(TerminalStatus.RESOLVED, "Incident closed with simulator evidence.")
                if call.name == "escalate_incident" and result.get("status") == "ok":
                    return self._outcome(TerminalStatus.ESCALATED, "Incident escalated with evidence.")

                # TODO(A1): distinguish retry, re-plan, abort, verify, and continue.

            return self._outcome(
                TerminalStatus.BUDGET_EXHAUSTED,
                "Agent budget exhausted before safe termination.",
            )
        except BudgetExceeded as exc:
            return self._outcome(TerminalStatus.BUDGET_EXHAUSTED, str(exc))
