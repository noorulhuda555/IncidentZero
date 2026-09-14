from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from incidentzero.approval.gateway import ApprovalGateway
from incidentzero.domain.models import AgentOutcome, ModelReply, ToolCall
from incidentzero.model.base import ModelClient
from incidentzero.model.errors import PermanentModelError, TransientModelError
from incidentzero.telemetry.budget import BudgetExceeded, BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry

from .planner import Planner
from .policies import LoopGuard, PostToolAction, ReplanPolicy, RiskPolicy, ToolOutcomeRouter
from .prompts import SYSTEM_PROMPT
from .recovery import RecoveryPolicy, RetryPolicy
from .state import AgentState, TerminalStatus
from .tool_gate import ToolExecutionGate

T = TypeVar("T")


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
        self.outcome_router = ToolOutcomeRouter(self.replan_policy)
        limits = json.loads(Path("configs/limits.json").read_text(encoding="utf-8"))
        max_retries = int(limits["max_consecutive_model_retries"])
        self.loop_guard = LoopGuard(max_same_action_repeats=int(limits["max_same_action_repeats"]))
        self.tool_gate = ToolExecutionGate(tools, self.risk, self.loop_guard)
        self.retry_policy = RetryPolicy(max_attempts=max_retries)
        self._max_tool_retries = max(0, max_retries - 1)
        self._pending_tool_retries: dict[str, int] = {}
        self._budget_warning_logged = False

    def _record_state(self, label: str) -> None:
        self.trace.record(label, self.state.to_snapshot())

    def _budget_snapshot(self) -> dict[str, Any]:
        return {
            "llm_calls": self.budget.llm_calls,
            "tool_calls": self.budget.tool_calls,
            "remaining_llm": self.budget.remaining_llm,
            "remaining_tools": self.budget.remaining_tools,
            "elapsed_seconds": round(self.budget.elapsed_seconds(), 2),
        }

    def _maybe_budget_warning(self) -> None:
        if self.budget.near_exhaustion() and not self._budget_warning_logged:
            self._budget_warning_logged = True
            self.trace.record("budget_warning", self._budget_snapshot())

    def _outcome(self, status: TerminalStatus, summary: str) -> AgentOutcome:
        self.state.set_terminal_status(status)
        self._record_state("terminal_state")
        self.trace.record("terminal_result", {"status": status.value, "summary": summary, **self._budget_snapshot()})
        return AgentOutcome(
            status=status.value,
            summary=summary,
            llm_calls=self.state.llm_call_count,
            tool_calls=self.state.tool_call_count,
            final_world_version=self.state.latest_world_version,
            evidence_ids=list(self.state.evidence_ids),
            trace_path=str(self.trace.path),
        )

    def _with_llm(self, fn: Callable[[], T], *, purpose: str) -> T:
        def on_retry(attempt: int, exc: TransientModelError) -> None:
            self.trace.record("model_retry", {"purpose": purpose, "attempt": attempt, "error": str(exc)})

        def attempt() -> T:
            self.state.consume_llm(self.budget)
            return fn()

        return self.retry_policy.call_model(attempt, on_retry=on_retry)

    def _model_decide(self) -> ModelReply:
        reply = self._with_llm(
            lambda: self.model.decide(self.state.messages, self.tools.groq_tools),
            purpose="decide",
        )
        self.trace.record(
            "model_request",
            {
                "purpose": "decide",
                "llm_calls": self.state.llm_call_count,
                "usage": reply.usage,
                "finish_reason": reply.finish_reason,
            },
        )
        return reply

    def _observe_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        before = self.state.latest_world_version
        self.state.observe_result(tool_name, result)
        after = self.state.latest_world_version
        if before is not None and after is not None and after > before:
            self.loop_guard.reset()

    def _budget_too_low_for_plan(self) -> bool:
        plan = self.state.plan
        if plan is None:
            return False
        pending = sum(1 for step in plan.steps if step.status != "done")
        if pending == 0:
            return False
        need_llm = max(2, pending + 1)
        need_tools = max(2, pending + 2)
        return self.budget.remaining_llm < need_llm or self.budget.remaining_tools < need_tools

    def _mark_plan_progress(self, tool_name: str, result: dict[str, Any]) -> None:
        plan = self.state.plan
        if plan is None or result.get("status") != "ok":
            return
        for step in plan.steps:
            if step.status == "done":
                continue
            if tool_name == "verify_recovery" and "verify" in step.step_id.lower():
                data = result.get("data") or {}
                if data.get("criteria_met"):
                    step.status = "done"
                return
            if tool_name in step.objective.lower() or tool_name.replace("_", " ") in step.objective.lower():
                step.status = "done"
                return

    def _path_re_observe(self) -> dict[str, Any] | None:
        """Refresh incident snapshot after the world changed."""
        if not self.state.has_budget(self.budget):
            return None
        fresh = self._execute_tool_call(ToolCall(id="re_observe", name="get_incident", arguments={}))
        if fresh.get("status") != "ok":
            return fresh
        self._observe_tool_result("get_incident", fresh)
        self.trace.record("re_observe", fresh)
        self.state.append_system_note(
            f"World changed — fresh incident observation (world_version={fresh.get('world_version')}): "
            f"{json.dumps(fresh, ensure_ascii=False)}"
        )
        return fresh

    def _path_replan(self, trigger_result: dict[str, Any], tool_name: str) -> None:
        if self.state.plan is None or not self.state.has_budget(self.budget):
            return
        trigger = {**trigger_result, "tool": tool_name}
        reason = self.replan_policy.replan_reason(trigger)
        self.trace.record("replan_trigger", {"reason": reason, "trigger": trigger})
        revised = self._with_llm(
            lambda: self.planner.revise(self.state.plan, trigger, self.state.run_summary()),
            purpose="plan_revise",
        )
        self.state.apply_revised_plan(revised)
        self.trace.record("plan_revised", {"revision": revised.revision, "plan": str(revised)})
        self.state.append_system_note(
            f"Plan revised (revision={revised.revision}, reason={reason}): {json.dumps(self.state._plan_snapshot(revised), ensure_ascii=False)}"
        )
        self._record_state("state_after_replan")

    def _path_retry_tool(self, call: ToolCall) -> dict[str, Any] | None:
        """Retry a transient telemetry/action error without a new LLM turn."""
        key = call.id
        attempts = self._pending_tool_retries.get(key, 0)
        if attempts >= self._max_tool_retries or not self.state.has_budget(self.budget):
            return None
        self._pending_tool_retries[key] = attempts + 1
        self.trace.record("retry_tool", {"tool": call.name, "attempt": attempts + 1})
        self.state.consume_tool(self.budget)
        result = self.tools.invoke(call.name, call.arguments)
        self.trace.record("tool_result", {"call": {"name": call.name, "arguments": call.arguments}, "result": result, "retry": True})
        return result

    def _handle_post_tool(self, call: ToolCall, result: dict[str, Any]) -> AgentOutcome | None:
        """Return a terminal outcome, or None to continue the main loop."""
        if RecoveryPolicy.verify_requires_replan(call.name, result):
            self._path_replan(result, call.name)
            return None

        msg = result.get("message") or ""
        if result.get("status") == "validation_error" and "Repeated identical action blocked" in msg:
            self._path_replan({"status": "loop_detected", "retryable": False, "message": msg}, call.name)
            return None

        self._mark_plan_progress(call.name, result)
        action = self.outcome_router.decide(
            call.name,
            result,
            budget_low=self._budget_too_low_for_plan(),
        )
        self.trace.record("post_tool_action", {"action": action.value, "tool": call.name, "status": result.get("status")})

        if action == PostToolAction.CONTINUE:
            return None
        if action == PostToolAction.RETRY:
            retry_result = self._path_retry_tool(call)
            if retry_result is None:
                return self._outcome(TerminalStatus.ABORTED, f"Transient error on {call.name} could not be retried.")
            self._observe_tool_result(call.name, retry_result)
            self._append_tool_result(call, retry_result)
            self._record_state("state_after_tool_retry")
            follow_up = self.outcome_router.decide(call.name, retry_result)
            if follow_up == PostToolAction.REPLAN:
                self._path_replan(retry_result, call.name)
            elif follow_up == PostToolAction.ABORT:
                return self._outcome(TerminalStatus.ABORTED, f"Tool {call.name} failed after retry.")
            return None
        if action == PostToolAction.RE_OBSERVE:
            self._path_re_observe()
            if self.replan_policy.should_replan({**result, "tool": call.name}):
                self._path_replan(result, call.name)
            return None
        if action == PostToolAction.REPLAN:
            self._path_replan(result, call.name)
            return None
        if action == PostToolAction.ABORT:
            return self._outcome(
                TerminalStatus.ABORTED,
                result.get("message") or f"Non-recoverable failure on {call.name}.",
            )
        return None

    def _close_allowed(self, call: ToolCall, result: dict[str, Any]) -> bool:
        if call.name != "close_incident" or result.get("status") != "ok":
            return False
        cited = call.arguments.get("evidence_ids") or []
        return (
            self.state.verification_criteria_met
            and self.state.last_verify_recovery_evidence_id is not None
            and self.state.last_verify_recovery_evidence_id in cited
        )

    def _execute_tool_call(self, call: ToolCall, *, skip_loop_check: bool = False) -> dict[str, Any]:
        """Pre-check, then invoke simulator (approval gate added in Phase 4)."""
        blocked = self.tool_gate.check(
            call.name,
            call.arguments,
            self.state,
            self.budget,
            skip_loop_check=skip_loop_check,
        )
        if blocked is not None:
            self.trace.record("validation_failure", {"call": {"name": call.name, "arguments": call.arguments}, "result": blocked})
            return blocked
        self.state.consume_tool(self.budget)
        if self.risk.requires_human_approval(call.name):
            justification = str(call.arguments.get("reason", "No evidence-based justification provided."))
            self.trace.record(
                "approval_request",
                {"tool": call.name, "arguments": call.arguments, "justification": justification},
            )
            approved = self.approval.approve(call.name, call.arguments, justification)
            self.trace.record("approval_result", {"tool": call.name, "approved": approved})
            if not approved:
                return {
                    "status": "approval_denied",
                    "tool": call.name,
                    "world_version": self.tools.environment.world_version,
                    "evidence_id": None,
                    "data": None,
                    "retryable": False,
                    "message": "Human approval denied for high/critical action.",
                }
        result = self.tools.invoke(call.name, call.arguments)
        self.state.record_action_fingerprint(call.name, call.arguments)
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
        self.loop_guard.reset()
        self._pending_tool_retries.clear()
        self._budget_warning_logged = False
        if self.budget._started_at is None:
            self.budget.start_clock()
        try:
            incident = self._execute_tool_call(ToolCall(id="bootstrap", name="get_incident", arguments={}))
            self._observe_tool_result("get_incident", incident)
            self.trace.record("bootstrap_incident", incident)
            self.state.append_system_note(f"Current incident evidence: {json.dumps(incident)}")
            self._record_state("state_after_bootstrap")

            self.state.set_plan(self._with_llm(lambda: self.planner.create(incident), purpose="plan_create"))
            self.trace.record("plan_created", {"plan": str(self.state.plan), "revision": self.state.plan.revision if self.state.plan else 0})
            self._record_state("state_after_plan")

            while self.state.has_budget(self.budget):
                if self.budget.runtime_exceeded():
                    return self._outcome(TerminalStatus.BUDGET_EXHAUSTED, "Wall-clock budget exceeded.")
                self._maybe_budget_warning()
                reply = self._model_decide()
                self._append_assistant(reply)
                self.trace.record("model_reply", {"content": reply.content, "tool_calls": [c.__dict__ if hasattr(c, "__dict__") else {"name": c.name, "arguments": c.arguments} for c in reply.tool_calls]})

                if not reply.tool_calls:
                    return self._outcome(
                        TerminalStatus.FAILED,
                        "Model stopped without a tool call; baseline controller cannot prove resolution.",
                    )

                call = reply.tool_calls[0]
                self.trace.record("tool_proposal", {"call": {"name": call.name, "arguments": call.arguments}})
                result = self._execute_tool_call(call)
                self._observe_tool_result(call.name, result)
                self._append_tool_result(call, result)
                self._record_state("state_after_tool")

                if self._close_allowed(call, result):
                    return self._outcome(TerminalStatus.RESOLVED, "Incident closed with verified simulator evidence.")
                if call.name == "escalate_incident" and result.get("status") == "ok":
                    return self._outcome(TerminalStatus.ESCALATED, "Incident escalated with evidence.")

                terminal = self._handle_post_tool(call, result)
                if terminal is not None:
                    return terminal

            return self._outcome(
                TerminalStatus.BUDGET_EXHAUSTED,
                "Agent budget exhausted before safe termination.",
            )
        except PermanentModelError as exc:
            return self._outcome(TerminalStatus.ABORTED, f"Permanent model error: {exc}")
        except TransientModelError as exc:
            return self._outcome(TerminalStatus.ABORTED, f"Model unavailable after retries: {exc}")
        except BudgetExceeded as exc:
            return self._outcome(TerminalStatus.BUDGET_EXHAUSTED, str(exc))
