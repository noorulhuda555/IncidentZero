import pytest

from incidentzero.agent.policies import LoopGuard, ReplanPolicy, RiskPolicy
from incidentzero.agent.recovery import RetryPolicy
from incidentzero.agent.state import AgentState
from incidentzero.agent.tool_gate import ToolExecutionGate
from incidentzero.approval.gateway import AlwaysDenyGateway
from incidentzero.domain.models import ModelReply, ToolCall
from incidentzero.model.errors import PermanentModelError, TransientModelError
from incidentzero.model.scripted import ScriptedModelClient
from incidentzero.telemetry.budget import BudgetExceeded, BudgetManager

from tests.student.conftest import build_controller, trace_events


def _sample_plan() -> dict:
    return {
        "hypothesis": "Checkout path degraded",
        "rationale_summary": "Investigate before acting.",
        "steps": [
            {"step_id": "gather", "objective": "get_metrics on checkout-service", "success_signal": "evidence collected"},
            {"step_id": "final_verification", "objective": "verify_recovery", "success_signal": "criteria_met true"},
        ],
    }


@pytest.mark.student
def test_gate_rejects_unknown_tool(registry):
    gate = ToolExecutionGate(registry, RiskPolicy(), LoopGuard())
    state = AgentState()
    state.latest_world_version = 1
    blocked = gate.check("not_a_real_tool", {}, state, BudgetManager())
    assert blocked is not None
    assert blocked["status"] == "validation_error"


@pytest.mark.student
def test_gate_rejects_stale_world_version(registry):
    gate = ToolExecutionGate(registry, RiskPolicy(), LoopGuard())
    state = AgentState()
    state.latest_world_version = 2
    blocked = gate.check(
        "restart_service",
        {"service": "checkout-service", "expected_world_version": 1, "reason": "testing stale gate path"},
        state,
        BudgetManager(),
    )
    assert blocked is not None
    assert blocked["status"] == "stale_precondition"


@pytest.mark.student
def test_gate_rejects_premature_close(registry):
    gate = ToolExecutionGate(registry, RiskPolicy(), LoopGuard())
    state = AgentState()
    state.latest_world_version = 1
    state.verification_criteria_met = False
    blocked = gate.check(
        "close_incident",
        {
            "summary": "Attempting premature close for test coverage",
            "evidence_ids": ["EV-0001"],
            "expected_world_version": 1,
            "reason": "testing close without verify evidence",
        },
        state,
        BudgetManager(),
    )
    assert blocked is not None
    assert "verify_recovery" in blocked["message"]


@pytest.mark.student
def test_gate_blocks_high_risk_without_evidence(registry):
    gate = ToolExecutionGate(registry, RiskPolicy(), LoopGuard())
    state = AgentState()
    state.latest_world_version = 1
    blocked = gate.check(
        "rollback_deployment",
        {
            "service": "checkout-service",
            "target_version": "v-prev",
            "expected_world_version": 1,
            "reason": "rollback without prior evidence ids",
        },
        state,
        BudgetManager(),
    )
    assert blocked is not None
    assert "evidence" in blocked["message"].lower()


@pytest.mark.student
def test_budget_manager_loads_limits_from_config():
    budget = BudgetManager.from_config()
    assert budget.max_llm_calls == 14
    assert budget.max_tool_calls == 28
    assert budget.warning_llm_calls_remaining == 3


@pytest.mark.student
def test_budget_manager_enforces_llm_cap():
    budget = BudgetManager(max_llm_calls=1, max_tool_calls=5)
    budget.consume_llm()
    with pytest.raises(BudgetExceeded):
        budget.consume_llm()


@pytest.mark.student
def test_retry_policy_retries_transient_only():
    calls = {"n": 0}
    retry = RetryPolicy(max_attempts=3, sleeper=lambda _s: None)

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientModelError("429")
        return "ok"

    assert retry.call_model(flaky) == "ok"
    assert calls["n"] == 3

    def permanent():
        raise PermanentModelError("bad")

    with pytest.raises(PermanentModelError):
        retry.call_model(permanent)


@pytest.mark.student
def test_replan_after_verify_failure():
    assert ReplanPolicy().should_replan(
        {"status": "ok", "tool": "verify_recovery", "data": {"criteria_met": False}}
    )


@pytest.mark.student
def test_scripted_stale_tool_triggers_reobserve(env, tmp_path):
    plan = _sample_plan()
    model = ScriptedModelClient(
        structured_outputs=[plan, plan],
        decisions=[
            ModelReply(
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="restart_service",
                        arguments={
                            "service": "checkout-service",
                            "expected_world_version": 1,
                            "reason": "stale version on purpose for test",
                        },
                    )
                ]
            ),
            ModelReply(content="done"),
        ],
    )
    controller, trace = build_controller(env, model, tmp_path)
    reg = controller.tools
    boot = reg.execute("get_incident", {})
    reg.execute(
        "restart_service",
        {
            "service": "auth-service",
            "expected_world_version": boot["world_version"],
            "reason": "bump world version for stale test",
        },
    )
    outcome = controller.run()
    events = trace_events(trace)
    assert "validation_failure" in events
    assert "re_observe" in events
    assert outcome.status in {"failed", "budget_exhausted", "aborted", "escalated", "resolved"}


@pytest.mark.student
def test_scripted_premature_close_blocked(env, tmp_path):
    plan = _sample_plan()
    model = ScriptedModelClient(
        structured_outputs=[plan],
        decisions=[
            ModelReply(
                tool_calls=[
                    ToolCall(
                        id="close1",
                        name="close_incident",
                        arguments={
                            "summary": "Trying to close without verify step completed",
                            "evidence_ids": ["EV-0001"],
                            "expected_world_version": 1,
                            "reason": "premature close attempt for test",
                        },
                    )
                ]
            ),
        ],
    )
    controller, trace = build_controller(env, model, tmp_path)
    outcome = controller.run()
    events = trace_events(trace)
    assert "validation_failure" in events
    assert outcome.status != "resolved"


@pytest.mark.student
def test_scripted_approval_denial_replans(env, tmp_path):
    plan = _sample_plan()
    state = AgentState()
    state.latest_world_version = 1
    state.evidence_ids.append("EV-0001")
    model = ScriptedModelClient(
        structured_outputs=[plan, plan],
        decisions=[
            ModelReply(
                tool_calls=[
                    ToolCall(
                        id="rb1",
                        name="shift_traffic",
                        arguments={
                            "service": "checkout-service",
                            "percent_to_secondary": 25,
                            "expected_world_version": 1,
                            "reason": "shift traffic with evidence for approval test",
                        },
                    )
                ]
            ),
            ModelReply(content="continue"),
        ],
    )
    controller, trace = build_controller(env, model, tmp_path, approval=AlwaysDenyGateway())
    outcome = controller.run()
    events = trace_events(trace)
    assert "approval_request" in events
    assert "approval_result" in events
    assert "replan_trigger" in events or "plan_revised" in events
    assert outcome.status != "resolved"


@pytest.mark.student
def test_scripted_verify_failure_triggers_replan(env, tmp_path):
    plan = _sample_plan()
    model = ScriptedModelClient(
        structured_outputs=[plan, plan],
        decisions=[
            ModelReply(tool_calls=[ToolCall(id="v1", name="verify_recovery", arguments={})]),
            ModelReply(content="continue after failed verify"),
        ],
    )
    controller, trace = build_controller(env, model, tmp_path)
    outcome = controller.run()
    events = trace_events(trace)
    assert "plan_revised" in events or "replan_trigger" in events
    assert outcome.status != "resolved"


@pytest.mark.student
def test_trace_contains_bootstrap_and_terminal(env, tmp_path):
    plan = _sample_plan()
    model = ScriptedModelClient(structured_outputs=[plan], decisions=[ModelReply(content="no tools")])
    controller, trace = build_controller(env, model, tmp_path, budget=BudgetManager(max_llm_calls=2, max_tool_calls=4))
    outcome = controller.run()
    events = trace_events(trace)
    assert "bootstrap_incident" in events
    assert "plan_created" in events
    assert "terminal_result" in events
    assert outcome.status == "failed"
