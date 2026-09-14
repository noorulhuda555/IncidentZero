import pytest

from incidentzero.agent.controller import AgentController
from incidentzero.approval.gateway import AlwaysApproveGateway
from incidentzero.environment.engine import SimulationEnvironment
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry


@pytest.fixture
def env():
    return SimulationEnvironment("TEST-STUDENT", "public-a")


@pytest.fixture
def registry(env):
    return ToolRegistry(env)


def build_controller(env, model, tmp_path, *, approval=None, budget=None) -> tuple[AgentController, TraceRecorder]:
    trace = TraceRecorder(tmp_path / "trace.jsonl")
    controller = AgentController(
        model=model,
        tools=ToolRegistry(env),
        approval=approval or AlwaysApproveGateway(),
        budget=budget or BudgetManager(max_llm_calls=8, max_tool_calls=12, warning_llm_calls_remaining=2),
        trace=trace,
    )
    return controller, trace


def trace_events(trace: TraceRecorder) -> list[str]:
    if not trace.path.exists():
        return []
    events = []
    for line in trace.path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            import json

            events.append(json.loads(line)["event"])
    return events
