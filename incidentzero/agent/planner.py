from __future__ import annotations

import json
from typing import Any

from incidentzero.domain.models import AgentPlan, PlanStep
from incidentzero.model.base import ModelClient
from incidentzero.model.errors import ModelError

FINAL_VERIFICATION_STEP_ID = "final_verification"

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "rationale_summary": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "success_signal": {"type": "string"},
                },
                "required": ["step_id", "objective", "success_signal"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hypothesis", "rationale_summary", "steps"],
    "additionalProperties": False,
}


def _verification_step() -> PlanStep:
    return PlanStep(
        step_id=FINAL_VERIFICATION_STEP_ID,
        objective="Run verify_recovery and confirm criteria_met before attempting close.",
        success_signal="verify_recovery returns criteria_met=true with a fresh evidence id.",
        status="pending",
    )


def _ensure_plan_shape(steps: list[PlanStep]) -> list[PlanStep]:
    """At least two subgoals plus an explicit final verification step."""
    if len(steps) < 2:
        steps = list(steps) + [
            PlanStep(
                step_id="gather_evidence",
                objective="Collect metrics, logs, and dependency evidence for the suspected path.",
                success_signal="At least two observation tools return ok evidence ids.",
                status="pending",
            ),
            PlanStep(
                step_id="mitigate_safely",
                objective="Apply one evidence-backed remediation if warranted.",
                success_signal="Remediation tool returns ok or a documented reason to escalate.",
                status="pending",
            ),
        ]
    has_verify = any(
        step.step_id == FINAL_VERIFICATION_STEP_ID or "verify_recovery" in step.objective.lower()
        for step in steps
    )
    if not has_verify:
        steps = list(steps) + [_verification_step()]
    return steps


def _default_plan(incident_observation: dict[str, Any], *, revision: int = 0, hypothesis: str | None = None) -> AgentPlan:
    """Generic plan used when the model cannot emit a valid structured plan."""
    data = incident_observation.get("data") or {}
    suspected = data.get("suspected_service") or "checkout-service"
    return AgentPlan(
        hypothesis=hypothesis
        or f"Symptoms near {suspected} need evidence-backed diagnosis; ticket cause is unverified.",
        rationale_summary="Fallback investigation plan: observe broadly, remediate only with evidence, verify, else escalate.",
        revision=revision,
        steps=_ensure_plan_shape(
            [
                PlanStep(
                    step_id="gather_evidence",
                    objective=f"Collect health, metrics, logs, deployments, and dependencies for {suspected} and critical path services.",
                    success_signal="Multiple observation tools return ok evidence ids.",
                    status="pending",
                ),
                PlanStep(
                    step_id="mitigate_safely",
                    objective="Apply one least-risk remediation supported by collected evidence, or escalate if unsafe.",
                    success_signal="Remediation ok with improved signals, or escalate_incident accepted.",
                    status="pending",
                ),
            ]
        ),
    )


def _plan_to_payload(plan: AgentPlan) -> dict[str, Any]:
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


def _steps_from_raw(raw: dict[str, Any]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for row in raw.get("steps") or []:
        if not isinstance(row, dict):
            continue
        if not all(k in row for k in ("step_id", "objective", "success_signal")):
            continue
        steps.append(PlanStep(step_id=row["step_id"], objective=row["objective"], success_signal=row["success_signal"]))
    return _ensure_plan_shape(steps)


class Planner:
    def __init__(self, model: ModelClient) -> None:
        self.model = model

    def create(self, incident_observation: dict[str, Any], context: list[dict[str, Any]] | None = None) -> AgentPlan:
        """Create an explicit initial plan."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Create a short SRE investigation-and-remediation plan with at least two subgoals "
                    "and a final verification step using verify_recovery. "
                    "Each steps[] item must be an object with string fields step_id, objective, success_signal only. "
                    "Do not nest objects under empty keys. "
                    "Do not assume the ticket's suspected root cause is correct."
                ),
            },
            {"role": "user", "content": f"Incident observation: {incident_observation}"},
        ]
        try:
            raw = self.model.structured(messages, "incident_plan", PLAN_SCHEMA)
            steps = _steps_from_raw(raw)
            return AgentPlan(
                hypothesis=str(raw.get("hypothesis") or "Unverified incident hypothesis"),
                steps=steps,
                rationale_summary=str(raw.get("rationale_summary") or "Investigate with evidence."),
                revision=0,
            )
        except (ModelError, KeyError, TypeError, ValueError):
            return _default_plan(incident_observation, revision=0)

    def revise(self, current: AgentPlan, trigger: dict[str, Any], state_summary: str) -> AgentPlan:
        completed = {step.step_id: step for step in current.steps if step.status == "done"}
        messages = [
            {
                "role": "system",
                "content": (
                    "Revise the incident plan. Keep completed steps when still valid. "
                    "Update the hypothesis when evidence contradicts it. "
                    "Do not repeat a failed approach blindly. Include verify_recovery before close. "
                    "Each steps[] item must include step_id, objective, success_signal as strings."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "replan_trigger": trigger,
                        "current_plan": _plan_to_payload(current),
                        "completed_step_ids": sorted(completed.keys()),
                        "state_summary": state_summary,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            raw = self.model.structured(messages, "incident_plan_revision", PLAN_SCHEMA)
            steps: list[PlanStep] = []
            for row in raw.get("steps") or []:
                if not isinstance(row, dict) or not all(k in row for k in ("step_id", "objective", "success_signal")):
                    continue
                prior = completed.get(row["step_id"])
                status = prior.status if prior is not None else "pending"
                steps.append(
                    PlanStep(
                        step_id=row["step_id"],
                        objective=row["objective"],
                        success_signal=row["success_signal"],
                        status=status,
                    )
                )
            steps = _ensure_plan_shape(steps)
            return AgentPlan(
                hypothesis=str(raw.get("hypothesis") or current.hypothesis),
                steps=steps,
                rationale_summary=str(raw.get("rationale_summary") or current.rationale_summary),
                revision=current.revision + 1,
            )
        except (ModelError, KeyError, TypeError, ValueError):
            fallback = _default_plan({"data": {}}, revision=current.revision + 1, hypothesis=current.hypothesis)
            for step in fallback.steps:
                if step.step_id in completed:
                    step.status = "done"
            return fallback
