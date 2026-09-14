from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from rich import print

from incidentzero.agent.controller import AgentController
from incidentzero.approval.gateway import AlwaysApproveGateway, ConsoleApprovalGateway
from incidentzero.environment.engine import SimulationEnvironment
from incidentzero.model.groq_client import GroqModelClient
from incidentzero.telemetry.budget import BudgetManager
from incidentzero.telemetry.trace import TraceRecorder
from incidentzero.tools.registry import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="incidentzero")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--student-id", required=True)
    run.add_argument("--scenario", default="public-a")
    run.add_argument("--model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    run.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve high/critical actions without interactive console (evaluation only).",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    if args.command == "run":
        env = SimulationEnvironment(args.student_id, args.scenario)
        registry = ToolRegistry(env)
        trace_path = Path("traces") / f"{args.student_id}_{args.scenario}.jsonl"
        approval = AlwaysApproveGateway() if args.auto_approve else ConsoleApprovalGateway()
        budget = BudgetManager.from_config()
        budget.start_clock()
        controller = AgentController(
            model=GroqModelClient(model=args.model),
            tools=registry,
            approval=approval,
            budget=budget,
            trace=TraceRecorder(trace_path),
        )
        outcome = controller.run()
        print("\n[bold]Outcome[/bold]")
        print(json.dumps(asdict(outcome), indent=2, default=str))
        print(f"\n[dim]Trace:[/dim] {trace_path}")
        print(f"[dim]Elapsed:[/dim] {budget.elapsed_seconds():.1f}s")
        print(f"[dim]Plan revision:[/dim] {controller.state.plan_revision}")
        print(f"[dim]Verify evidence:[/dim] {controller.state.last_verify_recovery_evidence_id}")


if __name__ == "__main__":
    main()
