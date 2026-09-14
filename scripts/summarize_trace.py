from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize(path: Path) -> dict:
    rows = load_events(path)
    events = [r["event"] for r in rows]
    counts = Counter(events)
    terminal = next((r["payload"] for r in rows if r["event"] == "terminal_result"), {})
    replans = [r["payload"] for r in rows if r["event"] == "replan_trigger"]
    approvals = [r["payload"] for r in rows if r["event"] == "approval_result"]
    proposals = [r["payload"]["call"]["name"] for r in rows if r["event"] == "tool_proposal"]
    tool_results = [
        (r["payload"]["call"]["name"], r["payload"]["result"].get("status"))
        for r in rows
        if r["event"] == "tool_result" and "call" in r["payload"]
    ]
    fingerprints: Counter[str] = Counter()
    for r in rows:
        if r["event"] == "tool_proposal":
            call = r["payload"]["call"]
            key = json.dumps({"tool": call["name"], "arguments": call["arguments"]}, sort_keys=True)
            fingerprints[key] += 1
    redundant = {k: v for k, v in fingerprints.items() if v >= 2}
    verify_ids = []
    for r in rows:
        if r["event"] == "tool_result":
            call = r["payload"].get("call", {})
            result = r["payload"].get("result", {})
            if call.get("name") == "verify_recovery" and result.get("status") == "ok":
                data = result.get("data") or {}
                if data.get("criteria_met"):
                    verify_ids.append(result.get("evidence_id"))
    return {
        "trace": str(path),
        "event_counts": dict(counts),
        "terminal": terminal,
        "replan_count": len(replans),
        "replan_reasons": [p.get("reason") for p in replans],
        "approvals": approvals,
        "proposed_tools": proposals,
        "tool_results": tool_results,
        "redundant_proposal_count": sum(v - 1 for v in fingerprints.values() if v > 1),
        "redundant_actions": [
            {"fingerprint": k, "count": v} for k, v in sorted(redundant.items(), key=lambda x: -x[1])[:8]
        ],
        "recovery_evidence_ids": verify_ids,
        "high_risk_approvals": [
            a for a in approvals
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize an IncidentZero JSONL trace")
    p.add_argument("trace")
    args = p.parse_args()
    print(json.dumps(summarize(Path(args.trace)), indent=2))


if __name__ == "__main__":
    main()
