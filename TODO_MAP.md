# TODO Map (scratch inventory)

Generated for Phase 0.5. Close each row when the assigned phase is fully implemented (not stubbed).

| Status | File:Line | Function / class | Short description | Phase |
|--------|-----------|------------------|-------------------|-------|
| [x] | `incidentzero/agent/controller.py` | `AgentController._with_llm` / `_model_decide` | Bounded model retries + LLM budget per attempt | 5, 7 |
| [ ] | `incidentzero/agent/controller.py:72` | `AgentController._execute_tool_call` | Call approval gateway before high/critical tools; LLM must not self-approve | 4 |
| [x] | `incidentzero/agent/controller.py` | `AgentController._handle_post_tool` | Separate paths: continue, retry tool, re-observe, replan, abort (`ToolOutcomeRouter`) | 2 (more in 5, 10) |
| [x] | `incidentzero/agent/planner.py:54` | `Planner.revise` | Grounded plan revision: increment revision, preserve evidence/history, avoid blind repeat | 2 |
| [x] | `incidentzero/agent/policies.py:23` | `ReplanPolicy.should_replan` | Re-plan on stale world, approval denied, contradictions, failed verification, non-retryable failures, low budget | 2 |
| [x] | `incidentzero/agent/policies.py:35` | `LoopGuard.record` | Stable action fingerprint; block or redirect on repeat threshold | 3, 5 |
| [x] | `incidentzero/agent/recovery.py:18` | `RetryPolicy.call_model` | Bounded backoff retry for `TransientModelError` only | 5, 7 |

**Total open (code markers):** 1 — run `python scripts/count_todos.py` (approval gate, Phase 4)

**Phase 1 (done):** `agent/state.py` — explicit state, terminal enum, counters, fingerprints, snapshots (no `TODO` marker in starter).

**Note:** `LoopGuard.record` (Phase 5) should reuse `action_fingerprint()` from `state.py` when implemented.
