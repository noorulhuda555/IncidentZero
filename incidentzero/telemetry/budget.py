from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetManager:
    """Hard limits from configs/limits.json"""

    max_llm_calls: int = 14
    max_tool_calls: int = 28
    max_consecutive_model_retries: int = 3
    max_same_action_repeats: int = 2
    max_runtime_seconds: float = 120.0
    warning_llm_calls_remaining: int = 3
    llm_calls: int = 0
    tool_calls: int = 0
    _started_at: float | None = None
    # Refuse starting another LLM/tool when less than this many seconds remain.
    _wall_reserve_seconds: float = 3.0

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/limits.json") -> BudgetManager:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(
            max_llm_calls=int(data["max_llm_calls"]),
            max_tool_calls=int(data["max_tool_calls"]),
            max_consecutive_model_retries=int(data["max_consecutive_model_retries"]),
            max_same_action_repeats=int(data["max_same_action_repeats"]),
            max_runtime_seconds=float(data["max_runtime_seconds"]),
            warning_llm_calls_remaining=int(data["warning_llm_calls_remaining"]),
        )

    def start_clock(self) -> None:
        self._started_at = time.monotonic()

    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def wall_time_remaining(self) -> float:
        return max(0.0, self.max_runtime_seconds - self.elapsed_seconds())

    def runtime_exceeded(self) -> bool:
        return self.elapsed_seconds() >= self.max_runtime_seconds

    def can_start_llm(self) -> bool:
        return self.remaining_llm > 0 and self.wall_time_remaining() > self._wall_reserve_seconds

    def can_start_tool(self) -> bool:
        return self.remaining_tools > 0 and not self.runtime_exceeded()

    def consume_llm(self) -> None:
        if self.runtime_exceeded():
            raise BudgetExceeded(
                f"Wall-clock budget exhausted ({self.max_runtime_seconds:.0f}s from configs/limits.json)"
            )
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(f"LLM-call budget exhausted (max {self.max_llm_calls} from configs/limits.json)")
        self.llm_calls += 1

    def consume_tool(self) -> None:
        if self.runtime_exceeded():
            raise BudgetExceeded(
                f"Wall-clock budget exhausted ({self.max_runtime_seconds:.0f}s from configs/limits.json)"
            )
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded(f"Tool-call budget exhausted (max {self.max_tool_calls} from configs/limits.json)")
        self.tool_calls += 1

    @property
    def remaining_llm(self) -> int:
        return self.max_llm_calls - self.llm_calls

    @property
    def remaining_tools(self) -> int:
        return self.max_tool_calls - self.tool_calls

    def near_exhaustion(self) -> bool:
        return (
            self.remaining_llm <= self.warning_llm_calls_remaining
            or self.remaining_tools <= self.warning_llm_calls_remaining
            or self.wall_time_remaining() <= max(15.0, self.max_runtime_seconds * 0.15)
        )

    def critical_exhaustion(self) -> bool:
        """Prefer verify/escalate only — stop exploratory remediations."""
        return (
            self.remaining_llm <= 2
            or self.remaining_tools <= 3
            or self.wall_time_remaining() <= max(10.0, self.max_runtime_seconds * 0.1)
        )
