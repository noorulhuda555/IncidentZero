from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetManager:
    max_llm_calls: int = 14
    max_tool_calls: int = 28
    llm_calls: int = 0
    tool_calls: int = 0
    warning_llm_calls_remaining: int = 3
    max_runtime_seconds: float = 120.0
    _started_at: float | None = None

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/limits.json") -> BudgetManager:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(
            max_llm_calls=int(data["max_llm_calls"]),
            max_tool_calls=int(data["max_tool_calls"]),
            warning_llm_calls_remaining=int(data["warning_llm_calls_remaining"]),
            max_runtime_seconds=float(data["max_runtime_seconds"]),
        )

    def start_clock(self) -> None:
        self._started_at = time.monotonic()

    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def runtime_exceeded(self) -> bool:
        return self.elapsed_seconds() >= self.max_runtime_seconds

    def consume_llm(self) -> None:
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded("LLM-call budget exhausted")
        self.llm_calls += 1

    def consume_tool(self) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("Tool-call budget exhausted")
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
        )

    def critical_exhaustion(self) -> bool:
        """Prefer verify/escalate only — stop exploratory remediations."""
        return self.remaining_llm <= 2 or self.remaining_tools <= 3
