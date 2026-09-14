from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from incidentzero.model.errors import PermanentModelError, TransientModelError

T = TypeVar("T")


class RetryPolicy:
    def __init__(self, max_attempts: int = 3, sleeper: Callable[[float], None] = time.sleep) -> None:
        self.max_attempts = max(1, max_attempts)
        self.sleeper = sleeper

    def call_model(
        self,
        fn: Callable[[], T],
        *,
        on_retry: Callable[[int, TransientModelError], None] | None = None,
    ) -> T:
        delay = 0.25
        last_exc: TransientModelError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except TransientModelError as exc:
                last_exc = exc
                if attempt >= self.max_attempts:
                    raise
                if on_retry is not None:
                    on_retry(attempt, exc)
                self.sleeper(delay)
                delay = min(delay * 2, 2.0)
        raise last_exc  # pragma: no cover


class RecoveryPolicy:
    """Maps failure outcomes to controller actions (Python-owned, not LLM-owned)."""

    @staticmethod
    def is_permanent_model_error(exc: BaseException) -> bool:
        return isinstance(exc, PermanentModelError)

    @staticmethod
    def verify_requires_replan(tool_name: str, result: dict[str, Any]) -> bool:
        if tool_name != "verify_recovery" or result.get("status") != "ok":
            return False
        data = result.get("data") or {}
        return data.get("criteria_met") is False
