from __future__ import annotations

import json
import os
import re
from typing import Any

from groq import Groq

from incidentzero.domain.models import ModelReply, ToolCall
from .errors import PermanentModelError, TransientModelError


class GroqModelClient:
    """Local function-calling only — no browser search, code execution, or remote MCP."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        if not key or key.strip() in {"", "replace_me"}:
            raise PermanentModelError("GROQ_API_KEY must be set in the environment (never hard-code it).")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.client = Groq(api_key=key)

    def _translate_error(self, exc: Exception) -> Exception:
        status = getattr(exc, "status_code", None)
        text = str(exc)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return TransientModelError(text)
        # Provider sometimes returns 400 when tool-call JSON is truncated; retry is useful.
        if status == 400 and "tool_use_failed" in text:
            return TransientModelError(text)
        if status == 400 and "json_validate_failed" in text:
            return TransientModelError(text)
        return PermanentModelError(text)

    def _sanitize_tool_name(self, name: str) -> str:
        # Some models append channel markers into the tool name string.
        if "<|" in name:
            name = name.split("<|", 1)[0]
        return name.strip()

    def _recover_tool_use_failed(self, exc: Exception) -> ModelReply | None:
        """If Groq failed parsing tool args, recover a best-effort ToolCall from the failed generation."""
        text = str(exc)
        if "tool_use_failed" not in text:
            return None
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
        if not name_match:
            return None
        name = self._sanitize_tool_name(name_match.group(1))
        if not name:
            return None
        args: dict[str, Any] = {}
        args_match = re.search(r'"arguments"\s*:\s*(\{.*?\})', text, flags=re.DOTALL)
        if args_match:
            try:
                parsed = json.loads(args_match.group(1))
                if isinstance(parsed, dict):
                    args = {k: v for k, v in parsed.items() if k != ""}
            except json.JSONDecodeError:
                args = {}
        return ModelReply(
            content=f"Recovered malformed tool proposal for {name}; controller will validate.",
            tool_calls=[ToolCall(id="recovered-0", name=name, arguments=args)],
            usage={},
            finish_reason="tool_calls",
        )

    def decide(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelReply:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0.1,
                reasoning_effort="low",
            )
        except Exception as exc:
            recovered = self._recover_tool_use_failed(exc)
            if recovered is not None:
                return recovered
            raise self._translate_error(exc) from exc
        msg = response.choices[0].message
        calls: list[ToolCall] = []
        for call in (msg.tool_calls or []):
            try:
                args = json.loads(call.function.arguments or "{}")
            except Exception:
                args = {"__malformed_arguments__": call.function.arguments}
            if not isinstance(args, dict):
                args = {"__malformed_arguments__": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=self._sanitize_tool_name(call.function.name), arguments=args))
        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
            }
        return ModelReply(
            content=msg.content,
            tool_calls=calls,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
        )

    def structured(self, messages: list[dict[str, Any]], schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                temperature=0.1,
                reasoning_effort="low",
            )
        except Exception as exc:
            raise self._translate_error(exc) from exc
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise TransientModelError(f"Model returned invalid structured JSON: {content[:160]}") from exc
