"""Tool loop for agents: schema validation, budget enforcement, and tracing.

The runtime, not the prompt, enforces the rules. A tool call that fails its
pydantic schema, names an unknown tool, exceeds a per-tool cap, or is refused
by its handler is written to the trace and returned to the model as an
error; it never runs. The loop stops when the dollar budget is spent.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, ValidationError

from linegate.agents.traces import TraceWriter
from linegate.dataio import ROOT


class ToolRefused(RuntimeError):
    """Raised by a tool handler to refuse a call without running it."""


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], object]

    def spec(self) -> dict:
        schema = self.input_model.model_json_schema()
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": schema}}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> Completion: ...


@dataclass
class Budget:
    max_usd: float
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0
    max_calls: dict[str, int] = field(default_factory=dict)
    spent_usd: float = 0.0
    calls: Counter = field(default_factory=Counter)

    def charge(self, completion: Completion) -> None:
        self.spent_usd += (completion.input_tokens * self.usd_per_mtok_in
                           + completion.output_tokens * self.usd_per_mtok_out) / 1e6

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.max_usd

    def take(self, tool: str) -> bool:
        if tool in self.max_calls and self.calls[tool] >= self.max_calls[tool]:
            return False
        self.calls[tool] += 1
        return True


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class OpenAIClient:
    def __init__(self, model: str):
        from openai import OpenAI

        load_env()
        self.model, self.client = model, OpenAI()

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools)
        message = response.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in (message.tool_calls or [])]
        usage = response.usage
        return Completion(message.content or "", calls, usage.prompt_tokens, usage.completion_tokens)


def assistant_message(completion: Completion) -> dict:
    message: dict = {"role": "assistant", "content": completion.text or None}
    if completion.tool_calls:
        message["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in completion.tool_calls]
    return message


def dispatch(call: ToolCall, tools: dict[str, Tool], budget: Budget, trace: TraceWriter) -> dict:
    tool = tools.get(call.name)
    if tool is None:
        trace.write("tool_rejected", tool=call.name, arguments=call.arguments, reason="unknown tool")
        return {"error": f"unknown tool {call.name}"}
    try:
        args = tool.input_model.model_validate_json(call.arguments or "{}")
    except ValidationError as exc:
        trace.write("tool_rejected", tool=call.name, arguments=call.arguments, reason=exc.errors(include_url=False))
        return {"error": "schema validation failed", "detail": exc.errors(include_url=False)}
    if not budget.take(call.name):
        trace.write("tool_refused", tool=call.name, arguments=args.model_dump(), reason="per-tool call cap reached")
        return {"error": f"call cap reached for {call.name}"}
    try:
        result = tool.handler(args)
    except ToolRefused as exc:
        trace.write("tool_refused", tool=call.name, arguments=args.model_dump(), reason=str(exc))
        return {"error": f"refused: {exc}"}
    except Exception as exc:  # a failing tool must not end the run; the model sees the error
        trace.write("tool_error", tool=call.name, arguments=args.model_dump(), error=repr(exc)[:2000])
        return {"error": f"tool failed: {str(exc)[:1000]}"}
    payload = result.model_dump() if isinstance(result, BaseModel) else result
    trace.write("tool_result", tool=call.name, arguments=args.model_dump(), result=payload)
    return {"result": payload}


def run_agent(client: LLMClient, system: str, task: str, tools: list[Tool], budget: Budget,
              trace: TraceWriter, max_turns: int, should_stop: Callable[[], str | None] = lambda: None,
              nudge: Callable[[], str | None] = lambda: None, max_nudges: int = 0) -> str:
    registry = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    trace.write("agent_start", task=task, tools=list(registry), max_usd=budget.max_usd)
    for turn in range(max_turns):
        completion = client.complete(messages, specs)
        budget.charge(completion)
        trace.write("model_turn", turn=turn, text=completion.text, spent_usd=round(budget.spent_usd, 4),
                    tool_calls=[{"name": c.name, "arguments": c.arguments} for c in completion.tool_calls])
        messages.append(assistant_message(completion))
        if not completion.tool_calls:
            message = nudge() if max_nudges > 0 else None
            if message is None:
                trace.write("agent_end", reason="no tool calls", text=completion.text)
                return completion.text
            max_nudges -= 1
            trace.write("nudge", message=message)
            messages.append({"role": "user", "content": message})
            continue
        for call in completion.tool_calls:
            content = json.dumps(dispatch(call, registry, budget, trace), default=str)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        if budget.exhausted:
            trace.write("budget_stop", spent_usd=budget.spent_usd, max_usd=budget.max_usd)
            return ""
        if (reason := should_stop()) is not None:
            trace.write("agent_end", reason=reason)
            return ""
    trace.write("agent_end", reason="max turns")
    return ""
