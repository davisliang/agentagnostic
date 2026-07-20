"""The one place anything reaches a model.

Every model call in the project — the task analyzer, the dataset generator, the
judge, and every call a candidate workflow makes — goes through `LLM.call`.
That is what makes cost measurable at a single chokepoint no matter what a
workflow's code does.
"""
import os
from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel

from .models import Catalog

# Server-side tools: they run on Anthropic's side, so there is nothing to
# execute locally — the model uses them and the results come back in the reply.
TOOL_DEFS = {
    "code_execution": {"type": "code_execution_20260521", "name": "code_execution"},
    # allowed_callers=["direct"] is REQUIRED for the cheap model: the _20260209
    # web tools default to being called from inside code execution (that's how
    # dynamic filtering works), and haiku can't do programmatic tool calling —
    # without this every web_search call on the default model dies with a 400.
    "web_search": {"type": "web_search_20260209", "name": "web_search",
                   "allowed_callers": ["direct"]},
}


@dataclass
class Call:
    """One completed model call. `blocks` is every content block from every API
    response the call made (tool calls, tool results, text), kept for the trace."""
    text: str
    blocks: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLM:
    """A thin, cache-aware wrapper over the Messages API."""

    def __init__(self, catalog: Catalog, call_cfg, client=None):
        if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY — every call here is a real API call.")
        self.catalog = catalog
        self.cfg = call_cfg
        self.client = client or anthropic.Anthropic()

    def call(self, model, prompt, system=None, tools=None, effort=None, schema=None) -> Call:
        """One call, resumed while a server-side tool keeps pausing the turn.

        The prompt and system prompt carry a cache breakpoint, so resending the
        SAME prompt to the SAME model bills the repeat at ~10% of the input rate.
        The cache is keyed by model, so a different model is always a fresh miss,
        and prompts under the model's floor (~1-4k tokens) don't cache at all.
        """
        request = self._request(model, prompt, system, tools, effort, schema)
        turn_texts, blocks = [], []
        usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}

        for _ in range(self.cfg.max_tool_turns):
            # Streamed because max_output_tokens is large: the SDK refuses a
            # non-streaming request whose ceiling could outlive the HTTP timeout.
            # get_final_message() reassembles the Message a create() would return.
            with self.client.messages.stream(**request) as stream:
                message = stream.get_final_message()

            for key, value in _usage_of(message).items():
                usage[key] += value
            blocks.extend(message.content)
            # Join the text blocks WITHIN this response (citations split one
            # message into several), but keep responses apart — see below.
            turn_texts.append("".join(b.text for b in message.content if b.type == "text"))

            if message.stop_reason != "pause_turn":
                break
            request["messages"].append({"role": "assistant", "content": message.content})

        # The LAST response holds the answer; earlier ones are the model working
        # up to it. Concatenating across responses would splice that preamble
        # onto the answer — and with `schema` set, produce unparseable JSON.
        return Call(text=turn_texts[-1] if turn_texts else "", blocks=blocks, usage=usage)

    def parse(self, model: str, prompt: str, schema_model: type[BaseModel]):
        """One structured call, validated straight into `schema_model`.

        The reply is guaranteed to match the schema as long as it fits in
        max_output_tokens — a reply cut off at the ceiling is invalid JSON, so
        keep expected output well under it.
        """
        return schema_model.model_validate_json(self.call(model, prompt, schema=schema_model).text)

    def _request(self, model, prompt, system, tools, effort, schema) -> dict:
        # cache_control "ephemeral" = an auto-expiring cache entry (typically 5m).
        request = {
            "model": model,
            "max_tokens": self.cfg.max_output_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": str(prompt),
                 "cache_control": {"type": "ephemeral"}}]}],
        }
        if system:
            request["system"] = [{"type": "text", "text": system,
                                  "cache_control": {"type": "ephemeral"}}]
        if tools:
            request["tools"] = [TOOL_DEFS[name] for name in tools]

        if effort and self.catalog.thinks(model):
            request["thinking"] = {"type": "adaptive"}
            request["output_config"] = {"effort": effort}
        else:
            request["thinking"] = {"type": "disabled"}   # default: the strategy is the only knob

        if schema is not None:
            # Constrains ONLY the text the model writes at the end — tool calls
            # and tool results in the same reply are untouched. Takes a Pydantic
            # class (what this package uses internally) or a raw JSON Schema dict
            # (what a workflow program passes, since the sandbox has no pydantic).
            request["output_format"] = (
                schema if isinstance(schema, type) and issubclass(schema, BaseModel)
                else {"type": "json_schema", "schema": schema})
        return request


def _usage_of(message) -> dict:
    """The token counts Anthropic returns, split so cached and fresh tokens can
    be priced differently."""
    return {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
        "cache_write": getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(message.usage, "cache_read_input_tokens", 0) or 0,
    }
