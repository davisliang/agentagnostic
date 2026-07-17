#!/usr/bin/env python3
"""Run the workflow-design agent in a clean subprocess.

Run this from the agent's scratch directory (its cwd). It reads
`proposer_config.json` from the cwd and drives a Claude Agent SDK session; the
agent (via the workflow-design / workflow-eval skills discovered under
`./.claude/skills/`) writes its selected programs to `programs.json` here.

Running as a subprocess isolates the async event loop from the notebook kernel
— a prior `nest_asyncio.apply()` monkeypatches `asyncio` for the whole kernel
session and breaks `asyncio.run` even from a worker thread.
"""
import asyncio
import json
import os
import sys

from claude_agent_sdk import (query, ClaudeAgentOptions, AssistantMessage,
                              ResultMessage, TextBlock, ToolUseBlock)

cfg = json.load(open("proposer_config.json"))


async def main():
    opts = ClaudeAgentOptions(
        model=cfg["model"],
        cwd=os.getcwd(),
        setting_sources=["project"],          # discover ./.claude/skills/<name>/
        skills=cfg["skills"],
        allowed_tools=cfg["allowed_tools"],
        permission_mode="bypassPermissions",
    )
    async for msg in query(prompt=cfg["prompt"], options=opts):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    print(b.text[:300], flush=True)
                elif isinstance(b, ToolUseBlock):
                    print(f"  [tool] {b.name}", flush=True)
        elif isinstance(msg, ResultMessage):
            print(f"[agent finished: {msg.subtype}]", flush=True)


try:
    asyncio.run(main())
except Exception as e:                          # let the notebook salvage candidates
    print(f"[proposer error] {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
