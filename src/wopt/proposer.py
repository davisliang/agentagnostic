"""Run one design-agent session. Entry point for the subprocess `designer` spawns.

Run from the agent's scratch directory (its cwd): it reads `proposer_config.json`
from there and drives a Claude Agent SDK session. The agent — via the skills
staged under `./.claude/skills/` — writes its picks to `programs.json`.

    python -m wopt.proposer
"""
import asyncio
import json
import os
import sys

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,
                              TextBlock, ToolUseBlock, query)


async def main() -> None:
    cfg = json.loads(open("proposer_config.json").read())
    options = ClaudeAgentOptions(
        model=cfg["model"],
        cwd=os.getcwd(),
        setting_sources=["project"],            # discovers ./.claude/skills/<name>/
        skills=cfg["skills"],
        allowed_tools=cfg["allowed_tools"],
        permission_mode="bypassPermissions",
    )
    async for message in query(prompt=cfg["prompt"], options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text[:300], flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"  [tool] {block.name}", flush=True)
        elif isinstance(message, ResultMessage):
            print(f"[agent finished: {message.subtype}]", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:                  # let the caller salvage candidates
        print(f"[proposer error] {type(error).__name__}: {error}", flush=True)
        sys.exit(1)
