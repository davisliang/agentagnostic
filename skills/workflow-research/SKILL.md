---
name: workflow-research
description: Research online what works for a given task before any workflow is designed — read as many sources as needed and record the findings in research_notes.md. Use at the start of a workflow-optimization run, before proposing candidate workflows.
---

# Research what works for this task

Before any workflow is written, find out what others actually do for a task of
this kind, and write it down. This runs once, ahead of design, and its only
output is `research_notes.md` in the working directory.

## What to do

1. **Search the web — this is mandatory.** Use `WebSearch` (and `WebFetch` to
   read promising results) to find how this kind of task is approached: known
   prompting strategies, published techniques, leaderboard write-ups, papers,
   blog posts, library docs, common failure modes and the tricks that fix them.
   Search for the specific task and for the general problem class behind it.
2. **Read as many sources as you need — no fixed number.** Follow leads until the
   picture stops changing. A narrow, well-understood task may take two or three
   sources; an open one may take many. Stop when further reading is repeating what
   you already have, not at a quota.
3. **Write `research_notes.md`** — the findings, for the design agent that runs
   next. Aim it at "what should I try, and what should I avoid, for THIS task."

## What research_notes.md should contain

- **Techniques that work for this task class**, concretely enough to build on —
  not "use chain-of-thought" but which decomposition, which verification step,
  what to route on, where sampling helps and where it doesn't.
- **Known failure modes and their fixes** — the mistakes that show up repeatedly
  on this kind of task, and what people do about them.
- **What tends to be wasted effort** — approaches reported to underperform, so the
  designer doesn't spend candidates rediscovering them.
- **Sources** — a short list of the pages you drew each point from (title + URL),
  so a claim can be traced back.

Keep it a briefing, not a literature review: dense, specific, skimmable. There is
no target length and no requirement to produce any particular number of ideas —
report what you actually found. If a genuine search turns up little, say so plainly
rather than padding.

Do **not** design, name, or test workflows here — that is the next phase. Your
final action is to write `research_notes.md` and stop.
