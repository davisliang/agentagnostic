"""ARC-AGI grid grading, ported from routerllm's `arc_agi_2/utils.py`.

Models wrap the answer grid in prose or markdown, so the LAST contiguous block
of all-numeric rows is extracted from the response and compared cell for cell
with the gold grid. Surrounding text and whitespace are ignored; the grid itself
must match exactly.

Kept byte-compatible with routerllm's `_last_grid` so a score here means what it
means there.
"""
import re

_NUMERIC_ROW = re.compile(r"^\s*\d+(?:[\s,]+\d+)*\s*$")


def _last_grid(text) -> str:
    """Extract the last contiguous run of numeric rows from a response.

    Args:
        text: The model's answer, or the gold grid.

    Returns:
        The rows joined by newlines, each normalized to single-space-separated
        digits. "" if the text contains no numeric block.
    """
    blocks, current = [], []
    for line in (text or "").splitlines():
        if _NUMERIC_ROW.match(line):
            current.append(" ".join(re.split(r"[\s,]+", line.strip())))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return "\n".join(blocks[-1]) if blocks else ""


def grade(prediction, item: dict) -> float:
    """Score one predicted grid.

    Args:
        prediction: What the workflow returned.
        item: The dataset example; its "answer" holds the gold grid.

    Returns:
        1.0 if the extracted grids match exactly, else 0.0.
    """
    gold = _last_grid(str(item.get("answer", "")))
    return 1.0 if gold and _last_grid(str(prediction)) == gold else 0.0
