"""Prompt text lives in `prompts/*.md`, not in the code.

Placeholders are `${name}` (`string.Template`, not `str.format`) so a prompt can
contain literal braces — JSON Schema, code samples — without escaping. Only the
template is scanned, so a `$` inside a substituted value is left alone.
"""
from string import Template

from .paths import PROMPTS_DIR


def render(name: str, **values: object) -> str:
    """Fill `prompts/<name>.md`. Raises KeyError if the file wants a value the
    caller didn't pass — a missing variable is a bug, not an empty string."""
    template = Template((PROMPTS_DIR / f"{name}.md").read_text())
    return template.substitute(**values).strip()
