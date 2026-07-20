"""Flexible containment match for open-domain QA, ported from routerllm's
`nq_open_gen/utils.py`.

Strict exact-match scores a chat model 0 for answering in a sentence ("The last
time was December 1972") even when the gold span ("December 1972") is right
there. An answer counts if any normalized gold alias appears in it as a whole
phrase.

The gold may be a single string or a list of accepted aliases — Natural
Questions ships several phrasings per answer.
"""
import re

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, and collapse whitespace.

    Args:
        text: Any answer text.

    Returns:
        The normalized form used for comparison.
    """
    text = _PUNCTUATION.sub(" ", text.lower())
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def grade(prediction, item: dict) -> float:
    """Score one answer against the accepted aliases.

    Args:
        prediction: What the workflow returned.
        item: The dataset example; its "answer" is a gold string, a list of
            aliases, or a JSON-encoded list of them.

    Returns:
        1.0 if any alias matches the whole prediction or appears in it as a whole
        phrase, else 0.0.
    """
    predicted = _normalize(str(prediction))
    if not predicted:
        return 0.0

    golds = item.get("answer", [])
    if isinstance(golds, str):
        stripped = golds.strip()
        if stripped.startswith("["):          # the export stores lists as JSON text
            try:
                import json
                golds = json.loads(stripped)
            except ValueError:
                golds = [golds]
        else:
            golds = [golds]

    for gold in golds if isinstance(golds, (list, tuple)) else [golds]:
        normalized = _normalize(str(gold))
        if normalized and (normalized == predicted
                           or re.search(rf"\b{re.escape(normalized)}\b", predicted)):
            return 1.0
    return 0.0
