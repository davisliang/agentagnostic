"""FanOutQA grading — the official string "loose accuracy", ported.

A fan-out answer is structured: usually a dict of entity -> value, sometimes a
list or a primitive. Grading walks that reference and checks what fraction of its
pieces appear in the model's answer text (dict counts keys AND values), returning
a score in [0, 1] — so a partial answer gets partial credit and the mean over the
dataset is FanOutQA's loose accuracy. No judge.

Ported from `fanoutqa/eval/string.py` (`answer_in_text`) and `fanoutqa/norm.py`,
with two deliberate deviations, both documented so a score here is understood:

1. The official `normalize` runs spaCy lemmatization and ftfy encoding-repair.
   Both are heavy dependencies and near-no-ops on this data — the answer pieces
   are overwhelmingly proper nouns and numbers, which lemmatize to themselves — so
   they are dropped to keep the harness dependency-free.
2. We also strip currency and percent symbols ($ € £ ¥ %). The official metric
   keeps them, and its `\b...\b` word-boundary check then never matches a value
   like "$1.027 billion" (a leading "$" has no word boundary before it), silently
   scoring every currency answer 0 — which is part of why FanOutQA also ships an
   LLM judge. Stripping them makes the loose metric fair on the ~1/5 of questions
   with monetary/percentage answers, at the cost of exact parity with the string
   scorer. We keep the string metric (not the judge) to stay judge-free.
"""
import itertools
import re


def normalize(text) -> str:
    """Normalize a string for loose string matching.

    Args:
        text: Any value; stringified first.

    Returns:
        Lowercased, with commas inside numbers removed, currency/percent symbols
        and punctuation dropped, and whitespace collapsed.
    """
    text = str(text).lower()
    text = re.sub(r"(\d+,)+\d+(\.\d+)?", lambda m: m[0].replace(",", ""), text)   # 1,027 -> 1027
    text = re.sub(r"[,.?!:;$€£¥%]", "", text)                                     # incl. currency/percent
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _found(reference, candidate: str) -> tuple[int, int]:
    """Count how many pieces of a (possibly nested) reference appear in candidate.

    Args:
        reference: The gold answer — dict, list, or primitive.
        candidate: The normalized model answer text.

    Returns:
        `(found, total)` — pieces present and total pieces. A dict contributes both
        its keys and its values.
    """
    if isinstance(reference, dict):
        found = total = 0
        for piece in itertools.chain(reference.keys(), reference.values()):
            f, t = _found(piece, candidate)
            found += f
            total += t
        return found, total
    if isinstance(reference, list):
        found = total = 0
        for piece in reference:
            f, t = _found(piece, candidate)
            found += f
            total += t
        return found, total
    if isinstance(reference, bool):
        reference = "yes" if reference else "no"
    norm = normalize(reference)
    present = bool(norm and re.search(rf"\b{re.escape(norm)}\b", candidate))
    return (1 if present else 0), 1


def grade(prediction, item: dict) -> float:
    """Score one FanOutQA answer against its structured reference.

    Args:
        prediction: The workflow's answer — a string that should state each entity
            and its value.
        item: The dataset example; its "answer" holds the structured gold answer.

    Returns:
        The fraction of reference pieces (keys and values) found in the prediction,
        in [0, 1].
    """
    reference = item["answer"]
    found, total = _found(reference, normalize(prediction))
    return found / total if total else 0.0
