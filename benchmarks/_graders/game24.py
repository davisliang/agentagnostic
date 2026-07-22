"""Game of 24 grading: the answer is an arithmetic expression, scored by whether
it uses each of the four given numbers exactly once and evaluates to 24.

There is no single gold string — many expressions are correct — so grading
EVALUATES the returned expression rather than string-matching it. The prediction
is expected to be a bare expression over the four numbers using + - * / and
parentheses; surrounding prose, a trailing "= 24", and the unicode operators
× ÷ − are tolerated. The expression is evaluated with a tiny arithmetic-only AST
walker, never `eval`, so a model-written string can't reach anything.
"""
import ast
import re
from collections import Counter

# The characters a bare arithmetic expression is made of. Used to pull the
# expression out of any prose the model wrapped around it.
_EXPR = re.compile(r"[0-9+\-*/(). ]+")


def _extract(prediction) -> str:
    """Pull the arithmetic expression out of whatever the workflow returned.

    Args:
        prediction: The returned answer, ideally a bare expression but possibly
            wrapped in prose or followed by "= 24".

    Returns:
        The longest arithmetic-looking span, unicode operators normalized to
        ASCII. "" if there is nothing expression-like.
    """
    text = str(prediction or "")
    text = (text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("·", "*"))
    text = re.sub(r"=\s*24(?:\.0)?\s*\.?\s*$", "", text.strip())   # drop a trailing "= 24"
    spans = _EXPR.findall(text)
    return max(spans, key=len).strip() if spans else ""


def _eval(node):
    """Evaluate an arithmetic-only AST node. Anything else raises.

    Args:
        node: An `ast` node from parsing the expression in "eval" mode.

    Returns:
        The numeric value.

    Raises:
        ValueError: The node is not a number, +-*/ binop, or unary +/-.
        ZeroDivisionError: Division by zero.
    """
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDivisionError
        return left / right
    raise ValueError("disallowed expression")


def grade(prediction, item: dict) -> float:
    """Score one Game-of-24 answer.

    Args:
        prediction: What the workflow returned — an arithmetic expression.
        item: The dataset example; its "numbers" holds the four given integers.

    Returns:
        1.0 if the expression uses each given number exactly once and evaluates to
        24 (within a small tolerance for division), else 0.0.
    """
    numbers = [int(n) for n in item["numbers"]]
    expression = _extract(prediction)
    if not expression:
        return 0.0
    try:
        value = _eval(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return 0.0
    used = [int(n) for n in re.findall(r"\d+", expression)]
    if Counter(used) != Counter(numbers):        # each given number used exactly once
        return 0.0
    return 1.0 if abs(value - 24) < 1e-6 else 0.0
