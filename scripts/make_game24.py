"""Generate a Game-of-24 benchmark: solvable four-number instances, hardest first.

Game of 24 is the canonical task where elaborate inference-time structure
(propose → verify → backtrack search) beats a single call or a majority vote,
because verification is trivial and errors across samples are correlated. We use
it to test whether the optimizer's search discovers that structure or collapses
to the usual single-call / vote / verify families.

Instances are found by a small exact solver (recursively combine pairs), so every
one is guaranteed solvable and its reference answer is a real solution. "Hardest"
is proxied by fewest distinct solutions — the instances an LLM is least likely to
stumble onto in one shot — which is the regime where the structure gap is widest.

    uv run python scripts/make_game24.py            # writes the benchmark + task file
    uv run python scripts/make_game24.py --n 60     # how many instances to keep

Deterministic: `random.Random(seed)` draws the number sets, so two runs produce
the same benchmark.
"""
import argparse
import json
import pathlib
import random
from collections import namedtuple

# One solvable Game-of-24 instance: the four numbers, how many distinct solutions
# it has (its difficulty proxy), and one reference solution expression.
Instance = namedtuple("Instance", "numbers n_solutions reference")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def solutions(numbers) -> set:
    """Every distinct expression over `numbers` (each used once) that makes 24.

    Args:
        numbers: The four integers.

    Returns:
        A set of fully-parenthesized expression strings evaluating to 24. Empty
        when the instance is unsolvable.
    """
    def combine(items):
        """Every (value, expression) reachable by combining these operands.

        Args:
            items: A list of (value, expression) pairs still to combine.

        Returns:
            A list of (value, expression) pairs — every result of repeatedly
            replacing two operands with one of their +, -, *, / combinations until
            a single operand remains.
        """
        if len(items) == 1:
            return [items[0]]
        out = []
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                (val_a, expr_a), (val_b, expr_b) = items[i], items[j]
                rest = [items[k] for k in range(len(items)) if k not in (i, j)]
                pairs = [(val_a + val_b, f"({expr_a}+{expr_b})"),
                         (val_a - val_b, f"({expr_a}-{expr_b})"),
                         (val_a * val_b, f"({expr_a}*{expr_b})")]
                if abs(val_b) > 1e-9:
                    pairs.append((val_a / val_b, f"({expr_a}/{expr_b})"))
                for value, expr in pairs:
                    out.extend(combine(rest + [(value, expr)]))
        return out

    found = set()
    for value, expr in combine([(float(n), str(n)) for n in numbers]):
        if abs(value - 24) < 1e-6:
            found.add(expr)
    return found


def build(n_keep: int, seed: int, low: int, high: int, pool_size: int) -> list[dict]:
    """Draw solvable instances and keep the `n_keep` hardest (fewest solutions).

    Args:
        n_keep: How many instances to keep.
        seed: RNG seed for reproducibility.
        low: Smallest number that may appear.
        high: Largest number that may appear.
        pool_size: How many distinct solvable instances to gather before ranking.

    Returns:
        `{"question", "answer", "numbers"}` dicts, hardest first.
    """
    rng = random.Random(seed)
    seen, pool = set(), []
    while len(pool) < pool_size:
        numbers = tuple(sorted(rng.randint(low, high) for _ in range(4)))
        if numbers in seen:
            continue
        seen.add(numbers)
        sols = solutions(numbers)
        if sols:
            pool.append(Instance(numbers, len(sols), min(sols, key=len)))  # shortest as reference

    pool.sort(key=lambda inst: (inst.n_solutions, inst.numbers))     # fewest solutions first
    rows = []
    for inst in pool[:n_keep]:
        a, b, c, d = inst.numbers
        rows.append({
            "question": (f"Use each of the numbers {a}, {b}, {c}, {d} exactly once, "
                         f"with the operators + - * / and parentheses, to make 24. "
                         f"Answer with a single arithmetic expression and nothing else."),
            "answer": inst.reference,
            "numbers": list(inst.numbers),
        })
    return rows


DESCRIPTION = (
    "Game of 24. You are given four numbers. Produce a single arithmetic "
    "expression that uses each of the four numbers exactly once, together with "
    "the operators + - * / and parentheses, and that evaluates to exactly 24. "
    "Answer with only the expression (for example, for 4, 7, 8, 8 the answer "
    "4*(7-8/8) evaluates to 24). The expression is graded by evaluating it: it "
    "scores 1 if it uses each given number exactly once and equals 24, else 0.")


def main() -> None:
    """Write the data, the benchmark descriptor, and the task config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="instances to keep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low", type=int, default=1)
    parser.add_argument("--high", type=int, default=13)
    parser.add_argument("--pool", type=int, default=800, help="solvable instances to rank")
    args = parser.parse_args()

    rows = build(args.n, args.seed, args.low, args.high, args.pool)

    bench_dir = ROOT / "benchmarks" / "game24"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (bench_dir / "benchmark.yaml").write_text(
        "# game24 — generated by scripts/make_game24.py (solvable instances, hardest first)\n"
        "name: game24\n"
        f"description: >-\n  {DESCRIPTION}\n"
        "source_dataset: generated\n"
        f"examples: {len(rows)}\n"
        "grading_supported: true\n"
        "check_type: exact\n"
        "grader: benchmarks/_graders/game24.py\n")

    (ROOT / "config" / "task" / "game24.yaml").write_text(
        "# Game of 24 — a task that needs elaborate (search / verify-loop) structure.\n"
        "# Closed-book on purpose (runtime.tools: []): no code execution, so the\n"
        "# WORKFLOW must do the reasoning, not a one-line brute-force tool call.\n"
        "task:\n"
        "  name: game24\n"
        f"  description: >-\n    {DESCRIPTION}\n"
        "  check_type: exact\n"
        "  dataset: benchmarks/game24/data.jsonl\n"
        "  grader: benchmarks/_graders/game24.py\n"
        "  answer_examples: ['4*(7-8/8)', '(4+8)*(3-1)']\n"
        "runtime:\n"
        "  tools: []          # closed-book: the structure must do the work, not a tool\n"
        "data:\n"
        "  n_examples: 40     # 24 dev / 16 test at dev_fraction 0.6\n")

    solved_lens = [len(solutions(r["numbers"])) for r in rows]
    print(f"wrote {len(rows)} instances to {bench_dir/'data.jsonl'}")
    print(f"difficulty (distinct solutions per instance): "
          f"min {min(solved_lens)}, median {sorted(solved_lens)[len(solved_lens)//2]}, "
          f"max {max(solved_lens)}")
    print(f"wrote {bench_dir/'benchmark.yaml'} and config/task/game24.yaml")


if __name__ == "__main__":
    main()
