"""Choosing between workflows on accuracy and cost.

Operates on anything with `.accuracy` and `.cost` — a `runtime.Result` from any
split, or a `Candidate`.
"""


def pareto_front(results: list) -> list:
    """The non-dominated results, cheapest first.

    Keep a result only if no OTHER result is at least as accurate AND at least as
    cheap (and strictly better on at least one of the two).
    """
    front = []
    for r in results:
        dominated = any(
            other is not r
            and other.cost <= r.cost
            and other.accuracy >= r.accuracy
            and (other.cost < r.cost or other.accuracy > r.accuracy)
            for other in results)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r.cost)


def best_under_budget(results: list, max_cost: float):
    """The most accurate result costing no more than `max_cost`."""
    affordable = [r for r in results if r.cost <= max_cost]
    return max(affordable, key=lambda r: r.accuracy, default=None)


def cheapest_above_accuracy(results: list, min_accuracy: float):
    """The cheapest result reaching `min_accuracy`."""
    good_enough = [r for r in results if r.accuracy >= min_accuracy]
    return min(good_enough, key=lambda r: r.cost, default=None)
