"""
Evaluation utilities.

Currently includes:
- compute_pass_at_k: standard pass@k metric used in code evaluation.
"""

import math
from typing import List, Dict


def compute_pass_at_k(results: List[bool], k_values: List[int]) -> Dict[int, float]:
    """
    Compute pass@k scores for a given set of model attempts.

    pass@k measures the probability that at least one correct solution is
    found in k samples drawn without replacement from the set of attempts.

    Args:
        results: List of booleans, where True means the attempt passed.
        k_values: List of k values (sample counts) to evaluate.

    Returns:
        Dict mapping each k in `k_values` to its pass@k score in [0, 1],
        rounded to two decimal places.

    Notes:
        - If no attempts are given (n == 0), returns 0.0 for all k.
        - If k > n, pass@k is 1.0 if there is at least one correct solution,
          otherwise 0.0.
        - Formula:
            pass@k = 1 - C(n - c, k) / C(n, k)
          where n = total attempts, c = correct attempts,
          and C(a, b) = "a choose b".
    """
    n = len(results)
    c = sum(results)
    scores: Dict[int, float] = {}

    def safe_comb(a: int, b: int) -> int:
        """Safe binomial coefficient, returns 0 if invalid args."""
        return math.comb(a, b) if 0 <= b <= a else 0

    for k in k_values:
        if n == 0:
            score = 0.0
        elif k > n:
            score = 1.0 if c > 0 else 0.0
        else:
            total = safe_comb(n, k)
            incorrect = safe_comb(n - c, k)
            score = 1.0 - (incorrect / total if total > 0 else 1.0)
        scores[k] = round(float(score), 2)

    return scores
