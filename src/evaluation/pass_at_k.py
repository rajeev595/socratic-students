"""
Evaluation utilities.

Currently includes:
- compute_pass_at_k: standard pass@k metric used in code evaluation.
"""

import math
import random
import numpy as np
from typing import List, Dict, Any


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


def bootstrap_aggregate(
    itemized: Dict[str, Dict[str, Any]],
    num_samples: int = 10_000,
    seed: int = 42
):
    """
    Args:
        itemized: mapping {example_id: {"results": {turn: {k: float}}}}
        num_samples: number of bootstrap resamples
        seed: RNG seed

    Returns:
        agg: {turn: {k: [mean_of_means, std_of_means]}}
    """
    rng = random.Random(seed)

    # Collect per-turn, per-k arrays across examples
    turns = None
    ks = None
    values: Dict[int, Dict[int, list]] = {}  # {turn: {k: [values]}}

    for _, ex_data in itemized.items():
        results = ex_data["results"]
        if turns is None:
            turns = sorted(map(int, results.keys()))
        if ks is None:
            # take k set from the first turn present
            first_turn_vals = next(iter(results.values()))
            ks = sorted(map(int, first_turn_vals.keys()))

        for t_str, t_res in results.items():
            t = int(t_str)
            for k_str, v in t_res.items():
                k = int(k_str)
                values.setdefault(t, {}).setdefault(k, []).append(float(v))

    if turns is None or ks is None:
        return {}

    # Bootstrap per (turn, k)
    agg: Dict[int, Dict[int, list]] = {}
    for t in turns:
        agg[t] = {}
        for k in ks:
            arr = np.array(values.get(t, {}).get(k, []), dtype=float)
            m = len(arr)
            if m == 0:
                agg[t][k] = [float("nan"), float("nan")]
                continue

            # resample with replacement from indices [0..m-1]
            samples = []
            for _ in range(num_samples):
                idxs = rng.choices(range(m), k=m)
                samples.append(float(arr[idxs].mean()))
            agg[t][k] = [float(np.mean(samples)), float(np.std(samples, ddof=0))]

    return agg
