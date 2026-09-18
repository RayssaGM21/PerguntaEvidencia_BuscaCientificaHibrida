from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean_difference: float
    ci_lower: float
    ci_upper: float
    p_value: float


def paired_bootstrap(
    baseline: list[float],
    candidate: list[float],
    samples: int = 10000,
    seed: int = 42,
) -> BootstrapResult:
    if len(baseline) != len(candidate):
        raise ValueError("Baseline and candidate arrays must have the same length.")
    if not baseline:
        raise ValueError("At least one paired observation is required.")

    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    differences = candidate_array - baseline_array
    mean_difference = float(np.mean(differences))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    sampled_means = np.mean(differences[indices], axis=1)
    ci_lower, ci_upper = np.percentile(sampled_means, [2.5, 97.5])
    if mean_difference >= 0:
        p_value = float(2 * min(np.mean(sampled_means <= 0), np.mean(sampled_means >= 0)))
    else:
        p_value = float(2 * min(np.mean(sampled_means >= 0), np.mean(sampled_means <= 0)))
    return BootstrapResult(
        mean_difference=mean_difference,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=min(p_value, 1.0),
    )
