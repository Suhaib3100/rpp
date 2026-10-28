"""Paired significance testing between two classifiers.

Both models are evaluated on the *same* 360 test images, so the comparison is
paired and the right test is McNemar's, not a two-sample proportion test. The
distinction matters: a two-sample test throws away the pairing and is
needlessly conservative on correlated predictions.

McNemar's test looks only at the discordant pairs - images one model gets right
and the other gets wrong - and asks whether the split between the two kinds of
disagreement is more lopsided than chance. With small discordant counts the
exact binomial version is used rather than the chi-square approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class McNemarResult:
    n: int
    both_correct: int
    only_a_correct: int      # model A right, model B wrong
    only_b_correct: int      # model B right, model A wrong
    both_wrong: int
    n_discordant: int
    statistic: float
    p_value: float
    method: str
    accuracy_a: float
    accuracy_b: float
    accuracy_delta: float
    odds_ratio: float | None
    significant_at_05: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"McNemar ({self.method}): b={self.only_a_correct}, c={self.only_b_correct}, "
                f"p={self.p_value:.3e}, Δacc={self.accuracy_delta*100:+.2f} pts")


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
            exact_threshold: int = 25) -> McNemarResult:
    """Paired comparison of two classifiers on identical samples.

    Args:
        y_true: ground-truth labels.
        pred_a: predictions of model A (the "before" / baseline).
        pred_b: predictions of model B (the "after" / proposed).
        exact_threshold: use the exact binomial test when the discordant count
            is at or below this, otherwise the continuity-corrected chi-square.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true

    both = int((a_ok & b_ok).sum())
    only_a = int((a_ok & ~b_ok).sum())
    only_b = int((~a_ok & b_ok).sum())
    neither = int((~a_ok & ~b_ok).sum())
    disc = only_a + only_b

    if disc == 0:
        # The models made identical correctness patterns; nothing to test.
        stat, p, method = 0.0, 1.0, "no discordant pairs"
    elif disc <= exact_threshold:
        # Exact binomial: under H0 each discordant pair is a fair coin flip.
        stat = float(min(only_a, only_b))
        p = float(stats.binomtest(min(only_a, only_b), disc, 0.5).pvalue)
        method = "exact binomial"
    else:
        stat = float((abs(only_a - only_b) - 1) ** 2 / disc)
        p = float(stats.chi2.sf(stat, df=1))
        method = "chi-square with continuity correction"

    odds = (only_b / only_a) if only_a > 0 else None

    return McNemarResult(
        n=int(len(y_true)),
        both_correct=both,
        only_a_correct=only_a,
        only_b_correct=only_b,
        both_wrong=neither,
        n_discordant=disc,
        statistic=stat,
        p_value=p,
        method=method,
        accuracy_a=float(a_ok.mean()),
        accuracy_b=float(b_ok.mean()),
        accuracy_delta=float(b_ok.mean() - a_ok.mean()),
        odds_ratio=odds,
        significant_at_05=bool(p < 0.05),
    )


def bootstrap_delta_ci(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                       n_boot: int = 10000, seed: int = 0,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired accuracy difference (B − A).

    Resampling is over *images*, keeping each image's pair of predictions
    together, which preserves the pairing the McNemar test relies on.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a_ok = (np.asarray(pred_a) == y_true).astype(float)
    b_ok = (np.asarray(pred_b) == y_true).astype(float)
    diff = b_ok - a_ok
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
