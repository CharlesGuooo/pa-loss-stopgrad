"""Phase 2.7b tests: statistical helpers for robustness analysis."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "p27b_stats", REPO_ROOT / "scripts" / "build_phase2_7b_stats.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def test_ci_contains_mean_and_is_deterministic():
    x = np.arange(100, dtype=float)
    S.RNG = np.random.default_rng(3407)
    a = S._ci(x)
    S.RNG = np.random.default_rng(3407)
    b = S._ci(x)
    assert a == b                                  # deterministic with fixed seed
    assert a[1] <= a[0] <= a[2]                     # CI brackets the point estimate
    assert abs(a[0] - x.mean()) < 1e-9


def test_rate_diff_sign():
    # full collides less often than other -> diff (full-other) negative
    full = np.array([0, 0, 0, 0, 1], bool)
    other = np.array([1, 1, 0, 0, 1], bool)
    S.RNG = np.random.default_rng(0)
    r = S._rate_diff_ci(full, other)
    assert r["rate_a"] < r["rate_b"]
    assert r["diff"] < 0


def test_mcnemar_discordant_counts():
    full = np.array([1, 0, 0, 0], bool)   # collides on frame 0
    other = np.array([0, 1, 1, 0], bool)  # collides on frames 1,2
    m = S._mcnemar(full, other)
    assert m["full_only"] == 1            # frame 0
    assert m["other_only"] == 2           # frames 1,2 -> favours full


def test_wilcoxon_detects_shift_and_handles_ties():
    # a consistently larger than b -> positive median diff
    a = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = S._wilcoxon(a, b)
    assert w["median_diff"] > 0
    assert w["effect_rank_biserial"] > 0
    # all-equal -> no crash, nan p, zero effect
    w2 = S._wilcoxon(a, a)
    assert w2["effect_rank_biserial"] == 0.0


def test_minsep_inf_capped_in_median():
    x = np.array([1.0, 2.0, np.inf, np.inf])
    med = S._median_ci(x)
    assert np.isfinite(med[0])
    assert med[0] <= S.MAXSEP


def test_nearmiss_threshold_counts():
    full = np.array([0.5, 1.5, 3.0, 10.0])   # <2m: frames 0,1
    other = np.array([0.5, 5.0, 5.0, 10.0])  # <2m: frame 0
    S.RNG = np.random.default_rng(0)
    r = S._rate_diff_ci(full < 2.0, other < 2.0)
    assert r["rate_a"] == 0.5
    assert r["rate_b"] == 0.25
