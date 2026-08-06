#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cross-seed stability metrics for Phase 0.2 synthetic probes."""

import math


def compute_mode_delta(
    seed_results,
    metric,
    numerator_mode="full",
    denominator_mode="stop_grad",
):
    return [
        float(seed_result[numerator_mode][metric]) - float(seed_result[denominator_mode][metric])
        for seed_result in seed_results
    ]


def compute_mode_ratio(
    seed_results,
    metric,
    numerator_mode="full",
    denominator_mode="wta_only",
):
    ratios = []
    for seed_result in seed_results:
        numerator = float(seed_result[numerator_mode][metric])
        denominator = max(float(seed_result[denominator_mode][metric]), 1e-12)
        ratios.append(numerator / denominator)
    return ratios


def summarize_values(values, lower_is_better=True):
    values = [float(value) for value in values]
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "win_rate_vs_zero": 0.0,
            "count": 0,
        }

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if lower_is_better:
        wins = sum(1 for value in values if value < 0.0)
    else:
        wins = sum(1 for value in values if value > 0.0)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "win_rate_vs_zero": wins / len(values),
        "count": len(values),
    }


def summarize_stability(seed_results):
    seed_results = _normalize_seed_results(seed_results)
    full_pa_grads = [
        float(seed_result["full"].get("pa_prediction_grad_norm", 0.0))
        for seed_result in seed_results
    ]
    full_pa_grad_positive_rate = (
        sum(1 for value in full_pa_grads if value > 0.0) / len(full_pa_grads)
        if full_pa_grads
        else 0.0
    )

    return {
        "full_vs_stop_grad_high_conflict_collision_risk_delta": summarize_values(
            compute_mode_delta(seed_results, "high_conflict_collision_risk", "full", "stop_grad"),
            lower_is_better=True,
        ),
        "full_vs_wta_high_conflict_collision_risk_delta": summarize_values(
            compute_mode_delta(seed_results, "high_conflict_collision_risk", "full", "wta_only"),
            lower_is_better=True,
        ),
        "full_vs_stop_grad_collision_risk_delta": summarize_values(
            compute_mode_delta(seed_results, "collision_risk", "full", "stop_grad"),
            lower_is_better=True,
        ),
        "full_vs_wta_mode_diversity_ratio": summarize_values(
            [ratio - 1.0 for ratio in compute_mode_ratio(seed_results, "mode_diversity", "full", "wta_only")],
            lower_is_better=False,
        ),
        "full_vs_wta_mode_diversity_ratio_raw": summarize_values(
            compute_mode_ratio(seed_results, "mode_diversity", "full", "wta_only"),
            lower_is_better=False,
        ),
        "full_pa_grad_positive_rate": full_pa_grad_positive_rate,
        "num_seeds": len(seed_results),
    }


def evaluate_phase0_2_gate(stability_summary, min_win_rate=0.6, max_diversity_drop=0.5):
    risk_delta = stability_summary["full_vs_stop_grad_high_conflict_collision_risk_delta"]
    diversity_ratio = stability_summary["full_vs_wta_mode_diversity_ratio_raw"]
    pa_grad_positive_rate = stability_summary["full_pa_grad_positive_rate"]

    checks = [
        (
            risk_delta["mean"] < 0.0,
            "full_vs_stop_grad_high_conflict_collision_risk_delta.mean < 0",
        ),
        (
            risk_delta["win_rate_vs_zero"] >= min_win_rate,
            f"full_vs_stop_grad win_rate_vs_zero >= {min_win_rate}",
        ),
        (
            pa_grad_positive_rate >= 0.8,
            "full_pa_grad_positive_rate >= 0.8",
        ),
        (
            diversity_ratio["mean"] >= max_diversity_drop,
            f"full_vs_wta_mode_diversity_ratio.mean >= {max_diversity_drop}",
        ),
    ]
    reasons = [
        ("PASS: " if passed else "FAIL: ") + label
        for passed, label in checks
    ]
    return {
        "go": all(passed for passed, _ in checks),
        "reasons": reasons,
        "thresholds": {
            "min_win_rate": min_win_rate,
            "max_diversity_drop": max_diversity_drop,
            "min_pa_grad_positive_rate": 0.8,
        },
    }


def _normalize_seed_results(seed_results):
    if isinstance(seed_results, dict):
        return list(seed_results.values())
    return list(seed_results)
