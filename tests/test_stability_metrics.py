def _seed_result(full_risk, stop_risk, wta_risk, full_diversity=1.0, wta_diversity=1.0, pa_grad=0.1):
    return {
        "full": {
            "high_conflict_collision_risk": full_risk,
            "collision_risk": full_risk,
            "mode_diversity": full_diversity,
            "pa_prediction_grad_norm": pa_grad,
        },
        "stop_grad": {
            "high_conflict_collision_risk": stop_risk,
            "collision_risk": stop_risk,
            "mode_diversity": wta_diversity,
            "pa_prediction_grad_norm": 0.0,
        },
        "wta_only": {
            "high_conflict_collision_risk": wta_risk,
            "collision_risk": wta_risk,
            "mode_diversity": wta_diversity,
            "pa_prediction_grad_norm": 0.0,
        },
        "no_pa": {
            "high_conflict_collision_risk": wta_risk,
            "collision_risk": wta_risk,
            "mode_diversity": wta_diversity,
            "pa_prediction_grad_norm": 0.0,
        },
    }


def test_compute_mode_delta_uses_negative_risk_delta_as_full_better():
    from pa_loss_stopgrad.eval.stability_metrics import compute_mode_delta

    seed_results = [_seed_result(full_risk=0.2, stop_risk=0.3, wta_risk=0.4)]

    deltas = compute_mode_delta(
        seed_results,
        "high_conflict_collision_risk",
        numerator_mode="full",
        denominator_mode="stop_grad",
    )

    assert deltas == [-0.09999999999999998]


def test_summarize_values_reports_negative_win_rate():
    from pa_loss_stopgrad.eval.stability_metrics import summarize_values

    summary = summarize_values([-0.2, -0.1, 0.1], lower_is_better=True)

    assert summary["mean"] < 0
    assert summary["win_rate_vs_zero"] == 2 / 3


def test_evaluate_phase0_2_gate_rejects_missing_gradients_and_mode_collapse():
    from pa_loss_stopgrad.eval.stability_metrics import evaluate_phase0_2_gate, summarize_stability

    good_summary = summarize_stability(
        [
            _seed_result(0.2, 0.3, 0.35, full_diversity=0.8, wta_diversity=1.0, pa_grad=0.2),
            _seed_result(0.1, 0.2, 0.25, full_diversity=0.7, wta_diversity=1.0, pa_grad=0.1),
        ]
    )
    missing_grad_summary = summarize_stability(
        [
            _seed_result(0.2, 0.3, 0.35, pa_grad=0.0),
            _seed_result(0.1, 0.2, 0.25, pa_grad=0.0),
        ]
    )
    collapsed_summary = summarize_stability(
        [
            _seed_result(0.2, 0.3, 0.35, full_diversity=0.1, wta_diversity=1.0),
            _seed_result(0.1, 0.2, 0.25, full_diversity=0.1, wta_diversity=1.0),
        ]
    )

    assert evaluate_phase0_2_gate(good_summary)["go"] is True
    assert evaluate_phase0_2_gate(missing_grad_summary)["go"] is False
    assert evaluate_phase0_2_gate(collapsed_summary)["go"] is False
