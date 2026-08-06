import torch


def test_collision_proxy_risk_is_higher_for_close_predictions():
    from pa_loss_stopgrad.eval.risk_metrics import collision_proxy_risk

    close_predictions = torch.zeros(1, 1, 1, 4, 2)
    far_predictions = torch.full((1, 1, 1, 4, 2), 10.0)
    ego_plans = {
        "trajectories": torch.zeros(1, 1, 4, 3),
        "scores": torch.zeros(1, 1),
    }

    close_risk = collision_proxy_risk(close_predictions, ego_plans)
    far_risk = collision_proxy_risk(far_predictions, ego_plans)

    assert close_risk > far_risk


def test_mode_diversity_increases_when_modes_spread_apart():
    from pa_loss_stopgrad.eval.risk_metrics import mode_diversity

    collapsed = torch.zeros(1, 1, 3, 4, 2)
    spread = collapsed.clone()
    spread[:, :, 1, :, 0] = 5.0
    spread[:, :, 2, :, 1] = 5.0

    assert mode_diversity(spread) > mode_diversity(collapsed)


def test_high_conflict_metrics_reports_subset_means():
    from pa_loss_stopgrad.eval.risk_metrics import high_conflict_metrics

    metrics = {
        "collision_risk": torch.tensor([1.0, 3.0, 5.0]),
        "wta_l1": torch.tensor([2.0, 4.0, 6.0]),
    }
    is_high_conflict = torch.tensor([True, False, True])

    result = high_conflict_metrics(metrics, is_high_conflict)

    assert result["high_conflict_collision_risk"] == 3.0
    assert result["low_conflict_collision_risk"] == 3.0
    assert result["high_conflict_wta_l1"] == 4.0
