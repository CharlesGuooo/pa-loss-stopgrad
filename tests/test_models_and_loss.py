import torch


def test_simple_predictor_returns_multimodal_futures_and_scores():
    from pa_loss_stopgrad.carriers.simple_predictor import SimplePredictor

    model = SimplePredictor(state_dim=9, hidden_dim=16, num_modes=4, horizon=6)
    agent_history = torch.randn(2, 3, 5, 9)

    output = model(agent_history)

    assert output["predicted_futures"].shape == (2, 3, 4, 6, 2)
    assert output["prediction_scores"].shape == (2, 3, 4)


def test_ego_planner_returns_batch_modes_and_scores():
    from pa_loss_stopgrad.planners.ego_planner import EgoPlanner

    planner = EgoPlanner(input_dim=32, hidden_dim=16, horizon=6, num_modes=3)
    context_features = torch.randn(2, 32)

    output = planner(context_features)

    assert output["trajectories"].shape == (2, 3, 6, 3)
    assert output["scores"].shape == (2, 3)


def test_planning_loss_assigns_higher_weight_to_risky_mode():
    from pa_loss_stopgrad.pa_loss.planning_loss import PlanningAwareLoss

    predicted_futures = torch.tensor(
        [[[[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
           [[8.0, 8.0], [8.0, 8.0], [8.0, 8.0]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    ego_plans = {
        "trajectories": torch.zeros(1, 1, 3, 3, requires_grad=True),
        "scores": torch.zeros(1, 1),
    }
    loss_fn = PlanningAwareLoss(w_collision=1.0, w_comfort=0.0, w_offroad=0.0)

    losses = loss_fn(predicted_futures, ego_plans, drivable_area_map=None)
    risk_weights = losses["risk_weights"]

    assert risk_weights.shape == (1, 1, 2)
    assert risk_weights[0, 0, 0] > risk_weights[0, 0, 1]


def test_stop_gradient_blocks_prediction_gradient():
    from pa_loss_stopgrad.pa_loss.planning_loss import PlanningAwareLoss

    loss_fn = PlanningAwareLoss(w_collision=1.0, w_comfort=0.0, w_offroad=0.0)

    full_predictions = torch.randn(1, 2, 3, 4, 2, requires_grad=True)
    full_plans = {
        "trajectories": torch.zeros(1, 1, 4, 3, requires_grad=True),
        "scores": torch.zeros(1, 1),
    }
    full_loss = loss_fn(full_predictions, full_plans, None)["total_loss"]
    full_loss.backward()

    stopped_predictions = full_predictions.detach().clone().requires_grad_(True)
    stopped_plans = {
        "trajectories": torch.zeros(1, 1, 4, 3, requires_grad=True),
        "scores": torch.zeros(1, 1),
    }
    stopped_loss = loss_fn(
        stopped_predictions,
        stopped_plans,
        None,
        detach_predicted_futures=True,
    )["total_loss"]
    stopped_loss.backward()

    assert full_predictions.grad is not None
    assert full_predictions.grad.abs().sum() > 0
    assert stopped_predictions.grad is None or stopped_predictions.grad.abs().sum() == 0
