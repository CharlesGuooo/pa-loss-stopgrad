import torch


def test_synthetic_batch_has_required_shapes_and_labels():
    from pa_loss_stopgrad._internal.synthetic_scenarios import make_synthetic_interaction_batch

    batch = make_synthetic_interaction_batch(
        batch_size=4,
        scenario_mix=["crossing", "cut_in", "safe_following", "near_collision"],
        history_len=5,
        horizon=6,
        state_dim=9,
        seed=3,
    )

    assert batch["agent_history"].shape == (4, 3, 5, 9)
    assert batch["gt_futures"].shape == (4, 3, 6, 2)
    assert batch["ego_plans"]["trajectories"].shape == (4, 1, 6, 3)
    assert batch["scenario_type"] == ["crossing", "cut_in", "safe_following", "near_collision"]
    assert batch["is_high_conflict"].dtype == torch.bool


def test_synthetic_scenarios_have_expected_conflict_labels():
    from pa_loss_stopgrad._internal.synthetic_scenarios import make_synthetic_interaction_batch

    batch = make_synthetic_interaction_batch(
        batch_size=3,
        scenario_mix=["crossing", "safe_following", "near_collision"],
        seed=5,
    )

    assert batch["is_high_conflict"].tolist() == [True, False, True]


def test_synthetic_difficulty_noise_is_seeded_and_preserves_labels():
    from pa_loss_stopgrad._internal.synthetic_scenarios import make_synthetic_interaction_batch

    first = make_synthetic_interaction_batch(
        batch_size=3,
        scenario_mix=["crossing", "safe_following", "near_collision"],
        noise_std=0.1,
        seed=7,
    )
    second = make_synthetic_interaction_batch(
        batch_size=3,
        scenario_mix=["crossing", "safe_following", "near_collision"],
        noise_std=0.1,
        seed=8,
    )

    assert first["gt_futures"].shape == second["gt_futures"].shape
    assert first["is_high_conflict"].tolist() == [True, False, True]
    assert second["is_high_conflict"].tolist() == [True, False, True]
    assert not torch.allclose(first["gt_futures"], second["gt_futures"])
