import json

import torch

from nuscenes_fixture import write_mini_fixture


def _event(sample_token="sample-1", closest_agent_token="agent-a", label="near_miss"):
    return {
        "event_id": "event_000000",
        "scene_token": "scene-1",
        "sample_token": sample_token,
        "window_index": 0,
        "label": label,
        "closest_agent_token": closest_agent_token,
        "closest_agent_index": 0,
        "temporal_min_distance": 0.0,
        "sync_min_distance": 0.0,
    }


def test_load_event_index_supports_limit(tmp_path):
    from pa_loss_stopgrad._internal.real_data_risk_probe import load_event_index

    event_path = tmp_path / "events.json"
    event_path.write_text(json.dumps([_event("sample-1"), _event("sample-2")]), encoding="utf-8")

    events = load_event_index(event_path, limit=1)

    assert len(events) == 1
    assert events[0]["event_id"] == "event_000000"


def test_find_event_window_matches_scene_sample_and_agent(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from pa_loss_stopgrad._internal.real_data_risk_probe import find_event_window

    data_root = write_mini_fixture(tmp_path / "fixture", num_samples=8, crossing=True)
    nusc = NuScenesMini(data_root=data_root)

    match = find_event_window(nusc, _event("sample-1", "agent-a"), history_len=2, future_len=4)

    assert match is not None
    assert match["event"]["event_id"] == "event_000000"
    assert match["target_agent_index"] == 0
    assert match["window"]["sample_token"] == "sample-1"
    assert match["window"]["agent_tokens"][0] == "agent-a"


def test_build_real_data_risk_batch_returns_phase0_tensors_and_metadata(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from pa_loss_stopgrad._internal.real_data_risk_probe import build_real_data_risk_batch

    data_root = write_mini_fixture(tmp_path / "batch", num_samples=8, crossing=True)
    nusc = NuScenesMini(data_root=data_root)

    batch = build_real_data_risk_batch(
        nusc,
        [_event("sample-1", "agent-a", label="high_conflict"), _event("missing", "agent-a")],
        history_len=2,
        future_len=4,
        max_agents=2,
    )

    assert batch["agent_history"].shape == (1, 2, 2, 9)
    assert batch["gt_futures"].shape == (1, 2, 4, 2)
    assert batch["ego_plans"]["trajectories"].shape == (1, 1, 4, 3)
    assert batch["is_high_conflict"].tolist() == [True]
    assert batch["event_ids"] == ["event_000000"]
    assert batch["closest_agent_tokens"] == ["agent-a"]
    assert batch["num_skipped_events"] == 1
    assert batch["skipped_events"][0]["reason"] == "window_not_found"
    assert torch.isfinite(batch["agent_history"]).all()


def test_build_real_data_risk_batch_pads_variable_agent_counts(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from pa_loss_stopgrad._internal.real_data_risk_probe import build_real_data_risk_batch

    data_root = write_mini_fixture(tmp_path / "variable-agents", num_samples=9, crossing=True)
    annotation_path = data_root / "v1.0-mini" / "sample_annotation.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = [
        annotation
        for annotation in annotations
        if not (annotation["instance_token"] == "agent-b" and annotation["sample_token"] in {"sample-0", "sample-1"})
    ]
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    nusc = NuScenesMini(data_root=data_root)

    batch = build_real_data_risk_batch(
        nusc,
        [_event("sample-1", "agent-a"), _event("sample-3", "agent-a")],
        history_len=2,
        future_len=4,
        max_agents=2,
    )

    assert batch["agent_history"].shape == (2, 2, 2, 9)
    assert batch["agent_mask"].tolist() == [[True, False], [True, True]]
    assert torch.equal(batch["gt_futures"][0, 1], torch.zeros(4, 2))
