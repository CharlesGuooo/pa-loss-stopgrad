from nuscenes_fixture import write_mini_fixture


def test_scene_tracks_group_agents_and_generate_windows(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini
    from pa_loss_stopgrad.data.nuscenes_trajectories import build_scene_tracks, iter_trajectory_windows

    data_root = write_mini_fixture(tmp_path, num_samples=8)
    nusc = NuScenesMini(data_root=data_root)

    tracks = build_scene_tracks(nusc, "scene-1")
    windows = list(iter_trajectory_windows(tracks, history_len=2, future_len=4, max_agents=2))

    assert tracks["ego_xy"].shape == (8, 2)
    assert set(tracks["agent_tracks"]) == {"agent-a", "agent-b"}
    assert windows
    assert windows[0]["ego_future"].shape == (4, 2)
    assert windows[0]["agent_futures"].shape == (2, 4, 2)
    assert windows[0]["agent_history"].shape == (2, 2, 9)
