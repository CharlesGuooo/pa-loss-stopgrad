from nuscenes_fixture import write_mini_fixture


def test_nuscenes_mini_loads_scene_chain_annotations_and_ego_pose(tmp_path):
    from pa_loss_stopgrad.data.nuscenes_mini import NuScenesMini

    data_root = write_mini_fixture(tmp_path, num_samples=4)
    nusc = NuScenesMini(data_root=data_root)

    scenes = nusc.list_scenes()
    samples = list(nusc.iter_scene_samples("scene-1"))
    annotations = nusc.get_sample_annotations("sample-0")
    ego_pose = nusc.get_ego_pose("sample-0")

    assert scenes[0]["token"] == "scene-1"
    assert [sample["token"] for sample in samples] == ["sample-0", "sample-1", "sample-2", "sample-3"]
    assert {ann["instance_token"] for ann in annotations} == {"agent-a", "agent-b"}
    assert ego_pose["translation"][:2] == [0.0, 0.0]
