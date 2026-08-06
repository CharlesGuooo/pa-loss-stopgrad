import json


def write_mini_fixture(
    root,
    num_samples=8,
    crossing=True,
    time_shifted=False,
    same_path_shifted=False,
    perpendicular_shifted=False,
    static_agent=False,
    short_track_agent=False,
):
    version_dir = root / "v1.0-mini"
    version_dir.mkdir(parents=True)

    scene = {
        "token": "scene-1",
        "name": "fixture-scene",
        "first_sample_token": "sample-0",
        "last_sample_token": f"sample-{num_samples - 1}",
        "nbr_samples": num_samples,
    }
    samples = []
    sample_data = []
    ego_poses = []
    annotations = []

    for idx in range(num_samples):
        sample_token = f"sample-{idx}"
        samples.append(
            {
                "token": sample_token,
                "timestamp": idx * 500000,
                "prev": f"sample-{idx - 1}" if idx > 0 else "",
                "next": f"sample-{idx + 1}" if idx < num_samples - 1 else "",
                "scene_token": "scene-1",
            }
        )
        ego_pose_token = f"ego-{idx}"
        sample_data.append(
            {
                "token": f"sd-{idx}",
                "sample_token": sample_token,
                "ego_pose_token": ego_pose_token,
                "is_key_frame": True,
            }
        )
        ego_poses.append(
            {
                "token": ego_pose_token,
                "timestamp": idx * 500000,
                "translation": [float(idx), 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        )
        if static_agent:
            x = 2.0
            y = 0.5
        elif same_path_shifted:
            x = float(max(idx - 4, 0))
            y = 0.0
        elif perpendicular_shifted:
            y = float(idx - 5) * 4.0
            x = 2.0
        elif time_shifted:
            y = float(idx - 4) * 4.0
            x = 2.0
        else:
            x = 2.0
            y = -2.0 + idx if crossing else 4.0
        if not short_track_agent or idx >= 2:
            annotations.append(
                {
                    "token": f"ann-a-{idx}",
                    "sample_token": sample_token,
                    "instance_token": "agent-a",
                    "translation": [x, y, 0.0],
                    "size": [1.0, 1.0, 1.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                }
            )
        annotations.append(
            {
                "token": f"ann-b-{idx}",
                "sample_token": sample_token,
                "instance_token": "agent-b",
                "translation": [10.0 + idx, 8.0, 0.0],
                "size": [1.0, 1.0, 1.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        )

    tables = {
        "scene.json": [scene],
        "sample.json": samples,
        "sample_data.json": sample_data,
        "ego_pose.json": ego_poses,
        "sample_annotation.json": annotations,
    }
    for filename, rows in tables.items():
        (version_dir / filename).write_text(json.dumps(rows), encoding="utf-8")
    return root
