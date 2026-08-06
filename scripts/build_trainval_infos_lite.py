#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Phase 1.10 lite nuScenes infos converter (oracle-GT only).

Standalone port of the GT block of SparseDrive's ``nuscenes_converter.py``
(``_fill_trainval_infos`` lines ~304-410). It produces the exact set of fields
the Planning-GAZE risk-probe pipeline consumes (``planning_gaze/phase1_8_gt_loader.py``)
and nothing else:

    gt_boxes, gt_names, gt_velocity, instance_inds, valid_flag,
    num_lidar_pts, num_radar_pts,
    gt_agent_fut_trajs, gt_agent_fut_masks,
    gt_ego_fut_trajs, gt_ego_fut_masks, gt_ego_fut_cmd, ego_status,
    token, scene_token, timestamp,
    lidar2ego_*, ego2global_*.

Deliberately dropped (not needed for oracle-GT Phase 1.9/1.10, and the source of
the local mmcv/map-plugin friction):

    * ``map_annos``       -> requires ``NuscMapExtractor`` (mmcv-full plugin).
    * ``cams`` / ``sweeps`` -> require sensor blobs.
    * the LiDAR-file existence checks (``mmcv.check_file_exist`` /
      ``get_available_scenes``) -> we never read the ``.pcd.bin`` blobs.

Only depends on ``nuscenes`` devkit + numpy + pyquaternion + pickle, so it runs
in an isolated py3.10 venv without the heavy mmdet3d/mmcv stack.

Usage (full trainval)::

    python scripts/build_trainval_infos_lite.py \
        --version v1.0-trainval \
        --dataroot $NUSCENES_ROOT/v1.0-trainval_meta \
        --canbus   $NUSCENES_ROOT/can_bus \
        --out-dir  data/infos/trainval

Usage (mini parity)::

    python scripts/build_trainval_infos_lite.py \
        --version v1.0-mini \
        --dataroot $NUSCENES_ROOT/v1.0-mini \
        --canbus   $NUSCENES_ROOT/v1.0-mini/can_bus \
        --out-dir  data/infos/mini_lite
"""

import argparse
import os
import pickle
import time
from os import path as osp

import numpy as np
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils import splits

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i - 1] < utimes[i] - utime):
        i -= 1
    return i


def get_ego_status(nusc, nusc_can_bus, sample):
    """10-dim ego status [accel(3), rotation_rate(3), vel(3), steering(1)]."""
    ego_status = []
    ref_scene = nusc.get("scene", sample["scene_token"])
    try:
        pose_msgs = nusc_can_bus.get_messages(ref_scene["name"], "pose")
        steer_msgs = nusc_can_bus.get_messages(ref_scene["name"], "steeranglefeedback")
        pose_uts = [msg["utime"] for msg in pose_msgs]
        steer_uts = [msg["utime"] for msg in steer_msgs]
        ref_utime = sample["timestamp"]
        pose_data = pose_msgs[locate_message(pose_uts, ref_utime)]
        steer_data = steer_msgs[locate_message(steer_uts, ref_utime)]
        ego_status.extend(pose_data["accel"])
        ego_status.extend(pose_data["rotation_rate"])
        ego_status.extend(pose_data["vel"])
        ego_status.append(steer_data["value"])
    except Exception:
        ego_status = [0] * 10
    return np.array(ego_status).astype(np.float32)


def get_global_sensor_pose(rec, nusc):
    lidar_sample_data = nusc.get("sample_data", rec["data"]["LIDAR_TOP"])
    pose_record = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    cs_record = nusc.get(
        "calibrated_sensor", lidar_sample_data["calibrated_sensor_token"]
    )
    ego2global = transform_matrix(
        pose_record["translation"], Quaternion(pose_record["rotation"]), inverse=False
    )
    sensor2ego = transform_matrix(
        cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False
    )
    return ego2global.dot(sensor2ego)


def _scene_splits(version):
    if version == "v1.0-trainval":
        return splits.train, splits.val
    if version == "v1.0-mini":
        return splits.mini_train, splits.mini_val
    if version == "v1.0-test":
        return splits.test, []
    raise ValueError(f"unknown version {version}")


def build_infos(nusc, nusc_can_bus, version, max_samples=None, fut_ts=12, ego_fut_ts=6):
    train_names, val_names = _scene_splits(version)
    name_by_token = {s["token"]: s["name"] for s in nusc.scene}
    train_names = set(train_names)
    val_names = set(val_names)

    predict_helper = PredictHelper(nusc)
    train_infos, val_infos = [], []

    samples = nusc.sample if max_samples is None else nusc.sample[:max_samples]
    total = len(samples)
    t0 = time.time()
    for s_idx, sample in enumerate(samples):
        if s_idx % 500 == 0:
            el = time.time() - t0
            print(f"  [{s_idx}/{total}] {el:.0f}s elapsed", flush=True)

        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        # boxes in the LiDAR sensor frame; does NOT touch the .pcd.bin blob.
        _, boxes, _ = nusc.get_sample_data(lidar_token)

        info = {
            "token": sample["token"],
            "scene_token": sample["scene_token"],
            "timestamp": sample["timestamp"],
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
        }

        e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix
        l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix

        annotations = [nusc.get("sample_annotation", t) for t in sample["anns"]]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array(
            [b.orientation.yaw_pitch_roll[0] for b in boxes]
        ).reshape(-1, 1)
        velocity = np.array(
            [nusc.box_velocity(t)[:2] for t in sample["anns"]]
        ).reshape(-1, 2)
        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        names = np.array([NameMapping.get(n, n) for n in names])
        valid_flag = np.array(
            [(a["num_lidar_pts"] + a["num_radar_pts"]) > 0 for a in annotations],
            dtype=bool,
        ).reshape(-1)
        # gt_boxes: [x, y, z, w(l-slot), l(w-slot), h, yaw] per upstream (dims[:, [1,0,2]]).
        gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

        instance_inds = [
            nusc.getind("instance", a["instance_token"]) for a in annotations
        ]

        # Agent future trajectories (offsets in current agent/lidar frame).
        num_box = len(boxes)
        gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
        gt_fut_masks = np.zeros((num_box, fut_ts))
        for i, anno in enumerate(annotations):
            fut_local = predict_helper.get_future_for_agent(
                anno["instance_token"],
                sample["token"],
                seconds=fut_ts / 2,
                in_agent_frame=True,
            )
            if fut_local.shape[0] > 0:
                box = boxes[i]
                trans = box.center
                rot = Quaternion(matrix=box.rotation_matrix)
                fut_scene = convert_local_coords_to_global(fut_local, trans, rot)
                valid_step = fut_scene.shape[0]
                gt_fut_trajs[i, 0] = fut_scene[0] - box.center[:2]
                gt_fut_trajs[i, 1:valid_step] = fut_scene[1:] - fut_scene[:-1]
                gt_fut_masks[i, :valid_step] = 1

        # Ego future trajectory + drive command + ego status.
        ego_fut_trajs = np.zeros((ego_fut_ts + 1, 3))
        ego_fut_masks = np.zeros((ego_fut_ts + 1))
        sample_cur = sample
        ego_status = get_ego_status(nusc, nusc_can_bus, sample_cur)
        pose_mat = None
        for i in range(ego_fut_ts + 1):
            pose_mat = get_global_sensor_pose(sample_cur, nusc)
            ego_fut_trajs[i] = pose_mat[:3, 3]
            ego_fut_masks[i] = 1
            if sample_cur["next"] == "":
                ego_fut_trajs[i + 1 :] = ego_fut_trajs[i]
                break
            sample_cur = nusc.get("sample", sample_cur["next"])
        # global -> ego
        ego_fut_trajs = ego_fut_trajs - np.array(pose_record["translation"])
        rot_mat = Quaternion(pose_record["rotation"]).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        # ego -> lidar
        ego_fut_trajs = ego_fut_trajs - np.array(cs_record["translation"])
        rot_mat = Quaternion(cs_record["rotation"]).inverse.rotation_matrix
        ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
        if ego_fut_trajs[-1][0] >= 2:
            command = np.array([1, 0, 0])  # right
        elif ego_fut_trajs[-1][0] <= -2:
            command = np.array([0, 1, 0])  # left
        else:
            command = np.array([0, 0, 1])  # straight
        ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]

        info["gt_boxes"] = gt_boxes
        info["gt_names"] = names
        info["gt_velocity"] = velocity.reshape(-1, 2)
        info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
        info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
        info["valid_flag"] = valid_flag
        info["instance_inds"] = instance_inds
        info["gt_agent_fut_trajs"] = gt_fut_trajs.astype(np.float32)
        info["gt_agent_fut_masks"] = gt_fut_masks.astype(np.float32)
        info["gt_ego_fut_trajs"] = ego_fut_trajs[:, :2].astype(np.float32)
        info["gt_ego_fut_masks"] = ego_fut_masks[1:].astype(np.float32)
        info["gt_ego_fut_cmd"] = command.astype(np.float32)
        info["ego_status"] = ego_status

        scene_name = name_by_token[sample["scene_token"]]
        if scene_name in train_names:
            train_infos.append(info)
        elif scene_name in val_names:
            val_infos.append(info)
    return train_infos, val_infos


def main():
    p = argparse.ArgumentParser(description="Lite nuScenes oracle-GT infos converter")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--dataroot", required=True)
    p.add_argument("--canbus", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--info-prefix", default="nuscenes")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    print(f"[lite-converter] version={args.version} dataroot={args.dataroot}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=args.canbus)

    train_infos, val_infos = build_infos(
        nusc, nusc_can_bus, args.version, max_samples=args.max_samples
    )
    print(f"[lite-converter] train={len(train_infos)} val={len(val_infos)}")

    os.makedirs(args.out_dir, exist_ok=True)
    metadata = dict(version=args.version)
    train_path = osp.join(args.out_dir, f"{args.info_prefix}_infos_train.pkl")
    val_path = osp.join(args.out_dir, f"{args.info_prefix}_infos_val.pkl")
    with open(train_path, "wb") as fh:
        pickle.dump(dict(infos=train_infos, metadata=metadata), fh)
    with open(val_path, "wb") as fh:
        pickle.dump(dict(infos=val_infos, metadata=metadata), fh)
    print(f"[lite-converter] wrote {train_path}\n[lite-converter] wrote {val_path}")


if __name__ == "__main__":
    main()
