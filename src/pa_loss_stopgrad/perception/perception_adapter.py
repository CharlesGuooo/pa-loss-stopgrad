"""Phase 2.5: frozen-SparseDrive perception -> GameFormer input adapter.

Consumes the Phase 1.6 perception cache (one pkl per keyframe, produced by
``SparseDrive/tools/export_perception_interface.py``) and produces GameFormer
nuScenes ``.npz`` inputs whose AGENT context and (optionally) MAP context come
from *frozen perception* instead of oracle GT.

Controlled-gap design (Option 1): the prediction TARGET and ego trajectory stay
oracle (taken from the existing oracle npz keyed by ``sample_token``); only the
perceived agent context (and optionally the map) is swapped in. The agent-of-
interest (neighbor[0]) is matched to its nearest perceived track so we predict
the SAME agent from perceived history -> the minADE delta vs oracle isolates the
effect of perception on the carrier.

This module is numpy-only (no nuscenes/torch) so it runs in the main env next to
the GameFormer eval. The ego-frame normalization mirrors
``preprocess_nuscenes.normalize_*`` exactly.

Perception pkl schema (per keyframe)::

    agents.boxes  : [N, 9]  decoded SparseDrive box [x,y,z,w,l,h,yaw,vx,vy] (LiDAR frame)
    agents.scores : [N]
    agents.labels : [N]     index into DET_CLASS_NAMES
    agents.track_ids : [N]  temporally-propagated instance ids
    map.vectors   : list of [P,2] polylines (ego/BEV frame); P=20
    map.labels    : [V]     0=ped_crossing, 1=divider, 2=boundary
    ego.ego2global_translation/rotation, lidar2ego_translation/rotation
    meta.sample_token, meta.timestamp
"""
from __future__ import annotations

import numpy as np

from pa_loss_stopgrad.carriers.config import (
    AGENT_DIM,
    CROSSWALK_FEAT_DIM,
    CROSSWALK_PTS,
    FUTURE_LEN,
    HIST_LEN,
    LANE_FEAT_DIM,
    LANE_PTS,
    NUM_CROSSWALKS,
    NUM_LANES,
    NUM_NEIGHBORS,
    NUM_PRED,
    TYPE_CYCLIST,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
)

DET_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
# DET label -> GameFormer agent type (0 = not an agent, dropped).
_LABEL_TO_TYPE = {
    0: TYPE_VEHICLE, 1: TYPE_VEHICLE, 2: TYPE_VEHICLE, 3: TYPE_VEHICLE, 4: TYPE_VEHICLE,
    5: 0,                      # barrier
    6: TYPE_CYCLIST, 7: TYPE_CYCLIST,  # motorcycle, bicycle
    8: TYPE_PEDESTRIAN,        # pedestrian
    9: 0,                      # traffic_cone
}

# decoded box columns
_BOX_X, _BOX_Y, _BOX_Z, _BOX_W, _BOX_L, _BOX_H, _BOX_YAW, _BOX_VX, _BOX_VY = range(9)

# scene segmentation: consecutive nuScenes keyframes are ~0.5 s apart (2 Hz)
_KEYFRAME_DT_US = 500_000
_SCENE_GAP_US = 1_500_000  # > this -> new scene
MAP_RADIUS = 50.0
AOI_MATCH_GATE = 3.0  # m, max distance to match oracle AoI to a perceived track


# --------------------------------------------------------------------------- #
# geometry (mirrors preprocess_nuscenes.normalize_*)
# --------------------------------------------------------------------------- #
def wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(q):
    """Yaw (rad) from quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def to_ego_xy(xy, center, angle):
    xy = np.asarray(xy, dtype=np.float64)
    pad = (xy[..., 0] == 0) & (xy[..., 1] == 0)
    dx = xy[..., 0] - center[0]
    dy = xy[..., 1] - center[1]
    c, s = np.cos(angle), np.sin(angle)
    nx = c * dx + s * dy
    ny = -s * dx + c * dy
    out = np.stack([nx, ny], axis=-1)
    out[pad] = 0.0
    return out


def normalize_agent_track(track, center, angle):
    """track [T,9] global -> ego frame (cols 5-8 untouched). Mirrors preprocess."""
    track = np.asarray(track, dtype=np.float64).copy()
    pad = (track[:, 0] == 0) & (track[:, 1] == 0)
    xy = to_ego_xy(track[:, :2], center, angle)
    h = np.where(pad, 0.0, wrap_to_pi(track[:, 2] - angle))
    c, s = np.cos(angle), np.sin(angle)
    nvx = np.where(pad, 0.0, c * track[:, 3] + s * track[:, 4])
    nvy = np.where(pad, 0.0, -s * track[:, 3] + c * track[:, 4])
    out = track.copy()
    out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4] = xy[:, 0], xy[:, 1], h, nvx, nvy
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# perception frame loading + LiDAR->global lift
# --------------------------------------------------------------------------- #
class Frame:
    """One parsed perception keyframe."""

    __slots__ = ("token", "timestamp", "boxes", "scores", "labels", "track_ids",
                 "vectors", "map_labels", "map_scores", "l2e_t", "l2e_yaw", "e2g_t", "e2g_yaw")

    def __init__(self, sample):
        self.token = sample["meta"]["sample_token"]
        self.timestamp = int(sample["meta"]["timestamp"])
        ag = sample.get("agents", {})
        self.boxes = np.asarray(ag.get("boxes", []), dtype=np.float64).reshape(-1, 9)
        self.scores = np.asarray(ag.get("scores", []), dtype=np.float64).reshape(-1)
        self.labels = np.asarray(ag.get("labels", []), dtype=np.int64).reshape(-1)
        self.track_ids = np.asarray(ag.get("track_ids", []), dtype=np.int64).reshape(-1)
        mp = sample.get("map", {})
        self.vectors = [np.asarray(v, dtype=np.float64).reshape(-1, 2) for v in mp.get("vectors", [])]
        self.map_labels = np.asarray(mp.get("labels", []), dtype=np.int64).reshape(-1)
        self.map_scores = np.asarray(mp.get("scores", []), dtype=np.float64).reshape(-1)
        ego = sample["ego"]
        self.l2e_t = np.asarray(ego["lidar2ego_translation"], dtype=np.float64)
        self.l2e_yaw = quat_to_yaw(ego["lidar2ego_rotation"])
        self.e2g_t = np.asarray(ego["ego2global_translation"], dtype=np.float64)
        self.e2g_yaw = quat_to_yaw(ego["ego2global_rotation"])

    def ego_center_angle(self):
        return self.e2g_t[:2], self.e2g_yaw

    def box_to_global(self, i):
        """Lift detection i (LiDAR frame) to a global [x,y,heading,vx,vy] row."""
        b = self.boxes[i]
        p_lidar = b[[_BOX_X, _BOX_Y]]
        p_ego = _rot(self.l2e_yaw) @ p_lidar + self.l2e_t[:2]
        p_g = _rot(self.e2g_yaw) @ p_ego + self.e2g_t[:2]
        heading_g = wrap_to_pi(b[_BOX_YAW] + self.l2e_yaw + self.e2g_yaw)
        v = np.array([b[_BOX_VX], b[_BOX_VY]], dtype=np.float64)
        v_g = _rot(self.e2g_yaw + self.l2e_yaw) @ v
        return p_g[0], p_g[1], heading_g, v_g[0], v_g[1]

    def box_size_type(self, i):
        b = self.boxes[i]
        length = max(b[_BOX_W], b[_BOX_L])
        width = min(b[_BOX_W], b[_BOX_L])
        tcode = _LABEL_TO_TYPE.get(int(self.labels[i]), 0)
        return length, width, b[_BOX_H], tcode


# --------------------------------------------------------------------------- #
# scene segmentation + history linking
# --------------------------------------------------------------------------- #
def segment_scenes(frames):
    """Order frames by timestamp, split into scenes by large timestamp gaps.

    Returns (ordered_frames, token_to_pos) where token_to_pos[token] = (scene_idx, pos).
    """
    ordered = sorted(frames, key=lambda f: f.timestamp)
    token_to_pos, scenes = {}, []
    cur = []
    for f in ordered:
        if cur and (f.timestamp - cur[-1].timestamp) > _SCENE_GAP_US:
            scenes.append(cur)
            cur = []
        cur.append(f)
    if cur:
        scenes.append(cur)
    for si, sc in enumerate(scenes):
        for pos, f in enumerate(sc):
            token_to_pos[f.token] = (si, pos)
    return scenes, token_to_pos


def build_track_history(scenes, token_to_pos, frames_by_token, token, track_id, center, angle):
    """[HIST_LEN, 9] global->ego-frame history of one track, padded at the front."""
    si, pos = token_to_pos[token]
    scene = scenes[si]
    start = max(0, pos - (HIST_LEN - 1))
    seq = scene[start: pos + 1]               # oldest..current within the scene
    track = np.zeros((HIST_LEN, AGENT_DIM), dtype=np.float64)
    offset = HIST_LEN - len(seq)
    for k, fr in enumerate(seq):
        idx = np.where(fr.track_ids == track_id)[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        x, y, h, vx, vy = fr.box_to_global(i)
        length, width, height, tcode = fr.box_size_type(i)
        track[offset + k] = (x, y, h, vx, vy, length, width, height, tcode)
    return normalize_agent_track(track, center, angle)


# --------------------------------------------------------------------------- #
# map (perceived) -> degraded 12-dim lane tensor + crosswalks
# --------------------------------------------------------------------------- #
def _polyline_heading(pts):
    if len(pts) < 2:
        return np.zeros(len(pts))
    seg = pts[1:] - pts[:-1]
    h = wrap_to_pi(np.arctan2(seg[:, 1], seg[:, 0]))
    return np.concatenate([h, h[-1:]])


def build_perceived_map(frame, pred_centers_ego, center, angle):
    """Approximate map_lanes/map_crosswalks from perceived divider/boundary/ped_crossing.

    No centerline in SparseDrive's online map: each divider/boundary polyline is
    placed as a lane self_line (cols 0:3); left/right stay zero. ped_crossing ->
    crosswalks. Vectors are in the current ego/BEV frame; we express them in the
    oracle ego frame (origin ego, +x heading) used by the carrier.
    """
    lane_polys, cw_polys = [], []
    for v, lab in zip(frame.vectors, frame.map_labels):
        # lift LiDAR/BEV -> ego frame (apply lidar2ego); vectors are 2D
        pe = (_rot(frame.l2e_yaw) @ v.T).T + frame.l2e_t[:2]
        if lab in (1, 2):       # divider / boundary -> lane line
            lane_polys.append(pe)
        elif lab == 0:          # ped_crossing
            cw_polys.append(pe)

    map_lanes = np.zeros((NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM), dtype=np.float32)
    map_cw = np.zeros((NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, CROSSWALK_FEAT_DIM), dtype=np.float32)

    for pi, pc in enumerate(pred_centers_ego):
        scored = sorted(lane_polys, key=lambda p: np.min(np.linalg.norm(p - pc, axis=-1)))
        for li, poly in enumerate(scored[:NUM_LANES]):
            n = min(len(poly), LANE_PTS)
            head = _polyline_heading(poly)[:n]
            map_lanes[pi, li, :n, 0:2] = poly[:n]
            map_lanes[pi, li, :n, 2] = head
            map_lanes[pi, li, :n, 9] = 1
        scored_cw = sorted(cw_polys, key=lambda p: np.min(np.linalg.norm(p - pc, axis=-1)))
        for ci, poly in enumerate(scored_cw[:NUM_CROSSWALKS]):
            n = min(len(poly), CROSSWALK_PTS)
            head = _polyline_heading(poly)[:n]
            map_cw[pi, ci, :n, 0:2] = poly[:n]
            map_cw[pi, ci, :n, 2] = head
    return map_lanes, map_cw


# --------------------------------------------------------------------------- #
# main per-token adaptation
# --------------------------------------------------------------------------- #
def adapt_token(token, oracle_npz, scenes, token_to_pos, frames_by_token,
                *, score_threshold=0.3, map_source="oracle"):
    """Build a perceived GameFormer npz dict for one token (controlled-gap).

    oracle_npz: dict-like (np.load) of the oracle sample (ego/futures/AoI target).
    Returns a dict in the oracle npz schema, or None if the token isn't in cache.
    """
    if token not in token_to_pos:
        return None
    frame = frames_by_token[token]
    center, angle = frame.ego_center_angle()

    # candidate perceived tracks (current frame), score-filtered, real agents only
    cand = []
    for i in range(len(frame.boxes)):
        if frame.scores[i] < score_threshold:
            continue
        if _LABEL_TO_TYPE.get(int(frame.labels[i]), 0) == 0:
            continue
        tid = int(frame.track_ids[i])
        if tid < 0:
            continue
        x, y, *_ = frame.box_to_global(i)
        ex, ey = to_ego_xy(np.array([x, y]), center, angle)
        cand.append((tid, float(np.hypot(ex, ey)), np.array([ex, ey])))
    # nearest unique track first
    cand.sort(key=lambda z: z[1])
    seen, uniq = set(), []
    for tid, dist, exy in cand:
        if tid in seen:
            continue
        seen.add(tid)
        uniq.append((tid, dist, exy))

    # match oracle AoI (neighbor0 current ego-frame xy) to nearest perceived track
    oracle_neigh = oracle_npz["neighbors"]
    aoi_xy = oracle_neigh[0, -1, :2].astype(np.float64)
    aoi_tid, best = None, AOI_MATCH_GATE
    for tid, _, exy in uniq:
        d = float(np.hypot(exy[0] - aoi_xy[0], exy[1] - aoi_xy[1]))
        if d < best:
            best, aoi_tid = d, tid

    # Slot 0 is reserved for the prediction target (oracle AoI). If perception
    # missed it (no match in gate), slot 0 stays zero -> honest "missed target".
    neighbors = np.zeros((NUM_NEIGHBORS, HIST_LEN, AGENT_DIM), dtype=np.float32)
    if aoi_tid is not None:
        neighbors[0] = build_track_history(
            scenes, token_to_pos, frames_by_token, token, aoi_tid, center, angle)
    slot = 1
    for tid, _, _ in uniq:
        if tid == aoi_tid or slot >= NUM_NEIGHBORS:
            continue
        neighbors[slot] = build_track_history(
            scenes, token_to_pos, frames_by_token, token, tid, center, angle)
        slot += 1

    # ego state + GT futures + AoI target + types from oracle (controlled)
    ego = oracle_npz["ego"].astype(np.float32)
    gt_future = oracle_npz["gt_future_states"].astype(np.float32)
    object_type = oracle_npz["object_type"].astype(np.float32)

    if map_source == "perceived":
        ego_center_ego = np.array([0.0, 0.0])
        nb0_center_ego = neighbors[0, -1, :2].astype(np.float64)
        map_lanes, map_cw = build_perceived_map(
            frame, [ego_center_ego, nb0_center_ego], center, angle)
    else:  # oracle map (isolates the detection effect)
        map_lanes = oracle_npz["map_lanes"].astype(np.float32)
        map_cw = oracle_npz["map_crosswalks"].astype(np.float32)

    return {
        "ego": ego,
        "neighbors": neighbors,
        "map_lanes": map_lanes,
        "map_crosswalks": map_cw,
        "gt_future_states": gt_future,
        "object_type": object_type,
        "meta": np.array(
            {"sample_token": token, "aoi_matched": bool(aoi_tid is not None),
             "n_perceived_agents": int(len(uniq)), "map_source": map_source},
            dtype=object,
        ),
    }
