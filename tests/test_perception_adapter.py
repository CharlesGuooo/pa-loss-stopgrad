"""Phase 2.5 unit tests: frozen-perception -> GameFormer adapter.

Numpy-only (no nuscenes/torch); validates geometry, track linking, AoI matching,
perceived-map routing, and output schema/shape parity with the oracle npz.
"""
import numpy as np
import pytest

from pa_loss_stopgrad.perception import perception_adapter as pa
from pa_loss_stopgrad.carriers.config import (
    AGENT_DIM, CROSSWALK_PTS, FUTURE_LEN, HIST_LEN, LANE_FEAT_DIM, LANE_PTS,
    NUM_CROSSWALKS, NUM_LANES, NUM_NEIGHBORS, NUM_PRED, TYPE_VEHICLE,
)


def _quat_from_yaw(yaw):
    return [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]


def _make_sample(token, ts, boxes, scores, labels, track_ids,
                 e2g_t=(0, 0, 0), e2g_yaw=0.0, l2e_t=(0, 0, 0), l2e_yaw=0.0,
                 vectors=None, map_labels=None):
    return {
        "meta": {"sample_token": token, "timestamp": ts},
        "agents": {"boxes": boxes, "scores": scores, "labels": labels, "track_ids": track_ids},
        "map": {"vectors": vectors or [], "labels": map_labels or [],
                "scores": [1.0] * len(vectors or [])},
        "ego": {
            "ego2global_translation": list(e2g_t),
            "ego2global_rotation": _quat_from_yaw(e2g_yaw),
            "lidar2ego_translation": list(l2e_t),
            "lidar2ego_rotation": _quat_from_yaw(l2e_yaw),
            "ego_status": [0.0] * 10,
            "command": [0.0, 0.0, 0.0],
        },
    }


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_quat_to_yaw_matches_known_angles():
    for yaw in (-2.5, -np.pi / 3, 0.0, 0.7, np.pi / 2, 3.0):
        got = pa.quat_to_yaw(_quat_from_yaw(yaw))
        assert pa.wrap_to_pi(got - yaw) == pytest.approx(0.0, abs=1e-9)


def test_box_to_global_pure_translation():
    # ego at global (10, 5), yaw 0; lidar at ego origin; box at lidar (3, 0)
    s = _make_sample("t", 0, [[3.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 1.0, 0.0]],
                     [0.9], [0], [7], e2g_t=(10, 5, 0), e2g_yaw=0.0)
    f = pa.Frame(s)
    x, y, h, vx, vy = f.box_to_global(0)
    assert (x, y) == pytest.approx((13.0, 5.0))
    assert pa.wrap_to_pi(h) == pytest.approx(0.0, abs=1e-9)
    assert (vx, vy) == pytest.approx((1.0, 0.0))


def test_box_to_global_with_ego_rotation():
    # ego yaw +90deg at origin; box ahead in lidar (+x=2) -> global +y=2
    s = _make_sample("t", 0, [[2.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]],
                     [0.9], [0], [1], e2g_t=(0, 0, 0), e2g_yaw=np.pi / 2)
    f = pa.Frame(s)
    x, y, h, _, _ = f.box_to_global(0)
    assert (x, y) == pytest.approx((0.0, 2.0), abs=1e-6)
    assert pa.wrap_to_pi(h - np.pi / 2) == pytest.approx(0.0, abs=1e-6)


def test_box_to_global_with_lidar_offset():
    # lidar offset +1m forward in ego; ego at origin yaw 0; box at lidar origin
    s = _make_sample("t", 0, [[0.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]],
                     [0.9], [0], [1], l2e_t=(1.0, 0.0, 0.0))
    f = pa.Frame(s)
    x, y, _, _, _ = f.box_to_global(0)
    assert (x, y) == pytest.approx((1.0, 0.0))


def test_normalize_agent_track_round_trip():
    # a global point placed relative to a known ego pose lands at expected ego xy
    center, angle = np.array([100.0, 50.0]), np.pi / 2
    track = np.zeros((HIST_LEN, AGENT_DIM))
    # global point 3m ahead of ego (ego heads +y) -> ego frame (3, 0)
    track[-1] = [100.0, 53.0, np.pi / 2, 0.0, 2.0, 4.5, 2.0, 1.6, TYPE_VEHICLE]
    out = pa.normalize_agent_track(track, center, angle)
    assert out[-1, 0] == pytest.approx(3.0, abs=1e-6)
    assert out[-1, 1] == pytest.approx(0.0, abs=1e-6)
    assert out[-1, 2] == pytest.approx(0.0, abs=1e-6)  # heading aligned with ego
    # padded rows stay zero
    assert np.all(out[:-1] == 0.0)


# --------------------------------------------------------------------------- #
# scene segmentation + track linking
# --------------------------------------------------------------------------- #
def test_segment_scenes_splits_on_gap():
    f0 = pa.Frame(_make_sample("a", 1_000_000, [], [], [], []))
    f1 = pa.Frame(_make_sample("b", 1_500_000, [], [], [], []))   # +0.5s same scene
    f2 = pa.Frame(_make_sample("c", 9_000_000, [], [], [], []))   # big gap -> new scene
    scenes, t2p = pa.segment_scenes([f2, f0, f1])
    assert len(scenes) == 2
    assert t2p["a"][0] == t2p["b"][0]      # same scene
    assert t2p["c"][0] != t2p["a"][0]
    assert t2p["a"][1] == 0 and t2p["b"][1] == 1   # ordered by timestamp


def test_build_track_history_links_across_frames():
    box = lambda x: [[x, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]]
    f0 = pa.Frame(_make_sample("a", 1_000_000, box(2.0), [0.9], [0], [42]))
    f1 = pa.Frame(_make_sample("b", 1_500_000, box(4.0), [0.9], [0], [42]))
    frames = {"a": f0, "b": f1}
    scenes, t2p = pa.segment_scenes([f0, f1])
    center, angle = np.array([0.0, 0.0]), 0.0
    hist = pa.build_track_history(scenes, t2p, frames, "b", 42, center, angle)
    assert hist.shape == (HIST_LEN, AGENT_DIM)
    # last two slots populated (history len 2), earlier padded
    assert hist[-1, 0] == pytest.approx(4.0)
    assert hist[-2, 0] == pytest.approx(2.0)
    assert np.all(hist[:-2] == 0.0)


def test_build_track_history_pads_missing_track():
    box = lambda x, tid: [[x, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]]
    f0 = pa.Frame(_make_sample("a", 1_000_000, box(2.0, 42), [0.9], [0], [99]))  # diff id
    f1 = pa.Frame(_make_sample("b", 1_500_000, box(4.0, 42), [0.9], [0], [42]))
    frames = {"a": f0, "b": f1}
    scenes, t2p = pa.segment_scenes([f0, f1])
    hist = pa.build_track_history(scenes, t2p, frames, "b", 42, np.array([0.0, 0.0]), 0.0)
    assert hist[-1, 0] == pytest.approx(4.0)
    assert np.all(hist[-2] == 0.0)   # track 42 absent in frame a


# --------------------------------------------------------------------------- #
# AoI matching + full adaptation
# --------------------------------------------------------------------------- #
def _oracle_stub(aoi_ego_xy):
    neighbors = np.zeros((NUM_NEIGHBORS, HIST_LEN, AGENT_DIM), dtype=np.float32)
    neighbors[0, -1, :2] = aoi_ego_xy
    return {
        "ego": np.zeros((HIST_LEN, AGENT_DIM), dtype=np.float32),
        "neighbors": neighbors,
        "map_lanes": np.zeros((NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM), dtype=np.float32),
        "map_crosswalks": np.zeros((NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3), dtype=np.float32),
        "gt_future_states": np.zeros((NUM_PRED, FUTURE_LEN, 5), dtype=np.float32),
        "object_type": np.array([1.0, 1.0], dtype=np.float32),
    }


def test_aoi_matches_nearest_perceived_track():
    # two tracks; oracle AoI sits on top of the one at ego-frame (5, 0)
    boxes = [[5.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0],
             [20.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]]
    s = _make_sample("b", 1_500_000, boxes, [0.9, 0.9], [0, 0], [7, 8])
    f = pa.Frame(s)
    scenes, t2p = pa.segment_scenes([f])
    out = pa.adapt_token("b", _oracle_stub([5.0, 0.0]), scenes, t2p, {"b": f},
                         map_source="oracle")
    m = out["meta"].item()
    assert m["aoi_matched"] is True
    # neighbor slot 0 (matched AoI) lands near (5,0)
    assert out["neighbors"][0, -1, 0] == pytest.approx(5.0, abs=1e-5)


def test_aoi_unmatched_when_no_track_in_gate():
    boxes = [[40.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]]
    s = _make_sample("b", 1_500_000, boxes, [0.9], [0], [7])
    f = pa.Frame(s)
    scenes, t2p = pa.segment_scenes([f])
    out = pa.adapt_token("b", _oracle_stub([5.0, 0.0]), scenes, t2p, {"b": f},
                         map_source="oracle")
    assert out["meta"].item()["aoi_matched"] is False
    assert np.all(out["neighbors"][0] == 0.0)   # missed target -> zero history


def test_low_score_and_nonagent_detections_dropped():
    boxes = [[5.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0],   # car, low score
             [6.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],   # traffic_cone (label 9)
             [7.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]]   # car, ok
    s = _make_sample("b", 1_500_000, boxes, [0.1, 0.99, 0.9], [0, 9, 0], [1, 2, 3])
    f = pa.Frame(s)
    scenes, t2p = pa.segment_scenes([f])
    out = pa.adapt_token("b", _oracle_stub([7.0, 0.0]), scenes, t2p, {"b": f},
                         map_source="oracle", score_threshold=0.3)
    assert out["meta"].item()["n_perceived_agents"] == 1   # only the high-score car


# --------------------------------------------------------------------------- #
# perceived map routing + schema parity
# --------------------------------------------------------------------------- #
def test_perceived_map_routes_labels():
    line = [[float(i), 0.0] for i in range(20)]
    cw = [[float(i), 5.0] for i in range(20)]
    s = _make_sample("b", 1_500_000,
                     [[5.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]], [0.9], [0], [7],
                     vectors=[line, cw, line], map_labels=[1, 0, 2])  # divider, ped, boundary
    f = pa.Frame(s)
    scenes, t2p = pa.segment_scenes([f])
    out = pa.adapt_token("b", _oracle_stub([5.0, 0.0]), scenes, t2p, {"b": f},
                         map_source="perceived")
    assert out["map_lanes"].shape == (NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM)
    assert out["map_crosswalks"].shape == (NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3)
    # divider+boundary -> 2 lane slots populated; ped_crossing -> 1 crosswalk
    lane_filled = (np.abs(out["map_lanes"][0]).sum(axis=(1, 2)) > 0).sum()
    cw_filled = (np.abs(out["map_crosswalks"][0]).sum(axis=(1, 2)) > 0).sum()
    assert lane_filled == 2
    assert cw_filled == 1
    assert np.all(np.isfinite(out["map_lanes"]))


def test_output_schema_parity_with_oracle():
    s = _make_sample("b", 1_500_000,
                     [[5.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0, 0.0, 0.0]], [0.9], [0], [7])
    f = pa.Frame(s)
    scenes, t2p = pa.segment_scenes([f])
    out = pa.adapt_token("b", _oracle_stub([5.0, 0.0]), scenes, t2p, {"b": f})
    expected = {
        "ego": (HIST_LEN, AGENT_DIM),
        "neighbors": (NUM_NEIGHBORS, HIST_LEN, AGENT_DIM),
        "map_lanes": (NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM),
        "map_crosswalks": (NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3),
        "gt_future_states": (NUM_PRED, FUTURE_LEN, 5),
        "object_type": (NUM_PRED,),
    }
    for k, shp in expected.items():
        assert out[k].shape == shp, f"{k}: {out[k].shape} != {shp}"
        assert np.all(np.isfinite(out[k]))
    assert "sample_token" in out["meta"].item()


def test_returns_none_for_unknown_token():
    f = pa.Frame(_make_sample("b", 1_500_000, [], [], [], []))
    scenes, t2p = pa.segment_scenes([f])
    assert pa.adapt_token("zzz", _oracle_stub([0, 0]), scenes, t2p, {"b": f}) is None
