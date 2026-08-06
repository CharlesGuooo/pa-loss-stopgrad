"""Phase 2.7 tests: open-loop planner + local PlanningMetric.

Covers:
  * command derivation agrees between the planner and the anchor-library builder
  * metric L2 matches the SparseDrive definition on a hand-checked example
  * collision detection: overlapping box -> hit; far box -> no hit; GT-collision
    subtraction (a collision the GT plan also incurs is not counted)
  * planner determinism + model-independence of the prior + collision-avoidance
    behaviour (a dangerous predicted neighbour shifts the chosen anchor away)
  * HC mask alignment by token
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_loss_stopgrad.planners import openloop_planner as olp# noqa: E402
from pa_loss_stopgrad.eval.planning_metric import PlanningMetric, get_yaw  # noqa: E402

# import the builder's derive_command without running it
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_ego_anchor_library", REPO_ROOT / "scripts" / "build_ego_anchor_library.py")
_blib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_blib)


# --------------------------------------------------------------------------- #
# command derivation consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y_end,expected", [(3.0, "left"), (-3.0, "right"), (0.2, "straight")])
def test_command_consistency(y_end, expected):
    traj = torch.zeros(6, 2)
    traj[:, 0] = torch.linspace(1, 6, 6)   # forward motion
    traj[5, 1] = y_end
    assert olp.derive_command(traj) == expected
    # builder uses numpy and indexes the 6th in-horizon waypoint identically
    assert _blib.derive_command(traj.numpy()) == expected


# --------------------------------------------------------------------------- #
# L2 metric definition
# --------------------------------------------------------------------------- #
def test_l2_matches_definition():
    plan = torch.zeros(6, 2)
    gt = torch.zeros(6, 2)
    gt[:, 0] = 1.0  # constant 1 m offset every step
    mask = torch.ones(6)
    m = PlanningMetric()
    m.update(plan, gt, mask, [np.zeros((0, 5), np.float32)] * 6)
    res = m.compute()
    # per-step L2 = 1.0 -> cumulative average at every step = 1.0 -> avg = 1.0
    assert abs(res["L2"]["avg"] - 1.0) < 1e-6
    assert abs(res["L2"]["at_3s"] - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# collision detection
# --------------------------------------------------------------------------- #
def _straight_plan(dx=2.0):
    p = torch.zeros(6, 2)
    p[:, 0] = torch.arange(1, 7) * dx
    return p


def test_collision_hit_and_miss():
    plan = _straight_plan()
    gt = _straight_plan() + 100.0  # GT far away -> no GT collision
    mask = torch.ones(6)

    # a box sitting exactly on the plan's 3rd step -> collision
    boxes_hit = [np.zeros((0, 5), np.float32) for _ in range(6)]
    boxes_hit[2] = np.array([[float(plan[2, 0]), float(plan[2, 1]), 0.0, 4.0, 1.8]], np.float32)
    m = PlanningMetric()
    m.update(plan, gt, mask, boxes_hit)
    assert m.compute()["coll_any_3s"] == 1.0

    # a box 50 m to the side -> no collision
    boxes_miss = [np.zeros((0, 5), np.float32) for _ in range(6)]
    boxes_miss[2] = np.array([[float(plan[2, 0]), 50.0, 0.0, 4.0, 1.8]], np.float32)
    m2 = PlanningMetric()
    m2.update(plan, gt, mask, boxes_miss)
    assert m2.compute()["coll_any_3s"] == 0.0


def test_gt_collision_subtracted():
    # plan == gt, both pass through the same box -> the planned collision is one the
    # GT plan also incurs, so coll (planned-minus-GT) must be 0.
    plan = _straight_plan()
    gt = _straight_plan()
    mask = torch.ones(6)
    boxes = [np.zeros((0, 5), np.float32) for _ in range(6)]
    boxes[2] = np.array([[float(plan[2, 0]), 0.0, 0.0, 4.0, 1.8]], np.float32)
    m = PlanningMetric()
    m.update(plan, gt, mask, boxes)
    res = m.compute()
    assert res["coll"]["avg"] == 0.0          # subtracted
    assert res["coll_gt"]["avg"] > 0.0        # but GT itself collided


# --------------------------------------------------------------------------- #
# planner: determinism, model-independent prior, collision avoidance
# --------------------------------------------------------------------------- #
def _toy_library():
    # straight command: two anchors, one going straight, one veering +y
    straight = torch.tensor([
        [[1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.]],    # centred
        [[1., 10.], [2., 10.], [3., 10.], [4., 10.], [5., 10.], [6., 10.]],  # offset lane
    ])
    proto = straight[0]
    lib = {"meta": {}, "commands": {
        "straight": {"anchors": straight, "prototype": proto},
        "left": {"anchors": straight, "prototype": proto},
        "right": {"anchors": straight, "prototype": proto},
    }}
    return lib


def test_planner_deterministic_and_prior():
    lib = _toy_library()
    M, H = 3, 6
    modes = torch.full((M, H, 2), 999.0)   # neighbours far away -> collision term ~0
    scores = torch.zeros(M)
    p1 = olp.plan(modes, scores, "straight", lib)
    p2 = olp.plan(modes, scores, "straight", lib)
    assert torch.allclose(p1, p2)                       # deterministic
    # with no danger, the prior wins -> the centred anchor (== prototype)
    assert torch.allclose(p1, lib["commands"]["straight"]["anchors"][0])


def test_cv_prior_is_speed_aware():
    # anchors at two speeds; CV prior from a fast ego must pick the matching-speed anchor
    fast = torch.stack([torch.arange(1, 7) * 4.0, torch.zeros(6)], dim=-1)   # 4 m/step
    slow = torch.stack([torch.arange(1, 7) * 1.0, torch.zeros(6)], dim=-1)   # 1 m/step
    anchors = torch.stack([slow, fast])
    lib = {"meta": {}, "commands": {"straight": {"anchors": anchors, "prototype": slow}}}
    # ego moving forward at 8 m/s -> CV rollout = 8*0.5=4 m/step -> matches 'fast'
    chosen = olp.plan(torch.full((3, 6, 2), 999.0), torch.zeros(3), "straight",
                      lib, ego_vel=torch.tensor([8.0, 0.0]))
    assert torch.allclose(chosen, fast)


def test_planner_avoids_predicted_danger():
    lib = _toy_library()
    M, H = 3, 6
    # one high-score neighbour mode parked right on the centred anchor's path
    modes = torch.full((M, H, 2), 999.0)
    modes[0] = lib["commands"]["straight"]["anchors"][0]  # overlaps centred anchor
    scores = torch.tensor([5.0, -5.0, -5.0])             # mode 0 dominant
    chosen = olp.plan(modes, scores, "straight", lib, w_imit=1.0, w_coll=20.0)
    # collision term should push the planner off the centred anchor onto the veering one
    assert torch.allclose(chosen, lib["commands"]["straight"]["anchors"][1])


# --------------------------------------------------------------------------- #
# get_yaw sanity
# --------------------------------------------------------------------------- #
def test_get_yaw_forward():
    traj = torch.zeros(6, 2)
    traj[:, 0] = torch.arange(1, 7)   # moving along +x -> yaw ~ 0
    yaw = get_yaw(traj)
    assert torch.allclose(yaw, torch.zeros(6), atol=1e-5)


def test_get_yaw_stationary_defaults_zero():
    traj = torch.zeros(6, 2)          # no movement
    yaw = get_yaw(traj)
    assert torch.allclose(yaw, torch.zeros(6))


# --------------------------------------------------------------------------- #
# HC mask alignment by token
# --------------------------------------------------------------------------- #
def test_hc_set_by_token(tmp_path):
    import json
    rec = {"tokens": ["a", "b", "c"], "labels": ["cruising", "high_conflict", "cruising"]}
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(rec))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_phase2_7_openloop_planning import load_hc_set
    hc = load_hc_set(str(p))
    assert hc == {"b"}
