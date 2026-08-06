"""Local replication of SparseDrive's nuScenes open-loop ``PlanningMetric``.

Mirrors ``SparseDrive/.../planning/planning_eval.py``:
  * L2@0.5..3.0 s = per-step ||plan - gt_ego|| (masked), reported as the
    cumulative average over steps; the headline ``avg`` = mean of the 1/2/3 s
    cumulative averages.
  * Collision rate = ego footprint (4.084 x 1.85, UniAD-style +0.5 m forward
    offset) intersecting any agent box at each future step; the planned-plan
    collisions that the *GT-ego* plan already incurs are subtracted
    (``logical_and(box_coll, ~gt_box_coll)``), exactly as SparseDrive does.
    Reported the same cumulative-average way, plus a simpler "any collision
    within 3 s" rate for interpretability.

Two intentional, numerically-equivalent deviations from SparseDrive, documented
for the defense:
  1. No ``trajs[...,0] = -trajs[...,0]`` flip. SparseDrive flips x to reconcile
     its planner frame with its box frame; here every quantity (ego plan, GT ego,
     agent boxes) lives in the *same* npz current-ego frame (x forward, y left),
     so the flip is unnecessary. L2 is flip-invariant; collision is correct as
     long as boxes and plan share one frame, which they do.
  2. Separating-Axis-Theorem overlap (reusing the Phase 2.4-tested
     ``_rect_corners`` / ``_boxes_overlap``) instead of shapely ``intersects``.
     For convex rectangles the boolean is identical; this drops the shapely dep.
"""
import numpy as np
import torch

from pa_loss_stopgrad.carriers.config import EGO_SIZE_LWH
from pa_loss_stopgrad.pa_loss.pa_loss_gameformer import _boxes_overlap, _rect_corners

# Standard nuScenes planning-metric ego footprint (matches SparseDrive: H=4.084 along
# heading, W=1.85 across). Note this W differs from EGO_SIZE_LWH[1]=1.730; the recognised
# planning number uses 1.85, so we keep 1.85 for the metric and note it.
EGO_PLAN_LW = (4.084, 1.85)
EGO_OFFSET = 0.5  # UniAD forward offset along heading
DEFAULT_AGENT_LW = (4.0, 1.8)


def get_yaw(traj_xy):
    """Per-step heading from trajectory tangent in the npz ego frame (x fwd, y left).

    traj_xy [T,2] -> [T]. Near-stationary (<0.5 m total travel) defaults to 0
    (facing forward), the correct rest heading in this frame.
    """
    traj = traj_xy if torch.is_tensor(traj_xy) else torch.as_tensor(traj_xy, dtype=torch.float32)
    T = traj.shape[0]
    if float(torch.linalg.norm(traj[-1] - traj[0])) < 0.5:
        return traj.new_zeros(T)
    pad = torch.cat([traj.new_zeros((1, 2)), traj], dim=0)  # [T+1,2]
    yaw = traj.new_zeros(T + 1)
    yaw[1:-1] = torch.atan2(pad[2:, 1] - pad[:-2, 1], pad[2:, 0] - pad[:-2, 0])
    yaw[-1] = torch.atan2(pad[-1, 1] - pad[-2, 1], pad[-1, 0] - pad[-2, 0])
    return yaw[1:]


def _ego_corners(traj_xy):
    """Ego footprint corners per step [T,4,2] with the UniAD +0.5m forward offset."""
    traj = traj_xy if torch.is_tensor(traj_xy) else torch.as_tensor(traj_xy, dtype=torch.float32)
    yaw = get_yaw(traj)
    cx = traj[:, 0] + EGO_OFFSET * torch.cos(yaw)
    cy = traj[:, 1] + EGO_OFFSET * torch.sin(yaw)
    center = torch.stack([cx, cy], dim=-1)                       # [T,2]
    lw = traj.new_tensor(EGO_PLAN_LW)
    return _rect_corners(center, yaw, lw)                        # [T,4,2]


def _agent_corners(centers, headings, lw):
    """centers [T,K,2], headings [T,K], lw [T,K,2] -> [T,K,4,2]."""
    return _rect_corners(centers, headings, lw)


class PlanningMetric:
    """Accumulate L2 and collisions over samples (one agent-box source).

    update() takes a single sample's plan/gt (already trimmed to n_future) and a
    per-step list of agent boxes (x, y, heading, L, W) in the npz ego frame.
    """

    def __init__(self, n_future=6):
        self.n_future = n_future
        self.reset()

    def reset(self):
        n = self.n_future
        self.L2 = torch.zeros(n)
        self.obj_col = torch.zeros(n)        # GT-ego plan collisions (context)
        self.obj_box_col = torch.zeros(n)    # planned-plan collisions (minus GT)
        self.any_col = 0.0                   # samples with >=1 planned collision in 3s
        self.total = 0

    @staticmethod
    def _step_collisions(traj_xy, agent_boxes):
        """Per-step bool [T]: does the ego footprint hit any agent box that step?"""
        T = traj_xy.shape[0]
        ego = _ego_corners(traj_xy)                              # [T,4,2]
        coll = torch.zeros(T, dtype=torch.bool)
        for t in range(T):
            boxes = agent_boxes[t]
            if boxes is None or len(boxes) == 0:
                continue
            b = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
            b = b.float()
            valid = b[:, 3:5].abs().sum(-1) > 1e-3               # drop padded zero boxes
            b = b[valid]
            if len(b) == 0:
                continue
            ac = _agent_corners(b[:, :2], b[:, 2], b[:, 3:5])    # [K,4,2]
            ego_t = ego[t][None].expand(b.shape[0], 4, 2)        # [K,4,2]
            coll[t] = bool(_boxes_overlap(ego_t, ac).any())
        return coll

    @torch.no_grad()
    def update(self, plan_xy, gt_xy, gt_mask, agent_boxes):
        """plan_xy/gt_xy [n_future,2]; gt_mask [n_future] (1=valid); agent_boxes: list len n_future."""
        plan = plan_xy if torch.is_tensor(plan_xy) else torch.as_tensor(plan_xy, dtype=torch.float32)
        gt = gt_xy if torch.is_tensor(gt_xy) else torch.as_tensor(gt_xy, dtype=torch.float32)
        plan, gt = plan.float(), gt.float()
        m = (gt_mask if torch.is_tensor(gt_mask) else torch.as_tensor(gt_mask)).float()

        l2 = torch.sqrt((((plan - gt) ** 2) * m[:, None]).sum(-1))   # [n_future]
        gt_coll = self._step_collisions(gt, agent_boxes)
        box_coll = self._step_collisions(plan, agent_boxes)
        box_coll = torch.logical_and(box_coll, torch.logical_not(gt_coll))

        self.L2 += l2
        self.obj_col += gt_coll.long()
        self.obj_box_col += box_coll.long()
        self.any_col += float(bool(box_coll.any()))
        self.total += 1

    def compute(self):
        if self.total == 0:
            return None
        l2 = (self.L2 / self.total).tolist()
        box_col = (self.obj_box_col / self.total).tolist()
        gt_col = (self.obj_col / self.total).tolist()
        return {
            "n": self.total,
            "L2": _summary(l2, pct=False),
            "coll": _summary(box_col, pct=True),       # headline (planned, minus GT)
            "coll_gt": _summary(gt_col, pct=True),     # GT-ego collisions (context)
            "coll_any_3s": self.any_col / self.total,  # fraction with >=1 planned collision
        }


def _summary(per_step, pct):
    """Cumulative-average at 0.5..3.0s + headline avg=mean(1s,2s,3s), like SparseDrive."""
    cum = [float(np.mean(per_step[: i + 1])) for i in range(len(per_step))]
    idx = [1, 3, 5]  # 1s, 2s, 3s
    avg = float(np.mean([cum[i] for i in idx if i < len(cum)]))
    scale = 100.0 if pct else 1.0
    return {
        "per_step_cumavg": [c * scale for c in cum],
        "at_1s": cum[1] * scale if len(cum) > 1 else float("nan"),
        "at_2s": cum[3] * scale if len(cum) > 3 else float("nan"),
        "at_3s": cum[5] * scale if len(cum) > 5 else float("nan"),
        "avg": avg * scale,
    }
