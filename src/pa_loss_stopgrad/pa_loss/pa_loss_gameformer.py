"""Planning-Aware Loss attached to the GameFormer carrier (Phase 2.3 / 2.4).

Plug-in port of the Phase 1.8 ``RiskProbePlanningLoss`` mechanism onto a strong
multimodal predictor. A differentiable collision proxy between the (frozen,
oracle) ego GT plan and the predicted neighbor modes, with mode-level
risk-weighted gradient allocation, adapted to GameFormer's *absolute* ego-frame
joint outputs (no delta cumsum).

GameFormer predicts 2 agents jointly: index 0 = ego, index 1 = neighbor0.
  outputs[f'level_{k}_interactions']: [B, 2, M, T, 4] (mu_x, mu_y, log_sx, log_sy)
  outputs[f'level_{k}_scores']:       [B, 2, M]

Risk object  = neighbor0 predicted modes (index 1).
Ego plan ref = ego_future[..., :2] (GT, fixed) -> gradients flow only through the
predicted neighbor modes, exactly mirroring Phase 1.8.

Phase 2.4 adds richer differentiable risk functions (``risk_kind``):
  gauss  : exp(-min_t d^2 / 2 sigma^2)              (Phase 2.3 baseline)
  ttc    : closing-speed time-to-collision          (Master Plan 3.7.1)
  boxiou : oriented-box overlap (Gaussian Bhattacharyya proxy)
plus a training-proxy-independent safety metric (near-miss rate / min-sep /
oriented-box overlap) for fair cross-kind comparison.

L_imitation stays GameFormer's own ``level_k_loss_nuscenes``; the fine-tune
script forms L_total = L_imit + lambda_plan * l_plan.
"""
import math

import torch
import torch.nn.functional as F

from pa_loss_stopgrad.carriers.config import DT, EGO_SIZE_LWH

EGO_LW = (EGO_SIZE_LWH[0], EGO_SIZE_LWH[1])  # (length, width)
DEFAULT_VEH_LW = (4.0, 1.8)
RISK_KINDS = ("gauss", "ttc", "boxiou")


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _step_valid(ego_future, neighbor_future):
    ego_valid = ego_future[..., :2].abs().sum(-1) != 0       # [B,T]
    nb_valid = neighbor_future[..., :2].abs().sum(-1) != 0   # [B,T]
    return ego_valid & nb_valid                              # [B,T]


def _neighbor_best_mode(neighbor_modes, neighbor_gt_xy, step_valid):
    """GT-closest mode of the neighbor (imitation WTA target). Detached."""
    d = (neighbor_modes - neighbor_gt_xy[:, None]).norm(dim=-1)  # [B,M,T]
    m = step_valid[:, None].float()
    cnt = m.sum(-1).clamp(min=1.0)
    ade = (d * m).sum(-1) / cnt
    return ade.argmin(dim=-1)  # [B]


def _seg_heading(traj, gt_heading, speed_thr=0.1):
    """Per-step heading from trajectory tangent, gated to GT below speed_thr.

    traj [..., T, 2]; gt_heading [..., T] (broadcastable). Returns [..., T].
    """
    v = traj[..., 1:, :] - traj[..., :-1, :]
    v = torch.cat([v[..., :1, :], v], dim=-2)  # forward-fill first step
    speed = v.norm(dim=-1)
    tangent = torch.atan2(v[..., 1], v[..., 0])
    use = speed > speed_thr
    return torch.where(use, tangent, gt_heading)


def _cov_terms(theta, hl, hw):
    """Oriented 2D covariance from half-length/half-width. Returns (s00,s01,s11)."""
    c, s = torch.cos(theta), torch.sin(theta)
    l2, w2 = hl ** 2, hw ** 2
    s00 = c * c * l2 + s * s * w2
    s11 = s * s * l2 + c * c * w2
    s01 = c * s * (l2 - w2)
    return s00, s01, s11


# --------------------------------------------------------------------------- #
# risk functions: each returns (cost_m [B,M], danger_m [B,M]) within horizon H
#   cost_m  : in [0,1], higher = more dangerous (enters L_plan)
#   danger_m: higher = more dangerous (drives risk-softmax alpha)
# --------------------------------------------------------------------------- #
def _gather_horizon(neighbor_modes, ego_plan, step_valid, horizon):
    H = min(horizon, neighbor_modes.shape[2], ego_plan.shape[1])
    return neighbor_modes[:, :, :H], ego_plan[:, :H], step_valid[:, :H]


def _risk_gauss(nb, ego, sv, sigma, **_):
    dist = (nb - ego[:, None]).norm(dim=-1)               # [B,M,H]
    big = torch.full_like(dist, 1e6)
    dist = torch.where(sv[:, None].expand_as(dist), dist, big)
    min_dist = dist.min(dim=-1).values                    # [B,M]
    cost = torch.exp(-(min_dist ** 2) / (2.0 * sigma ** 2))
    return cost, -min_dist


def _risk_ttc(nb, ego, sv, tau_ttc=2.0, soft_tau=0.5, **_):
    r = nb - ego[:, None]                                 # [B,M,H,2]
    dist = r.norm(dim=-1)                                 # [B,M,H]
    rdot = (r[:, :, 1:] - r[:, :, :-1]) / DT
    rdot = torch.cat([rdot[:, :, :1], rdot], dim=2)       # [B,M,H,2]
    closing = -(r * rdot).sum(-1) / (dist + 1e-3)         # [B,M,H], + = approaching
    ttc = dist / closing.clamp(min=1e-2)
    big = torch.full_like(ttc, 1e6)
    ttc = torch.where((closing > 0) & sv[:, None].expand_as(ttc), ttc, big)
    # soft-min over time
    ttc_m = -soft_tau * torch.logsumexp(-ttc / soft_tau, dim=-1)  # [B,M]
    cost = torch.exp(-ttc_m / tau_ttc)
    return cost, -ttc_m


def _risk_boxiou(nb, ego, sv, nb_heading, ego_heading, nb_lw, ego_lw, **_):
    # headings within horizon
    H = nb.shape[2]
    th_nb = nb_heading[:, :, :H]                          # [B,M,H]
    th_ego = ego_heading[:, :H]                           # [B,H]
    hl_n, hw_n = nb_lw[..., 0:1] / 2, nb_lw[..., 1:2] / 2  # [B,1]
    hl_e, hw_e = ego_lw[0] / 2, ego_lw[1] / 2
    a00, a01, a11 = _cov_terms(th_nb, hl_n[:, None], hw_n[:, None])      # [B,M,H]
    b00, b01, b11 = _cov_terms(th_ego, hl_e, hw_e)                       # [B,H]
    b00, b01, b11 = b00[:, None], b01[:, None], b11[:, None]             # [B,1,H]
    # mixture covariance
    m00, m01, m11 = (a00 + b00) / 2, (a01 + b01) / 2, (a11 + b11) / 2
    det_m = (m00 * m11 - m01 ** 2).clamp(min=1e-6)
    det_a = (a00 * a11 - a01 ** 2).clamp(min=1e-6)
    det_b = (b00 * b11 - b01 ** 2).clamp(min=1e-6)
    d = nb - ego[:, None]                                 # [B,M,H,2]
    dx, dy = d[..., 0], d[..., 1]
    quad = (m11 * dx ** 2 - 2 * m01 * dx * dy + m00 * dy ** 2) / det_m
    db = quad / 8.0 + 0.5 * torch.log(det_m / torch.sqrt(det_a * det_b))
    bc = torch.exp(-db.clamp(min=0.0))                    # [B,M,H], in (0,1]
    bc = torch.where(sv[:, None].expand_as(bc), bc, torch.zeros_like(bc))
    # max overlap over time (bounded in (0,1]; grad routes through the worst step)
    cost = bc.max(dim=-1).values
    return cost, cost


# --------------------------------------------------------------------------- #
# main loss
# --------------------------------------------------------------------------- #
def pa_loss_gameformer(
    outputs,
    ego_future,
    neighbor_future,
    *,
    level,
    mode_weighting="risk_softmax",
    stop_gradient=False,
    risk_kind="gauss",
    sigma=5.0,
    tau=5.0,
    tau_ttc=2.0,
    collision_horizon=6,
    neighbor_size=None,
    ego_size=EGO_LW,
):
    """Differentiable planning-risk term + eval proxies on the GameFormer outputs.

    Returns dict with trainable ``l_plan`` plus no-grad gauss eval proxies
    (kept identical to Phase 2.3 for comparability across risk kinds).
    """
    if risk_kind not in RISK_KINDS:
        raise ValueError(f"unknown risk_kind {risk_kind}")
    inter = outputs[f"level_{level}_interactions"]
    scores = outputs[f"level_{level}_scores"]
    neighbor_modes = inter[:, 1, :, :, :2]          # [B,M,T,2]
    ego_plan = ego_future[..., :2]                  # [B,T,2]
    nb_scores = scores[:, 1]                         # [B,M]
    step_valid = _step_valid(ego_future, neighbor_future)

    plan_modes = neighbor_modes.detach() if stop_gradient else neighbor_modes
    nb, ego, sv = _gather_horizon(plan_modes, ego_plan, step_valid, collision_horizon)
    has_valid = sv.any(dim=-1)

    if risk_kind == "gauss":
        cost, danger = _risk_gauss(nb, ego, sv, sigma=sigma)
    elif risk_kind == "ttc":
        cost, danger = _risk_ttc(nb, ego, sv, tau_ttc=tau_ttc)
    else:  # boxiou
        M, H = nb.shape[1], nb.shape[2]
        nb_gt_h = neighbor_future[..., 2][:, None, :].expand(-1, M, -1)  # [B,M,T] GT heading
        nb_heading = _seg_heading(plan_modes, nb_gt_h)                   # [B,M,T]
        ego_heading = ego_future[..., 2]                                 # [B,T]
        if neighbor_size is None:
            neighbor_size = ego_future.new_tensor(DEFAULT_VEH_LW).expand(ego_future.shape[0], 2)
        ego_lw_t = ego_future.new_tensor(ego_size)
        cost, danger = _risk_boxiou(nb, ego, sv, nb_heading[:, :, :H], ego_heading[:, :H],
                                    neighbor_size, ego_lw_t)

    if mode_weighting == "wta":
        bm = _neighbor_best_mode(neighbor_modes.detach(), neighbor_future[..., :2], step_valid)
        alpha = F.one_hot(bm, num_classes=neighbor_modes.shape[1]).to(cost.dtype)
    else:  # risk_softmax
        alpha = F.softmax(danger / tau, dim=-1)

    valid_f = has_valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    per_sample = (alpha * cost).sum(dim=-1)
    l_plan = (per_sample * valid_f).sum() / n_valid

    with torch.no_grad():
        # gauss collision proxy kept for cross-phase / cross-kind comparability
        g_nb, g_ego, g_sv = _gather_horizon(neighbor_modes, ego_plan, step_valid, collision_horizon)
        g_cost, _ = _risk_gauss(g_nb, g_ego, g_sv, sigma=sigma)
        worst = g_cost.max(dim=-1).values
        collision_proxy = (worst * valid_f).sum() / n_valid
        top_mode = nb_scores.argmax(dim=-1)
        dist = (g_nb - g_ego[:, None]).norm(dim=-1)
        big = torch.full_like(dist, 1e6)
        dist = torch.where(g_sv[:, None].expand_as(dist), dist, big)
        top_min = dist.gather(1, top_mode[:, None, None].expand(-1, 1, dist.shape[2])).squeeze(1).min(-1).values
        top_cost = torch.exp(-(top_min ** 2) / (2.0 * sigma ** 2))
        topmode_collision = (top_cost * valid_f).sum() / n_valid

    return {
        "l_plan": l_plan,
        "collision_proxy": collision_proxy,
        "topmode_collision": topmode_collision,
        "n_valid": n_valid,
    }


@torch.no_grad()
def collision_proxy_per_sample(
    outputs, ego_future, neighbor_future, *, level, sigma=5.0, collision_horizon=6
):
    """Per-sample worst-mode gauss collision proxy (Phase 2.3 eval). (worst [B], has_valid [B])."""
    inter = outputs[f"level_{level}_interactions"]
    neighbor_modes = inter[:, 1, :, :, :2]
    ego_plan = ego_future[..., :2]
    step_valid = _step_valid(ego_future, neighbor_future)
    nb, ego, sv = _gather_horizon(neighbor_modes, ego_plan, step_valid, collision_horizon)
    cost, _ = _risk_gauss(nb, ego, sv, sigma=sigma)
    return cost.max(dim=-1).values, sv.any(dim=-1)


# --------------------------------------------------------------------------- #
# training-proxy-independent safety metric (top-score neighbor mode vs ego plan)
# --------------------------------------------------------------------------- #
def _rect_corners(center, heading, lw):
    """center [...,2], heading [...], lw [...,2] -> corners [...,4,2]."""
    hl, hw = lw[..., 0] / 2, lw[..., 1] / 2
    c, s = torch.cos(heading), torch.sin(heading)
    # local corner offsets (FL, FR, RR, RL)
    sx = torch.stack([hl, hl, -hl, -hl], dim=-1)   # [...,4]
    sy = torch.stack([hw, -hw, -hw, hw], dim=-1)
    gx = center[..., 0:1] + sx * c[..., None] - sy * s[..., None]
    gy = center[..., 1:2] + sx * s[..., None] + sy * c[..., None]
    return torch.stack([gx, gy], dim=-1)


def _boxes_overlap(c1, c2):
    """SAT overlap test for two convex rect corner sets c1,c2 [...,4,2] -> bool [...]."""
    overlap = torch.ones(c1.shape[:-2], dtype=torch.bool, device=c1.device)
    for corners in (c1, c2):
        edges = corners.roll(-1, dims=-2) - corners        # [...,4,2]
        # axis = edge normal
        ax = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)  # [...,4,2]
        for k in range(4):
            axis = ax[..., k, :]                            # [...,2]
            p1 = (c1 * axis[..., None, :]).sum(-1)          # [...,4]
            p2 = (c2 * axis[..., None, :]).sum(-1)
            sep = (p1.min(-1).values > p2.max(-1).values) | (p2.min(-1).values > p1.max(-1).values)
            overlap = overlap & (~sep)
    return overlap


@torch.no_grad()
def safety_metrics_per_sample(
    outputs, ego_future, neighbor_future, *, level, neighbor_size=None, ego_size=EGO_LW,
    collision_horizon=6,
):
    """Proxy-independent safety from the TOP-score neighbor mode vs GT ego plan.

    Returns dict of per-sample tensors: min_sep [B], box_overlap [B] (0/1),
    has_valid [B]. near-miss rates are thresholds applied on min_sep downstream.
    """
    inter = outputs[f"level_{level}_interactions"]
    scores = outputs[f"level_{level}_scores"]
    neighbor_modes = inter[:, 1, :, :, :2]          # [B,M,T,2]
    ego_plan = ego_future[..., :2]
    step_valid = _step_valid(ego_future, neighbor_future)
    nb, ego, sv = _gather_horizon(neighbor_modes, ego_plan, step_valid, collision_horizon)
    B, M, H, _ = nb.shape

    top_mode = scores[:, 1].argmax(dim=-1)          # [B]
    top = nb.gather(1, top_mode[:, None, None, None].expand(-1, 1, H, 2)).squeeze(1)  # [B,H,2]
    dist = (top - ego).norm(dim=-1)                 # [B,H]
    big = torch.full_like(dist, 1e6)
    dist = torch.where(sv, dist, big)
    min_sep = dist.min(dim=-1).values               # [B]
    has_valid = sv.any(dim=-1)

    # oriented-box overlap at the closest step, using GT headings + sizes
    closest = dist.argmin(dim=-1)                    # [B]
    top_c = top.gather(1, closest[:, None, None].expand(-1, 1, 2)).squeeze(1)   # [B,2]
    ego_c = ego.gather(1, closest[:, None, None].expand(-1, 1, 2)).squeeze(1)
    nb_head_gt = neighbor_future[..., 2].gather(1, closest[:, None]).squeeze(1)  # [B]
    ego_head_gt = ego_future[..., 2].gather(1, closest[:, None]).squeeze(1)
    if neighbor_size is None:
        neighbor_size = ego_future.new_tensor(DEFAULT_VEH_LW).expand(B, 2)
    ego_lw_t = ego_future.new_tensor(ego_size).expand(B, 2)
    overlap = _boxes_overlap(
        _rect_corners(top_c, nb_head_gt, neighbor_size),
        _rect_corners(ego_c, ego_head_gt, ego_lw_t),
    ).float()
    return {"min_sep": min_sep, "box_overlap": overlap, "has_valid": has_valid}
