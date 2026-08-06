"""Fixed, model-independent trajectory-library open-loop planner (Phase 2.7).

The planner is identical for every model variant: it consumes a model's predicted
neighbor modes + scores and chooses an ego plan from a pre-built GT-ego anchor
library by minimising a fixed cost

    cost(anchor) = w_imit * dev_to_prior(anchor, command)
                 + w_coll * collision(anchor, predicted neighbor modes)

  * ``dev_to_prior`` = ADE between the anchor and the command-conditioned GT-ego
    *prototype* (mean trajectory). The prototype is GT-derived and identical across
    variants, so it keeps L2 plausible and cross-variant comparable -- it does NOT
    use the per-sample GT (no leakage of the answer), only the command class mean.
  * ``collision`` = score-weighted soft proximity to the predicted neighbour modes.
    This is the ONLY term that differs across model variants, so any change in L2 /
    collision is attributable purely to prediction quality. This is where PA-Loss
    should help: better high-conflict neighbour modes -> the planner avoids danger.

All quantities are absolute positions in the npz current-ego frame (x forward,
y left), the same frame as ``gt_future_states`` and the model's neighbour modes.
Pure torch, deterministic, no side effects.
"""
import json

import torch
import torch.nn.functional as F

PLAN_HORIZON = 6
DT = 0.5
COMMANDS = ("left", "straight", "right")


def cv_rollout(ego_vel, horizon=PLAN_HORIZON, dt=DT):
    """Constant-velocity prior in the ego frame from the observed current velocity.

    ego_vel [2] = (vx, vy) at the current step (an *input* feature, identical across
    model variants). Returns [horizon, 2] absolute positions: pos[t] = vel*dt*(t+1).
    """
    v = ego_vel if torch.is_tensor(ego_vel) else torch.as_tensor(ego_vel, dtype=torch.float32)
    steps = torch.arange(1, horizon + 1, dtype=v.dtype, device=v.device)[:, None]  # [H,1]
    return steps * dt * v[None, :]                                                 # [H,2]


def derive_command(gt_ego_fut_xy, lateral_thresh=1.5):
    """left / straight / right from GT-ego lateral (y) offset at 3 s.

    Kept identical to ``build_ego_anchor_library.derive_command`` (asserted by the
    Phase 2.7 unit test). Model-independent: depends only on the GT-ego trajectory.
    """
    t = min(PLAN_HORIZON, gt_ego_fut_xy.shape[0]) - 1
    y_end = float(gt_ego_fut_xy[t, 1])
    if y_end > lateral_thresh:
        return "left"
    if y_end < -lateral_thresh:
        return "right"
    return "straight"


def load_library(path):
    """Load the anchor library JSON into torch tensors keyed by command."""
    with open(path) as f:
        lib = json.load(f)
    out = {"meta": lib.get("meta", {}), "commands": {}}
    for c, d in lib["commands"].items():
        out["commands"][c] = {
            "anchors": torch.tensor(d["anchors"], dtype=torch.float32),      # [K,6,2]
            "prototype": torch.tensor(d["prototype"], dtype=torch.float32),  # [6,2]
        }
    return out


def plan(neighbor_modes, neighbor_scores, command, library,
         ego_vel=None, w_imit=1.0, w_coll=4.0, sigma=2.0):
    """Select the lowest-cost ego anchor. Returns ego_plan [PLAN_HORIZON, 2].

    neighbor_modes  [M, T, 2] predicted neighbour-0 xy (absolute, ego frame)
    neighbor_scores [M]       predicted mode scores (pre-softmax logits)
    command         one of COMMANDS
    library         output of load_library
    ego_vel         [2] observed current ego (vx,vy); when given the imitation
                    prior is the constant-velocity rollout (speed-aware, model-
                    independent). When None it falls back to the command prototype.
    """
    cmd = library["commands"][command]
    anchors = cmd["anchors"]                                  # [K,6,2]
    K, H, _ = anchors.shape
    device = anchors.device

    if ego_vel is not None:
        prior = cv_rollout(ego_vel, H).to(device)            # [H,2] constant-velocity
    else:
        prior = cmd["prototype"].to(device)                  # [H,2] command mean

    nb = neighbor_modes.to(device).float()[:, :H, :]          # [M,H,2]
    scores = neighbor_scores.to(device).float()
    p = F.softmax(scores, dim=-1)                             # [M]

    # imitation: ADE(anchor, fixed prior) -- same prior for all variants
    imit = (anchors - prior[None]).norm(dim=-1).mean(dim=-1)       # [K]

    # collision: score-weighted soft proximity of each anchor to the predicted modes
    # anchors [K,1,H,2] vs nb [1,M,H,2] -> dist [K,M,H]
    dist = (anchors[:, None] - nb[None]).norm(dim=-1)             # [K,M,H]
    prox = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2)).max(dim=-1).values  # [K,M] worst step
    coll = (prox * p[None]).sum(dim=-1)                          # [K]

    cost = w_imit * imit + w_coll * coll                         # [K]
    best = int(torch.argmin(cost))
    return anchors[best].clone()
