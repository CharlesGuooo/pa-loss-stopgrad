"""Phase 2.8: frozen, shared, prediction-sensitive ego planner (learned cost over anchors).

Motivation
----------
The Phase 2.7 fixed library planner was dominated by the model-independent CV
imitation prior, so its hand-coded collision term almost never flipped the chosen
anchor -> the open-loop planning metrics were insensitive to prediction quality
(an honest null). This module keeps EVERYTHING that made 2.7 confounding-free
(the same 66-anchor GT-ego library, the same model-independent CV imitation prior,
the same nuScenes ``PlanningMetric``) but replaces ONLY the hand-coded collision
term with a small learned, prediction-conditioned interaction head:

    cost(anchor_k) = imit_k                      (model-independent CV-prior ADE)
                   + risk_scale * risk_k          (LEARNED, function of predictions)

    risk_k = sum_m softmax(score)_m * softplus(MLP(feat(anchor_k, mode_m)))

The planner is trained ONCE on a neutral prediction source (the baseline Phase 2.2
predictor, the common ancestor of every PA-Loss variant), then FROZEN. At eval the
identical frozen weights are applied to every variant's predictions, so any change
in L2 / collision is attributable purely to prediction quality -- exactly the clean
attribution Phase 2.7 had, now with a planner that actually reacts to predictions.

Structural prediction-sensitivity: the imitation term is fixed and the ONLY learned,
prediction-dependent quantity is ``risk_k``. With ``risk_scale=0`` (or no neighbours)
the planner degenerates to the pure CV-prior selection. Phase 2.8a's Gate G-A checks
empirically that the trained head makes the chosen anchor flip on conflict frames.

All quantities are absolute positions in the npz current-ego frame (x forward,
y left), the same frame as ``gt_future_states`` and the model's neighbour modes.
Deterministic at eval (hard argmin); differentiable at train (soft argmin).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pa_loss_stopgrad.planners.openloop_planner import (  # reuse 2.7 primitives (DRY)
    COMMANDS,
    PLAN_HORIZON,
    cv_rollout,
    derive_command,
    load_library,
)

DIST_SCALE = 10.0  # normalise metre distances into ~O(1) before the MLP


class LearnedAnchorPlanner(nn.Module):
    """Learned cost over the fixed GT-ego anchor library.

    The only learnable component is a per-(anchor, mode) risk MLP. The imitation
    term (CV-prior ADE) is fixed and model-independent, so the learned part is the
    only thing that depends on the predicted neighbour modes.
    """

    def __init__(self, hidden=32, sigmas=(1.0, 2.0, 4.0)):
        super().__init__()
        self.sigmas = tuple(float(s) for s in sigmas)
        self.n_feat = 5 + len(self.sigmas) + 1  # dists(5) + prox(len sigmas) + prob(1)
        self.head = nn.Sequential(
            nn.Linear(self.n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    # ------------------------------------------------------------------ #
    def _features(self, anchors, nb, prob):
        """Per-(anchor, mode) features. Broadcasts over optional leading batch dims.

        anchors [..., K, H, 2]; nb [..., M, H, 2]; prob [..., M] -> feat [..., K, M, n_feat].
        """
        a = anchors.unsqueeze(-3)                          # [...,K,1,H,2]
        m = nb.unsqueeze(-4)                               # [...,1,M,H,2]
        dist = (a - m).norm(dim=-1)                        # [...,K,M,H]
        d_min = dist.min(dim=-1).values                    # [...,K,M]
        d_mean = dist.mean(dim=-1)
        d_start = dist[..., 0]
        d_end = dist[..., -1]
        closing = d_start - d_end                          # >0 => neighbour approaching anchor
        prox = [torch.exp(-(d_min ** 2) / (2.0 * s * s)) for s in self.sigmas]
        prob_km = prob.unsqueeze(-2).expand_as(d_min)      # [...,K,M]
        feats = torch.stack(
            [d_min / DIST_SCALE, d_mean / DIST_SCALE, d_start / DIST_SCALE,
             d_end / DIST_SCALE, closing / DIST_SCALE, *prox, prob_km], dim=-1)
        return feats                                       # [...,K,M,n_feat]

    def anchor_costs(self, anchors, neighbor_modes, neighbor_scores, prior, risk_scale=1.0):
        """Return (imit, risk, cost), each [..., K]. Shapes broadcast over batch dims.

        anchors [...,K,H,2]; neighbor_modes [...,M,Tn,2] (>=H); neighbor_scores [...,M];
        prior [...,H,2].
        """
        H = anchors.shape[-2]
        nb = neighbor_modes[..., :H, :]
        prob = F.softmax(neighbor_scores, dim=-1)                       # [...,M]
        feats = self._features(anchors, nb, prob)                       # [...,K,M,n_feat]
        risk_km = F.softplus(self.head(feats).squeeze(-1))              # [...,K,M]
        risk = (risk_km * prob.unsqueeze(-2)).sum(dim=-1)               # [...,K]
        imit = (anchors - prior.unsqueeze(-3)).norm(dim=-1).mean(dim=-1)  # [...,K]
        cost = imit + risk_scale * risk
        return imit, risk, cost

    def forward(self, anchors, neighbor_modes, neighbor_scores, prior,
                temperature=0.5, risk_scale=1.0):
        """Differentiable soft plan for training. Returns (soft_plan [...,H,2], cost [...,K])."""
        imit, risk, cost = self.anchor_costs(anchors, neighbor_modes, neighbor_scores, prior, risk_scale)
        w = F.softmax(-cost / temperature, dim=-1)                      # [...,K]
        soft_plan = (w.unsqueeze(-1).unsqueeze(-1) * anchors).sum(dim=-3)  # [...,H,2]
        return soft_plan, cost

    # ------------------------------------------------------------------ #
    def _device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def plan(self, neighbor_modes, neighbor_scores, command, library,
             ego_vel=None, risk_scale=1.0, return_idx=False):
        """Hard-argmin ego plan for eval. Mirrors ``openloop_planner.plan`` signature.

        Returns ego_plan [PLAN_HORIZON, 2] (or (plan, anchor_idx) when return_idx).
        """
        dev = self._device()
        cmd = library["commands"][command]
        anchors = cmd["anchors"].to(dev).float()                        # [K,H,2]
        H = anchors.shape[-2]
        if ego_vel is not None:
            prior = cv_rollout(ego_vel, H).to(dev)                      # [H,2]
        else:
            prior = cmd["prototype"].to(dev).float()
        nb = neighbor_modes.to(dev).float()
        sc = neighbor_scores.to(dev).float()
        _, _, cost = self.anchor_costs(anchors, nb, sc, prior, risk_scale)
        best = int(torch.argmin(cost))
        plan = anchors[best].clone()
        return (plan, best) if return_idx else plan


def load_planner(path, map_location="cpu"):
    """Load a frozen LearnedAnchorPlanner from a checkpoint saved by train_shared_planner."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt.get("config", {})
    planner = LearnedAnchorPlanner(
        hidden=cfg.get("hidden", 32), sigmas=tuple(cfg.get("sigmas", (1.0, 2.0, 4.0))))
    planner.load_state_dict(ckpt["model_states"])
    planner.eval()
    return planner, ckpt


__all__ = [
    "LearnedAnchorPlanner", "load_planner", "load_library",
    "derive_command", "cv_rollout", "PLAN_HORIZON", "COMMANDS",
]
