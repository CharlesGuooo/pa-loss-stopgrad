"""Phase 2.8 tests: frozen, shared, learned-cost-over-anchors planner.

Covers the properties the defense relies on:
  * determinism at eval (frozen hard-argmin is reproducible)
  * risk_scale=0 degenerates to the model-independent CV-prior selection
    (proves the imitation path is unchanged from 2.7 and prediction-free)
  * prediction actually enters the cost (risk responds to neighbour modes) ->
    the structural basis for Gate G-A
  * frozen-eval equivalence: plan() == hard argmin of anchor_costs' cost
  * no per-sample GT leakage: the returned plan is always a library anchor
  * the soft plan is differentiable w.r.t. the head parameters (trainable)
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_loss_stopgrad.planners import openloop_planner as olp# noqa: E402
from pa_loss_stopgrad.planners.learned_planner import LearnedAnchorPlanner  # noqa: E402


def _toy_library():
    # straight command: centred anchor + an offset-lane anchor (matches 2.7 toy)
    straight = torch.tensor([
        [[1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.]],        # centred
        [[1., 10.], [2., 10.], [3., 10.], [4., 10.], [5., 10.], [6., 10.]],  # offset lane
    ])
    proto = straight[0]
    return {"meta": {}, "commands": {c: {"anchors": straight, "prototype": proto}
                                     for c in ("left", "straight", "right")}}


def _planner(seed=0):
    torch.manual_seed(seed)
    p = LearnedAnchorPlanner(hidden=16)
    p.eval()
    return p


def test_plan_deterministic():
    p = _planner()
    lib = _toy_library()
    modes = torch.full((3, 6, 2), 5.0)
    scores = torch.zeros(3)
    a = p.plan(modes, scores, "straight", lib, ego_vel=torch.tensor([4.0, 0.0]))
    b = p.plan(modes, scores, "straight", lib, ego_vel=torch.tensor([4.0, 0.0]))
    assert torch.allclose(a, b)


def test_risk_scale_zero_is_prior_only():
    # With risk_scale=0 the learned term vanishes -> selection is the CV-prior nearest
    # anchor, identical to the model-independent imitation path (no prediction used).
    p = _planner()
    lib = _toy_library()
    anchors = lib["commands"]["straight"]["anchors"]
    ego_vel = torch.tensor([4.0, 0.0])                      # CV rollout -> 2 m/step, centred
    # a dangerous mode sitting on the centred anchor would matter only if risk_scale>0
    modes = anchors[0].unsqueeze(0).repeat(3, 1, 1)
    scores = torch.tensor([9.0, -9.0, -9.0])
    chosen = p.plan(modes, scores, "straight", lib, ego_vel=ego_vel, risk_scale=0.0)
    # prior-only -> the centred anchor (closest to the forward CV rollout) wins
    imit = (anchors - olp.cv_rollout(ego_vel, 6)[None]).norm(dim=-1).mean(dim=-1)
    assert torch.allclose(chosen, anchors[int(imit.argmin())])
    assert torch.allclose(chosen, anchors[0])


def test_prediction_enters_cost():
    # risk must be a non-trivial function of the predicted modes (seeded, deterministic):
    # placing a mode on anchor 0 vs far away changes the risk vector.
    p = _planner(seed=1)
    lib = _toy_library()
    anchors = lib["commands"]["straight"]["anchors"]
    prior = olp.cv_rollout(torch.tensor([4.0, 0.0]), 6)
    scores = torch.tensor([9.0, -9.0, -9.0])               # mode 0 dominant

    modes_on = anchors[0].unsqueeze(0).repeat(3, 1, 1)      # mode 0 on centred anchor
    modes_far = torch.full((3, 6, 2), 99.0)
    _, risk_on, _ = p.anchor_costs(anchors, modes_on, scores, prior)
    _, risk_far, _ = p.anchor_costs(anchors, modes_far, scores, prior)
    assert not torch.allclose(risk_on, risk_far)           # prediction is used


def test_plan_matches_hard_argmin():
    p = _planner(seed=2)
    lib = _toy_library()
    anchors = lib["commands"]["straight"]["anchors"]
    modes = torch.randn(3, 6, 2)
    scores = torch.randn(3)
    prior = olp.cv_rollout(torch.tensor([4.0, 0.0]), 6)
    _, _, cost = p.anchor_costs(anchors, modes, scores, prior)
    plan, idx = p.plan(modes, scores, "straight", lib,
                       ego_vel=torch.tensor([4.0, 0.0]), return_idx=True)
    assert idx == int(cost.argmin())
    assert torch.allclose(plan, anchors[idx])


def test_plan_returns_library_anchor_no_gt_leak():
    # the eval plan is always one of the fixed anchors -> it can never be the per-sample
    # GT-ego trajectory (no test-time answer leakage beyond the shared 3-way command).
    p = _planner()
    lib = _toy_library()
    anchors = lib["commands"]["straight"]["anchors"]
    plan = p.plan(torch.randn(3, 6, 2), torch.randn(3), "straight", lib,
                  ego_vel=torch.tensor([3.0, 1.0]))
    assert any(torch.allclose(plan, a) for a in anchors)


def test_soft_plan_differentiable():
    p = LearnedAnchorPlanner(hidden=16)
    p.train()
    anchors = _toy_library()["commands"]["straight"]["anchors"]
    prior = olp.cv_rollout(torch.tensor([4.0, 0.0]), 6)
    soft_plan, _ = p(anchors, torch.randn(3, 6, 2), torch.randn(3), prior)
    loss = (soft_plan ** 2).mean()
    loss.backward()
    grads = [param.grad for param in p.head.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_batched_matches_single():
    # batched anchor_costs equals the per-sample loop (training uses the batched path).
    p = _planner(seed=3)
    K, H, M = 4, 6, 3
    anchors = torch.randn(K, H, 2)
    prior = torch.randn(H, 2)
    modes = torch.randn(2, M, H, 2)        # batch of 2
    scores = torch.randn(2, M)
    ab = anchors.unsqueeze(0).expand(2, K, H, 2)
    pb = prior.unsqueeze(0).expand(2, H, 2)
    _, _, cost_b = p.anchor_costs(ab, modes, scores, pb)
    for i in range(2):
        _, _, cost_i = p.anchor_costs(anchors, modes[i], scores[i], prior)
        assert torch.allclose(cost_b[i], cost_i, atol=1e-5)
