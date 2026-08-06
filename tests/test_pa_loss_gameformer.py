"""Phase 2.3 tests: PA-Loss on GameFormer (risk weighting, stop-grad, routing, conflict labels)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_loss_stopgrad.carriers.config import (  # noqa: E402
    CROSSWALK_PTS,
    FUTURE_LEN,
    HIST_LEN,
    LANE_FEAT_DIM,
    LANE_PTS,
    MODALITIES,
    NUM_CROSSWALKS,
    NUM_LANES,
    NUM_NEIGHBORS,
    NUM_PRED,
)
from pa_loss_stopgrad.carriers.loss_nuscenes import level_k_loss_nuscenes  # noqa: E402
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402
from pa_loss_stopgrad.pa_loss.pa_loss_gameformer import pa_loss_gameformer  # noqa: E402
from pa_loss_stopgrad.eval.conflict_mining import classify_frame  # noqa: E402

T = FUTURE_LEN


def _outputs(neighbor_modes, nb_scores, level=1):
    """Build a minimal GameFormer-style outputs dict. neighbor_modes [B,M,T,2]."""
    B, M = neighbor_modes.shape[0], neighbor_modes.shape[1]
    inter = torch.zeros(B, NUM_PRED, M, T, 4)
    inter[:, 1, :, :, :2] = neighbor_modes
    scores = torch.zeros(B, NUM_PRED, M)
    scores[:, 1] = nb_scores
    return {f"level_{level}_interactions": inter, f"level_{level}_scores": scores}


def test_risk_softmax_weights_dangerous_mode_higher():
    # ego plan goes straight +x; mode0 safe (far -y), mode1 dangerous (on ego plan), low score
    ego_future = torch.zeros(1, T, 5)
    ego_future[0, :, 0] = torch.linspace(1, T, T)  # x = 1..T
    neighbor_future = torch.zeros(1, T, 5)
    neighbor_future[0, :, 1] = torch.linspace(1, T, T)  # neighbor GT crosses in +y (valid steps)

    modes = torch.zeros(1, 2, T, 2)
    modes[0, 0, :, 1] = -50.0                          # safe: far away in -y
    modes[0, 1, :, 0] = torch.linspace(1, T, T)        # dangerous: rides the ego plan
    nb_scores = torch.tensor([[2.0, -2.0]])            # dangerous mode has LOW score

    out = _outputs(modes, nb_scores, level=1)
    # inspect alpha via the internal risk-softmax: replicate min_dist then softmax
    res = pa_loss_gameformer(out, ego_future, neighbor_future, level=1,
                             mode_weighting="risk_softmax", stop_gradient=False)
    # dangerous-mode cost must dominate the proxy
    assert res["collision_proxy"] > 0.5
    # recompute alpha to assert dangerous weighted higher
    neigh = out["level_1_interactions"][:, 1, :, :, :2]
    d = (neigh[:, :, :6] - ego_future[:, :6, :2][:, None]).norm(dim=-1).min(-1).values
    alpha = F.softmax(-d / 5.0, dim=-1)
    assert alpha[0, 1] > alpha[0, 0]


def test_collision_increases_when_neighbor_approaches_ego():
    ego_future = torch.zeros(1, T, 5)
    ego_future[0, :, 0] = torch.linspace(1, T, T)
    neighbor_future = torch.zeros(1, T, 5)
    neighbor_future[0, :, 0] = torch.linspace(1, T, T)

    near = torch.zeros(1, 1, T, 2)
    near[0, 0, :, 0] = torch.linspace(1, T, T)  # on the ego plan
    far = torch.zeros(1, 1, T, 2)
    far[0, 0, :, 0] = torch.linspace(1, T, T)
    far[0, 0, :, 1] = 100.0                     # 100 m to the side

    r_near = pa_loss_gameformer(_outputs(near, torch.zeros(1, 1), 1), ego_future, neighbor_future,
                                level=1, mode_weighting="risk_softmax", stop_gradient=False)
    r_far = pa_loss_gameformer(_outputs(far, torch.zeros(1, 1), 1), ego_future, neighbor_future,
                               level=1, mode_weighting="risk_softmax", stop_gradient=False)
    assert r_near["l_plan"] > r_far["l_plan"]


def test_stop_gradient_blocks_neighbor_mode_grad():
    ego_future = torch.zeros(1, T, 5)
    ego_future[0, :, 0] = torch.linspace(1, T, T)
    neighbor_future = torch.zeros(1, T, 5)
    neighbor_future[0, :, 0] = torch.linspace(1, T, T)

    def run(stop_gradient):
        inter = torch.zeros(1, NUM_PRED, MODALITIES, T, 4, requires_grad=True)
        out = {"level_1_interactions": inter, "level_1_scores": torch.zeros(1, NUM_PRED, MODALITIES)}
        res = pa_loss_gameformer(out, ego_future, neighbor_future, level=1,
                                 mode_weighting="risk_softmax", stop_gradient=stop_gradient)
        l_plan = res["l_plan"]
        if not l_plan.requires_grad:
            return False, 0.0
        l_plan.backward()
        return True, (0.0 if inter.grad is None else float(inter.grad.abs().sum()))

    # Full: planning term carries gradient to the predicted neighbor modes.
    requires_grad_full, gnorm_full = run(stop_gradient=False)
    assert requires_grad_full and gnorm_full > 0
    # stop-grad: planning term is constant w.r.t. the predictor (zero gradient contribution).
    requires_grad_stop, _ = run(stop_gradient=True)
    assert not requires_grad_stop


def _synthetic_batch(B=2):
    g = torch.Generator().manual_seed(2)
    ego = torch.randn(B, HIST_LEN, 9, generator=g)
    ego[..., 8] = 1
    neighbors = torch.randn(B, NUM_NEIGHBORS, HIST_LEN, 9, generator=g)
    neighbors[..., 8] = torch.randint(0, 4, neighbors[..., 8].shape).float()
    lanes = torch.randn(B, NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM, generator=g)
    lanes[..., 9:12] = torch.randint(0, 4, lanes[..., 9:12].shape).float()
    crosswalks = torch.randn(B, NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3, generator=g)
    ego_future = torch.randn(B, T, 5, generator=g) * 0.5
    neighbor_future = torch.randn(B, T, 5, generator=g) * 0.5
    return {"ego_state": ego, "neighbors_state": neighbors, "map_lanes": lanes,
            "map_crosswalks": crosswalks, "ego_future": ego_future, "neighbor_future": neighbor_future}


def test_paloss_grad_routes_through_real_model():
    torch.manual_seed(0)
    levels = 1
    model = build_nuscenes_gameformer(future_len=T, modalities=MODALITIES, decoder_levels=levels)
    model.train()
    batch = _synthetic_batch(B=2)
    outputs = model(batch)
    l_imit, _ = level_k_loss_nuscenes(outputs, batch["ego_future"], batch["neighbor_future"],
                                      levels=levels, gmm=True)
    pa = pa_loss_gameformer(outputs, batch["ego_future"], batch["neighbor_future"], level=levels,
                            mode_weighting="risk_softmax", stop_gradient=False)
    (l_imit + 5.0 * pa["l_plan"]).backward()
    total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert total > 0


def test_conflict_labeling_crossing_is_high_conflict():
    # ego straight +x, neighbor crossing +y through the ego path -> high_conflict
    ego = torch.zeros(T, 2)
    ego[:, 0] = torch.linspace(0.5, 0.5 * T, T)
    nb = torch.zeros(T, 2)
    nb[:, 0] = 0.5 * T / 2.0                       # fixed x near ego midpoint
    nb[:, 1] = torch.linspace(-0.5 * T, 0.5 * T, T)  # sweeps across in y
    sample = {"ego_fut_abs": ego.numpy(), "gt_agent_fut_abs": nb.numpy()[None], "ego_history_abs": None}
    assert classify_frame(sample)["label"] == "high_conflict"
