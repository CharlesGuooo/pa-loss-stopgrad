"""Phase 2.4 tests: stronger risk functions (ttc, boxiou) + proxy-independent safety metric."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

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
from pa_loss_stopgrad.pa_loss.pa_loss_gameformer import (  # noqa: E402
    _boxes_overlap,
    _rect_corners,
    _risk_boxiou,
    _risk_ttc,
    pa_loss_gameformer,
    safety_metrics_per_sample,
)

T = FUTURE_LEN


def _outputs(neighbor_modes, nb_scores, level=1):
    B, M = neighbor_modes.shape[0], neighbor_modes.shape[1]
    inter = torch.zeros(B, NUM_PRED, M, T, 4)
    inter[:, 1, :, :, :2] = neighbor_modes
    scores = torch.zeros(B, NUM_PRED, M)
    scores[:, 1] = nb_scores
    return {f"level_{level}_interactions": inter, f"level_{level}_scores": scores}


# --------------------------------------------------------------------------- #
# TTC monotonicity
# --------------------------------------------------------------------------- #
def test_ttc_danger_increases_with_closing_speed():
    H = 6
    ego = torch.zeros(1, H, 2)  # ego sits at origin
    sv = torch.ones(1, H, dtype=torch.bool)

    def danger(x_start, x_end):
        nb = torch.zeros(1, 1, H, 2)
        nb[0, 0, :, 0] = torch.linspace(x_start, x_end, H)
        _, d = _risk_ttc(nb, ego, sv)
        return float(d[0, 0])

    slow = danger(5.0, 4.0)   # creeps toward ego
    fast = danger(5.0, 1.0)   # closes fast
    assert fast > slow        # smaller TTC -> higher danger


def test_ttc_receding_is_low_danger():
    H = 6
    ego = torch.zeros(1, H, 2)
    sv = torch.ones(1, H, dtype=torch.bool)
    nb_app = torch.zeros(1, 1, H, 2); nb_app[0, 0, :, 0] = torch.linspace(5.0, 2.0, H)
    nb_rec = torch.zeros(1, 1, H, 2); nb_rec[0, 0, :, 0] = torch.linspace(2.0, 6.0, H)
    _, d_app = _risk_ttc(nb_app, ego, sv)
    _, d_rec = _risk_ttc(nb_rec, ego, sv)
    assert float(d_rec[0, 0]) < float(d_app[0, 0])


# --------------------------------------------------------------------------- #
# box-IoU overlap monotonicity
# --------------------------------------------------------------------------- #
def test_boxiou_overlap_high_when_aligned_and_coincident():
    H = 4
    ego = torch.zeros(1, H, 2)
    sv = torch.ones(1, H, dtype=torch.bool)
    lw = torch.tensor([[4.0, 1.8]])
    ego_lw = torch.tensor([4.0, 1.8])
    th_ego = torch.zeros(1, H)

    nb_on = torch.zeros(1, 1, H, 2)             # coincident with ego, aligned
    th_on = torch.zeros(1, 1, H)
    cost_on, _ = _risk_boxiou(nb_on, ego, sv, th_on, th_ego, lw, ego_lw)

    nb_far = torch.zeros(1, 1, H, 2); nb_far[0, 0, :, 1] = 50.0   # 50 m to the side
    th_far = torch.full((1, 1, H), 3.14159 / 2)                  # orthogonal heading
    cost_far, _ = _risk_boxiou(nb_far, ego, sv, th_far, th_ego, lw, ego_lw)

    assert float(cost_on[0, 0]) > float(cost_far[0, 0])
    assert float(cost_far[0, 0]) < 0.1


# --------------------------------------------------------------------------- #
# differentiability + grad routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["ttc", "boxiou"])
def test_risk_kind_differentiable(kind):
    H = 6
    ego = torch.zeros(1, H, 2)
    sv = torch.ones(1, H, dtype=torch.bool)
    nb = torch.zeros(1, 1, H, 2, requires_grad=True)
    nb_drive = nb.clone()
    nb_drive = nb + torch.linspace(3.0, 0.5, H)[None, None, :, None]  # approaching, differentiable
    if kind == "ttc":
        cost, _ = _risk_ttc(nb_drive, ego, sv)
    else:
        lw = torch.tensor([[4.0, 1.8]]); ego_lw = torch.tensor([4.0, 1.8])
        cost, _ = _risk_boxiou(nb_drive, ego, sv, torch.zeros(1, 1, H), torch.zeros(1, H), lw, ego_lw)
    cost.sum().backward()
    assert nb.grad is not None and torch.isfinite(nb.grad).all() and float(nb.grad.abs().sum()) > 0


def _synthetic_batch(B=2):
    g = torch.Generator().manual_seed(2)
    ego = torch.randn(B, HIST_LEN, 9, generator=g); ego[..., 8] = 1
    neighbors = torch.randn(B, NUM_NEIGHBORS, HIST_LEN, 9, generator=g)
    neighbors[..., 8] = torch.randint(0, 4, neighbors[..., 8].shape).float()
    lanes = torch.randn(B, NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM, generator=g)
    lanes[..., 9:12] = torch.randint(0, 4, lanes[..., 9:12].shape).float()
    crosswalks = torch.randn(B, NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3, generator=g)
    ego_future = torch.randn(B, T, 5, generator=g) * 0.5
    neighbor_future = torch.randn(B, T, 5, generator=g) * 0.5
    return {"ego_state": ego, "neighbors_state": neighbors, "map_lanes": lanes,
            "map_crosswalks": crosswalks, "ego_future": ego_future, "neighbor_future": neighbor_future}


@pytest.mark.parametrize("kind", ["gauss", "ttc", "boxiou"])
def test_paloss_grad_routes_through_real_model(kind):
    torch.manual_seed(0)
    levels = 1
    model = build_nuscenes_gameformer(future_len=T, modalities=MODALITIES, decoder_levels=levels)
    model.train()
    batch = _synthetic_batch(B=2)
    outputs = model(batch)
    l_imit, _ = level_k_loss_nuscenes(outputs, batch["ego_future"], batch["neighbor_future"],
                                      levels=levels, gmm=True)
    pa = pa_loss_gameformer(outputs, batch["ego_future"], batch["neighbor_future"], level=levels,
                            mode_weighting="risk_softmax", stop_gradient=False, risk_kind=kind)
    (l_imit + 5.0 * pa["l_plan"]).backward()
    total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    assert total > 0


@pytest.mark.parametrize("kind", ["gauss", "ttc", "boxiou"])
def test_stop_gradient_blocks_neighbor_mode_grad(kind):
    ego_future = torch.zeros(1, T, 5); ego_future[0, :, 0] = torch.linspace(1, T, T)
    neighbor_future = torch.zeros(1, T, 5); neighbor_future[0, :, 0] = torch.linspace(1, T, T)
    inter = torch.zeros(1, NUM_PRED, MODALITIES, T, 4, requires_grad=True)
    out = {"level_1_interactions": inter, "level_1_scores": torch.zeros(1, NUM_PRED, MODALITIES)}
    res = pa_loss_gameformer(out, ego_future, neighbor_future, level=1,
                             mode_weighting="risk_softmax", stop_gradient=True, risk_kind=kind)
    assert not res["l_plan"].requires_grad


# --------------------------------------------------------------------------- #
# SAT box overlap + safety metric correctness
# --------------------------------------------------------------------------- #
def test_boxes_overlap_sat():
    c = _rect_corners(torch.tensor([0.0, 0.0]), torch.tensor(0.0), torch.tensor([4.0, 1.8]))
    c_same = _rect_corners(torch.tensor([0.0, 0.0]), torch.tensor(0.0), torch.tensor([4.0, 1.8]))
    c_far = _rect_corners(torch.tensor([0.0, 50.0]), torch.tensor(0.0), torch.tensor([4.0, 1.8]))
    assert bool(_boxes_overlap(c, c_same))
    assert not bool(_boxes_overlap(c, c_far))


def test_safety_metric_near_miss_vs_clear():
    ego_future = torch.zeros(1, T, 5); ego_future[0, :, 0] = torch.linspace(1, T, T)
    neighbor_future = torch.zeros(1, T, 5); neighbor_future[0, :, 0] = torch.linspace(1, T, T)

    # top mode (high score) rides the ego plan -> near miss
    modes = torch.zeros(1, 2, T, 2)
    modes[0, 0, :, 0] = torch.linspace(1, T, T)      # on ego plan
    modes[0, 1, :, 1] = 100.0                        # far away
    out_hit = _outputs(modes, torch.tensor([[2.0, -2.0]]), level=1)
    sm_hit = safety_metrics_per_sample(out_hit, ego_future, neighbor_future, level=1)
    assert float(sm_hit["min_sep"][0]) < 1.0
    assert float(sm_hit["box_overlap"][0]) == 1.0

    # top mode far away -> clear
    out_clear = _outputs(modes, torch.tensor([[-2.0, 2.0]]), level=1)
    sm_clear = safety_metrics_per_sample(out_clear, ego_future, neighbor_future, level=1)
    assert float(sm_clear["min_sep"][0]) > 2.0
    assert float(sm_clear["box_overlap"][0]) == 0.0
