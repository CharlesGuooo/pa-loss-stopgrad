"""Phase 2.2 tests: TF-free metrics correctness, train-step loss decrease, finite eval."""
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
from pa_loss_stopgrad.eval.metrics_nuscenes import (  # noqa: E402
    MotionMetricsAccumulator,
    evaluate_all_levels,
    predicted_xy,
    stack_gt,
)
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402


def _gt(ego_xy, nbr_xy):
    """Build [1, 2, T, 5] gt from xy lists; other cols zero."""
    T = len(ego_xy)
    gt = torch.zeros(1, 2, T, 5)
    gt[0, 0, :, :2] = torch.tensor(ego_xy, dtype=torch.float32)
    gt[0, 1, :, :2] = torch.tensor(nbr_xy, dtype=torch.float32)
    return gt


def test_metrics_known_values_and_masking():
    # ego all valid; nbr step 1 invalid (zeros)
    gt = _gt([[1, 0], [2, 0], [3, 0]], [[1, 0], [0, 0], [3, 0]])
    d = 1.0
    # single mode: pred = gt xy + (d, 0) at every step
    pred = (gt[..., :2] + torch.tensor([d, 0.0])).unsqueeze(2)  # [1,2,1,T,2]

    acc = MotionMetricsAccumulator()
    acc.update(pred, gt)
    r = acc.result()
    assert abs(r["minADE_ego"] - d) < 1e-5
    assert abs(r["minFDE_ego"] - d) < 1e-5
    assert abs(r["minADE_nbr"] - d) < 1e-5  # averaged over 2 valid steps only
    assert abs(r["minFDE_nbr"] - d) < 1e-5  # last valid step = index 2
    assert r["MR_mean"] == 0.0  # d < 2.0


def test_metrics_missrate_threshold():
    gt = _gt([[1, 0], [2, 0], [3, 0]], [[1, 0], [2, 0], [3, 0]])
    pred = (gt[..., :2] + torch.tensor([3.0, 0.0])).unsqueeze(2)  # fde = 3 > 2
    acc = MotionMetricsAccumulator()
    acc.update(pred, gt)
    r = acc.result()
    assert r["MR_mean"] == 1.0


def test_metrics_min_over_modes():
    gt = _gt([[1, 0], [2, 0], [3, 0]], [[1, 0], [2, 0], [3, 0]])
    gtxy = gt[..., :2]
    bad = gtxy + 5.0
    good = gtxy.clone()
    pred = torch.stack([bad, good], dim=2)  # [1,2,2,T,2] -> min picks the exact mode
    acc = MotionMetricsAccumulator()
    acc.update(pred, gt)
    r = acc.result()
    assert r["minADE_mean"] < 1e-5
    assert r["minFDE_mean"] < 1e-5


def _synthetic_batch(B=4):
    g = torch.Generator().manual_seed(1)
    ego = torch.randn(B, HIST_LEN, 9, generator=g)
    ego[..., 8] = 1
    neighbors = torch.randn(B, NUM_NEIGHBORS, HIST_LEN, 9, generator=g)
    neighbors[..., 8] = torch.randint(0, 4, neighbors[..., 8].shape).float()
    lanes = torch.randn(B, NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM, generator=g)
    lanes[..., 9:12] = torch.randint(0, 4, lanes[..., 9:12].shape).float()
    crosswalks = torch.randn(B, NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3, generator=g)
    # small, structured futures so the loss is learnable
    ego_future = torch.randn(B, FUTURE_LEN, 5, generator=g) * 0.5
    neighbor_future = torch.randn(B, FUTURE_LEN, 5, generator=g) * 0.5
    return {
        "ego_state": ego, "neighbors_state": neighbors, "map_lanes": lanes,
        "map_crosswalks": crosswalks, "ego_future": ego_future, "neighbor_future": neighbor_future,
    }


def test_overfit_one_batch_loss_decreases():
    torch.manual_seed(0)
    levels = 1
    model = build_nuscenes_gameformer(future_len=FUTURE_LEN, modalities=MODALITIES, decoder_levels=levels)
    model.train()
    batch = _synthetic_batch(B=4)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = model(batch)
        loss, _ = level_k_loss_nuscenes(out, batch["ego_future"], batch["neighbor_future"],
                                        levels=levels, gmm=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        losses.append(float(loss.detach()))

    assert min(losses[-5:]) < losses[0], f"loss did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}"


def test_eval_metrics_finite_all_levels():
    levels = 2
    model = build_nuscenes_gameformer(future_len=FUTURE_LEN, modalities=MODALITIES, decoder_levels=levels)
    model.eval()
    batch = _synthetic_batch(B=3)
    with torch.no_grad():
        out = model(batch)
    res = evaluate_all_levels(out, batch["ego_future"], batch["neighbor_future"], levels)
    assert set(res.keys()) == set(range(levels + 1))
    import math
    for k, m in res.items():
        assert math.isfinite(m["minADE_mean"]) and math.isfinite(m["minFDE_mean"])
