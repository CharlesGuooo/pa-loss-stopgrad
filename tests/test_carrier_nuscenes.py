"""Phase 2.1 tests: nuScenes preprocessor outputs, lane encoder, model forward, loss+grad.

The preprocessor itself runs in the devkit venv; here we validate its already
generated npz artifacts (no devkit needed). The model/loss tests use synthetic
batches so they are independent of whether preprocessing has been run.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pa_loss_stopgrad.carriers.config import (  # noqa: E402
    CROSSWALK_PTS,
    DECODER_LEVELS,
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
from pa_loss_stopgrad.data.lane_encoder_nuscenes import NuScenesLaneEncoder  # noqa: E402
from pa_loss_stopgrad.carriers.loss_nuscenes import level_k_loss_nuscenes  # noqa: E402
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402

MINI_DIR = REPO_ROOT / "data" / "gameformer_nuscenes" / "mini"


def _mini_files():
    return sorted(glob.glob(str(MINI_DIR / "*.npz")))


# --------------------------------------------------------------------------- #
# G1: preprocessor outputs
# --------------------------------------------------------------------------- #
def test_preprocess_outputs_parity():
    files = _mini_files()
    if len(files) < 20:
        pytest.skip(f"need >=20 preprocessed npz, found {len(files)}; run preprocess_nuscenes")

    expected = {
        "ego": (HIST_LEN, 9),
        "neighbors": (NUM_NEIGHBORS, HIST_LEN, 9),
        "map_lanes": (NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM),
        "map_crosswalks": (NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3),
        "gt_future_states": (NUM_PRED, FUTURE_LEN, 5),
        "object_type": (NUM_PRED,),
    }

    locations, lanes_nonempty = set(), 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        for key, shape in expected.items():
            assert d[key].shape == shape, f"{f}:{key} {d[key].shape} != {shape}"
            assert np.all(np.isfinite(d[key])), f"{f}:{key} non-finite"
        ml = d["map_lanes"]
        if np.any((ml[..., 0] != 0) | (ml[..., 1] != 0)):
            lanes_nonempty += 1
        locations.add(d["meta"].item()["location"])

    assert len(locations) >= 2, f"expected >=2 map locations, got {locations}"
    frac = lanes_nonempty / len(files)
    assert frac >= 0.8, f"lanes non-empty fraction {frac:.2f} < 0.8"


# --------------------------------------------------------------------------- #
# G2: lane encoder
# --------------------------------------------------------------------------- #
def test_lane_encoder_shape_and_finite():
    enc = NuScenesLaneEncoder(max_len=LANE_PTS)
    x = torch.randn(2, NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM)
    # type columns must be valid embedding indices
    x[..., 9:12] = torch.randint(0, 4, x[..., 9:12].shape).float()
    out = enc(x)
    assert out.shape == (2, NUM_PRED, NUM_LANES, LANE_PTS, 256)
    assert torch.isfinite(out).all()


def _synthetic_batch(B=2):
    g = torch.Generator().manual_seed(0)
    ego = torch.randn(B, HIST_LEN, 9, generator=g)
    ego[..., 8] = 1  # type
    neighbors = torch.randn(B, NUM_NEIGHBORS, HIST_LEN, 9, generator=g)
    neighbors[..., 8] = torch.randint(0, 4, neighbors[..., 8].shape).float()
    lanes = torch.randn(B, NUM_PRED, NUM_LANES, LANE_PTS, LANE_FEAT_DIM, generator=g)
    lanes[..., 9:12] = torch.randint(0, 4, lanes[..., 9:12].shape).float()
    crosswalks = torch.randn(B, NUM_PRED, NUM_CROSSWALKS, CROSSWALK_PTS, 3, generator=g)
    ego_future = torch.randn(B, FUTURE_LEN, 5, generator=g)
    neighbor_future = torch.randn(B, FUTURE_LEN, 5, generator=g)
    return {
        "ego_state": ego,
        "neighbors_state": neighbors,
        "map_lanes": lanes,
        "map_crosswalks": crosswalks,
        "ego_future": ego_future,
        "neighbor_future": neighbor_future,
    }


# --------------------------------------------------------------------------- #
# G3: model forward shapes
# --------------------------------------------------------------------------- #
def test_model_forward_shapes():
    model = build_nuscenes_gameformer(
        future_len=FUTURE_LEN, modalities=MODALITIES, decoder_levels=DECODER_LEVELS,
    )
    model.eval()
    batch = _synthetic_batch(B=2)
    with torch.no_grad():
        out = model(batch)
    for k in range(DECODER_LEVELS + 1):
        assert out[f"level_{k}_interactions"].shape == (2, NUM_PRED, MODALITIES, FUTURE_LEN, 4)
        assert out[f"level_{k}_scores"].shape == (2, NUM_PRED, MODALITIES)
        assert torch.isfinite(out[f"level_{k}_interactions"]).all()
        assert torch.isfinite(out[f"level_{k}_scores"]).all()


# --------------------------------------------------------------------------- #
# G4: loss + gradient flow
# --------------------------------------------------------------------------- #
def test_loss_finite_and_grads_flow():
    model = build_nuscenes_gameformer(
        future_len=FUTURE_LEN, modalities=MODALITIES, decoder_levels=DECODER_LEVELS,
    )
    model.train()
    batch = _synthetic_batch(B=2)
    out = model(batch)
    loss, _ = level_k_loss_nuscenes(
        out, batch["ego_future"], batch["neighbor_future"], levels=DECODER_LEVELS, gmm=True,
    )
    assert torch.isfinite(loss)

    model.zero_grad()
    loss.backward()

    def has_finite_grad(module):
        seen = False
        for p in module.parameters():
            if p.grad is not None:
                seen = True
                if not torch.isfinite(p.grad).all():
                    return False
        return seen

    assert has_finite_grad(model.encoder.lane_encoder), "no/NaN grad in lane_encoder"
    assert has_finite_grad(model.decoder), "no/NaN grad in decoder"


def test_loss_imitation_mode_finite():
    model = build_nuscenes_gameformer(
        future_len=FUTURE_LEN, modalities=MODALITIES, decoder_levels=DECODER_LEVELS,
    )
    model.train()
    batch = _synthetic_batch(B=2)
    out = model(batch)
    loss, _ = level_k_loss_nuscenes(
        out, batch["ego_future"], batch["neighbor_future"], levels=DECODER_LEVELS, gmm=False,
    )
    assert torch.isfinite(loss)
    loss.backward()
