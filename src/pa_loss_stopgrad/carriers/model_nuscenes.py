"""Build a GameFormer instance adapted for nuScenes (Phase 2.1).

Constructs the upstream GameFormer model unchanged, then swaps its Waymo
``LaneEncoder`` for ``NuScenesLaneEncoder``. The upstream package is imported by
adding the cloned repo root to ``sys.path``; the clone itself stays pristine.
"""
import os
import sys

import torch.nn as nn

from pa_loss_stopgrad.carriers.config import (
    DECODER_LEVELS,
    ENCODER_LAYERS,
    FUTURE_LEN,
    LANE_PTS,
    MODALITIES,
    NEIGHBORS_TO_PREDICT,
)
from pa_loss_stopgrad.data.lane_encoder_nuscenes import NuScenesLaneEncoder
from pa_loss_stopgrad.paths import GAMEFORMER_ROOT


def _gameformer_repo_root() -> str:
    # GameFormer is not redistributed with this repository: upstream publishes
    # no licence, so it must be cloned by the user. See
    # third_party/setup_gameformer.md, or set GAMEFORMER_ROOT.
    root = str(GAMEFORMER_ROOT)
    if not os.path.isdir(os.path.join(root, "model")):
        raise FileNotFoundError(
            f"GameFormer checkout not found at {root}\n"
            "  Clone it and/or set GAMEFORMER_ROOT -- see third_party/setup_gameformer.md")
    return root


def _import_upstream_gameformer():
    root = _gameformer_repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from model.GameFormer import GameFormer  # noqa: WPS433 (deferred import)

    return GameFormer


def build_nuscenes_gameformer(
    future_len: int = FUTURE_LEN,
    modalities: int = MODALITIES,
    neighbors_to_predict: int = NEIGHBORS_TO_PREDICT,
    encoder_layers: int = ENCODER_LAYERS,
    decoder_levels: int = DECODER_LEVELS,
    lane_pts: int = LANE_PTS,
) -> nn.Module:
    GameFormer = _import_upstream_gameformer()
    model = GameFormer(
        modalities=modalities,
        neighbors_to_predict=neighbors_to_predict,
        future_len=future_len,
        encoder_layers=encoder_layers,
        decoder_levels=decoder_levels,
    )
    # surgical swap: identical output shape ([..., P, 256]) keeps fusion intact
    model.encoder.lane_encoder = NuScenesLaneEncoder(max_len=lane_pts)
    return model
