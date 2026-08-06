"""Shared, dependency-free constants for the GameFormer nuScenes adaptation.

These are imported by both the devkit-based preprocessor and the torch-based
model/dataset, so this module must NOT import torch or nuscenes.
"""

# Temporal sampling (nuScenes keyframes are 2 Hz -> dt = 0.5 s).
DT = 0.5
HIST_LEN = 5      # current + 4 past = 2.0 s of history
FUTURE_LEN = 12   # 6.0 s prediction horizon

# Agent tensor layout: [x, y, heading, vx, vy, length, width, height, type]
AGENT_DIM = 9
NUM_NEIGHBORS = 32

# Ground-truth future layout: [x, y, heading, vx, vy]
FUTURE_DIM = 5

# Map context (per predicted agent). Point counts MUST be divisible by the
# encoder's segment stride (10): see GameFormer Encoder.segment_map.
NUM_LANES = 6
LANE_PTS = 100
# Reduced nuScenes lane feature vector:
#   [0:3] self_line (x, y, heading)
#   [3:6] left_line  (x, y, heading)
#   [6:9] right_line (x, y, heading)
#   [9]   self_type  (0=pad, 1=lane, 2=lane_connector)
#   [10]  left_type  (0=none/pad, 1=divider present)
#   [11]  right_type (0=none/pad, 1=divider present)
LANE_FEAT_DIM = 12

NUM_CROSSWALKS = 4
CROSSWALK_PTS = 50
CROSSWALK_FEAT_DIM = 3  # [x, y, heading]

# Number of jointly-predicted agents (ego + 1 nearest neighbor).
NEIGHBORS_TO_PREDICT = 1
NUM_PRED = NEIGHBORS_TO_PREDICT + 1  # = 2

# Model defaults
MODALITIES = 6
ENCODER_LAYERS = 3
DECODER_LEVELS = 2  # kept small for the Phase 2.1 smoke (upstream default is 4)

# nuScenes type codes used in agent[..., 8]
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3

# Standard nuScenes ego vehicle box (Renault Zoe), [length, width, height] in m.
EGO_SIZE_LWH = (4.084, 1.730, 1.562)
