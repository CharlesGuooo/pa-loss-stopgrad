# Third-party code and data

This repository contains **only code written for this project**. Nothing here is
a copy of an upstream repository. Everything external is fetched by the user
from its own source, for the licensing reasons set out below.

---

## GameFormer — referenced, deliberately not redistributed

| | |
|---|---|
| Repository | https://github.com/MCZhi/GameFormer |
| Paper | Huang et al., *GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning*, ICCV 2023 (Oral) |
| Licence | **None published.** The GitHub API reports `license: null` and the repository contains no `LICENSE` file. |
| Pinned commit | `fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128` (2024-03-08) — the commit these results were produced against |

Under default copyright, source published without a licence grants no
redistribution rights. **This repository therefore ships no GameFormer code.**
`src/pa_loss_stopgrad/carriers/model_nuscenes.py` constructs the upstream model
by adding a user-supplied checkout to `sys.path`; the checkout is never modified.

See [`third_party/setup_gameformer.md`](third_party/setup_gameformer.md).

What *is* ours, and is released here:

- `carriers/loss_nuscenes.py` — an independent, TensorFlow-free reimplementation
  of the level-*k* / GMM / imitation losses, reparameterised for the 12-step,
  2 Hz nuScenes horizon. Upstream's `utils/inter_pred_utils` cannot be imported
  because it pulls in TensorFlow and the Waymo Open Dataset.
- `data/lane_encoder_nuscenes.py` — a nuScenes lane encoder replacing upstream's
  Waymo `LaneEncoder`.
- Everything under `pa_loss/`, `planners/`, `eval/`, `perception/`.

## SparseDrive — referenced, MIT

| | |
|---|---|
| Repository | https://github.com/swc-17/SparseDrive |
| Licence | MIT, Copyright (c) 2024 swc-17 |

MIT would permit redistribution with attribution, but the useful artefacts are
the ~2 GB of released weights (`sparsedrive_stage1.pth`, `sparsedrive_stage2.pth`,
`resnet50-19c8e357.pth`), which belong in the upstream release rather than in a
git repository. Install SparseDrive from upstream and fetch its weights there.

`src/pa_loss_stopgrad/perception/export_perception_interface.py` **is our code**,
written to run inside a SparseDrive environment (it imports `mmcv`/`mmdet3d`).
Copy it into `SparseDrive/tools/` of your checkout. It writes the frozen
per-frame detection, tracking and online-map cache that
`perception/perception_adapter.py` consumes.

We make no claim about other differences between our SparseDrive working copy
and upstream: that copy was not tracked as an independent git checkout, so a
verified diff cannot be produced. The exported cache format is documented in the
adapter's module docstring, which is the interface that actually matters for
reproduction.

## nuScenes — not redistributed

| | |
|---|---|
| Source | https://www.nuscenes.org/nuscenes |
| Licence | CC BY-NC-SA 4.0 (non-commercial) |

Download it yourself and point `NUSCENES_ROOT` at it. Neither the raw release
nor the preprocessed `.npz` cache derived from it is committed here.

---

## Summary

| Component | Shipped here? | Why |
|---|---|---|
| PA-Loss, stop-gradient control, planners, adapter, metrics | ✅ | ours, MIT |
| GameFormer model code | ❌ | no upstream licence |
| SparseDrive code and weights | ❌ | upstream release is the right source |
| nuScenes data | ❌ | CC BY-NC-SA, download from source |
