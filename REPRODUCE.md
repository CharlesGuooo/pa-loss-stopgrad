# Reproducing the paper

Every claim in the paper is listed below with the command that produces it and
the number to expect. Section numbers refer to the manuscript.

Hyperparameters are given **explicitly** in every command. Some script defaults
differ from the values used in the paper (notably `--lambda-plan`, whose default
is `5.0` while the paper uses `20`), so do not rely on defaults.

---

## 0. Setup

```bash
pip install -e ".[dev]"                      # add ",data" to regenerate the cache
git clone https://github.com/MCZhi/GameFormer.git third_party/GameFormer
cd third_party/GameFormer && git checkout fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128 && cd ../..
export NUSCENES_ROOT=/path/to/nuscenes       # Windows: set NUSCENES_ROOT=...
pytest tests/ -q                             # expect: 103 passed, 1 skipped
```

`pytest` needs no data and no GPU. It is the fastest check that the loss, the
control and the metrics behave as described.

### The instrument, in isolation

The paper's central construction is one line
(`src/pa_loss_stopgrad/pa_loss/pa_loss_gameformer.py:173`):

```python
plan_modes = neighbor_modes.detach() if stop_gradient else neighbor_modes
```

Both arms compute the same objective; only the backward path differs. On
synthetic tensors, with no dataset:

```bash
python scripts/check_pa_loss_gradients.py --batch-size 4 --num-modes 6 --horizon 12
```

Expect a non-zero `d L_plan / d theta` for the full arm and exactly zero for the
stop-gradient arm, with identical loss values. This is §III-C.

---

## 1. Data (§IV-A)

```bash
python -m pa_loss_stopgrad.data.preprocess_nuscenes \
    --version v1.0-trainval --split train --num-workers 4
python -m pa_loss_stopgrad.data.preprocess_nuscenes \
    --version v1.0-trainval --split val --num-workers 4
```

Expect **19,695** training and **4,207** validation frames (frames whose scene
tail lacks a full six-second future are dropped).

High-conflict subset — expect **120 of 4,207 frames (2.85 %)**:

```bash
python -m pa_loss_stopgrad.eval.conflict_subset
```

**Perceived domain.** Copy
`src/pa_loss_stopgrad/perception/export_perception_interface.py` into
`SparseDrive/tools/` of a SparseDrive checkout and run it there to cache the
frozen per-frame detections, tracks and online map. It requires `mmcv`/`mmdet3d`
from SparseDrive's own environment. An export-time check reports **86.1 M
perception parameters, 0 trainable**. See `THIRD_PARTY.md`.

---

## 2. Carrier (§IV-B)

```bash
python scripts/train_gameformer_nuscenes.py \
    --epochs 30 --batch-size 16 --lr 1e-4 --seed 3407 \
    --decoder-levels 3 --encoder-layers 3
```

Expect minADE **1.80**, minFDE **3.68**, miss rate **0.41** on validation.

---

## 3. Main predictor leg (§V-A, Table I, Fig. 3)

The two arms differ only in `--variant`. Everything else is identical.

```bash
for V in full stopgrad; do
  python scripts/finetune_gameformer_paloss.py \
      --variant $V --risk-kind gauss \
      --epochs 6 --batch-size 32 --lambda-plan 20 \
      --sigma 5.0 --tau 5.0 --out-phase phase2_3 \
      --init-ckpt outputs/gameformer_nuscenes/best.pth
done
```

Working point: epoch 3, minADE **2.99**, inside the guard-cap of **3.02**
(`cap = 1.5 x` the control's final minADE).

| Domain | metric | stop-grad | full | change |
|---|---|---|---|---|
| Oracle | collision proxy, global | 0.357 | **0.223** | −37.5 % |
| Oracle | collision proxy, HC | 0.768 | **0.577** | −24.9 % |
| Perceived (full-perception) | global / HC | — | — | −35.4 % / −21.7 % |
| Perceived (detection-only) | global / HC | — | — | −61.3 % / −63.9 % |

The accuracy cost is real: mean minADE runs **1.5–1.8×** the control's at the
working point (≈2.3× at an unconstrained checkpoint).

---

## 4. Robustness (§V-B)

**Risk surrogate** — rerun step 3 with `--risk-kind ttc` and `--risk-kind boxiou`
(use `--out-phase phase2_4`). Global collision reduction:

| surrogate | reduction |
|---|---|
| `gauss` | 37.5 % |
| `ttc` | 10.8 % |
| `boxiou` | 7.2 % |

The gate holds in all three; the magnitudes span a factor of five, so the
direction generalizes and the size does not.

**Bootstrap** — 2,000 resamples, fixed seed:

```bash
python scripts/build_phase2_7b_stats.py
```

Expect **four of four non-overlapping** intervals. Headline cell (oracle,
global): full `0.222 [0.213, 0.231]` vs control `0.355 [0.345, 0.364]`. Note
0.222 is the bootstrap central estimate of the same cell whose working-point
point estimate is 0.223 — the two are different estimators, not a discrepancy.

**Second carrier** — the lightweight recurrent probe
(`carriers/simple_predictor.py`): −38.6 % on a small validation split, −8.94 %
at full scale. Each variant was trained once, so this gap should not be read as
a data-scale effect.

---

## 5. Risk-weighted allocation: a carrier-dependent null (§V-C)

Compare `--variant full` (risk-softmax) against `--variant wta`.

| carrier | risk-softmax | winner-take-all | verdict |
|---|---|---|---|
| recurrent | 0.067 | 0.107 | helps |
| GameFormer (HC) | 0.578 | 0.558 | **null** — slight edge to WTA |

Reported as a failed gate on the strong carrier.

---

## 6. Planning leg I — fixed library planner (§V-D)

```bash
python scripts/run_phase2_7_openloop_planning.py
```

Expect a **null**: no statistically significant difference on any planning
metric. Global open-loop L2 ≈ **0.975** for all variants, near-miss differences
zero at 1 m and 2 m, min-separation not significant (p = 0.18 perceived,
p = 0.29 oracle). Sweeping the planner's collision weight 1→16 leaves the
high-conflict collision count within a single frame, so the null is not an
artefact of one weight.

---

## 7. Planning leg II — frozen shared planner (§V-E, Table II, Fig. 4)

Train the probe **once**, on predictions from the common baseline ancestor of
every variant, then freeze it and apply it unchanged to all conditions:

```bash
python scripts/build_ego_anchor_library.py
python scripts/export_planner_train_pred.py
python scripts/train_shared_planner.py --epochs 20 --batch-size 64 --w-coll 1.0
python scripts/run_phase2_8_planner_probe.py --domains oracle,perceived
python scripts/run_phase2_8_perframe.py
python scripts/build_phase2_8_summary.py
```

Sensitivity gate first: the chosen anchor must flip on **32.0 %** of the most
safety-relevant decile against a 20 % threshold, and only **5.9 %** of frames
globally.

Paired per-frame difference in open-loop L2 (full − stop-grad; negative favours
the full model):

| Domain | Subset | ΔL2 | 95 % CI | Wilcoxon p |
|---|---|---|---|---|
| Perceived | Global | −0.043 m | [−0.054, −0.031] | 4.6e−15 |
| Perceived | HC | −0.164 m | [−0.336, −0.004] | 0.019 |
| Oracle | Global | −0.005 m | [−0.013, +0.003] | 0.17 |
| Oracle | HC | +0.001 m | [−0.051, +0.062] | 0.42 |

**Boundaries.** The collision *rate* is a tie in both domains. The reversal is
on displacement and the ghost-braking reading, not on collision rate. The
min-separation difference has a negative rank-biserial correlation — the full
model planned *closer* to other agents, which is consistent with less
unnecessary avoidance but is **not** a separation-safety gain.

---

## 8. Figures

- **Fig. 3** (bootstrap intervals): `figures/make_fig3.py`
- **Fig. 4** (ghost-braking BEV pair): rendered from the cached scene data by the
  visualization tool; see `figures/README.md`. Frame token
  `af67f465f5994ac7bab19825336db644`, perceived domain, ΔL2 ≈ 0.66 m.

---

## What this repository cannot reproduce on its own

Stated plainly, because it bounds what a reader can check:

- **Closed-loop results** — none exist. Every result is open-loop.
- **Multi-seed variance** — each variant was trained once. The central
  comparison is a matched pair (same initialization, data order and
  hyperparameters, differing in one operation), not a cross-run comparison, but
  the magnitudes are one realization.
- **Perceived domain without SparseDrive** — the cache must be exported from a
  SparseDrive checkout with its own environment and released weights.
- **The raw dataset** — nuScenes is CC BY-NC-SA and is downloaded separately.
