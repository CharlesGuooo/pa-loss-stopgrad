"""Phase 2.7b: per-frame dump for statistical robustness (no retraining).

Re-runs the existing Phase 2.7 ckpts (oracle + perceived domains) ONE more time
and records, *per validation frame*, the quantities needed to turn the small
"0 vs 1 collision" headline into statistically-powered statements:

  prediction leg (strengthens 2.6):
    minADE_ego, minADE_nbr        per-sample minADE over the full 12-step horizon
    collision_proxy               worst-mode gauss proxy (the 2.3/2.6 training metric)
  planning leg (strengthens 2.7):
    plan_L2                       per-sample L2 avg (1/2/3 s) of the planned ego
    coll_neighbor / coll_full     per-sample box-collision flag (any within 3 s)
    minsep_neighbor / minsep_full per-sample min centre-to-centre clearance (m) to the
                                  nearest GT agent over the horizon -> the continuous
                                  near-miss signal that the binary collision rate lacks
  sensitivity (c):
    grid_coll_neighbor / grid_minsep_neighbor  same neighbour metrics under a grid of
                                  planner collision weights w_coll, to show the
                                  Full>stop-grad direction is robust to the one free knob

All variants in a domain share the same val npz order, so frames are paired across
variants by token (stored) for the paired tests in build_phase2_7b_stats.py.

Writes outputs/phase2_7b/perframe/{domain}__{variant}.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pa_loss_stopgrad.carriers.config import FUTURE_LEN, MODALITIES  # noqa: E402
from pa_loss_stopgrad.data.dataset_nuscenes import NuScenesDrivingData  # noqa: E402
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402
from pa_loss_stopgrad.planners.openloop_planner import derive_command, load_library, plan  # noqa: E402
from pa_loss_stopgrad.pa_loss.pa_loss_gameformer import collision_proxy_per_sample  # noqa: E402
from pa_loss_stopgrad.eval.planning_metric import _ego_corners  # noqa: E402
from pa_loss_stopgrad.pa_loss.pa_loss_gameformer import _boxes_overlap, _rect_corners  # noqa: E402

# reuse the registry + loaders from the Phase 2.7 driver
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "p27", os.path.join(REPO_ROOT, "scripts", "run_phase2_7_openloop_planning.py"))
_p27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p27)

PLAN_HORIZON = 6
WCOLL_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_WCOLL = 4.0


def per_sample_minade(pred, gt):
    """pred [B,2,M,T,2], gt [B,2,T,5] -> minADE [B,2], has_valid [B,2]."""
    gtxy = gt[..., :2]
    valid = gtxy.abs().sum(-1) != 0                         # [B,2,T]
    dist = (pred - gtxy.unsqueeze(2)).norm(dim=-1)          # [B,2,M,T]
    m = valid.unsqueeze(2).float()
    cnt = m.sum(-1).clamp(min=1.0)
    ade = (dist * m).sum(-1) / cnt                          # [B,2,M]
    return ade.min(dim=-1).values, valid.any(-1)           # [B,2], [B,2]


def _collided(plan_xy, boxes_list):
    """box SAT collision (any step within 3 s) of the planned ego vs agent boxes."""
    ego = _ego_corners(plan_xy)                             # [6,4,2]
    for t in range(PLAN_HORIZON):
        b = boxes_list[t]
        if b is None or len(b) == 0:
            continue
        b = torch.as_tensor(b, dtype=torch.float32)
        b = b[b[:, 3:5].abs().sum(-1) > 1e-3]
        if len(b) == 0:
            continue
        ac = _rect_corners(b[:, :2], b[:, 2], b[:, 3:5])    # [K,4,2]
        if bool(_boxes_overlap(ego[t][None].expand(b.shape[0], 4, 2), ac).any()):
            return True
    return False


def _min_sep(plan_xy, boxes_list):
    """min centre-to-centre clearance (m) of planned ego to nearest agent over horizon."""
    best = float("inf")
    for t in range(PLAN_HORIZON):
        b = boxes_list[t]
        if b is None or len(b) == 0:
            continue
        b = torch.as_tensor(b, dtype=torch.float32)
        b = b[b[:, 3:5].abs().sum(-1) > 1e-3]
        if len(b) == 0:
            continue
        d = (b[:, :2] - plan_xy[t][None]).norm(dim=-1).min().item()
        best = min(best, d)
    return best  # inf if no agent ever present


@torch.no_grad()
def _single_forward(model, val_glob, level, device):
    """ONE pass: per-sample minADE/proxy + the tensors the planner needs (CPU)."""
    import glob as _glob
    from pa_loss_stopgrad.eval.metrics_nuscenes import predicted_xy, stack_gt
    files = sorted(_glob.glob(val_glob))
    tokens = [os.path.splitext(os.path.basename(f))[0] for f in files]
    ds = NuScenesDrivingData(val_glob)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    modes, scores, ego_fut, nb_fut, nb_lw, ego_vel = [], [], [], [], [], []
    minade_ego, minade_nbr, proxy = [], [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        pred = predicted_xy(out, level)                       # [B,2,M,T,2]
        gt = stack_gt(b["ego_future"], b["neighbor_future"])
        ade, _ = per_sample_minade(pred, gt)
        minade_ego.append(ade[:, 0].cpu()); minade_nbr.append(ade[:, 1].cpu())
        worst, _ = collision_proxy_per_sample(out, b["ego_future"], b["neighbor_future"], level=level)
        proxy.append(worst.cpu())
        modes.append(pred[:, 1, :, :PLAN_HORIZON, :].cpu())
        scores.append(out[f"level_{level}_scores"][:, 1].cpu())
        ego_fut.append(batch["ego_future"])
        nb_fut.append(batch["neighbor_future"])
        nb_lw.append(batch["neighbors_state"][:, 0, -1, 5:7])
        ego_vel.append(batch["ego_state"][:, -1, 3:5])
    return (tokens, torch.cat(modes), torch.cat(scores), torch.cat(ego_fut),
            torch.cat(nb_fut), torch.cat(nb_lw), torch.cat(ego_vel),
            torch.cat(minade_ego).numpy(), torch.cat(minade_nbr).numpy(),
            torch.cat(proxy).numpy())


def eval_perframe(model, val_glob, level, device, hc_set, full_boxes, library):
    (tokens, modes, scores, ego_fut, nb_fut, nb_lw, ego_vel,
     minade_ego, minade_nbr, proxy) = _single_forward(model, val_glob, level, device)

    n = len(tokens)
    rec = {k: np.zeros(n, np.float32) for k in
           ("plan_L2", "minsep_neighbor", "minsep_full")}
    coll_n = np.zeros(n, bool); coll_f = np.zeros(n, bool); is_hc = np.zeros(n, bool)
    grid_coll = np.zeros((n, len(WCOLL_GRID)), bool)
    grid_minsep = np.zeros((n, len(WCOLL_GRID)), np.float32)

    grid_minsep[:] = np.inf
    for i, tok in enumerate(tokens):
        ego6 = ego_fut[i, :PLAN_HORIZON, :2]
        gt_mask = (ego6.abs().sum(-1) != 0).float()
        cmd = derive_command(ego6)
        hc_i = (hc_set is not None and tok in hc_set)
        is_hc[i] = hc_i
        nb_boxes = _p27.neighbor_boxes(nb_fut[i], nb_lw[i])
        fb = full_boxes.get(tok) if full_boxes is not None else None

        ego_plan = plan(modes[i], scores[i], cmd, library, ego_vel=ego_vel[i], w_coll=DEFAULT_WCOLL)
        l2 = torch.sqrt((((ego_plan - ego6) ** 2) * gt_mask[:, None]).sum(-1))  # [6]
        cum = [float(l2[: k + 1].mean()) for k in range(PLAN_HORIZON)]
        rec["plan_L2"][i] = float(np.mean([cum[1], cum[3], cum[5]]))
        coll_n[i] = _collided(ego_plan, nb_boxes)
        rec["minsep_neighbor"][i] = _min_sep(ego_plan, nb_boxes)
        if fb is not None:
            coll_f[i] = _collided(ego_plan, fb)
            rec["minsep_full"][i] = _min_sep(ego_plan, fb)
        else:
            rec["minsep_full"][i] = np.inf

        # w_coll sensitivity (c) is only needed on the HC subset -> skip the 4087 others
        if hc_i:
            for gi, wc in enumerate(WCOLL_GRID):
                gp = plan(modes[i], scores[i], cmd, library, ego_vel=ego_vel[i], w_coll=wc)
                grid_coll[i, gi] = _collided(gp, nb_boxes)
                grid_minsep[i, gi] = _min_sep(gp, nb_boxes)

    return {
        "tokens": np.array(tokens, dtype=object), "is_hc": is_hc,
        "minADE_ego": minade_ego, "minADE_nbr": minade_nbr, "collision_proxy": proxy,
        "plan_L2": rec["plan_L2"],
        "coll_neighbor": coll_n, "coll_full": coll_f,
        "minsep_neighbor": rec["minsep_neighbor"], "minsep_full": rec["minsep_full"],
        "grid_wcoll": np.array(WCOLL_GRID, np.float32),
        "grid_coll_neighbor": grid_coll, "grid_minsep_neighbor": grid_minsep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["oracle", "perceived"],
                    choices=["oracle", "perceived"])
    ap.add_argument("--decoder-levels", type=int, default=3)
    ap.add_argument("--encoder-layers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--library", default=_p27._p("outputs", "phase2_7", "ego_anchor_library.json"))
    ap.add_argument("--full-boxes", default=_p27._p("outputs", "phase2_7", "val_fut_boxes.npz"))
    ap.add_argument("--ckpt-map", default=None,
                    help="JSON {domain:{variant:ckpt_path}} overriding REGISTRY (working-point eval)")
    ap.add_argument("--out", default=_p27._p("outputs", "phase2_7b", "perframe"))
    args = ap.parse_args()

    _p27.apply_ckpt_map(args.ckpt_map)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    library = load_library(args.library)
    full_boxes = _p27.load_full_boxes(args.full_boxes)
    os.makedirs(args.out, exist_ok=True)
    print(f"[2.7b] device={device} full_boxes={'none' if full_boxes is None else len(full_boxes)}")

    for domain in args.domains:
        cfg = _p27.REGISTRY[domain]
        hc_set = _p27.load_hc_set(cfg["conflict_json"])
        for variant, ckpt_path in cfg["variants"].items():
            if not os.path.exists(ckpt_path):
                print(f"[2.7b] SKIP {domain}/{variant}: missing {ckpt_path}")
                continue
            model = build_nuscenes_gameformer(
                future_len=FUTURE_LEN, modalities=MODALITIES,
                encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_states"])
            model.eval()
            rec = eval_perframe(model, cfg["val_glob"], args.decoder_levels, device,
                                hc_set, full_boxes, library)
            out_path = os.path.join(args.out, f"{domain}__{variant}.npz")
            np.savez(out_path, **rec)
            hc = rec["is_hc"]
            print(f"[2.7b] {domain}/{variant}: n={len(rec['tokens'])} hc={int(hc.sum())} "
                  f"minADE_nbr={rec['minADE_nbr'].mean():.3f} "
                  f"collN_hc={int(rec['coll_neighbor'][hc].sum())}/{int(hc.sum())} "
                  f"minsepN_hc={np.median(rec['minsep_neighbor'][hc]):.2f} -> {out_path}")


if __name__ == "__main__":
    main()
