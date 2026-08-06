"""Phase 2.8c: per-frame dump through the FROZEN SHARED learned planner.

Mirrors run_phase2_7b_perframe.py (same output schema, so build_phase2_7b_stats.py
consumes it unchanged) but the planner is the frozen LearnedAnchorPlanner. The only
semantic change: the planner has no w_coll knob, so the sensitivity grid sweeps the
learned-risk SCALE (grid_wcoll holds the risk_scale values) -- the analogue robustness
check that the Full vs stop-grad direction does not hinge on the planner's one free knob.

Writes outputs/phase2_8/perframe/{domain}__{variant}.npz
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pa_loss_stopgrad.carriers.config import FUTURE_LEN, MODALITIES  # noqa: E402
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402
from pa_loss_stopgrad.planners.openloop_planner import derive_command, load_library  # noqa: E402
from pa_loss_stopgrad.planners.learned_planner import load_planner  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, "scripts", rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p27 = _load("p27", "run_phase2_7_openloop_planning.py")
_p27b = _load("p27b", "run_phase2_7b_perframe.py")

PLAN_HORIZON = 6
RISK_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]   # learned-risk scale sweep (replaces w_coll grid)


def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)


def eval_perframe(model, planner, library, val_glob, level, device, hc_set, full_boxes):
    (tokens, modes, scores, ego_fut, nb_fut, nb_lw, ego_vel,
     minade_ego, minade_nbr, proxy) = _p27b._single_forward(model, val_glob, level, device)

    n = len(tokens)
    rec = {k: np.zeros(n, np.float32) for k in ("plan_L2", "minsep_neighbor", "minsep_full")}
    coll_n = np.zeros(n, bool); coll_f = np.zeros(n, bool); is_hc = np.zeros(n, bool)
    grid_coll = np.zeros((n, len(RISK_GRID)), bool)
    grid_minsep = np.full((n, len(RISK_GRID)), np.inf, np.float32)

    for i, tok in enumerate(tokens):
        ego6 = ego_fut[i, :PLAN_HORIZON, :2]
        gt_mask = (ego6.abs().sum(-1) != 0).float()
        cmd = derive_command(ego6)
        hc_i = (hc_set is not None and tok in hc_set)
        is_hc[i] = hc_i
        nb_boxes = _p27.neighbor_boxes(nb_fut[i], nb_lw[i])
        fb = full_boxes.get(tok) if full_boxes is not None else None

        ego_plan = planner.plan(modes[i], scores[i], cmd, library, ego_vel=ego_vel[i], risk_scale=1.0)
        l2 = torch.sqrt((((ego_plan - ego6) ** 2) * gt_mask[:, None]).sum(-1))
        cum = [float(l2[: k + 1].mean()) for k in range(PLAN_HORIZON)]
        rec["plan_L2"][i] = float(np.mean([cum[1], cum[3], cum[5]]))
        coll_n[i] = _p27b._collided(ego_plan, nb_boxes)
        rec["minsep_neighbor"][i] = _p27b._min_sep(ego_plan, nb_boxes)
        if fb is not None:
            coll_f[i] = _p27b._collided(ego_plan, fb)
            rec["minsep_full"][i] = _p27b._min_sep(ego_plan, fb)
        else:
            rec["minsep_full"][i] = np.inf

        if hc_i:  # risk-scale sensitivity only on the HC subset
            for gi, rs in enumerate(RISK_GRID):
                gp = planner.plan(modes[i], scores[i], cmd, library, ego_vel=ego_vel[i], risk_scale=rs)
                grid_coll[i, gi] = _p27b._collided(gp, nb_boxes)
                grid_minsep[i, gi] = _p27b._min_sep(gp, nb_boxes)

    return {
        "tokens": np.array(tokens, dtype=object), "is_hc": is_hc,
        "minADE_ego": minade_ego, "minADE_nbr": minade_nbr, "collision_proxy": proxy,
        "plan_L2": rec["plan_L2"], "coll_neighbor": coll_n, "coll_full": coll_f,
        "minsep_neighbor": rec["minsep_neighbor"], "minsep_full": rec["minsep_full"],
        "grid_wcoll": np.array(RISK_GRID, np.float32),   # = risk_scale grid (key kept for stats reuse)
        "grid_coll_neighbor": grid_coll, "grid_minsep_neighbor": grid_minsep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["oracle", "perceived"],
                    choices=["oracle", "perceived"])
    ap.add_argument("--decoder-levels", type=int, default=3)
    ap.add_argument("--encoder-layers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--planner", default=_p("outputs", "phase2_8", "planner.pth"))
    ap.add_argument("--library", default=_p("outputs", "phase2_7", "ego_anchor_library.json"))
    ap.add_argument("--full-boxes", default=_p("outputs", "phase2_7", "val_fut_boxes.npz"))
    ap.add_argument("--ckpt-map", default=_p("outputs", "phase2_7b", "ckpt_map.json"))
    ap.add_argument("--out", default=_p("outputs", "phase2_8", "perframe"))
    args = ap.parse_args()

    _p27.apply_ckpt_map(args.ckpt_map)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    # big predictor on `device`; tiny planner on CPU (forward tensors + metrics are CPU)
    planner, _ = load_planner(args.planner, map_location="cpu")
    planner.eval()
    library = load_library(args.library)
    full_boxes = _p27.load_full_boxes(args.full_boxes)
    os.makedirs(args.out, exist_ok=True)
    print(f"[2.8b] device={device} planner={args.planner} "
          f"full_boxes={'none' if full_boxes is None else len(full_boxes)}")

    for domain in args.domains:
        cfg = _p27.REGISTRY[domain]
        hc_set = _p27.load_hc_set(cfg["conflict_json"])
        for variant, ckpt_path in cfg["variants"].items():
            if not os.path.exists(ckpt_path):
                print(f"[2.8b] SKIP {domain}/{variant}: missing {ckpt_path}")
                continue
            model = build_nuscenes_gameformer(
                future_len=FUTURE_LEN, modalities=MODALITIES,
                encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_states"])
            model.eval()
            rec = eval_perframe(model, planner, library, cfg["val_glob"], args.decoder_levels,
                                device, hc_set, full_boxes)
            out_path = os.path.join(args.out, f"{domain}__{variant}.npz")
            np.savez(out_path, **rec)
            hc = rec["is_hc"]
            print(f"[2.8b] {domain}/{variant}: n={len(rec['tokens'])} hc={int(hc.sum())} "
                  f"plan_L2={rec['plan_L2'].mean():.3f} "
                  f"collN_hc={int(rec['coll_neighbor'][hc].sum())}/{int(hc.sum())} "
                  f"minsepN_hc={np.median(rec['minsep_neighbor'][hc]):.2f} -> {out_path}")


if __name__ == "__main__":
    main()
