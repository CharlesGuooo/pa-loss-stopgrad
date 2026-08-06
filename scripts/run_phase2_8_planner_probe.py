"""Phase 2.8c driver: open-loop planning eval through the FROZEN SHARED learned planner.

Identical to run_phase2_7_openloop_planning.py except the fixed hand-coded planner is
replaced by the frozen LearnedAnchorPlanner (trained once, variant-blind, on the
baseline-2.2 source). The same frozen weights are applied to every variant's
predictions, so any L2 / collision difference is attributable purely to prediction
quality -- a prediction-sensitive planner with the SAME confounding-free guarantee
as Phase 2.7's fixed planner.

Writes outputs/phase2_8/{domain}/{variant}/planning.json (+ summary.json).
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pa_loss_stopgrad.carriers.config import FUTURE_LEN, MODALITIES  # noqa: E402
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402
from pa_loss_stopgrad.planners.openloop_planner import derive_command, load_library  # noqa: E402
from pa_loss_stopgrad.planners.learned_planner import load_planner  # noqa: E402
from pa_loss_stopgrad.eval.planning_metric import PlanningMetric  # noqa: E402

# reuse the Phase 2.7 registry + loaders (forward_all, neighbor_boxes, hc/full-box loaders)
_spec = importlib.util.spec_from_file_location(
    "p27", os.path.join(REPO_ROOT, "scripts", "run_phase2_7_openloop_planning.py"))
_p27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p27)

PLAN_HORIZON = 6


def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)


def eval_variant(model, planner, library, val_glob, level, device, hc_set, full_boxes):
    tokens, modes, scores, ego_fut, nb_fut, nb_lw, ego_vel = _p27.forward_all(model, val_glob, level, device)
    metrics = {"neighbor_global": PlanningMetric(), "neighbor_hc": PlanningMetric(),
               "full_global": PlanningMetric(), "full_hc": PlanningMetric()}
    n_full = 0
    for i, tok in enumerate(tokens):
        ego6 = ego_fut[i, :PLAN_HORIZON, :2]
        gt_mask = (ego6.abs().sum(-1) != 0).float()
        cmd = derive_command(ego6)
        ego_plan = planner.plan(modes[i], scores[i], cmd, library, ego_vel=ego_vel[i])
        is_hc = (hc_set is not None and tok in hc_set)
        nb_boxes = _p27.neighbor_boxes(nb_fut[i], nb_lw[i])
        metrics["neighbor_global"].update(ego_plan, ego6, gt_mask, nb_boxes)
        if is_hc:
            metrics["neighbor_hc"].update(ego_plan, ego6, gt_mask, nb_boxes)
        if full_boxes is not None and tok in full_boxes:
            fb = full_boxes[tok]
            metrics["full_global"].update(ego_plan, ego6, gt_mask, fb)
            if is_hc:
                metrics["full_hc"].update(ego_plan, ego6, gt_mask, fb)
            n_full += 1
    return {
        "n_samples": len(tokens), "n_full_boxes": n_full,
        "neighbor": {"global": metrics["neighbor_global"].compute(),
                     "high_conflict": metrics["neighbor_hc"].compute()},
        "full": {"global": metrics["full_global"].compute(),
                 "high_conflict": metrics["full_hc"].compute()},
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
    ap.add_argument("--ckpt-map", default=_p("outputs", "phase2_7b", "ckpt_map.json"),
                    help="working-point ckpts (minADE-guarded), reused from Phase 2.7b")
    ap.add_argument("--out", default=_p("outputs", "phase2_8"))
    args = ap.parse_args()

    _p27.apply_ckpt_map(args.ckpt_map)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    # the big predictor runs on `device`; forward_all returns CPU tensors, so the tiny
    # planner runs on CPU too (matches Phase 2.7 -- all metric tensors stay on CPU).
    planner, pck = load_planner(args.planner, map_location="cpu")
    planner.eval()
    library = load_library(args.library)
    full_boxes = _p27.load_full_boxes(args.full_boxes)
    print(f"[2.8] device={device} planner={args.planner} "
          f"(GA pass={pck.get('gate_GA', {}).get('pass')}) "
          f"full_boxes={'none' if full_boxes is None else len(full_boxes)}")

    results = {"planner": args.planner, "ckpt_map": args.ckpt_map, "domains": {}}
    for domain in args.domains:
        cfg = _p27.REGISTRY[domain]
        hc_set = _p27.load_hc_set(cfg["conflict_json"])
        results["domains"][domain] = {}
        for variant, ckpt_path in cfg["variants"].items():
            if not os.path.exists(ckpt_path):
                print(f"[2.8] SKIP {domain}/{variant}: missing {ckpt_path}")
                continue
            model = build_nuscenes_gameformer(
                future_len=FUTURE_LEN, modalities=MODALITIES,
                encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_states"])
            model.eval()
            res = eval_variant(model, planner, library, cfg["val_glob"], args.decoder_levels,
                               device, hc_set, full_boxes)
            res.update({"domain": domain, "variant": variant, "ckpt": ckpt_path,
                        "epoch": ckpt.get("epoch")})
            out_dir = os.path.join(args.out, domain, variant)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "planning.json"), "w") as f:
                json.dump(res, f, indent=2)
            ng, nh = res["neighbor"]["global"], res["neighbor"]["high_conflict"]
            results["domains"][domain][variant] = {
                "L2_global": ng["L2"]["avg"], "coll_global": ng["coll"]["avg"],
                "L2_hc": nh["L2"]["avg"] if nh else None,
                "coll_hc": nh["coll"]["avg"] if nh else None}
            print(f"[2.8] {domain}/{variant}: n={res['n_samples']} "
                  f"L2g={ng['L2']['avg']:.3f} collNg={ng['coll']['avg']:.3f}% "
                  f"L2hc={nh['L2']['avg']:.3f} collNhc={nh['coll']['avg']:.3f}% -> {out_dir}")

    with open(os.path.join(args.out, "probe_summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
