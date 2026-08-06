"""Phase 2.7 driver: open-loop planning eval for each model variant.

For every (domain, variant) checkpoint: forward on that variant's val npz ->
predicted neighbor0 modes+scores -> fixed library planner -> ego plan -> local
nuScenes PlanningMetric (L2 + collision). Collision is reported two ways:
  * coll_neighbor (PRIMARY): ego plan vs neighbor0 GT box (built from the npz).
  * coll_full     (CONTEXT): ego plan vs all GT agent boxes (from val_fut_boxes.npz,
                  devkit-exported; only over tokens present there).
Each split into global and high-conflict (HC) via the conflict-label json, keyed
by token so it aligns across the oracle and perceived val dirs.

Writes outputs/phase2_7/{domain}/{variant}/planning.json.
The planner / library / metric are identical for every variant -> the ONLY
variable is prediction quality.
"""
import argparse
import glob
import json
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
from pa_loss_stopgrad.eval.planning_metric import PlanningMetric  # noqa: E402

PLAN_HORIZON = 6


def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)


REGISTRY = {
    "oracle": {
        "val_glob": _p("data", "gameformer_nuscenes", "trainval", "val", "*.npz"),
        "conflict_json": _p("data", "gameformer_nuscenes", "trainval", "val_conflict_labels.json"),
        "variants": {
            "oracle": _p("outputs", "phase2_2", "gf_nusc_full", "best.pth"),
            "full": _p("outputs", "phase2_3", "full", "best.pth"),
            "stopgrad": _p("outputs", "phase2_3", "stopgrad", "best.pth"),
            "wta": _p("outputs", "phase2_3", "wta", "best.pth"),
        },
    },
    "perceived": {
        "val_glob": _p("outputs", "phase2_5", "trainval_val", "perceived_perceived", "*.npz"),
        "conflict_json": _p("data", "gameformer_nuscenes", "trainval", "val_conflict_labels.json"),
        "variants": {
            "full": _p("outputs", "phase2_6", "perceived_perceived", "full", "best.pth"),
            "stopgrad": _p("outputs", "phase2_6", "perceived_perceived", "stopgrad", "best.pth"),
            "wta": _p("outputs", "phase2_6", "perceived_perceived", "wta", "best.pth"),
        },
    },
}


def apply_ckpt_map(ckpt_map_path):
    """Override REGISTRY ckpt paths from a JSON {domain: {variant: path}}."""
    if not (ckpt_map_path and os.path.exists(ckpt_map_path)):
        return
    with open(ckpt_map_path) as f:
        cmap = json.load(f)
    for domain, variants in cmap.items():
        for variant, path in variants.items():
            REGISTRY[domain]["variants"][variant] = path
            print(f"[ckpt-map] {domain}/{variant} -> {path}")


def load_hc_set(conflict_json):
    if not (conflict_json and os.path.exists(conflict_json)):
        return None
    with open(conflict_json) as f:
        rec = json.load(f)
    return {t for t, lab in zip(rec["tokens"], rec["labels"]) if lab == "high_conflict"}


def load_full_boxes(path):
    if not (path and os.path.exists(path)):
        return None
    d = np.load(path, allow_pickle=True)
    return {str(t): list(b) for t, b in zip(d["tokens"], d["boxes"])}


@torch.no_grad()
def forward_all(model, val_glob, level, device, batch_size=16):
    """Return tokens[N], modes[N,M,6,2], scores[N,M], ego_fut[N,12,5],
    nb_fut[N,12,5], nb_lw[N,2] -- all CPU, in sorted-file order."""
    files = sorted(glob.glob(val_glob))
    tokens = [os.path.splitext(os.path.basename(f))[0] for f in files]
    ds = NuScenesDrivingData(val_glob)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    modes, scores, ego_fut, nb_fut, nb_lw, ego_vel = [], [], [], [], [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        inter = out[f"level_{level}_interactions"]            # [B,2,M,T,4]
        sc = out[f"level_{level}_scores"]                     # [B,2,M]
        modes.append(inter[:, 1, :, :PLAN_HORIZON, :2].cpu())
        scores.append(sc[:, 1].cpu())
        ego_fut.append(batch["ego_future"])
        nb_fut.append(batch["neighbor_future"])
        nb_lw.append(batch["neighbors_state"][:, 0, -1, 5:7])
        ego_vel.append(batch["ego_state"][:, -1, 3:5])        # observed current (vx,vy)
    return (tokens, torch.cat(modes), torch.cat(scores),
            torch.cat(ego_fut), torch.cat(nb_fut), torch.cat(nb_lw), torch.cat(ego_vel))


def neighbor_boxes(nb_fut_i, nb_lw_i):
    """Per-step list (len 6) of neighbor0 GT boxes [Ki,5]; empty where padded."""
    L, W = float(nb_lw_i[0]), float(nb_lw_i[1])
    if L <= 0 or W <= 0:
        L, W = 4.0, 1.8
    out = []
    for t in range(PLAN_HORIZON):
        xy = nb_fut_i[t, :2]
        if float(xy.abs().sum()) > 0:
            out.append(np.array([[float(xy[0]), float(xy[1]), float(nb_fut_i[t, 2]), L, W]],
                                dtype=np.float32))
        else:
            out.append(np.zeros((0, 5), dtype=np.float32))
    return out


def eval_variant(model, val_glob, level, device, hc_set, full_boxes):
    tokens, modes, scores, ego_fut, nb_fut, nb_lw, ego_vel = forward_all(model, val_glob, level, device)
    metrics = {
        "neighbor_global": PlanningMetric(), "neighbor_hc": PlanningMetric(),
        "full_global": PlanningMetric(), "full_hc": PlanningMetric(),
    }
    library = eval_variant.library
    n_full = 0
    for i, tok in enumerate(tokens):
        ego6 = ego_fut[i, :PLAN_HORIZON, :2]
        gt_mask = (ego6.abs().sum(-1) != 0).float()
        cmd = derive_command(ego6)
        ego_plan = plan(modes[i], scores[i], cmd, library, ego_vel=ego_vel[i])
        is_hc = (hc_set is not None and tok in hc_set)

        nb_boxes = neighbor_boxes(nb_fut[i], nb_lw[i])
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
        "n_samples": len(tokens),
        "n_full_boxes": n_full,
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
    ap.add_argument("--library", default=_p("outputs", "phase2_7", "ego_anchor_library.json"))
    ap.add_argument("--full-boxes", default=_p("outputs", "phase2_7", "val_fut_boxes.npz"))
    ap.add_argument("--ckpt-map", default=None,
                    help="JSON {domain:{variant:ckpt_path}} overriding REGISTRY (working-point eval)")
    ap.add_argument("--out", default=_p("outputs", "phase2_7"))
    args = ap.parse_args()

    apply_ckpt_map(args.ckpt_map)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    eval_variant.library = load_library(args.library)
    full_boxes = load_full_boxes(args.full_boxes)
    print(f"[2.7] device={device} library={args.library} "
          f"full_boxes={'none' if full_boxes is None else len(full_boxes)}")

    for domain in args.domains:
        cfg = REGISTRY[domain]
        hc_set = load_hc_set(cfg["conflict_json"])
        for variant, ckpt_path in cfg["variants"].items():
            if not os.path.exists(ckpt_path):
                print(f"[2.7] SKIP {domain}/{variant}: missing {ckpt_path}")
                continue
            model = build_nuscenes_gameformer(
                future_len=FUTURE_LEN, modalities=MODALITIES,
                encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_states"])
            model.eval()
            res = eval_variant(model, cfg["val_glob"], args.decoder_levels, device,
                               hc_set, full_boxes)
            res.update({"domain": domain, "variant": variant, "ckpt": ckpt_path,
                        "epoch": ckpt.get("epoch")})
            out_dir = os.path.join(args.out, domain, variant)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "planning.json"), "w") as f:
                json.dump(res, f, indent=2)
            ng = res["neighbor"]["global"]
            print(f"[2.7] {domain}/{variant}: n={res['n_samples']} "
                  f"L2avg={ng['L2']['avg']:.3f} collN={ng['coll']['avg']:.3f}% "
                  f"-> {out_dir}/planning.json")


if __name__ == "__main__":
    main()
