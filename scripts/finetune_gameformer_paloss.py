"""Phase 2.3: fine-tune the trained GameFormer with the Planning-Aware Loss.

Loads the Phase 2.2 checkpoint and fine-tunes one of three variants from the
identical init:
  full     = risk_softmax mode weighting, planning grads reach the predictor
  stopgrad = risk_softmax, but stop_gradient(predicted_futures) in the plan term
  wta      = winner-take-all (only GT-closest neighbor mode gets safety gradient)

Per-epoch eval reports the collision proxy globally and on the high-conflict
subset, plus minADE/minFDE/MR as a no-collapse guard. The Full > stop-grad and
risk > WTA comparisons are assembled across variants by build_phase2_3_summary.
"""
import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pa_loss_stopgrad.carriers.config import FUTURE_LEN, MODALITIES  # noqa: E402
from pa_loss_stopgrad.eval.conflict_subset import load_conflict_mask  # noqa: E402
from pa_loss_stopgrad.data.dataset_nuscenes import NuScenesDrivingData  # noqa: E402
from pa_loss_stopgrad.carriers.loss_nuscenes import level_k_loss_nuscenes  # noqa: E402
from pa_loss_stopgrad.eval.metrics_nuscenes import (  # noqa: E402
    MotionMetricsAccumulator,
    predicted_xy,
    stack_gt,
)
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402
from pa_loss_stopgrad.pa_loss.pa_loss_gameformer import (  # noqa: E402
    collision_proxy_per_sample,
    pa_loss_gameformer,
    safety_metrics_per_sample,
)

VARIANTS = {
    "full": ("risk_softmax", False),
    "stopgrad": ("risk_softmax", True),
    "wta": ("wta", False),
}


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def neighbor_size_from_batch(batch):
    """L/W of neighbor0 at the last history step -> [B,2], clamped positive."""
    ns = batch["neighbors_state"]  # [B, N, T_hist, F]
    lw = ns[:, 0, -1, 5:7].clone()
    lw = torch.where(lw > 0.1, lw, torch.full_like(lw, 0.0))
    fallback = lw.new_tensor([4.0, 1.8])
    lw = torch.where(lw > 0.1, lw, fallback)
    return lw


def train_one_epoch(model, loader, optimizer, levels, device, mw, sg, lam, pa_kw):
    model.train()
    losses, plans, t0, seen = [], [], time.time(), 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch)
        l_imit, _ = level_k_loss_nuscenes(
            outputs, batch["ego_future"], batch["neighbor_future"], levels=levels, gmm=True)
        pa = pa_loss_gameformer(
            outputs, batch["ego_future"], batch["neighbor_future"],
            level=levels, mode_weighting=mw, stop_gradient=sg,
            neighbor_size=neighbor_size_from_batch(batch), **pa_kw)
        loss = l_imit + lam * pa["l_plan"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(loss.item())
        plans.append(float(pa["l_plan"].detach()))
        seen += 1
        if seen % 200 == 0:
            logging.info("  step %d/%d loss=%.3f l_plan=%.4f (%.3fs/it)",
                         seen, len(loader), sum(losses) / len(losses),
                         sum(plans) / len(plans), (time.time() - t0) / seen)
    return sum(losses) / max(len(losses), 1), sum(plans) / max(len(plans), 1)


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


@torch.no_grad()
def validate(model, loader, levels, device, hc_mask, pa_kw):
    model.eval()
    acc = MotionMetricsAccumulator()
    g_sum = g_cnt = hc_sum = hc_cnt = 0.0
    # proxy-independent safety accumulators (global / high-conflict)
    s = {k: 0.0 for k in (
        "g_nm1", "g_nm2", "g_sep", "g_box", "g_n", "hc_nm1", "hc_nm2", "hc_sep", "hc_box", "hc_n")}
    idx = 0
    for batch in loader:
        bsz = batch["ego_future"].shape[0]
        batch = to_device(batch, device)
        outputs = model(batch)
        worst, has_valid = collision_proxy_per_sample(
            outputs, batch["ego_future"], batch["neighbor_future"],
            level=levels, sigma=pa_kw["sigma"], collision_horizon=pa_kw["collision_horizon"])
        sm = safety_metrics_per_sample(
            outputs, batch["ego_future"], batch["neighbor_future"],
            level=levels, neighbor_size=neighbor_size_from_batch(batch),
            collision_horizon=pa_kw["collision_horizon"])
        worst = worst.cpu().numpy()
        has_valid = has_valid.cpu().numpy()
        min_sep = sm["min_sep"].cpu().numpy()
        box_ov = sm["box_overlap"].cpu().numpy()
        acc.update(predicted_xy(outputs, levels), stack_gt(batch["ego_future"], batch["neighbor_future"]))
        for j in range(bsz):
            if not has_valid[j]:
                idx += 1
                continue
            g_sum += worst[j]
            g_cnt += 1
            nm1 = 1.0 if min_sep[j] < 1.0 else 0.0
            nm2 = 1.0 if min_sep[j] < 2.0 else 0.0
            s["g_nm1"] += nm1; s["g_nm2"] += nm2; s["g_sep"] += min_sep[j]; s["g_box"] += box_ov[j]; s["g_n"] += 1
            if hc_mask is not None and idx < len(hc_mask) and hc_mask[idx]:
                hc_sum += worst[j]
                hc_cnt += 1
                s["hc_nm1"] += nm1; s["hc_nm2"] += nm2; s["hc_sep"] += min_sep[j]; s["hc_box"] += box_ov[j]; s["hc_n"] += 1
            idx += 1
    metrics = acc.result()
    return {
        "collision_global": float(g_sum / max(g_cnt, 1)),
        "collision_hc": float(hc_sum / hc_cnt) if hc_cnt else float("nan"),
        "hc_count": int(hc_cnt),
        "minADE_mean": float(metrics["minADE_mean"]),
        "minFDE_mean": float(metrics["minFDE_mean"]),
        "MR_mean": float(metrics["MR_mean"]),
        "near_miss1_global": _safe_div(s["g_nm1"], s["g_n"]),
        "near_miss2_global": _safe_div(s["g_nm2"], s["g_n"]),
        "min_sep_global": _safe_div(s["g_sep"], s["g_n"]),
        "box_overlap_global": _safe_div(s["g_box"], s["g_n"]),
        "near_miss1_hc": _safe_div(s["hc_nm1"], s["hc_n"]),
        "near_miss2_hc": _safe_div(s["hc_nm2"], s["hc_n"]),
        "min_sep_hc": _safe_div(s["hc_sep"], s["hc_n"]),
        "box_overlap_hc": _safe_div(s["hc_box"], s["hc_n"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-glob",
                    default=os.path.join(REPO_ROOT, "data", "gameformer_nuscenes", "trainval", "train", "*.npz"))
    ap.add_argument("--val-glob",
                    default=os.path.join(REPO_ROOT, "data", "gameformer_nuscenes", "trainval", "val", "*.npz"))
    ap.add_argument("--conflict-json",
                    default=os.path.join(REPO_ROOT, "data", "gameformer_nuscenes", "trainval", "val_conflict_labels.json"))
    ap.add_argument("--init-ckpt",
                    default=os.path.join(REPO_ROOT, "outputs", "phase2_2", "gf_nusc_full", "best.pth"))
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--risk-kind", default="gauss", choices=["gauss", "ttc", "boxiou"])
    ap.add_argument("--out-phase", default="phase2_3",
                    help="output subdir under outputs/ (phase2_3 for gauss, phase2_4 for ttc/boxiou)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda-plan", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--tau-ttc", type=float, default=2.0)
    ap.add_argument("--collision-horizon", type=int, default=6)
    ap.add_argument("--decoder-levels", type=int, default=3)
    ap.add_argument("--encoder-layers", type=int, default=3)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--val-subset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="also save epoch_{k}.pth each epoch (for minADE-guarded working-point selection)")
    args = ap.parse_args()

    name = args.name or (os.path.join(args.risk_kind, args.variant) if args.risk_kind != "gauss" else args.variant)
    out_dir = os.path.join(REPO_ROOT, "outputs", args.out_phase, name)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(out_dir, "train.log"), mode="w"), logging.StreamHandler()])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    mw, sg = VARIANTS[args.variant]
    pa_kw = {"sigma": args.sigma, "tau": args.tau, "collision_horizon": args.collision_horizon,
             "risk_kind": args.risk_kind, "tau_ttc": args.tau_ttc}

    train_ds = NuScenesDrivingData(args.train_glob)
    if args.subset and args.subset < len(train_ds):
        train_ds = Subset(train_ds, list(range(args.subset)))
    val_full = NuScenesDrivingData(args.val_glob)
    hc_mask = load_conflict_mask(args.conflict_json) if os.path.exists(args.conflict_json) else None
    val_cap = args.val_subset or (args.subset // 4 if args.subset else 0)
    if val_cap and val_cap < len(val_full):
        val_ds = Subset(val_full, list(range(val_cap)))
        hc_mask = hc_mask[:val_cap] if hc_mask is not None else None
    else:
        val_ds = val_full
    logging.info("variant=%s risk=%s (mw=%s sg=%s) train=%d val=%d hc=%s device=%s lambda=%.1f",
                 args.variant, args.risk_kind, mw, sg, len(train_ds), len(val_ds),
                 int(hc_mask.sum()) if hc_mask is not None else "NA", device, args.lambda_plan)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=args.workers, drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = build_nuscenes_gameformer(
        future_len=FUTURE_LEN, modalities=MODALITIES,
        encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels).to(device)
    if os.path.exists(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_states"])
        logging.info("loaded init ckpt %s (epoch %s)", args.init_ckpt, ckpt.get("epoch"))
    else:
        logging.warning("init ckpt missing (%s) -> training from scratch", args.init_ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    csv_path = os.path.join(out_dir, "train_log.csv")
    history, best = [], {"epoch": -1, "collision_global": float("inf")}
    for epoch in range(args.epochs):
        logging.info("[%s] Epoch %d/%d", args.variant, epoch + 1, args.epochs)
        train_loss, train_plan = train_one_epoch(
            model, train_loader, optimizer, args.decoder_levels, device, mw, sg, args.lambda_plan, pa_kw)
        ev = validate(model, val_loader, args.decoder_levels, device, hc_mask, pa_kw)
        row = {"epoch": epoch + 1, "train_loss": round(train_loss, 4), "train_l_plan": round(train_plan, 5),
               "collision_global": round(ev["collision_global"], 5), "collision_hc": round(ev["collision_hc"], 5),
               "hc_count": ev["hc_count"], "minADE_mean": round(ev["minADE_mean"], 4),
               "minFDE_mean": round(ev["minFDE_mean"], 4), "MR_mean": round(ev["MR_mean"], 4),
               "near_miss1_global": round(ev["near_miss1_global"], 5),
               "near_miss2_global": round(ev["near_miss2_global"], 5),
               "min_sep_global": round(ev["min_sep_global"], 4),
               "box_overlap_global": round(ev["box_overlap_global"], 5),
               "near_miss1_hc": round(ev["near_miss1_hc"], 5),
               "near_miss2_hc": round(ev["near_miss2_hc"], 5),
               "min_sep_hc": round(ev["min_sep_hc"], 4),
               "box_overlap_hc": round(ev["box_overlap_hc"], 5)}
        history.append(row)
        logging.info("  collision g=%.5f hc=%.5f | minADE=%.4f | nm2 g=%.3f hc=%.3f sep g=%.2f hc=%.2f box hc=%.3f",
                     ev["collision_global"], ev["collision_hc"], ev["minADE_mean"],
                     ev["near_miss2_global"], ev["near_miss2_hc"], ev["min_sep_global"],
                     ev["min_sep_hc"], ev["box_overlap_hc"])
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(row.keys())
            w.writerow(row.values())
        if ev["collision_global"] < best["collision_global"]:
            best = {"epoch": epoch + 1, **ev}
            torch.save({"model_states": model.state_dict(), "epoch": epoch + 1, "eval": ev},
                       os.path.join(out_dir, "best.pth"))
        if args.save_every_epoch:
            torch.save({"model_states": model.state_dict(), "epoch": epoch + 1, "eval": ev},
                       os.path.join(out_dir, f"epoch_{epoch + 1}.pth"))

    summary = {
        "variant": args.variant, "risk_kind": args.risk_kind, "mode_weighting": mw, "stop_gradient": sg,
        "lambda_plan": args.lambda_plan, "epochs": args.epochs, "lr": args.lr,
        "sigma": args.sigma, "tau": args.tau, "tau_ttc": args.tau_ttc,
        "collision_horizon": args.collision_horizon,
        "train_size": len(train_ds), "val_size": len(val_ds), "device": str(device),
        "best": best, "final": history[-1] if history else {}, "history": history,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("DONE variant=%s final collision global=%.5f hc=%.5f -> %s",
                 args.variant, history[-1]["collision_global"], history[-1]["collision_hc"], out_dir)


if __name__ == "__main__":
    main()
