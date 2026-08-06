"""Phase 2.2 single-GPU GameFormer training on nuScenes (TF-free).

Trains the nuScenes-adapted GameFormer from scratch on the preprocessed npz and
evaluates minADE/minFDE/MissRate per decoder level each epoch. Mirrors the
upstream optimizer recipe (AdamW lr=1e-4, MultiStepLR, grad-clip 5) but uses a
single-process loop and TF-free metrics.
"""
import argparse
import csv
import json
import logging
import os
import sys
import time

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pa_loss_stopgrad.carriers.config import FUTURE_LEN, MODALITIES  # noqa: E402
from pa_loss_stopgrad.data.dataset_nuscenes import NuScenesDrivingData  # noqa: E402
from pa_loss_stopgrad.carriers.loss_nuscenes import level_k_loss_nuscenes  # noqa: E402
from pa_loss_stopgrad.eval.metrics_nuscenes import (  # noqa: E402
    MotionMetricsAccumulator,
    predicted_xy,
    stack_gt,
)
from pa_loss_stopgrad.carriers.model_nuscenes import build_nuscenes_gameformer  # noqa: E402

INPUT_KEYS = ("ego_state", "neighbors_state", "map_lanes", "map_crosswalks")


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def milestones_for(epochs):
    ms = sorted({int(epochs * f) for f in (0.66, 0.73, 0.80, 0.87, 0.93) if int(epochs * f) >= 1})
    return ms or [max(epochs - 1, 1)]


def train_one_epoch(model, loader, optimizer, levels, device):
    model.train()
    losses, t0, seen = [], time.time(), 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss, _ = level_k_loss_nuscenes(
            outputs, batch["ego_future"], batch["neighbor_future"], levels=levels, gmm=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(loss.item())
        seen += 1
        if seen % 200 == 0:
            logging.info("  train step %d/%d loss=%.3f (%.3fs/it)",
                         seen, len(loader), sum(losses) / len(losses), (time.time() - t0) / seen)
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def validate(model, loader, levels, device):
    model.eval()
    accs = {k: MotionMetricsAccumulator() for k in range(levels + 1)}
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(batch)
        loss, _ = level_k_loss_nuscenes(
            outputs, batch["ego_future"], batch["neighbor_future"], levels=levels, gmm=True,
        )
        losses.append(loss.item())
        gt = stack_gt(batch["ego_future"], batch["neighbor_future"])
        for k in range(levels + 1):
            accs[k].update(predicted_xy(outputs, k), gt)
    per_level = {k: accs[k].result() for k in accs}
    return sum(losses) / max(len(losses), 1), per_level


def maybe_subset(ds, n):
    if n and n < len(ds):
        return Subset(ds, list(range(n)))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-glob",
                    default=os.path.join(REPO_ROOT, "data", "gameformer_nuscenes", "trainval", "train", "*.npz"))
    ap.add_argument("--val-glob",
                    default=os.path.join(REPO_ROOT, "data", "gameformer_nuscenes", "trainval", "val", "*.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--decoder-levels", type=int, default=3)
    ap.add_argument("--encoder-layers", type=int, default=3)
    ap.add_argument("--subset", type=int, default=0, help="cap train+val size for smoke runs")
    ap.add_argument("--val-subset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default="gf_nusc")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    out_dir = os.path.join(REPO_ROOT, "outputs", "phase2_2", args.name)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(out_dir, "train.log"), mode="w"), logging.StreamHandler()],
    )
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    train_ds = maybe_subset(NuScenesDrivingData(args.train_glob), args.subset)
    val_cap = args.val_subset or (args.subset // 4 if args.subset else 0)
    val_ds = maybe_subset(NuScenesDrivingData(args.val_glob), val_cap)
    logging.info("train=%d val=%d device=%s levels=%d", len(train_ds), len(val_ds), device, args.decoder_levels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = build_nuscenes_gameformer(
        future_len=FUTURE_LEN, modalities=MODALITIES,
        encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_for(args.epochs), gamma=0.5)
    logging.info("params=%.2fM milestones=%s", sum(p.numel() for p in model.parameters()) / 1e6,
                 milestones_for(args.epochs))

    csv_path = os.path.join(out_dir, "train_log.csv")
    top = args.decoder_levels
    history, best = [], {"epoch": -1, "minADE": float("inf")}

    for epoch in range(args.epochs):
        logging.info("Epoch %d/%d (lr=%.2e)", epoch + 1, args.epochs, optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(model, train_loader, optimizer, args.decoder_levels, device)
        val_loss, per_level = validate(model, val_loader, args.decoder_levels, device)
        scheduler.step()

        m_top, m_l0 = per_level[top], per_level[0]
        row = {
            "epoch": epoch + 1, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "minADE_mean": round(m_top["minADE_mean"], 4), "minFDE_mean": round(m_top["minFDE_mean"], 4),
            "MR_mean": round(m_top["MR_mean"], 4),
            "minADE_ego": round(m_top["minADE_ego"], 4), "minADE_nbr": round(m_top["minADE_nbr"], 4),
            "minADE_mean_L0": round(m_l0["minADE_mean"], 4),
        }
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "per_level": per_level})
        logging.info("  val_loss=%.4f minADE(top L%d)=%.4f minFDE=%.4f MR=%.4f | minADE(L0)=%.4f",
                     val_loss, top, m_top["minADE_mean"], m_top["minFDE_mean"], m_top["MR_mean"],
                     m_l0["minADE_mean"])

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(row.keys())
            w.writerow(row.values())

        if m_top["minADE_mean"] == m_top["minADE_mean"] and m_top["minADE_mean"] < best["minADE"]:
            best = {"epoch": epoch + 1, "minADE": m_top["minADE_mean"]}
            torch.save({"model_states": model.state_dict(), "epoch": epoch + 1, "metrics": m_top},
                       os.path.join(out_dir, "best.pth"))

    summary = {
        "name": args.name, "epochs": args.epochs, "train_size": len(train_ds), "val_size": len(val_ds),
        "decoder_levels": args.decoder_levels, "encoder_layers": args.encoder_layers,
        "batch_size": args.batch_size, "lr": args.lr, "device": str(device),
        "best_epoch": best["epoch"], "best_minADE_mean": best["minADE"],
        "final_per_level": history[-1]["per_level"] if history else {},
        "level_k_benefit": (
            history[-1]["per_level"][top]["minADE_mean"] <= history[-1]["per_level"][0]["minADE_mean"]
            if history else None
        ),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("DONE best_epoch=%d best_minADE_mean=%.4f -> %s",
                 best["epoch"], best["minADE"], os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
