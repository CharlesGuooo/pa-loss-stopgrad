"""Phase 2.8a: export a neutral training source for the shared planner.

Forwards the BASELINE Phase 2.2 predictor (the common ancestor of every PA-Loss
variant -> the most neutral, shareable prediction source) over the train npz and
dumps, per frame, exactly what train_shared_planner.py needs:

  tokens   [N]              sample id (sorted-file order)
  modes    [N,M,H,2]        baseline predicted neighbour0 modes (first H=6 steps)
  scores   [N,M]            predicted mode scores (pre-softmax logits)
  ego_vel  [N,2]            observed current (vx,vy) -> CV imitation prior
  ego_fut  [N,12,5]         GT-ego future (imitation target + command derivation)

The planner is trained ONLY on this baseline source (never on full/stopgrad/wta),
which is what keeps the frozen planner variant-blind and confounding-free.

Writes outputs/phase2_8/planner_trainset.npz
"""
import argparse
import glob
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

PLAN_HORIZON = 6


def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)


@torch.no_grad()
def export(model, data_glob, level, device, batch_size=16):
    files = sorted(glob.glob(data_glob))
    tokens = [os.path.splitext(os.path.basename(f))[0] for f in files]
    ds = NuScenesDrivingData(data_glob)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    modes, scores, ego_vel, ego_fut = [], [], [], []
    seen = 0
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(b)
        inter = out[f"level_{level}_interactions"]            # [B,2,M,T,4]
        sc = out[f"level_{level}_scores"]                     # [B,2,M]
        modes.append(inter[:, 1, :, :PLAN_HORIZON, :2].cpu())
        scores.append(sc[:, 1].cpu())
        ego_vel.append(batch["ego_state"][:, -1, 3:5])        # observed (vx,vy)
        ego_fut.append(batch["ego_future"])                   # [B,12,5]
        seen += b["ego_future"].shape[0]
        if seen % 3200 == 0:
            print(f"  forwarded {seen}/{len(files)}")
    return (np.array(tokens, dtype=object),
            torch.cat(modes).numpy().astype(np.float32),
            torch.cat(scores).numpy().astype(np.float32),
            torch.cat(ego_vel).numpy().astype(np.float32),
            torch.cat(ego_fut).numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-glob",
                    default=_p("data", "gameformer_nuscenes", "trainval", "train", "*.npz"))
    ap.add_argument("--ckpt", default=_p("outputs", "phase2_2", "gf_nusc_full", "best.pth"))
    ap.add_argument("--decoder-levels", type=int, default=3)
    ap.add_argument("--encoder-layers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=_p("outputs", "phase2_8", "planner_trainset.npz"))
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = build_nuscenes_gameformer(
        future_len=FUTURE_LEN, modalities=MODALITIES,
        encoder_layers=args.encoder_layers, decoder_levels=args.decoder_levels).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_states"])
    model.eval()
    print(f"[2.8-export] baseline ckpt={args.ckpt} (epoch {ckpt.get('epoch')}) device={device}")

    tokens, modes, scores, ego_vel, ego_fut = export(model, args.train_glob, args.decoder_levels, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, tokens=tokens, modes=modes, scores=scores,
             ego_vel=ego_vel, ego_fut=ego_fut)
    print(f"[2.8-export] n={len(tokens)} modes={modes.shape} scores={scores.shape} -> {args.out}")


if __name__ == "__main__":
    main()
