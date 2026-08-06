"""Phase 2.8a/b: train the frozen shared LearnedAnchorPlanner.

Trains ONLY on the neutral baseline-2.2 prediction source dumped by
export_planner_train_pred.py (never on full/stopgrad/wta) -> the resulting frozen
planner is variant-blind, so the 2.8c probe stays confounding-free.

Objective per frame (soft-argmin makes anchor selection differentiable):
    loss = w_imit * ADE(soft_plan, GT-ego)                 (stay realistic / low L2)
         + w_coll * score-weighted soft proximity(soft_plan, predicted modes)
The predicted modes are jittered with Gaussian noise during training so the planner
learns to react to *imperfect* predictions (what it meets at eval).

Checkpoint selection is variant-blind: the best epoch minimises the SAME objective
on a held-out slice of the baseline source (no PA variant ever involved).

Gate G-A (sensitivity): after training, on the held-out slice, measure how often the
chosen anchor flips when the neighbour predictions are zeroed, globally and on the
"danger decile" (frames where the predicted neighbour comes closest to the CV prior).
Phase 2.8b only proceeds if the danger-decile flip rate clears the threshold.

Writes outputs/phase2_8/planner.pth (+ training_summary.json).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pa_loss_stopgrad.planners.learned_planner import LearnedAnchorPlanner  # noqa: E402
from pa_loss_stopgrad.planners.openloop_planner import (  # noqa: E402
    COMMANDS, PLAN_HORIZON, derive_command, load_library,
)

DT = 0.5


def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)


def cv_rollout_batch(ego_vel, horizon, dt=DT):
    """ego_vel [B,2] -> CV prior [B,H,2]: pos[t] = vel*dt*(t+1)."""
    steps = torch.arange(1, horizon + 1, dtype=ego_vel.dtype, device=ego_vel.device)  # [H]
    return steps[None, :, None] * dt * ego_vel[:, None, :]                            # [B,H,2]


def build_tensors(npz_path, library, device):
    """Vectorise the trainset into command-indexed anchor tensors + per-frame fields."""
    d = np.load(npz_path, allow_pickle=True)
    modes = torch.tensor(d["modes"], dtype=torch.float32)        # [N,M,H,2]
    scores = torch.tensor(d["scores"], dtype=torch.float32)      # [N,M]
    ego_vel = torch.tensor(d["ego_vel"], dtype=torch.float32)    # [N,2]
    ego_fut = torch.tensor(d["ego_fut"], dtype=torch.float32)    # [N,12,5]
    gt_ego = ego_fut[:, :PLAN_HORIZON, :2]                        # [N,H,2]
    gt_mask = (gt_ego.abs().sum(-1) != 0).float()                # [N,H]

    cmd_anchors = torch.stack(
        [library["commands"][c]["anchors"].float() for c in COMMANDS], dim=0)  # [3,K,H,2]
    cmd_idx = torch.tensor(
        [COMMANDS.index(derive_command(gt_ego[i])) for i in range(gt_ego.shape[0])],
        dtype=torch.long)                                        # [N]
    return {
        "modes": modes, "scores": scores, "ego_vel": ego_vel,
        "gt_ego": gt_ego, "gt_mask": gt_mask, "cmd_idx": cmd_idx,
        "cmd_anchors": cmd_anchors.to(device),
    }


def _objective(planner, anchors, modes, scores, prior, gt_ego, gt_mask,
               w_imit, w_coll, sigma_coll, temperature, risk_scale=1.0):
    soft_plan, _ = planner(anchors, modes, scores, prior,
                           temperature=temperature, risk_scale=risk_scale)  # [B,H,2]
    # imitation: masked ADE to GT-ego
    step_l2 = ((soft_plan - gt_ego) ** 2).sum(-1).clamp_min(1e-12).sqrt()   # [B,H]
    denom = gt_mask.sum(-1).clamp_min(1.0)
    l_imit = ((step_l2 * gt_mask).sum(-1) / denom).mean()
    # collision: fixed score-weighted soft proximity of the plan to predicted modes
    dist = (soft_plan.unsqueeze(1) - modes[..., :PLAN_HORIZON, :]).norm(dim=-1)  # [B,M,H]
    prox = torch.exp(-(dist ** 2) / (2.0 * sigma_coll ** 2)).max(dim=-1).values  # [B,M]
    prob = F.softmax(scores, dim=-1)
    l_coll = (prox * prob).sum(-1).mean()
    return w_imit * l_imit + w_coll * l_coll, l_imit.detach(), l_coll.detach()


def gate_ga(planner, T, idx, library, device):
    """Anchor-flip rate (real vs zeroed neighbour preds), global + danger decile."""
    planner.eval()
    modes = T["modes"][idx].to(device)
    scores = T["scores"][idx].to(device)
    prior = cv_rollout_batch(T["ego_vel"][idx].to(device), PLAN_HORIZON)
    anchors = T["cmd_anchors"][T["cmd_idx"][idx].to(device)]      # [B,K,H,2]
    zeros = torch.full_like(modes, 1e3)                          # neighbours "absent"

    with torch.no_grad():
        _, _, cost_real = planner.anchor_costs(anchors, modes, scores, prior)
        _, _, cost_zero = planner.anchor_costs(anchors, zeros, scores, prior)
        idx_real = cost_real.argmin(-1)
        idx_zero = cost_zero.argmin(-1)
        flip = (idx_real != idx_zero).float()                    # [B]
        # danger proxy: closest predicted-mode approach to the CV prior (smaller=worse)
        dist = (prior.unsqueeze(1) - modes[..., :PLAN_HORIZON, :]).norm(dim=-1)  # [B,M,H]
        danger = -dist.min(dim=-1).values.min(dim=-1).values     # [B] larger=closer
    n = flip.shape[0]
    k = max(1, n // 10)
    danger_idx = torch.topk(danger, k).indices
    return {
        "n_holdout": int(n),
        "flip_rate_global": float(flip.mean()),
        "flip_rate_danger_decile": float(flip[danger_idx].mean()),
        "danger_decile_n": int(k),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainset", default=_p("outputs", "phase2_8", "planner_trainset.npz"))
    ap.add_argument("--library", default=_p("outputs", "phase2_7", "ego_anchor_library.json"))
    ap.add_argument("--out", default=_p("outputs", "phase2_8", "planner.pth"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--w-imit", type=float, default=1.0)
    ap.add_argument("--w-coll", type=float, default=4.0)
    ap.add_argument("--sigma-coll", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--noise-std", type=float, default=0.5, help="Gaussian jitter on modes (m), train only")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--ga-threshold", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    library = load_library(args.library)
    T = build_tensors(args.trainset, library, device)
    N = T["modes"].shape[0]

    # deterministic variant-blind train / holdout split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_hold = int(round(args.holdout_frac * N))
    hold_idx = torch.tensor(perm[:n_hold], dtype=torch.long)
    train_idx = torch.tensor(perm[n_hold:], dtype=torch.long)
    print(f"[2.8-train] N={N} train={len(train_idx)} holdout={len(hold_idx)} device={device}")

    planner = LearnedAnchorPlanner(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(planner.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))

    def gather(idx, jitter):
        modes = T["modes"][idx].to(device)
        if jitter:
            modes = modes + args.noise_std * torch.randn_like(modes)
        return (T["cmd_anchors"][T["cmd_idx"][idx].to(device)],
                modes, T["scores"][idx].to(device),
                cv_rollout_batch(T["ego_vel"][idx].to(device), PLAN_HORIZON),
                T["gt_ego"][idx].to(device), T["gt_mask"][idx].to(device))

    @torch.no_grad()
    def holdout_obj():
        planner.eval()
        a, m, s, pr, g, gm = gather(hold_idx, jitter=False)
        obj, li, lc = _objective(planner, a, m, s, pr, g, gm,
                                  args.w_imit, args.w_coll, args.sigma_coll, args.temperature)
        return float(obj), float(li), float(lc)

    history, best = [], {"epoch": -1, "holdout_obj": float("inf")}
    for ep in range(args.epochs):
        planner.train()
        tl = []
        for (bidx,) in loader:
            a, m, s, pr, g, gm = gather(bidx, jitter=True)
            obj, _, _ = _objective(planner, a, m, s, pr, g, gm,
                                   args.w_imit, args.w_coll, args.sigma_coll, args.temperature)
            opt.zero_grad(); obj.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), 5.0)
            opt.step()
            tl.append(float(obj.detach()))
        h_obj, h_imit, h_coll = holdout_obj()
        row = {"epoch": ep + 1, "train_obj": round(float(np.mean(tl)), 5),
               "holdout_obj": round(h_obj, 5), "holdout_imit": round(h_imit, 5),
               "holdout_coll": round(h_coll, 5)}
        history.append(row)
        print(f"[2.8-train] ep{ep+1}/{args.epochs} train={row['train_obj']:.4f} "
              f"hold={h_obj:.4f} (imit={h_imit:.4f} coll={h_coll:.4f})")
        if h_obj < best["holdout_obj"]:
            best = {"epoch": ep + 1, "holdout_obj": h_obj, "state": {
                k: v.detach().cpu().clone() for k, v in planner.state_dict().items()}}

    # restore variant-blind best, freeze, run Gate G-A
    planner.load_state_dict(best["state"])
    ga = gate_ga(planner, T, hold_idx, library, device)
    ga_pass = ga["flip_rate_danger_decile"] >= args.ga_threshold
    print(f"[2.8-train] BEST epoch={best['epoch']} holdout_obj={best['holdout_obj']:.4f}")
    print(f"[2.8-train] Gate G-A: flip_global={ga['flip_rate_global']:.3f} "
          f"flip_danger_decile={ga['flip_rate_danger_decile']:.3f} "
          f"(thr={args.ga_threshold}) -> {'PASS' if ga_pass else 'FAIL'}")

    config = {"hidden": args.hidden, "sigmas": list(planner.sigmas), "w_imit": args.w_imit,
              "w_coll": args.w_coll, "sigma_coll": args.sigma_coll, "temperature": args.temperature,
              "noise_std": args.noise_std, "epochs": args.epochs, "lr": args.lr,
              "batch_size": args.batch_size, "holdout_frac": args.holdout_frac, "seed": args.seed,
              "train_source": "baseline phase2_2 (variant-blind)"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_states": best["state"], "config": config,
                "selection": {"best_epoch": best["epoch"], "holdout_obj": best["holdout_obj"]},
                "gate_GA": {**ga, "threshold": args.ga_threshold, "pass": ga_pass},
                "history": history}, args.out)
    with open(_p("outputs", "phase2_8", "training_summary.json"), "w") as f:
        json.dump({"config": config, "selection": {"best_epoch": best["epoch"],
                   "holdout_obj": best["holdout_obj"]}, "gate_GA": {**ga, "pass": ga_pass},
                   "history": history}, f, indent=2)
    print(f"[2.8-train] saved {args.out}")


if __name__ == "__main__":
    main()
