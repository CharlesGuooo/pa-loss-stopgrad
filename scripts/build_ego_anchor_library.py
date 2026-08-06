"""Phase 2.7: build a fixed, model-independent ego trajectory-anchor library.

Clusters the first 6 future steps (3 s @ 0.5 s) of the train GT-ego trajectories,
grouped by a GT-derived driving command (left / straight / right), into K anchors
per command. The open-loop planner (``pa_loss_stopgrad.planners.openloop_planner``)
selects among these anchors; because the library is built once from GT-ego and is
identical for every model variant, the planner is model-independent -- the only
thing that changes across variants is the predicted-neighbor collision term.

Anchors and per-command prototypes (mean trajectory) are absolute positions in the
current-ego frame, exactly matching the npz ``gt_future_states[0, :6, :2]`` layout.

Pure numpy (no torch / sklearn): a small deterministic k-means keeps this runnable
in any environment.
"""
import argparse
import glob
import json
import os

import numpy as np

PLAN_HORIZON = 6          # 3 s @ 0.5 s, matches the nuScenes planning metric
COMMANDS = ("left", "straight", "right")


def derive_command(ego_fut_xy, lateral_thresh=1.5):
    """left / straight / right from the GT-ego lateral (y) displacement at 3 s.

    Current-ego frame: x forward, y left (nuScenes convention). Uses the final
    in-horizon waypoint's lateral offset. Model-independent (GT-derived), the same
    convention nuScenes uses to build ``gt_ego_fut_cmd``.
    """
    y_end = float(ego_fut_xy[min(PLAN_HORIZON, len(ego_fut_xy)) - 1, 1])
    if y_end > lateral_thresh:
        return "left"
    if y_end < -lateral_thresh:
        return "right"
    return "straight"


def _kmeans(X, k, iters=50, seed=3407):
    """Tiny deterministic k-means. X [N, D] -> centers [k, D]."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n <= k:
        # pad by repeating to keep a fixed library size
        idx = np.arange(n)
        reps = int(np.ceil(k / max(n, 1)))
        idx = np.tile(idx, reps)[:k]
        return X[idx].copy()
    centers = X[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None]) ** 2).sum(-1)   # [N, k]
        assign = d.argmin(1)
        new = centers.copy()
        for j in range(k):
            sel = X[assign == j]
            if len(sel):
                new[j] = sel.mean(0)
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    return centers


def main():
    ap = argparse.ArgumentParser()
    repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--train-glob",
                    default=os.path.join(repo, "data", "gameformer_nuscenes", "trainval", "train", "*.npz"))
    ap.add_argument("--k-per-command", type=int, default=22,
                    help="anchors per command (left/straight/right); ~22*3=66 total")
    ap.add_argument("--lateral-thresh", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(repo, "outputs", "phase2_7", "ego_anchor_library.json"))
    args = ap.parse_args()

    files = sorted(glob.glob(args.train_glob))
    if not files:
        raise FileNotFoundError(f"no train npz for {args.train_glob}")
    by_cmd = {c: [] for c in COMMANDS}
    skipped = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        ego = d["gt_future_states"].astype(np.float32)[0, :PLAN_HORIZON, :2]  # [6,2]
        if ego.shape[0] < PLAN_HORIZON or (np.abs(ego).sum(-1) == 0).any():
            skipped += 1
            continue
        by_cmd[derive_command(ego, args.lateral_thresh)].append(ego)

    library = {"commands": {}, "meta": {
        "plan_horizon": PLAN_HORIZON, "dt": 0.5, "k_per_command": args.k_per_command,
        "lateral_thresh": args.lateral_thresh, "n_train": len(files), "n_skipped": skipped,
        "frame": "current-ego (x forward, y left), absolute positions",
    }}
    counts = {}
    for c in COMMANDS:
        arr = np.stack(by_cmd[c], 0) if by_cmd[c] else np.zeros((0, PLAN_HORIZON, 2), np.float32)
        counts[c] = int(len(arr))
        flat = arr.reshape(len(arr), -1) if len(arr) else np.zeros((0, PLAN_HORIZON * 2), np.float32)
        centers = _kmeans(flat, args.k_per_command) if len(arr) else np.zeros((args.k_per_command, PLAN_HORIZON * 2), np.float32)
        prototype = arr.mean(0) if len(arr) else np.zeros((PLAN_HORIZON, 2), np.float32)
        library["commands"][c] = {
            "anchors": centers.reshape(-1, PLAN_HORIZON, 2).tolist(),
            "prototype": prototype.tolist(),
            "n_train_frames": counts[c],
        }
    library["meta"]["command_counts"] = counts

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(library, f)
    print(f"[anchors] commands={counts} skipped={skipped} k_per_cmd={args.k_per_command} -> {args.out}")


if __name__ == "__main__":
    main()
