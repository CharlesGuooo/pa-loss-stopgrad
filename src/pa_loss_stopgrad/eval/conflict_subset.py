"""GameFormer-native high-conflict subset for Phase 2.3.

Labels each preprocessed val npz by reusing the validated Phase 1.9 conflict
classifier (`pa_loss_stopgrad.eval.conflict_mining.classify_frame`) on the two
agents GameFormer actually predicts: ego (agent 0) and neighbor0 (agent 1), both
already in the current-ego frame. Produces a label JSON aligned to the sorted
npz order so the fine-tune eval can split metrics into global vs high-conflict.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from pa_loss_stopgrad.eval.conflict_mining import STRATA, classify_frame

MIN_VALID_STEPS = 2


def _sample_from_npz(path):
    d = np.load(path, allow_pickle=True)
    gt = d["gt_future_states"].astype(np.float32)  # [2, T, 5]
    ego = gt[0, :, :2]
    nb = gt[1, :, :2]
    nb_valid = int((np.abs(nb).sum(-1) != 0).sum())
    token = os.path.splitext(os.path.basename(path))[0]
    if "meta" in d:
        try:
            token = d["meta"].item().get("sample_token", token)
        except Exception:  # noqa: BLE001
            pass
    return ego, nb, nb_valid, token


def label_val_npz(val_glob, min_valid_steps: int = MIN_VALID_STEPS):
    files = sorted(glob.glob(val_glob))
    if not files:
        raise FileNotFoundError(f"No npz for glob: {val_glob}")
    tokens, labels = [], []
    for f in files:
        ego, nb, nvalid, token = _sample_from_npz(f)
        if nvalid < min_valid_steps:
            label = "cruising"
        else:
            sample = {"ego_fut_abs": ego, "gt_agent_fut_abs": nb[None], "ego_history_abs": None}
            label = classify_frame(sample)["label"]
        tokens.append(token)
        labels.append(label)
    counts = {name: int(labels.count(name)) for name in STRATA}
    num = max(len(labels), 1)
    hc_idx = [i for i, lab in enumerate(labels) if lab == "high_conflict"]
    return {
        "val_glob": val_glob,
        "num_frames": len(labels),
        "counts": counts,
        "fractions": {name: counts[name] / num for name in STRATA},
        "prevalence_high_conflict": counts["high_conflict"] / num,
        "tokens": tokens,
        "labels": labels,
        "high_conflict_indices": hc_idx,
    }


def load_conflict_mask(json_path):
    """Boolean mask (sorted-npz order) marking high-conflict frames."""
    with open(json_path) as f:
        rec = json.load(f)
    return np.array([lab == "high_conflict" for lab in rec["labels"]], dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    from pa_loss_stopgrad.paths import gf_split

    trainval = gf_split("trainval")
    default_glob = str(trainval / "val" / "*.npz")
    default_out = str(trainval / "val_conflict_labels.json")
    ap.add_argument("--val-glob", default=default_glob)
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    rec = label_val_npz(args.val_glob)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f)
    print(f"DONE frames={rec['num_frames']} counts={rec['counts']} "
          f"high_conflict={rec['prevalence_high_conflict']:.3%} out={args.out}")


if __name__ == "__main__":
    main()
