"""Phase 2.7b: statistical robustness for the planning + prediction legs.

Consumes the per-frame dumps from run_phase2_7b_perframe.py and produces:

  (b) bootstrap 95% CIs per variant (global + HC) for minADE_nbr, collision_proxy,
      plan_L2, coll_neighbor rate, minsep_neighbor median.
  (a) paired significance full-vs-stopgrad and full-vs-baseline on the HC subset:
        - Wilcoxon signed-rank on per-frame minsep_neighbor and minADE_nbr (+ effect size)
        - near-miss rate at {1,2,3,5} m with paired-bootstrap CI on the difference
        - collision-rate difference with paired-bootstrap CI + McNemar discordant counts
  (c) w_coll sensitivity: HC collision count + median min-sep gap (full - stopgrad)
      across the planner collision-weight grid, to show the direction is robust.

Honest by construction: it reports p-values / CIs / effect sizes rather than
asserting significance. The verdict only flags DIRECTION consistency.

Writes outputs/phase2_7b/stats.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MAXSEP = 50.0          # cap inf min-sep (no agent present) for rank/median stats
NEARMISS_THR = [1.0, 2.0, 3.0, 5.0]
RNG = np.random.default_rng(3407)


def _load_domain(perframe_dir, domain):
    out = {}
    for p in sorted(glob.glob(os.path.join(perframe_dir, f"{domain}__*.npz"))):
        variant = os.path.basename(p)[len(domain) + 2:-4]
        out[variant] = dict(np.load(p, allow_pickle=True))
    return out


def _ci(x, n_boot=2000):
    """bootstrap 95% CI of the mean of x (1-D)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan"), float("nan"), float("nan")]
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _median_ci(x, n_boot=2000):
    x = np.asarray(x, float)
    x = np.minimum(x[np.isfinite(x) | (x == np.inf)], MAXSEP)
    if len(x) == 0:
        return [float("nan")] * 3
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    return [float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def _wilcoxon(a, b):
    """paired Wilcoxon signed-rank a vs b (cap inf). Returns p + rank-biserial effect."""
    a = np.minimum(np.asarray(a, float), MAXSEP)
    b = np.minimum(np.asarray(b, float), MAXSEP)
    d = a - b
    finite = np.isfinite(d)
    a, b, d = a[finite], b[finite], d[finite]
    if len(d) == 0 or np.all(d == 0):
        return {"p": float("nan"), "effect_rank_biserial": 0.0, "n": int(len(d)),
                "median_diff": 0.0}
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p = float(res.pvalue)
    except ValueError:
        p = float("nan")
    nz = d[d != 0]
    pos = float((nz > 0).mean()) if len(nz) else 0.5
    return {"p": p, "effect_rank_biserial": 2 * pos - 1, "n": int(len(d)),
            "median_diff": float(np.median(d))}


def _rate_diff_ci(a_flag, b_flag, n_boot=2000):
    """paired bootstrap CI on rate(a)-rate(b) for boolean arrays."""
    a = np.asarray(a_flag, float); b = np.asarray(b_flag, float)
    n = len(a)
    if n == 0:
        return {"rate_a": float("nan"), "rate_b": float("nan"), "diff": float("nan"),
                "ci": [float("nan"), float("nan")]}
    idx = RNG.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    return {"rate_a": float(a.mean()), "rate_b": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def _mcnemar(a_flag, b_flag):
    """discordant counts for two paired boolean outcomes (a=full, b=other)."""
    a = np.asarray(a_flag, bool); b = np.asarray(b_flag, bool)
    a_only = int((a & ~b).sum())     # full collides, other doesn't (bad for full)
    b_only = int((~a & b).sum())     # other collides, full doesn't (good for full)
    return {"full_only": a_only, "other_only": b_only,
            "note": "other_only > full_only favours full"}


def _variant_cis(rec, mask):
    sub = lambda k: rec[k][mask]
    return {
        "n": int(mask.sum()),
        "minADE_nbr": _ci(sub("minADE_nbr")),
        "minADE_ego": _ci(sub("minADE_ego")),
        "collision_proxy": _ci(sub("collision_proxy")),
        "plan_L2": _ci(sub("plan_L2")),
        "coll_neighbor_rate": _ci(sub("coll_neighbor").astype(float)),
        "minsep_neighbor_median": _median_ci(sub("minsep_neighbor")),
    }


def _pair(full, other, mask, label):
    f = lambda r, k: r[k][mask]
    out = {"vs": label, "n": int(mask.sum())}
    out["minsep_neighbor_wilcoxon"] = _wilcoxon(f(full, "minsep_neighbor"), f(other, "minsep_neighbor"))
    out["minADE_nbr_wilcoxon"] = _wilcoxon(f(other, "minADE_nbr"), f(full, "minADE_nbr"))  # other-full: >0 => full better
    out["collision_rate"] = _rate_diff_ci(f(full, "coll_neighbor"), f(other, "coll_neighbor"))
    out["collision_mcnemar"] = _mcnemar(f(full, "coll_neighbor"), f(other, "coll_neighbor"))
    out["nearmiss"] = {}
    for thr in NEARMISS_THR:
        out["nearmiss"][f"<{thr}m"] = _rate_diff_ci(
            f(full, "minsep_neighbor") < thr, f(other, "minsep_neighbor") < thr)
    # direction consistency (full no worse): larger clearance, lower collision, lower minADE
    out["direction_consistent"] = bool(
        out["minsep_neighbor_wilcoxon"]["median_diff"] >= 0
        and out["collision_rate"]["diff"] <= 0
        and out["minADE_nbr_wilcoxon"]["median_diff"] >= 0)
    return out


def _sensitivity(full, other, mask):
    grid = full["grid_wcoll"].tolist()
    fc = full["grid_coll_neighbor"][mask]; oc = other["grid_coll_neighbor"][mask]
    fs = np.minimum(full["grid_minsep_neighbor"][mask], MAXSEP)
    os_ = np.minimum(other["grid_minsep_neighbor"][mask], MAXSEP)
    rows = []
    for gi, wc in enumerate(grid):
        rows.append({
            "w_coll": wc,
            "coll_full": int(fc[:, gi].sum()), "coll_other": int(oc[:, gi].sum()),
            "minsep_median_gap": float(np.median(fs[:, gi]) - np.median(os_[:, gi])),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perframe", default=os.path.join(REPO_ROOT, "outputs", "phase2_7b", "perframe"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "phase2_7b", "stats.json"))
    args = ap.parse_args()

    summary = {"phase": "2.7b", "maxsep_cap_m": MAXSEP, "nearmiss_thresholds_m": NEARMISS_THR,
               "seed": 3407, "domains": {}}
    for domain in ("oracle", "perceived"):
        recs = _load_domain(args.perframe, domain)
        if "full" not in recs:
            continue
        # verify token alignment across variants
        toks = {v: list(r["tokens"]) for v, r in recs.items()}
        ref = toks["full"]
        for v, t in toks.items():
            assert t == ref, f"token order mismatch {domain}/{v}"
        baseline = "oracle" if "oracle" in recs else ("stopgrad" if "stopgrad" in recs else None)
        hc = recs["full"]["is_hc"].astype(bool)
        glob_mask = np.ones(len(ref), bool)

        dom = {"baseline": baseline, "variant_cis": {}, "pairs": {}, "sensitivity": {}}
        for v, r in recs.items():
            dom["variant_cis"][v] = {"global": _variant_cis(r, glob_mask),
                                     "high_conflict": _variant_cis(r, hc)}
        for subset, mask in (("global", glob_mask), ("high_conflict", hc)):
            dom["pairs"][subset] = {}
            if "stopgrad" in recs:
                dom["pairs"][subset]["full_vs_stopgrad"] = _pair(recs["full"], recs["stopgrad"], mask, "stopgrad")
            if baseline and baseline in recs and baseline != "stopgrad":
                dom["pairs"][subset]["full_vs_baseline"] = _pair(recs["full"], recs[baseline], mask, baseline)
        if "stopgrad" in recs:
            dom["sensitivity"]["full_vs_stopgrad_HC"] = _sensitivity(recs["full"], recs["stopgrad"], hc)
        summary["domains"][domain] = dom

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[2.7b-stats] -> {args.out}")
    for domain, dom in summary["domains"].items():
        pr = dom["pairs"].get("high_conflict", {}).get("full_vs_stopgrad")
        if pr:
            ms = pr["minsep_neighbor_wilcoxon"]; nm = pr["nearmiss"]["<2.0m"]
            print(f"  {domain}/HC full_vs_stopgrad: minsep med_diff={ms['median_diff']:+.2f}m "
                  f"p={ms['p']:.3f} | nearmiss<2m diff={nm['diff']*100:+.2f}% "
                  f"CI[{nm['ci'][0]*100:+.2f},{nm['ci'][1]*100:+.2f}]% | dir_ok={pr['direction_consistent']}")


if __name__ == "__main__":
    main()
