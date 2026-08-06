"""Phase 2.7-WP: pick the minADE-GUARDED working-point epoch for the retrained
gradient-flowing variants, and emit a ckpt-map JSON for the 2.7 / 2.7b drivers.

Working-point rule (same as Phase 2.3):
    among the variant's epochs, choose the one with the LOWEST collision_global
    subject to  minADE_mean <= GUARD * stopgrad_final_minADE  (GUARD = 1.5).
    If no epoch satisfies the guard, fall back to the lowest-minADE epoch.

stop-grad and the no-PA baseline are gradient-blocked / frozen -> reused as-is
(their best.pth is already at a stable minADE), so they are NOT remapped.
"""
import argparse
import json
import os

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GUARD = 1.5


def _p(*parts):
    return os.path.join(REPO_ROOT, *parts)


def _final_minade(summary_path):
    with open(summary_path) as f:
        s = json.load(f)
    return float(s["final"]["minADE_mean"])


def _pick_working_point(summary_path, guard_minade):
    with open(summary_path) as f:
        s = json.load(f)
    hist = s["history"]
    ok = [r for r in hist if r["minADE_mean"] <= guard_minade]
    pool = ok if ok else hist
    best = min(pool, key=lambda r: r["collision_global"])
    return best["epoch"], best, bool(ok)


# (domain, retrained out-dir, stopgrad summary for the guard)
DOMAINS = {
    "oracle": {
        "dir": _p("outputs", "phase2_3b"),
        "stopgrad_summary": _p("outputs", "phase2_3", "stopgrad", "summary.json"),
        "variants": {"full": "full", "wta": "wta"},
    },
    "perceived": {
        "dir": _p("outputs", "phase2_6b", "perceived_perceived"),
        "stopgrad_summary": _p("outputs", "phase2_6", "perceived_perceived", "stopgrad", "summary.json"),
        "variants": {"full": "full", "wta": "wta"},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", type=float, default=GUARD)
    ap.add_argument("--out", default=_p("outputs", "phase2_7b", "ckpt_map.json"))
    args = ap.parse_args()

    cmap, report = {}, {}
    for domain, cfg in DOMAINS.items():
        sg_minade = _final_minade(cfg["stopgrad_summary"])
        guard_minade = args.guard * sg_minade
        cmap[domain], report[domain] = {}, {"stopgrad_minADE": sg_minade,
                                            "guard_minADE": guard_minade, "variants": {}}
        for variant, sub in cfg["variants"].items():
            summ = os.path.join(cfg["dir"], sub, "summary.json")
            if not os.path.exists(summ):
                print(f"[wp] MISSING {summ} -- run retrain first"); continue
            epoch, row, satisfied = _pick_working_point(summ, guard_minade)
            ckpt = os.path.join(cfg["dir"], sub, f"epoch_{epoch}.pth")
            cmap[domain][variant] = ckpt
            report[domain]["variants"][variant] = {
                "epoch": epoch, "guard_satisfied": satisfied,
                "minADE_mean": row["minADE_mean"], "collision_global": row["collision_global"],
                "collision_hc": row["collision_hc"], "ckpt": ckpt}
            print(f"[wp] {domain}/{variant}: epoch={epoch} guard_ok={satisfied} "
                  f"minADE={row['minADE_mean']:.3f} (<= {guard_minade:.3f}) "
                  f"coll_g={row['collision_global']:.4f} coll_hc={row['collision_hc']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cmap, f, indent=2)
    with open(args.out.replace(".json", "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[wp] ckpt-map -> {args.out}")


if __name__ == "__main__":
    main()
