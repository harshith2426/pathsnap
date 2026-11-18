"""Stage 6 experiments: classifier-only vs classifier + HMM/Viterbi.

1. Decoder parameters (beta, emission weight, top-K) are chosen by a small
   grid search on TRAIN using out-of-fold classifier probabilities (so the
   scores are as noisy as on unseen trips). 30 random train trips are used to
   keep the search fast.
2. The chosen setting is evaluated on STUDENT_TEST and compared with
   classifier-only Top-1 picks: Top-1 accuracy, route continuity, route arc F1,
   per-trip differences.
3. Folium map of a student-test trip: GPS, reference, classifier-only, HMM.

Run:  python -m src.run_hmm      (needs python -m src.tune first)
"""
import itertools
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, DERIVED_DIR, MAPS_DIR, RANDOM_SEED, RESULTS_DIR
from .data_io import load_trip
from .features import load_table
from .hmm import GAP_S, RoadGraph, classifier_only, viterbi_trip
from .maps import trip_map
from .metrics import route_metrics

OUT = RESULTS_DIR / "hmm"
GRID = {"beta": [3.0, 10.0, 30.0], "emission_weight": [0.5, 1.0, 2.0], "top_k": [5, 8]}
N_TUNE_TRIPS = 30


def decode_partition(partition, table, prob, trips, params, use_hmm=True):
    """Return per-point picks (trip_id, point_idx, arc_id) for the given trips."""
    out = []
    for tid in trips:
        mask = (table.trip_id == tid).to_numpy()
        cand = table[mask].reset_index(drop=True)
        if cand.empty:
            continue
        if use_hmm:
            trip = load_trip(partition, tid, with_route=False)
            res, _ = viterbi_trip(trip, cand, prob[mask], params, RoadGraph(trip))
        else:
            res = classifier_only(cand, prob[mask])
        res.insert(0, "trip_id", tid)
        out.append(res[["trip_id", "point_idx", "arc_id"]])
    return pd.concat(out, ignore_index=True)


def with_truth(picks, points):
    lab = points[["trip_id", "point_idx", "true_arc"]]
    p = picks.merge(lab, on=["trip_id", "point_idx"], how="left")
    p["is_true"] = (p.arc_id == p.true_arc).astype(int)
    return p.drop(columns="true_arc")


def per_trip_accuracy(picks, points):
    rel = points[points.label_reliable]
    m = rel.merge(picks, on=["trip_id", "point_idx"], how="left")
    m["ok"] = (m.arc_id == m.true_arc)
    return m.groupby("trip_id").ok.mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, train_pts = load_table("train")
    test, test_pts = load_table("student_test")
    oof = pd.read_pickle(DERIVED_DIR / "oof_train_tuned.pkl").prob.to_numpy()
    test_prob = pd.read_pickle(DERIVED_DIR / "test_prob_tuned.pkl").prob.to_numpy()

    # ---- 1. grid search on train (OOF probabilities)
    rng = np.random.default_rng(RANDOM_SEED)
    tune_trips = sorted(rng.choice(sorted(train.trip_id.unique()), N_TUNE_TRIPS, replace=False))
    sub_pts = train_pts[train_pts.trip_id.isin(tune_trips)]
    rows = []
    base = with_truth(decode_partition("train", train, oof, tune_trips, None, use_hmm=False), sub_pts)
    rows.append({"setting": "classifier only", **route_metrics(base, sub_pts, "train")})
    for beta, w, k in itertools.product(GRID["beta"], GRID["emission_weight"], GRID["top_k"]):
        params = {"beta": beta, "emission_weight": w, "top_k": k}
        t0 = time.time()
        picks = with_truth(decode_partition("train", train, oof, tune_trips, params), sub_pts)
        rows.append({"setting": "hmm", **params, **route_metrics(picks, sub_pts, "train"),
                     "decode_s": time.time() - t0})
        print(params, f"top1={rows[-1]['top1_accuracy']:.4f}")
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "hmm_grid_train_oof.csv", index=False)
    best = grid[grid.setting == "hmm"].sort_values("top1_accuracy", ascending=False).iloc[0]
    best_params = {"beta": float(best.beta), "emission_weight": float(best.emission_weight),
                   "top_k": int(best.top_k)}
    print("best HMM params:", best_params)

    # ---- 2. student-test comparison
    trips = sorted(test.trip_id.unique())
    cls = with_truth(decode_partition("student_test", test, test_prob, trips, None, use_hmm=False), test_pts)
    t0 = time.time()
    hmm = with_truth(decode_partition("student_test", test, test_prob, trips, best_params), test_pts)
    decode_time = time.time() - t0
    comp = pd.DataFrame([{"method": "classifier only (tuned LightGBM)", **route_metrics(cls, test_pts, "student_test")},
                         {"method": "classifier + HMM/Viterbi", **route_metrics(hmm, test_pts, "student_test")}])
    comp.to_csv(OUT / "classifier_vs_hmm_student_test.csv", index=False)
    per_trip = pd.DataFrame({"classifier_only": per_trip_accuracy(cls, test_pts),
                             "hmm": per_trip_accuracy(hmm, test_pts)})
    per_trip["change"] = per_trip.hmm - per_trip.classifier_only
    per_trip.to_csv(OUT / "per_trip_student_test.csv")
    print(comp.round(4).to_string(index=False))
    print(per_trip.round(4).to_string())

    # Also train (all 80 trips, OOF) for a larger sample
    all_train = sorted(train.trip_id.unique())
    cls_tr = with_truth(decode_partition("train", train, oof, all_train, None, use_hmm=False), train_pts)
    hmm_tr = with_truth(decode_partition("train", train, oof, all_train, best_params), train_pts)
    comp_tr = pd.DataFrame([{"method": "classifier only (OOF)", **route_metrics(cls_tr, train_pts, "train")},
                            {"method": "classifier + HMM (OOF)", **route_metrics(hmm_tr, train_pts, "train")}])
    comp_tr.to_csv(OUT / "classifier_vs_hmm_train_oof.csv", index=False)
    per_trip_tr = pd.DataFrame({"classifier_only": per_trip_accuracy(cls_tr, train_pts),
                                "hmm": per_trip_accuracy(hmm_tr, train_pts)})
    per_trip_tr["change"] = per_trip_tr.hmm - per_trip_tr.classifier_only
    per_trip_tr.to_csv(OUT / "per_trip_train_oof.csv")
    print(comp_tr.round(4).to_string(index=False))

    # Save decoder configuration for the API
    with open(ARTIFACTS_DIR / "hmm_config.json", "w") as fh:
        json.dump({**best_params, "gap_s": GAP_S, "tuned_on": f"{N_TUNE_TRIPS} train trips, OOF probabilities",
                   "student_test_decode_time_s": decode_time}, fh, indent=1)

    # ---- plots
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(per_trip))
    ax.bar(x - 0.2, per_trip.classifier_only, 0.4, label="classifier only")
    ax.bar(x + 0.2, per_trip.hmm, 0.4, label="classifier + HMM")
    ax.set_xticks(x, per_trip.index, rotation=30); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 accuracy"); ax.set_title("Per-trip accuracy, student-test"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "per_trip_student_test.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_trip_tr.change * 100, bins=30, color="tab:purple")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("HMM - classifier Top-1 accuracy (percentage points)"); ax.set_ylabel("train trips")
    ax.set_title("Effect of HMM per trip (train, OOF probabilities)")
    fig.tight_layout(); fig.savefig(OUT / "hmm_gain_per_trip_train.png", dpi=130); plt.close(fig)

    # ---- map of the student-test trip where the HMM changed most
    tid = per_trip.change.abs().idxmax()
    trip = load_trip("student_test", tid)
    m = trip_map(trip, route_arcs=trip.route, predicted_arcs=hmm[hmm.trip_id == tid].arc_id.to_numpy(),
                 title=f"Trip {tid} (student_test): HMM route in red",
                 extra_layers=[("Classifier-only picks", cls[cls.trip_id == tid].arc_id.to_numpy(), "orange")])
    m.save(str(MAPS_DIR / f"hmm_trip_{tid}.html"))
    print("map:", MAPS_DIR / f"hmm_trip_{tid}.html")


if __name__ == "__main__":
    main()
