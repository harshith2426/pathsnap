"""Stages 3-4: baseline + comparison of six candidate classifiers.

Protocol
  - Training data: TRAIN partition, reliable points only.
  - 5-fold GroupKFold by trip (a trip is never split between train and
    validation folds) -> CV macro F1, CV Top-1 accuracy, CV log loss.
  - To keep run times comparable every model is fitted on the same random
    sample of at most 600,000 candidate rows (whole points are sampled, so a
    point keeps all its candidates). KNN uses 50,000 rows (its prediction cost grows with the stored rows).
  - Final model per algorithm: fitted on the sample of the full TRAIN set and
    evaluated once on STUDENT_TEST.
  - Baseline (no learning): pick the nearest candidate arc.

Outputs: results/models/*.csv|png|json, data_derived/models/*.joblib,
         data_derived/scores_student_test.pkl
Run:  python -m src.train_models                       (all models, then merge)
      python -m src.train_models fast LogisticRegression GaussianNB KNN
      python -m src.train_models trees RandomForest LightGBM MLP
      python -m src.train_models merge                 (after parallel runs)
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import DERIVED_DIR, RANDOM_SEED, RESULTS_DIR
from .features import FEATURES, load_table
from .metrics import pair_metrics, reference_continuity, route_metrics, top1_accuracy, top1_picks
from .models import EpochMLP, make_models

OUT = RESULTS_DIR / "models"
MODEL_DIR = DERIVED_DIR / "models"
MAX_TRAIN_ROWS = 600_000
N_FOLDS = 5


def sample_rows(table, max_rows, seed=RANDOM_SEED):
    """Random sample of whole points until about max_rows candidate rows."""
    if len(table) <= max_rows:
        return table
    keys = table[["trip_id", "point_idx"]].drop_duplicates()
    frac = max_rows / len(table)
    keep = keys.sample(frac=frac, random_state=seed)
    return table.merge(keep, on=["trip_id", "point_idx"])


def xy(table):
    return table[FEATURES].to_numpy(dtype=np.float32), table.is_true.to_numpy()


def fit(model, table, max_rows):
    X, y = xy(sample_rows(table, max_rows))
    t0 = time.time()
    model.fit(X, y)
    return time.time() - t0


def cross_validate(name, spec, train_rel, train_pts):
    folds = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tr, va) in enumerate(gkf.split(train_rel, groups=train_rel.trip_id)):
        model = make_models()[name]["model"]            # fresh, unfitted copy
        fit(model, train_rel.iloc[tr], spec.get("max_train_rows", MAX_TRAIN_ROWS))
        val = train_rel.iloc[va]
        prob = model.predict_proba(val[FEATURES].to_numpy(dtype=np.float32))[:, 1]
        pm = pair_metrics(val.is_true.to_numpy(), prob)
        pts = train_pts[train_pts.trip_id.isin(val.trip_id.unique())]
        folds.append({"model": name, "fold": k, "macro_f1": pm["macro_f1"],
                      "log_loss": pm["log_loss"],
                      "top1_accuracy": top1_accuracy(top1_picks(val, prob), pts)})
        print(f"  {name} fold {k}: macroF1={pm['macro_f1']:.4f} top1={folds[-1]['top1_accuracy']:.4f}")
    return folds


def baseline_nearest(train_rel, train_pts, test, test_pts):
    """No learning: the closest candidate is predicted as the true arc."""
    pred = (test.dist_rank == 0).astype(float).to_numpy()
    rel = test.label_reliable.to_numpy()
    pm = pair_metrics(test.is_true.to_numpy()[rel], pred[rel])
    picks = top1_picks(test, -test.dist_m.to_numpy())
    rm = route_metrics(picks, test_pts, "student_test")
    cv_pred = (train_rel.dist_rank == 0).astype(float).to_numpy()
    cv_pm = pair_metrics(train_rel.is_true.to_numpy(), cv_pred)
    return {"model": "Nearest arc (baseline)", "family": "Rule (no learning)",
            "config": "argmin distance", "cv_macro_f1": cv_pm["macro_f1"], "cv_macro_f1_std": 0.0,
            "cv_top1_accuracy": top1_accuracy(top1_picks(train_rel, -train_rel.dist_m.to_numpy()), train_pts),
            "cv_log_loss": np.nan, "test_accuracy": pm["accuracy"], "test_precision": pm["precision"],
            "test_recall": pm["recall"], "test_f1": pm["f1"], "test_macro_f1": pm["macro_f1"],
            "test_log_loss": np.nan, "test_top1_accuracy": rm["top1_accuracy"],
            "test_route_continuity": rm["route_continuity"], "test_route_arc_f1": rm["route_arc_f1"],
            "fit_time_s": 0.0, "inference_ms_per_1k_points": np.nan,
            "confusion": pm["confusion"]}


def learning_curves(train_rel):
    """Per-epoch / per-iteration log loss for MLP and LightGBM on an 80/20 split
    of TRAIN trips (only these two models produce training histories)."""
    trips = np.array(sorted(train_rel.trip_id.unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    val_trips = rng.choice(trips, size=len(trips) // 5, replace=False)
    tr = sample_rows(train_rel[~train_rel.trip_id.isin(val_trips)], MAX_TRAIN_ROWS)
    va = train_rel[train_rel.trip_id.isin(val_trips)]
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)

    mlp = EpochMLP(hidden=(64, 32), epochs=20).fit(Xtr, ytr, Xva, yva)
    lgbm = make_models()["LightGBM"]["model"]
    lgbm.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "validation"],
             eval_metric="binary_logloss")
    hist = lgbm.evals_result_

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, len(mlp.history_["train_loss"]) + 1)
    axes[0].plot(ep, mlp.history_["train_loss"], label="train")
    axes[0].plot(ep, mlp.history_["val_loss"], label="validation (held-out train trips)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("log loss"); axes[0].set_title("MLP")
    axes[0].legend()
    axes[1].plot(hist["train"]["binary_logloss"], label="train")
    axes[1].plot(hist["validation"]["binary_logloss"], label="validation (held-out train trips)")
    axes[1].set_xlabel("boosting iteration"); axes[1].set_ylabel("log loss"); axes[1].set_title("LightGBM")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "learning_curves_mlp_lightgbm.png", dpi=130); plt.close(fig)
    return {"mlp": mlp.history_,
            "lightgbm": {k: v["binary_logloss"] for k, v in hist.items()},
            "validation_trips": val_trips.tolist()}


def plots(comp, folds):
    names = comp.model.tolist()
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, comp.cv_macro_f1, yerr=comp.cv_macro_f1_std, capsize=4, color="tab:blue")
    for i, v in enumerate(comp.cv_macro_f1):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=20, ha="right"); ax.set_ylim(0, 1)
    ax.set_ylabel("macro F1 (mean ± std over 5 folds)"); ax.set_title("Cross-validation macro F1 (train)")
    fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    ax.bar(x - w / 2, comp.test_top1_accuracy, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, comp.test_macro_f1, w, label="macro F1")
    for i in range(len(names)):
        ax.text(i - w / 2, comp.test_top1_accuracy[i] + 0.01, f"{comp.test_top1_accuracy[i]:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, comp.test_macro_f1[i] + 0.01, f"{comp.test_macro_f1[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, names, rotation=20, ha="right"); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Student-test performance")
    fig.tight_layout(); fig.savefig(OUT / "student_test_top1_macro_f1.png", dpi=130); plt.close(fig)

    ll = comp.dropna(subset=["test_log_loss"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ll.model, ll.cv_log_loss, 0.4, label="CV", align="edge")
    ax.bar(ll.model, ll.test_log_loss, -0.4, label="student-test", align="edge")
    ax.set_yscale("log"); ax.set_ylabel("log loss (log scale, lower is better)")
    ax.tick_params(axis="x", rotation=20); ax.legend(); ax.set_title("Log loss")
    fig.tight_layout(); fig.savefig(OUT / "log_loss.png", dpi=130); plt.close(fig)

    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow))
    axes = axes.ravel()
    for ax in axes[len(names):]:
        ax.axis("off")
    for ax, (_, r) in zip(axes, comp.iterrows()):
        cm = np.array(r.confusion)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(r.model, fontsize=9)
    fig.suptitle("Confusion matrices, student-test candidate pairs (threshold 0.5)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    t = comp.dropna(subset=["inference_ms_per_1k_points"])
    ax.bar(t.model, t.inference_ms_per_1k_points, color="tab:gray")
    ax.set_yscale("log"); ax.set_ylabel("ms per 1,000 GPS points (log)")
    ax.tick_params(axis="x", rotation=20); ax.set_title("Inference time (student-test)")
    fig.tight_layout(); fig.savefig(OUT / "inference_time.png", dpi=130); plt.close(fig)


def main(names=None, tag="all"):
    """Train/evaluate the given models (default: all) and save results with a tag.
    Several tags can run in parallel; `merge()` combines them into the final tables/plots."""
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, train_pts = load_table("train")
    test, test_pts = load_table("student_test")
    train_rel = train[train.label_reliable].reset_index(drop=True)
    test_rel_mask = test.label_reliable.to_numpy()
    X_test = test[FEATURES].to_numpy(dtype=np.float32)
    n_test_points = test_pts.shape[0]

    specs = make_models()
    names = names or list(specs)
    rows = [baseline_nearest(train_rel, train_pts, test, test_pts)]
    all_folds, log, scores = [], {}, {}
    for name in names:
        spec = specs[name]
        print(f"== {name}")
        folds = cross_validate(name, spec, train_rel, train_pts)
        all_folds += folds
        model = spec["model"]
        fit_time = fit(model, train_rel, spec.get("max_train_rows", MAX_TRAIN_ROWS))
        t0 = time.time()
        prob = model.predict_proba(X_test)[:, 1]
        infer = time.time() - t0
        scores[name] = prob
        pm = pair_metrics(test.is_true.to_numpy()[test_rel_mask], prob[test_rel_mask])
        rm = route_metrics(top1_picks(test, prob), test_pts, "student_test")
        f = pd.DataFrame(folds)
        rows.append({"model": name, "family": spec["family"], "config": spec["config"],
                     "cv_macro_f1": f.macro_f1.mean(), "cv_macro_f1_std": f.macro_f1.std(),
                     "cv_top1_accuracy": f.top1_accuracy.mean(), "cv_log_loss": f.log_loss.mean(),
                     "test_accuracy": pm["accuracy"], "test_precision": pm["precision"],
                     "test_recall": pm["recall"], "test_f1": pm["f1"], "test_macro_f1": pm["macro_f1"],
                     "test_log_loss": pm["log_loss"], "test_top1_accuracy": rm["top1_accuracy"],
                     "test_route_continuity": rm["route_continuity"], "test_route_arc_f1": rm["route_arc_f1"],
                     "fit_time_s": fit_time, "inference_ms_per_1k_points": 1000 * infer / n_test_points * 1000,
                     "confusion": pm["confusion"]})
        log[name] = {"family": spec["family"], "config": spec["config"], "seed": RANDOM_SEED,
                     "params": {k: str(v) for k, v in model.get_params().items()},
                     "max_train_rows": spec.get("max_train_rows", MAX_TRAIN_ROWS)}
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"  test top1={rm['top1_accuracy']:.4f} macroF1={pm['macro_f1']:.4f} "
              f"logloss={pm['log_loss']:.4f} fit={fit_time:.1f}s")

    comp = pd.DataFrame(rows)
    comp["confusion"] = comp.confusion.apply(json.dumps)
    comp.to_csv(OUT / f"part_comparison_{tag}.csv", index=False)
    pd.DataFrame(all_folds).to_csv(OUT / f"part_cv_folds_{tag}.csv", index=False)
    pd.DataFrame(scores).to_pickle(DERIVED_DIR / f"part_scores_{tag}.pkl")
    if "LightGBM" in names or "MLP" in names:
        log["curves"] = learning_curves(train_rel)
    with open(OUT / f"part_log_{tag}.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    print("done", tag)


def merge():
    """Combine all part_* results into the final comparison table, logs and plots."""
    _, test_pts = load_table("student_test")
    test, _ = load_table("student_test")
    order = ["Nearest arc (baseline)"] + list(make_models())
    comp = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("part_comparison_*.csv"))])
    comp = comp.drop_duplicates("model").set_index("model").loc[
        [m for m in order if m in set(comp.model)]].reset_index()
    comp["confusion"] = comp.confusion.apply(json.loads)
    folds = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("part_cv_folds_*.csv"))])
    scores = pd.concat([pd.read_pickle(p) for p in sorted(DERIVED_DIR.glob("part_scores_*.pkl"))], axis=1)
    log = {}
    for p in sorted(OUT.glob("part_log_*.json")):
        log.update(json.load(open(p)))
    log["reference_route_continuity_student_test"] = reference_continuity(test_pts, "student_test")
    log["protocol"] = {"cv": f"GroupKFold({N_FOLDS}) by trip on train reliable points",
                       "max_train_rows": MAX_TRAIN_ROWS, "seed": RANDOM_SEED, "features": FEATURES}
    comp.drop(columns="confusion").to_csv(OUT / "model_comparison.csv", index=False)
    folds.to_csv(OUT / "cv_folds.csv", index=False)
    scores.assign(trip_id=test.trip_id.values, point_idx=test.point_idx.values,
                  arc_id=test.arc_id.values).to_pickle(DERIVED_DIR / "scores_student_test.pkl")
    with open(OUT / "experiment_log.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    plots(comp, folds)
    cols = ["model", "cv_macro_f1", "cv_macro_f1_std", "cv_top1_accuracy", "cv_log_loss", "test_macro_f1",
            "test_top1_accuracy", "test_log_loss", "test_route_continuity", "test_route_arc_f1",
            "fit_time_s", "inference_ms_per_1k_points"]
    print(comp[cols].round(4).to_string(index=False))
    print("reference continuity (labels):", round(log["reference_route_continuity_student_test"], 4))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge()
    elif len(sys.argv) > 2:
        main(sys.argv[2:], tag=sys.argv[1])        # python -m src.train_models <tag> <model> ...
    else:
        main()
        merge()
