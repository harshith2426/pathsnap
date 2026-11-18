"""Stage 5: Bayesian hyperparameter optimisation of the selected model (LightGBM).

Method   : Optuna with the TPE sampler (Tree-structured Parzen Estimator), a
           Bayesian method: it builds a probability model of which parameter
           values gave good scores so far and proposes the next trial where
           improvement is most likely (instead of a blind grid/random search).
Objective: maximise mean Top-1 accuracy over 3-fold GroupKFold (by trip) on
           TRAIN reliable points. Log loss and macro F1 are logged per trial.
Data     : every fold trains on the same 600k-row sample size as Stage 4, so
           tuned vs untuned is a fair comparison. STUDENT_TEST is only used
           after tuning to report the result. Evaluator data: never.
Also     : saves out-of-fold (OOF) probabilities of the tuned model on TRAIN,
           used to tune the HMM without using in-sample (overfitted) scores.

Run:  python -m src.tune
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from .config import ARTIFACTS_DIR, DERIVED_DIR, RANDOM_SEED, RESULTS_DIR
from .features import FEATURES, RADIUS_M, load_table
from .metrics import pair_metrics, route_metrics, top1_accuracy, top1_picks
from .train_models import MAX_TRAIN_ROWS, sample_rows, xy

OUT = RESULTS_DIR / "tuning"
N_TRIALS = 25
N_FOLDS_TUNE = 3


def search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def make_lgbm(params):
    return LGBMClassifier(**params, subsample_freq=1, n_jobs=-1,
                          random_state=RANDOM_SEED, verbose=-1)


def cv_score(params, train_rel, train_pts, n_folds):
    tops, f1s, lls = [], [], []
    for tr, va in GroupKFold(n_splits=n_folds).split(train_rel, groups=train_rel.trip_id):
        X, y = xy(sample_rows(train_rel.iloc[tr], MAX_TRAIN_ROWS))
        model = make_lgbm(params).fit(X, y)
        val = train_rel.iloc[va]
        prob = model.predict_proba(val[FEATURES].to_numpy(dtype=np.float32))[:, 1]
        pm = pair_metrics(val.is_true.to_numpy(), prob)
        pts = train_pts[train_pts.trip_id.isin(val.trip_id.unique())]
        tops.append(top1_accuracy(top1_picks(val, prob), pts))
        f1s.append(pm["macro_f1"]); lls.append(pm["log_loss"])
    return np.mean(tops), np.mean(f1s), np.mean(lls)


def out_of_fold(params, train, train_rel, n_folds=5):
    """OOF probabilities for EVERY train candidate row (incl. unreliable points)."""
    oof = np.zeros(len(train))
    trip_folds = {}
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(train_rel, groups=train_rel.trip_id)):
        for t in train_rel.trip_id.iloc[va].unique():
            trip_folds[t] = k
    fold_of_row = train.trip_id.map(trip_folds).to_numpy()
    for k in range(n_folds):
        fit_rows = train_rel[train_rel.trip_id.map(trip_folds) != k]
        X, y = xy(sample_rows(fit_rows, MAX_TRAIN_ROWS))
        model = make_lgbm(params).fit(X, y)
        mask = fold_of_row == k
        oof[mask] = model.predict_proba(train.loc[mask, FEATURES].to_numpy(dtype=np.float32))[:, 1]
    return oof


def evaluate_on_test(model, test, test_pts):
    X = test[FEATURES].to_numpy(dtype=np.float32)
    t0 = time.time()
    prob = model.predict_proba(X)[:, 1]
    infer = time.time() - t0
    rel = test.label_reliable.to_numpy()
    pm = pair_metrics(test.is_true.to_numpy()[rel], prob[rel])
    rm = route_metrics(top1_picks(test, prob), test_pts, "student_test")
    return {"test_macro_f1": pm["macro_f1"], "test_log_loss": pm["log_loss"],
            "test_top1_accuracy": rm["top1_accuracy"], "test_route_continuity": rm["route_continuity"],
            "inference_ms_per_1k_points": 1000 * infer / len(test_pts) * 1000}, prob


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, train_pts = load_table("train")
    test, test_pts = load_table("student_test")
    train_rel = train[train.label_reliable].reset_index(drop=True)

    untuned_params = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                      "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                      "reg_lambda": 0.0}

    def objective(trial):
        top1, f1, ll = cv_score(search_space(trial), train_rel, train_pts, N_FOLDS_TUNE)
        trial.set_user_attr("cv_macro_f1", f1)
        trial.set_user_attr("cv_log_loss", ll)
        return top1

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.enqueue_trial({k: v for k, v in untuned_params.items() if k != "reg_lambda"} | {"reg_lambda": 1e-3})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_time = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = study.best_params
    print("best params:", best, "cv top1:", round(study.best_value, 5), f"search {search_time:.0f}s")

    # Same 3-fold CV for the untuned config, then 5-fold for both (comparable to Stage 4)
    rows = []
    for name, params in [("LightGBM untuned", untuned_params), ("LightGBM tuned", best)]:
        top1, f1, ll = cv_score(params, train_rel, train_pts, 5)
        X, y = xy(sample_rows(train_rel, MAX_TRAIN_ROWS))
        t1 = time.time()
        model = make_lgbm(params).fit(X, y)
        fit_time = time.time() - t1
        res, prob = evaluate_on_test(model, test, test_pts)
        rows.append({"model": name, "cv_macro_f1": f1, "cv_top1_accuracy": top1, "cv_log_loss": ll,
                     **res, "fit_time_s": fit_time})
        if name == "LightGBM tuned":
            tuned_model, tuned_prob = model, prob
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(comp.round(4).to_string(index=False))

    # Export the final classifier for the API
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(tuned_model, ARTIFACTS_DIR / "classifier.joblib")
    with open(ARTIFACTS_DIR / "classifier_config.json", "w") as fh:
        json.dump({"model": "LightGBM (Optuna-tuned)", "params": best, "features": FEATURES,
                   "candidate_radius_m": RADIUS_M, "max_train_rows": MAX_TRAIN_ROWS,
                   "seed": RANDOM_SEED}, fh, indent=1)

    # OOF train scores + student-test scores for the HMM stage
    pd.DataFrame({"prob": out_of_fold(best, train, train_rel)}).to_pickle(DERIVED_DIR / "oof_train_tuned.pkl")
    pd.DataFrame({"prob": tuned_prob}).to_pickle(DERIVED_DIR / "test_prob_tuned.pkl")

    with open(OUT / "tuning_log.json", "w") as fh:
        json.dump({"sampler": "TPE", "n_trials": N_TRIALS, "cv_folds": N_FOLDS_TUNE,
                   "objective": "mean CV Top-1 accuracy (maximise)", "best_params": best,
                   "best_cv_top1": study.best_value, "search_time_s": search_time,
                   "untuned_params": untuned_params}, fh, indent=1)

    # Optimisation history plot
    vals = trials.value.to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(np.arange(len(vals)), vals, s=18, label="trial")
    ax.plot(np.maximum.accumulate(vals), color="red", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 accuracy (3-fold)")
    ax.set_title("Optuna (TPE) optimisation history - LightGBM"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "optimisation_history.png", dpi=130); plt.close(fig)

    # Parameter vs score scatter
    params = [c for c in trials.columns if c.startswith("params_")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), params):
        ax.scatter(trials[col], vals, s=14)
        ax.set_xlabel(col.replace("params_", "")); ax.set_ylabel("CV Top-1")
        if col.replace("params_", "") in {"learning_rate", "num_leaves", "min_child_samples", "reg_lambda"}:
            ax.set_xscale("log")
    for ax in axes.ravel()[len(params):]:
        ax.axis("off")
    fig.suptitle("Hyperparameter value vs CV Top-1 accuracy (each dot = one trial)")
    fig.tight_layout(); fig.savefig(OUT / "params_vs_score.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
