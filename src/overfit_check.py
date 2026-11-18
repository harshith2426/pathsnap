"""Training-set (in-sample) scores of every final model, to compare with CV.

A large gap between the training score and the cross-validation score means
overfitting; low scores on both mean underfitting. Each model is scored on
exactly the rows it was fitted on (same seeded sample as in train_models.py).

Output: results/models/train_vs_cv.csv
Run:    python -m src.overfit_check
"""
import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, RESULTS_DIR
from .features import FEATURES, load_table
from .metrics import pair_metrics, top1_accuracy, top1_picks
from .models import make_models
from .train_models import MAX_TRAIN_ROWS, MODEL_DIR, sample_rows


def score_in_sample(model, sample, points):
    prob = model.predict_proba(sample[FEATURES].to_numpy(dtype=np.float32))[:, 1]
    pm = pair_metrics(sample.is_true.to_numpy(), prob)
    keys = sample[["trip_id", "point_idx"]].drop_duplicates()
    pts = points.merge(keys, on=["trip_id", "point_idx"])
    return pm["macro_f1"], pm["log_loss"], top1_accuracy(top1_picks(sample, prob), pts)


def main():
    train, train_pts = load_table("train")
    train_rel = train[train.label_reliable].reset_index(drop=True)
    comp = pd.read_csv(RESULTS_DIR / "models" / "model_comparison.csv").set_index("model")
    rows = []
    specs = make_models()
    for name, spec in specs.items():
        sample = sample_rows(train_rel, spec.get("max_train_rows", MAX_TRAIN_ROWS))
        model = joblib.load(MODEL_DIR / f"{name}.joblib")
        f1, ll, top1 = score_in_sample(model, sample, train_pts)
        c = comp.loc[name]
        rows.append({"model": name, "train_macro_f1": f1, "cv_macro_f1": c.cv_macro_f1,
                     "gap_macro_f1": f1 - c.cv_macro_f1, "train_top1": top1, "cv_top1": c.cv_top1_accuracy,
                     "train_log_loss": ll, "cv_log_loss": c.cv_log_loss})
        print(name, {k: round(v, 4) for k, v in rows[-1].items() if k != "model"})
    # tuned LightGBM (final classifier)
    tuned = joblib.load(ARTIFACTS_DIR / "classifier.joblib")
    sample = sample_rows(train_rel, MAX_TRAIN_ROWS)
    f1, ll, top1 = score_in_sample(tuned, sample, train_pts)
    tv = pd.read_csv(RESULTS_DIR / "tuning" / "tuned_vs_untuned.csv").set_index("model").loc["LightGBM tuned"]
    rows.append({"model": "LightGBM (tuned)", "train_macro_f1": f1, "cv_macro_f1": tv.cv_macro_f1,
                 "gap_macro_f1": f1 - tv.cv_macro_f1, "train_top1": top1, "cv_top1": tv.cv_top1_accuracy,
                 "train_log_loss": ll, "cv_log_loss": tv.cv_log_loss})
    print("LightGBM (tuned)", {k: round(v, 4) for k, v in rows[-1].items() if k != "model"})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "models" / "train_vs_cv.csv", index=False)


if __name__ == "__main__":
    main()
