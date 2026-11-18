"""Summaries and plots of the candidate/feature tables (TRAIN for plots).

Run:  python -m src.feature_eda   -> results/candidates/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .features import FEATURES, load_table

OUT = RESULTS_DIR / "candidates"


def summary_rows():
    rows = []
    for part in ["train", "student_test"]:
        table, pts = load_table(part)
        rel = pts[pts.label_reliable]
        rows.append({
            "partition": part,
            "points": len(pts),
            "reliable_points": len(rel),
            "candidate_rows": len(table),
            "positive_rate_%": 100 * table.is_true.mean(),
            "mean_candidates_per_point": len(table) / len(pts),
            "points_without_candidates_%": 100 * (pts.n_candidates == 0).mean(),
            "miss_rate_reliable_%": 100 * (~rel.true_in_candidates).mean(),
            "miss_rate_all_%": 100 * (~pts.true_in_candidates).mean(),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = summary_rows()
    summary.to_csv(OUT / "candidate_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    table, _ = load_table("train")
    table = table[table.label_reliable]
    means = table.groupby("is_true")[FEATURES].mean().T
    means.columns = ["wrong_candidate_mean", "true_candidate_mean"]
    means.to_csv(OUT / "feature_means_by_label_train.csv")
    print(means.round(2).to_string())

    # Distributions of the most informative features, true vs wrong candidates
    plots = [("dist_m", "distance to arc (m)", (0, 50)),
             ("heading_diff", "heading difference (deg)", (0, 180)),
             ("window_dist_mean", "mean distance of 5 neighbouring points (m)", (0, 80)),
             ("dist_rank", "distance rank among candidates", (0, 15))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, label, rng) in zip(axes.ravel(), plots):
        for val, color, name in [(0, "tab:red", "wrong candidate"), (1, "tab:green", "true candidate")]:
            ax.hist(table.loc[table.is_true == val, col].clip(*rng), bins=40, range=rng,
                    density=True, alpha=0.55, color=color, label=name)
        ax.set_xlabel(label); ax.set_ylabel("density")
    axes[0, 0].legend()
    fig.suptitle("Feature distributions, true vs wrong candidates (train, reliable points)")
    fig.tight_layout(); fig.savefig(OUT / "feature_distributions_train.png", dpi=130); plt.close(fig)

    # Number of candidates per point
    _, pts = load_table("train")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pts.n_candidates.clip(upper=40), bins=np.arange(0, 42), color="tab:blue")
    ax.set_xlabel("candidates per GPS point (radius 50 m, clipped at 40)")
    ax.set_ylabel("GPS points"); ax.set_title("Candidate set size (train)")
    fig.tight_layout(); fig.savefig(OUT / "candidates_per_point_train.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
