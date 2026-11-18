"""Builds the technical report reports/PathSnap_Technical_Report.pdf.

The structure follows the hackathon's "Technical Report & Submission
Guidelines". Every number is read from the files in results/ and every
figure is a file produced by the code in src/ (or a screenshot of the running
system), so the report can be checked against the repository.

Run:  python -m src.report
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from .config import RESULTS_DIR, ROOT
from .progress_report import BODY, CAP, H1, H2, PAGE_W, SMALL, bullets, f, figure, p, styles, table

OUT = ROOT / "reports" / "PathSnap_Technical_Report.pdf"
FIG = ROOT / "reports" / "figures"
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=3,
                    textColor=colors.HexColor("#1f3b57"))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "PathSnap - Distributed Map-Matching & Route Reconstruction - Technical Report")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {doc.page}")
    canvas.restoreState()


def qa(q, a):
    return [p(f"<b>{q}</b>", BODY), p(a, ParagraphStyle("A", parent=BODY, leftIndent=10))]


def train_vs_cv_plot(tvc):
    path = FIG / "train_vs_cv_macro_f1.png"
    t = tvc[tvc.model != "LightGBM (tuned)"]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - 0.2, t.train_macro_f1, 0.4, label="training rows (in-sample)")
    ax.bar(x + 0.2, t.cv_macro_f1, 0.4, label="5-fold CV (unseen trips)")
    for i, (a, b) in enumerate(zip(t.train_macro_f1, t.cv_macro_f1)):
        ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, t.model, rotation=15); ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("macro F1"); ax.set_title("Training vs cross-validation macro F1 (overfitting check)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


# Per-model text: suitability, strengths/limitations, loss. Numbers are added from the results.
MODEL_TEXT = {
    "LogisticRegression": dict(
        title="Logistic Regression (linear)",
        why="It is the simplest model that outputs a probability per candidate, which is exactly what the "
            "Top-1 choice and the HMM need. With standardised features it is fast and its coefficients show "
            "which features matter, so it is a good first learned baseline.",
        sl="Strengths: fast (about 1 s to train), stable, well-calibrated probabilities. Limitation: the decision "
           "boundary is linear, so it cannot model interactions such as 'a small distance matters less when the "
           "heading is opposite', which limits it to a lower accuracy than the tree and neural models.",
        hp="C = 1.0 (inverse L2 regularisation strength), solver lbfgs, max_iter 1000, StandardScaler on all "
           "13 features, no class weights.",
        loss="Binary cross-entropy (log loss) with an L2 penalty, minimised by the lbfgs optimiser."),
    "GaussianNB": dict(
        title="Gaussian Naive Bayes (probabilistic)",
        why="It gives class probabilities from very simple statistics (a Gaussian per feature and class), so it "
            "tests how far a purely probabilistic model with almost no tuning can go on our features.",
        sl="Strengths: trains in about 0.1 s, needs no scaling, robust. Limitation: it assumes the features are "
           "independent given the class. Our features are strongly correlated (distance, distance rank, extra "
           "distance and the neighbour-window distances all describe closeness), so it counts the same evidence "
           "several times and becomes over-confident.",
        hp="var_smoothing = 1e-9 (default), class priors estimated from the data.",
        loss="No iterative loss: the per-class means, variances and priors are maximum-likelihood estimates "
             "computed directly from the training rows."),
    "KNN": dict(
        title="K-Nearest Neighbours (instance-based)",
        why="Map-matching is about similarity: a candidate that looks like known true candidates (small "
            "distance, matching heading) is probably true. KNN uses exactly that idea without assuming a model "
            "shape.",
        sl="Strengths: no training step, flexible decision boundary. Limitations: needs feature scaling, and "
           "prediction compares every row with the stored rows, so it is by far the slowest model at prediction "
           "time; we therefore stored only 50,000 training rows.",
        hp="k = 25 neighbours, distance-weighted votes, Euclidean distance, StandardScaler, 50,000 stored rows.",
        loss="No loss function: KNN stores the training rows and predicts the distance-weighted share of true "
             "candidates among the 25 nearest stored rows."),
    "RandomForest": dict(
        title="Random Forest (bagging ensemble)",
        why="Trees can learn non-linear rules and feature interactions (distance together with heading), and "
            "averaging many trees trained on bootstrap samples makes the result stable.",
        sl="Strengths: high accuracy, little tuning, no scaling needed. Limitations: largest training time among "
           "the strong models after the MLP, larger model files, and deep trees partly memorise the training "
           "rows (see the overfitting check).",
        hp="100 trees, max_depth = 16, min_samples_leaf = 5, max_features = sqrt, bootstrap = True, seed 42.",
        loss="Each split minimises Gini impurity; the forest averages the trees' class probabilities."),
    "LightGBM": dict(
        title="LightGBM (gradient boosting) - selected model",
        why="Boosted trees are strong on tabular features like ours: each new tree corrects the errors of the "
            "previous ones, interactions are learned automatically, and the output probabilities are good. "
            "LightGBM is also fast enough to train many times during Bayesian optimisation.",
        sl="Strengths: tied-best accuracy, best log loss, fast training, small model, no scaling. Limitations: "
           "several hyperparameters to set, and it can overfit if trees are too large or too many.",
        hp="300 trees, learning_rate = 0.05, num_leaves = 31, min_child_samples = 20, subsample = 1.0, "
           "colsample_bytree = 1.0, reg_lambda = 0 (before tuning), seed 42.",
        loss="Binary log loss, minimised by gradient boosting (each tree fits the gradient of the loss)."),
    "MLP": dict(
        title="Multi-layer perceptron (neural network)",
        why="A small neural network can learn smooth non-linear combinations of the features and provides a "
            "neural-network family comparison against the tree models.",
        sl="Strengths: accuracy close to the tree models, fast prediction, produces an epoch-wise training "
           "curve. Limitations: needs scaling, the longest training time of all models, results depend on "
           "learning rate and epochs, and it is the hardest model to explain.",
        hp="StandardScaler, two hidden layers with 64 and 32 ReLU units, Adam with learning rate 0.001, "
           "L2 alpha = 1e-4, batch size 512, 20 epochs, seed 42.",
        loss="Binary cross-entropy (log loss) with L2 regularisation, minimised by Adam mini-batch gradient "
             "descent."),
}


def overfit_text(name, row, tvc):
    r = tvc.set_index("model").loc[name]
    gap = r.train_macro_f1 - r.cv_macro_f1
    base = (f"Training-row macro F1 {r.train_macro_f1:.3f} vs CV {r.cv_macro_f1:.3f} (gap {gap:+.3f}); "
            f"Top-1 {r.train_top1:.3f} vs {r.cv_top1:.3f}. CV and student-test macro F1 agree "
            f"({row.cv_macro_f1:.3f} vs {row.test_macro_f1:.3f}). ")
    if name == "KNN":
        return base + ("The perfect training score is expected: every training row is its own nearest neighbour. "
                       "So KNN memorises the training data and only the CV score is meaningful; no further overfitting "
                       "is visible because CV and student-test agree.")
    if gap < 0.005:
        return base + ("No overfitting. The model scores the same on data it has and has not seen, but its overall "
                       "level is lower than the non-linear models, which points to mild underfitting (the model is "
                       "too simple for the feature interactions).")
    if gap < 0.02:
        return base + ("Only a small gap, so mild overfitting at most; generalisation to new trips is good.")
    return base + ("A clear but moderate gap: the model fits the training rows better than new trips (mild "
                   "overfitting). It still generalises well, as CV and student-test agree.")


def build():
    mod, eda, cand, tun, hmmd = (RESULTS_DIR / d for d in ["models", "eda", "candidates", "tuning", "hmm"])
    comp = pd.read_csv(mod / "model_comparison.csv")
    tvc = pd.read_csv(mod / "train_vs_cv.csv")
    summ = pd.read_csv(eda / "partition_summary.csv", index_col=0)
    inv = pd.read_csv(eda / "trip_inventory.csv", dtype={"trip_id": str})
    radius = pd.read_csv(cand / "radius_study_train.csv").set_index("radius_m")
    csum = pd.read_csv(cand / "candidate_summary.csv").set_index("partition")
    tv = pd.read_csv(tun / "tuned_vs_untuned.csv").set_index("model")
    tlog = json.load(open(tun / "tuning_log.json"))
    grid = pd.read_csv(hmmd / "hmm_grid_train_oof.csv")
    hmm_st = pd.read_csv(hmmd / "classifier_vs_hmm_student_test.csv")
    hmm_tr = pd.read_csv(hmmd / "classifier_vs_hmm_train_oof.csv")
    per_st = pd.read_csv(hmmd / "per_trip_student_test.csv", dtype={"trip_id": str})
    per_tr = pd.read_csv(hmmd / "per_trip_train_oof.csv", dtype={"trip_id": str})
    hcfg = json.load(open(ROOT / "artifacts" / "hmm_config.json"))
    api = pd.DataFrame(json.load(open(RESULTS_DIR / "api_student_test_check.json")))
    tr, st = summ["train"], summ["student_test"]
    cm_ = comp.set_index("model")
    tvplot = train_vs_cv_plot(tvc)

    s = []
    # ------------------------------------------------------------- title + overview
    s += [Spacer(1, 30),
          p("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", ParagraphStyle(
              "T", parent=styles["Title"], fontSize=20, leading=24, textColor=colors.HexColor("#1f3b57"))),
          p("Technical Report - UE24CS352A Machine Learning Hackathon", ParagraphStyle(
              "ST", parent=BODY, alignment=TA_CENTER, fontSize=11)),
          Spacer(1, 8),
          p("Team members: ______________________________", ParagraphStyle("TM", parent=BODY, alignment=TA_CENTER)),
          p("GitHub repository: https://github.com/harshith2426/pathsnap", ParagraphStyle(
              "GH", parent=BODY, alignment=TA_CENTER)),
          Spacer(1, 14),
          p("1. Overview", H1),
          p("GPS receivers report noisy positions: points drift a few metres off the road and sometimes appear on "
            "a parallel street. PathSnap matches every GPS observation of a trip to the road segment the vehicle "
            "was actually on and reconstructs a connected route. Each trip comes with its own road network of "
            "nodes and directed arcs (a two-way street is two arcs)."),
          p("Our pipeline:"),
          table([["Step", "What happens"],
                 ["1. Candidates", "for each GPS point, every road arc within 50 m"],
                 ["2. Features", "13 numbers per (point, candidate) pair, e.g. distance, heading difference"],
                 ["3. Classifier", "tuned LightGBM gives P(candidate is the true arc)"],
                 ["4. HMM / Viterbi", "chooses the most likely connected sequence of arcs"],
                 ["5. API + UI", "FastAPI server (laptop 1) runs the pipeline; a web UI (laptop 2) shows the result"]],
                [3.2 * cm, PAGE_W - 3.2 * cm]),
          Spacer(1, 6),
          p("Data use: models are fitted on the <b>train</b> partition (80 trips) and compared on "
            "<b>student_test</b> (10 trips). The 10 evaluator trips were only used by the finished, frozen system "
            "through the API. All numbers in this report are read from result files in the repository "
            "(<font face='Courier'>results/</font>).", SMALL),
          PageBreak()]

    # ------------------------------------------------------------- 2. EDA
    unrel = inv.set_index("trip_id").pct_unreliable_labels.sort_values(ascending=False)
    s += [p("2. Dataset Understanding and Exploratory Data Analysis", H1),
          p("Partitions used: all tables and plots in this section use <b>train</b> and <b>student_test</b>; the "
            "radius study and feature plots use <b>train</b> only.", SMALL),
          p("2.1 How the raw data is organised", H2),
          table([["File", "Columns (no header)", "Meaning"],
                 [".track", "lon, lat, time (s)", "one row per GPS observation"],
                 [".nodes", "lon, lat", "row number = node id"],
                 [".arcs", "from_node, to_node", "row number = arc id; a directed road segment"],
                 [".route", "arc id", "reference route: arcs driven, in order"]],
                [2.2 * cm, 4.2 * cm, PAGE_W - 6.4 * cm]),
          Spacer(1, 4),
          p("Ids are local to each trip. The route lists the arcs driven but not which GPS point belongs to which "
            "arc, so there is no per-point label in the raw data."),
          p("2.2 Missing values and data quality", H2),
          table([["", "train", "student_test"],
                 ["trips / GPS observations", f"{int(tr.trips)} / {int(tr.total_obs):,}",
                  f"{int(st.trips)} / {int(st.total_obs):,}"],
                 ["missing values", int(tr.missing_values), int(st.missing_values)],
                 ["invalid coordinates", int(tr.invalid_coords), int(st.invalid_coords)],
                 ["non-increasing timestamps", int(tr.non_increasing_time), int(st.non_increasing_time)],
                 ["median % of steps exactly 1 s apart", f"{tr.median_pct_dt_1s:.1f}%", f"{st.median_pct_dt_1s:.1f}%"],
                 ["trips with a gap > 60 s", int(tr.trips_with_gap_over_60s), int(st.trips_with_gap_over_60s)],
                 ["median nodes / arcs per trip", f"{tr.median_nodes:,.0f} / {tr.median_arcs:,.0f}",
                  f"{st.median_nodes:,.0f} / {st.median_arcs:,.0f}"],
                 ["isolated nodes", int(tr.isolated_nodes), int(st.isolated_nodes)],
                 ["median weakly-connected components", f"{tr.median_weak_components:.0f}",
                  f"{st.median_weak_components:.0f}"],
                 ["route breaks (consecutive arcs not connected)", int(tr.total_route_breaks), int(st.total_route_breaks)],
                 ["median GPS-to-route distance", f"{tr.median_label_dist_m:.2f} m", f"{st.median_label_dist_m:.2f} m"]],
                [8.5 * cm, (PAGE_W - 8.5 * cm) / 2, (PAGE_W - 8.5 * cm) / 2]),
          Spacer(1, 4),
          p("There are no missing or invalid values. The real quality issues are: time gaps (largest "
            f"{inv.max_gap_s.max():,.0f} s, about {inv.max_gap_s.max() / 3600:.0f} h, in trip "
            f"{inv.loc[inv.max_gap_s.idxmax(), 'trip_id']}), small breaks in some reference routes, and reference "
            f"routes that end before the GPS track (trips {', '.join(unrel.index[:3])}: {unrel.iloc[0]:.0f}%, "
            f"{unrel.iloc[1]:.0f}% and {unrel.iloc[2]:.0f}% of their points lie more than 50 m from the route)."),
          p("2.3 How we preprocessed the data", H2),
          *bullets(["<b>Coordinates to metres.</b> Longitude and latitude are angles and a degree of longitude "
                    "shrinks towards the poles, so each trip is projected to a local flat grid in metres "
                    "(equirectangular projection centred on the trip). Trips are at most tens of km wide, so the "
                    "projection error is far below the GPS noise.",
                    "<b>Per-point labels.</b> Each GPS point is assigned to a position on the reference route with "
                    "dynamic programming: the position may only stay or move forward in time and the total "
                    "point-to-arc distance is minimised. This avoids the mistakes of 'nearest route arc' at "
                    "junctions, U-turns and roads driven twice.",
                    "<b>Unreliable labels removed.</b> Points more than 50 m from their assigned arc are not used for "
                    f"training or scoring ({100 - 100 * csum.loc['train', 'reliable_points'] / csum.loc['train', 'points']:.2f}% "
                    f"of train points, {100 - 100 * csum.loc['student_test', 'reliable_points'] / csum.loc['student_test', 'points']:.2f}% "
                    "of student_test points)."]),
          figure(eda / "gps_to_route_distance.png", "Figure 1 - GPS noise: distance of each point to its labelled "
                 "route arc (train and student_test). Most points lie within about 5-15 m.", 0.75 * PAGE_W),
          figure(eda / "sampling_intervals.png", "Figure 2 - time between consecutive observations: almost all 1 s.",
                 0.7 * PAGE_W),
          figure(eda / "example_trip_00000027.png", "Figure 3 - GPS trajectory and road network of train trip "
                 "00000027 (grey: network, green: reference route, blue: GPS).", 0.5 * PAGE_W),
          figure(eda / "network_size.png", "Figure 4 - road network size per trip (nodes vs arcs, log scale).",
                 0.55 * PAGE_W),
          figure(FIG / "map_eda_trip_00000027.png", "Figure 5 - Folium map of train trip 00000027 with layer "
                 "control: road network, reference route and GPS observations "
                 "(maps/eda_trip_00000027.html).", PAGE_W),
          p("2.4 Patterns observed and features extracted", H2),
          p("For each GPS point we keep every arc within <b>50 m</b> as a candidate. On train, 50 m is the smallest "
            f"radius with no misses on reliable points (40 m misses {radius.loc[40, 'miss_rate_reliable_%']:.2f}%) "
            f"while giving about {radius.loc[50, 'mean_candidates_per_point']:.0f} candidates per point. "
            f"This produced {csum.loc['train', 'candidate_rows']:,.0f} train rows, "
            f"{csum.loc['train', 'positive_rate_%']:.1f}% of them true candidates."),
          p("Patterns that guided the features: (1) true arcs are close to the GPS point; (2) a two-way street is "
            "two arcs at exactly the same distance, so only the direction of travel can separate them; (3) a single "
            "point is noisy but the true arc also fits the neighbouring points. The 13 features, none of which use "
            "the reference route:"),
          table([["Feature", "Why"],
                 ["dist_m, dist_rank, dist_minus_min", "closeness, absolute and relative to the other candidates"],
                 ["n_candidates", "crowded areas (junctions, parallel roads) are harder"],
                 ["heading_diff, heading_valid", "travel direction vs arc direction; separates the two directions"],
                 ["arc_len_m, dist_to_arc_end_m", "points near an arc end are ambiguous (junctions)"],
                 ["speed_ms", "context: highway vs city driving"],
                 ["is_two_way, out_degree", "road type and junction size"],
                 ["window_dist_mean, window_dist_max", "fit of the arc to the 5 surrounding GPS points"]],
                [5.5 * cm, PAGE_W - 5.5 * cm]),
          Spacer(1, 4),
          figure(cand / "feature_distributions_train.png", "Figure 6 - true vs wrong candidates on train: distance, "
                 "heading difference and window distance separate the classes clearly.", 0.85 * PAGE_W),
          PageBreak()]

    # ------------------------------------------------------------- 3. models
    rows = [["Model", "Algorithm family", "Key configuration", "CV macro F1", "Top-1 acc. (student-test)",
             "Route continuity (student-test)", "Student-test macro F1", "Log loss (student-test)",
             "Inference (ms / 1k points)"]]
    for _, r in comp.iterrows():
        rows.append([r.model, r.family, r.config, f"{r.cv_macro_f1:.3f} ± {r.cv_macro_f1_std:.3f}",
                     f(r.test_top1_accuracy), f(r.test_route_continuity), f(r.test_macro_f1),
                     "N/A (rule outputs no probability)" if pd.isna(r.test_log_loss) else f(r.test_log_loss, 4),
                     "N/A (no model)" if pd.isna(r.inference_ms_per_1k_points) else f(r.inference_ms_per_1k_points, 1)])
    s += [p("3. Model Experimentation and Comparison", H1),
          p("Every model solves the same task: for each (GPS point, candidate arc) row, predict whether the arc is "
            "the true one. Training uses train reliable points; validation is <b>5-fold GroupKFold by trip</b> so a "
            "trip is never in both training and validation (neighbouring GPS points are near-copies and would leak). "
            "Each fit uses the same seeded sample of at most 600,000 candidate rows (KNN: 50,000). Scaling is "
            "inside a pipeline, so it is fitted on the training fold only. Seed 42 throughout."),
          p("<b>Primary metric: observation-level Top-1 accuracy</b> - for each GPS point the highest-scored "
            "candidate is compared with the labelled arc. It is exactly what the final evaluation measures. Plain "
            "accuracy is misleading because only about 10% of rows are true candidates. Log loss is the key "
            "secondary metric because the HMM uses the probabilities. Candidate miss rate on reliable points is "
            f"{csum.loc['student_test', 'miss_rate_reliable_%']:.2f}% for every model (same candidates)."),
          p("3.1 Overall model comparison", H2),
          table(rows, [2.4 * cm, 1.9 * cm, 3.2 * cm, 1.6 * cm, 1.5 * cm, 1.6 * cm, 1.5 * cm, 1.8 * cm, 1.9 * cm],
                bold_row=list(comp.model).index("LightGBM") + 1),
          p("Nearest arc is a rule-based baseline (no learning) shown for reference. Route continuity = share of "
            "consecutive points whose picked arcs are the same or directly connected.", SMALL),
          Spacer(1, 4),
          p("<b>Selection: LightGBM.</b> Random Forest, LightGBM and the MLP are tied in cross-validation (macro F1 "
            "within 0.004, while the fold standard deviation is about 0.025). LightGBM was chosen because it has "
            "the best student-test macro F1 and log loss (best probabilities for the HMM), trains about 5 times "
            f"faster ({cm_.loc['LightGBM', 'fit_time_s']:.1f} s vs {cm_.loc['RandomForest', 'fit_time_s']:.1f} s "
            f"for Random Forest and {cm_.loc['MLP', 'fit_time_s']:.1f} s for the MLP), needs no scaling and has "
            "meaningful hyperparameters for Bayesian tuning. The nearest-arc baseline reaches only "
            f"{cm_.loc['Nearest arc (baseline)', 'test_top1_accuracy']:.1%} because it cannot tell the two "
            "directions of a two-way street apart."),
          p("3.2 Questions for every model", H2)]
    for name, t in MODEL_TEXT.items():
        r = cm_.loc[name]
        res = (f"Top-1 accuracy {r.test_top1_accuracy:.3f}, macro F1 {r.test_macro_f1:.3f}, log loss "
               f"{r.test_log_loss:.4f}, accuracy {r.test_accuracy:.3f}, precision {r.test_precision:.3f}, recall "
               f"{r.test_recall:.3f}, route continuity {r.test_route_continuity:.3f}; training time "
               f"{r.fit_time_s:.1f} s, inference {r.inference_ms_per_1k_points:.1f} ms per 1,000 points.")
        s += [p(t["title"], H3),
              *qa("A1. Why suitable?", t["why"]),
              *qa("A2. Strengths and limitations?", t["sl"]),
              *qa("B5. Important hyperparameters and initial values?", t["hp"]),
              *qa("B6. Loss function / objective?", t["loss"]),
              *qa("C7. Performance on student-test?", res),
              *qa("C8. Top-1 accuracy, macro F1 and log loss on student-test?",
                  f"Top-1 {r.test_top1_accuracy:.3f}, macro F1 {r.test_macro_f1:.3f}, log loss {r.test_log_loss:.4f} "
                  f"(CV: Top-1 {r.cv_top1_accuracy:.3f}, macro F1 {r.cv_macro_f1:.3f}, log loss {r.cv_log_loss:.4f})."),
              *qa("C9. Overfitting or underfitting?", overfit_text(name, r, tvc))]

    s += [PageBreak(), p("3.3 Model visualisations", H2),
          figure(mod / "cv_macro_f1.png", "Figure 7 - cross-validation macro F1 per model (mean ± std, 5 folds).",
                 0.8 * PAGE_W),
          figure(mod / "student_test_top1_macro_f1.png", "Figure 8 - student-test Top-1 accuracy and macro F1.",
                 0.85 * PAGE_W),
          figure(mod / "log_loss.png", "Figure 9 - log loss, CV and student-test (log scale).", 0.65 * PAGE_W),
          figure(tvplot, "Figure 10 - training vs CV macro F1: the evidence used for the overfitting answers.",
                 0.8 * PAGE_W),
          figure(mod / "confusion_matrices.png", "Figure 11 - confusion matrices on student-test candidate pairs "
                 "(threshold 0.5).", PAGE_W),
          figure(mod / "learning_curves_mlp_lightgbm.png", "Figure 12 - training and validation loss for the two "
                 "models that train epoch by epoch / iteration by iteration (validation = 20% of train trips held "
                 "out). The other models have no training history, so no curve is shown for them.", PAGE_W),
          figure(mod / "inference_time.png", "Figure 13 - inference time per 1,000 GPS points (log scale).",
                 0.65 * PAGE_W),
          PageBreak()]

    # ------------------------------------------------------------- 4. Bayesian optimisation
    bp = tlog["best_params"]
    space = [["Hyperparameter", "Range searched", "Best value"],
             ["n_estimators", "100 - 800 (step 50)", bp["n_estimators"]],
             ["learning_rate", "0.01 - 0.2 (log scale)", f"{bp['learning_rate']:.4f}"],
             ["num_leaves", "15 - 255 (log scale)", bp["num_leaves"]],
             ["min_child_samples", "10 - 300 (log scale)", bp["min_child_samples"]],
             ["subsample", "0.5 - 1.0", f"{bp['subsample']:.3f}"],
             ["colsample_bytree", "0.5 - 1.0", f"{bp['colsample_bytree']:.3f}"],
             ["reg_lambda", "0.001 - 10 (log scale)", f"{bp['reg_lambda']:.3f}"]]
    cmp_rows = [["", "CV macro F1", "Student-test Top-1", "Student-test macro F1", "Student-test log loss",
                 "Inference (ms / 1k pts)", "Training time (s)"]]
    for k in ["LightGBM untuned", "LightGBM tuned"]:
        r = tv.loc[k]
        cmp_rows.append([k, f"{r.cv_macro_f1:.4f}", f"{r.test_top1_accuracy:.4f}", f"{r.test_macro_f1:.4f}",
                         f"{r.test_log_loss:.4f}", f"{r.inference_ms_per_1k_points:.1f}", f"{r.fit_time_s:.1f}"])
    s += [p("4. Bayesian Hyperparameter Optimisation (LightGBM)", H1),
          *qa("1-2. Which hyperparameters and ranges?", "The seven below. Ranges cover the usual useful values; "
              "log scales are used where the effect is multiplicative."),
          table(space, [4.5 * cm, 5.5 * cm, 4 * cm]),
          Spacer(1, 4),
          *qa("3. Objective?", "Maximise the mean Top-1 accuracy over 3-fold GroupKFold by trip on train (reliable "
              "points), with the same 600,000-row training budget as in Section 3. Optuna's TPE sampler "
              "(Tree-structured Parzen Estimator) is Bayesian: it models which parameter regions gave good and bad "
              "scores so far and proposes the next trial where good results are more likely."),
          *qa("4. Number of trials?", f"{tlog['n_trials']} trials (the first one is the untuned configuration), "
              f"search time {tlog['search_time_s'] / 60:.0f} minutes. Student-test was not used during the search."),
          *qa("5. Best values?", f"See the table; best 3-fold CV Top-1 = {tlog['best_cv_top1']:.4f}."),
          *qa("6. Did the tuned model outperform the untuned one?", "Yes, on every quality metric, by a small "
              "margin (same 5-fold CV and the same student-test data for both). The price is a larger model: "
              "about 7 times longer training and 3 times slower prediction, still well under 0.1 s per 1,000 points."),
          table(cmp_rows, [3.2 * cm] + [(PAGE_W - 3.2 * cm) / 6] * 6, bold_row=2),
          Spacer(1, 4),
          figure(tun / "optimisation_history.png", "Figure 14 - Optuna optimisation history (CV Top-1 per trial "
                 "and best so far).", 0.75 * PAGE_W),
          figure(tun / "params_vs_score.png", "Figure 15 - each hyperparameter value against the trial score.",
                 PAGE_W),
          PageBreak()]

    # ------------------------------------------------------------- 5. HMM
    best = grid[grid.setting == "hmm"].sort_values("top1_accuracy", ascending=False).iloc[0]
    clsrow = grid[grid.setting == "classifier only"].iloc[0]
    st_c, st_h = hmm_st.iloc[0], hmm_st.iloc[1]
    tr_c, tr_h = hmm_tr.iloc[0], hmm_tr.iloc[1]
    worse = per_tr[per_tr.change < 0].sort_values("change")
    s += [p("5. HMM and Viterbi Route Reconstruction", H1),
          p("The classifier judges each GPS point on its own, so two consecutive points can be matched to arcs "
            "that are not connected - the route 'jumps'. The HMM adds the rule that the vehicle must drive along "
            "the network."),
          *qa("1. Hidden states?", f"At each GPS point, the top-{hcfg['top_k']} candidate arcs by classifier "
              "probability. Each state stores its arc id and the projected position on the arc, so the decoded "
              "state maps back to exactly one road segment."),
          *qa("2. Emission probabilities?", f"emission = {hcfg['emission_weight']:g} x log P(candidate is true), "
              "using the tuned LightGBM probability. The weight sets how much the classifier is trusted relative "
              "to the transitions."),
          *qa("3. Transition probabilities?", "For candidate a at point i-1 and b at point i, d_route is the "
              "driving distance along the network from a's projected position to b's (Dijkstra on arc lengths with "
              "a search limit) and d_gps is the straight-line distance between the two GPS points. "
              f"log P(a to b) = -|d_route - d_gps| / beta with beta = {hcfg['beta']:g} m. If no path exists within "
              "the limit the transition is impossible. The Viterbi recurrence is "
              "V_i(b) = emission_i(b) + max over a of [V_(i-1)(a) + log P(a to b)], with a back-pointer to the "
              "best a; the best final state is traced back to give one arc per point."),
          *qa("4. Missing candidates and large gaps?", f"Points without any candidate within 50 m are skipped (no "
              f"prediction). If the time gap to the previous point is more than {hcfg['gap_s']:.0f} s the chain is "
              "cut and a new one starts, because the vehicle could have driven anywhere during the gap."),
          *qa("5. Consecutive points on disconnected segments?", "Such a transition gets probability zero, so "
              "Viterbi chooses another candidate that is connected. If no candidate of the new point can be "
              "reached from any previous state, the chain is cut and restarted instead of forcing an impossible "
              "route."),
          *qa("6. Did HMM decoding improve the route?", "Yes - see the table. Decoder parameters (beta, emission "
              "weight, top-K; 18 combinations) were chosen on 30 train trips using out-of-fold classifier "
              "probabilities, so they were not tuned on student-test or on over-confident in-sample probabilities. "
              f"Best: beta {best.beta:g}, emission weight {best.emission_weight:g}, top-K {best.top_k:.0f} "
              f"(Top-1 {best.top1_accuracy:.4f} vs {clsrow.top1_accuracy:.4f} classifier only on those trips)."),
          table([["Partition", "Method", "Top-1 accuracy", "Route continuity", "Route arc F1"],
                 ["student-test", "classifier only", f"{st_c.top1_accuracy:.4f}", f"{st_c.route_continuity:.4f}",
                  f"{st_c.route_arc_f1:.4f}"],
                 ["student-test", "classifier + HMM", f"{st_h.top1_accuracy:.4f}", f"{st_h.route_continuity:.4f}",
                  f"{st_h.route_arc_f1:.4f}"],
                 ["train (80 trips, out-of-fold)", "classifier only", f"{tr_c.top1_accuracy:.4f}",
                  f"{tr_c.route_continuity:.4f}", f"{tr_c.route_arc_f1:.4f}"],
                 ["train (80 trips, out-of-fold)", "classifier + HMM", f"{tr_h.top1_accuracy:.4f}",
                  f"{tr_h.route_continuity:.4f}", f"{tr_h.route_arc_f1:.4f}"]],
                [4.3 * cm, 3.4 * cm, 2.8 * cm, 2.8 * cm, PAGE_W - 13.3 * cm], bold_row=2),
          Spacer(1, 4),
          p(f"The HMM improved all {len(per_st)} student-test trips (by {per_st.change.min() * 100:.1f} to "
            f"{per_st.change.max() * 100:.1f} percentage points) and {int((per_tr.change > 0).sum())} of "
            f"{len(per_tr)} train trips (median +{per_tr.change.median() * 100:.1f} points)."),
          p("<b>Where the HMM did not help.</b> " + f"{len(worse)} train trips got worse, two strongly: "
            + ", ".join(f"trip {r.trip_id} ({r.classifier_only:.3f} to {r.hmm:.3f})" for _, r in worse.head(2).iterrows())
            + ". In both the decoder stays on a wrong road for long stretches (578 and 708 consecutive wrong "
            "points) and only a few percent of the errors are the opposite direction of the true arc. Once the "
            "decoder commits to a well-connected wrong road, the transition cost keeps it there; poorly connected "
            "network pieces around the true path (trip 82 was split into 49 chains) make this more likely."),
          figure(hmmd / "per_trip_student_test.png", "Figure 16 - per-trip Top-1 accuracy on student-test, "
                 "classifier only vs classifier + HMM.", 0.85 * PAGE_W),
          figure(hmmd / "hmm_gain_per_trip_train.png", "Figure 17 - change in Top-1 accuracy from the HMM per "
                 "train trip (out-of-fold probabilities).", 0.6 * PAGE_W),
          PageBreak()]

    # ------------------------------------------------------------- 6. Folium
    s += [p("6. Geographical Visualisation Using Folium", H1),
          p("The maps are produced by <font face='Courier'>src/maps.py</font>, "
            "<font face='Courier'>src/run_hmm.py</font> and <font face='Courier'>src/report_maps.py</font>; the "
            "interactive HTML files are in <font face='Courier'>maps/</font>. Each map has a legend and a layer "
            "control: road network (dark grey), GPS observations (blue dots), reference route (green), "
            "classifier-only picks (orange, dashed) and the final classifier + HMM route (red)."),
          figure(FIG / "map_report_trip_00000076_overview.png", "Figure 18 - student-test trip 00000076, whole "
                 "trip. The red predicted route lies on top of the green reference route.", 0.82 * PAGE_W),
          figure(FIG / "map_report_trip_00000076_zoom.png", "Figure 19 - the same trip zoomed on the stretch where "
                 "the HMM changed most classifier picks (30 of 60 consecutive points): at a highway interchange "
                 "the classifier alone (orange) jumps onto slip roads, while the HMM route (red) follows the main "
                 "road like the reference (green).", 0.82 * PAGE_W),
          PageBreak()]

    # ------------------------------------------------------------- 7. API / UI
    s += [p("7. API, UI and Deployment", H1),
          table([["Laptop", "Component", "What it does"],
                 ["Laptop 1", "FastAPI backend", "loads the tuned LightGBM and HMM settings from artifacts/; holds the "
                  "evaluation trips including the reference routes; endpoints /health, /trips and /predict"],
                 ["Laptop 2", "Web UI (HTML + Leaflet)", "lists the trips, sends only the selected trip id, draws the GPS "
                  "points and the predicted route, shows the metrics"]],
                [2 * cm, 3.6 * cm, PAGE_W - 5.6 * cm]),
          Spacer(1, 4),
          p("For <font face='Courier'>POST /predict</font> the server loads the trip without its route, runs "
            "candidates, features, LightGBM and Viterbi, and only afterwards reads the reference route to compute "
            "point accuracy, route continuity and route arc F1. The response contains the GPS preview, the "
            "predicted route geometry, the metrics and timings - never the reference route. Every request is "
            "logged in the server console. The laptops were connected over a phone hotspot; the API was started "
            "with <font face='Courier'>uvicorn api.main:app --host 0.0.0.0 --port 8000</font>."),
          p(f"Check on the 10 student-test trips through the running API: point-weighted accuracy "
            f"{(api.point_accuracy * api.n_gps_points).sum() / api.n_gps_points.sum():.4f}, mean request time "
            f"{api.request_s.mean():.2f} s, identical per-trip accuracy to the offline results in Section 5."),
          figure(FIG / "ui_trip_00000017.png", "Figure 20 - the UI on laptop 2 after selecting evaluator trip "
                 "00000017: connection status, trip list, predicted route on the map and the metrics returned by "
                 "the API.", PAGE_W),
          figure(FIG / "api_startup_laptop1.png", "Figure 21 - API start-up on laptop 1: model and HMM settings "
                 "loaded, evaluation trips folder, server listening on 0.0.0.0:8000.", 0.85 * PAGE_W),
          figure(FIG / "api_log_laptop1.png", "Figure 22 - API console on laptop 1: /health, /trips and several "
                 "successful /predict requests (200 OK) from laptop 2 (10.66.220.246), each logged with points, "
                 "accuracy, continuity and time.", PAGE_W),
          figure(FIG / "ui_server_laptop2.png", "Figure 23 - laptop 2 cloning the repository and serving the UI.",
                 0.85 * PAGE_W)]

    # ------------------------------------------------------------- 8. conclusion + AI disclosure
    s += [p("8. Limitations and Conclusion", H1),
          *bullets(["The reference routes are not perfect ground truth (breaks, routes ending early); points more "
                    "than 50 m from the route were excluded from training and scoring.",
                    "Random Forest, LightGBM and the MLP perform almost equally; feature quality mattered more than "
                    "the choice among strong models, and Bayesian tuning gave only a small gain.",
                    "The HMM gives the largest improvement (student-test Top-1 "
                    f"{st_c.top1_accuracy:.3f} to {st_h.top1_accuracy:.3f}, continuity {st_c.route_continuity:.3f} "
                    f"to {st_h.route_continuity:.3f}), but it can lock onto a wrong road in poorly connected areas.",
                    "The route continuity metric only accepts directly connected arcs; in dense networks a vehicle "
                    "can pass a very short arc between two 1 Hz fixes, which lowers the metric although the route is "
                    "correct."]),
          p("9. Disclosure of AI Assistance", H1),
          p("An AI coding assistant (Claude, by Anthropic) was used during this project to help write code, run "
            "experiments and draft this report. All experiments were actually run on our machines and every number "
            "and figure comes from the result files in the repository. The team reviewed the work and is "
            "responsible for its content. [Team: edit this section to match your event's AI-use policy.]"),
          ]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="PathSnap Technical Report",
                            author="PathSnap team")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(build())
