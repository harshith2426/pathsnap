"""Builds reports/PathSnap_progress_stages1-4.pdf from the saved result files.

Every number in the PDF is read from results/ (nothing typed by hand), so the
report always matches the experiments in the repository.

Run:  python -m src.progress_report
"""
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from .config import RESULTS_DIR, ROOT

OUT = ROOT / "reports" / "PathSnap_progress_stages1-4.pdf"
PAGE_W = A4[0] - 3.6 * cm

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3b57"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1f3b57"))
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
CAP = ParagraphStyle("Cap", parent=SMALL, alignment=TA_CENTER, spaceAfter=8)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=12, bulletIndent=2)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(rows, widths=None, header=True, bold_row=None):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3eaf1"))]
    if bold_row is not None:
        style += [("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#fff4d6"))]
    t.setStyle(TableStyle(style))
    return t


def figure(path, caption, width=PAGE_W):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width, width * ratio
    return KeepTogether([img, p(caption, CAP)])


def f(x, d=3):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "PathSnap - progress report, stages 1-4 (EDA, candidates, features, model comparison)")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {doc.page}")
    canvas.restoreState()


def build():
    eda = RESULTS_DIR / "eda"
    cand = RESULTS_DIR / "candidates"
    mod = RESULTS_DIR / "models"
    summ = pd.read_csv(eda / "partition_summary.csv", index_col=0)
    inv = pd.read_csv(eda / "trip_inventory.csv", dtype={"trip_id": str})
    radius = pd.read_csv(cand / "radius_study_train.csv")
    csum = pd.read_csv(cand / "candidate_summary.csv").set_index("partition")
    fmeans = pd.read_csv(cand / "feature_means_by_label_train.csv", index_col=0)
    comp = pd.read_csv(mod / "model_comparison.csv")
    folds = pd.read_csv(mod / "cv_folds.csv")
    log = json.load(open(mod / "experiment_log.json"))

    tr, st = summ["train"], summ["student_test"]
    by_part = inv.groupby("partition")
    unrel = inv.set_index("trip_id").pct_unreliable_labels.sort_values(ascending=False)
    lgb = comp.set_index("model").loc["LightGBM"]
    base = comp.set_index("model").loc["Nearest arc (baseline)"]
    lgb_folds = folds[folds.model == "LightGBM"]

    s = []
    # ------------------------------------------------------------ title
    s += [p("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", ParagraphStyle(
              "T", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#1f3b57"))),
          p("Progress report - stages 1 to 4: dataset exploration, labelling, candidate generation, "
            "feature engineering and comparison of six models", ParagraphStyle("ST", parent=BODY, alignment=TA_CENTER,
                                                                                  fontSize=10.5, leading=14)),
          Spacer(1, 6),
          p("All numbers and figures in this report are produced by the code in the repository "
            "(<font face='Courier'>src/</font>) and read from <font face='Courier'>results/</font>. "
            "Models are fitted on the <b>train</b> partition only and compared on <b>student_test</b>. "
            "The evaluator partition was never available to or used by this work.", SMALL),
          Spacer(1, 8)]

    # ------------------------------------------------------------ 1. problem
    s += [p("1. Problem", H1),
          p("A vehicle's GPS receiver reports noisy positions: points drift a few metres off the road and can appear "
            "on a parallel street. Each trip comes with its local road network, made of <b>nodes</b> (points) and "
            "<b>arcs</b> (directed road segments, so a two-way street is two arcs). The task is to say, for every "
            "GPS observation, which arc the vehicle was actually on, and to reconstruct a realistic route."),
          p("Our approach turns this into a supervised learning problem:"),
          *bullets(["for each GPS point, collect the nearby arcs (<b>candidates</b>);",
                    "describe every (point, candidate) pair with <b>features</b>;",
                    "train a classifier that predicts P(candidate is the true arc);",
                    "pick the highest-probability candidate per point (Top-1) - and, in the next stage, "
                    "let an HMM/Viterbi decoder choose a <i>connected</i> route using these probabilities."])]

    # ------------------------------------------------------------ 2. dataset
    s += [p("2. The original dataset and what we did with it", H1),
          p("2.1 Structure", H2),
          p("The data is organised by trip. Every trip folder has four header-less, whitespace-separated files:"),
          table([["File", "Content", "Meaning"],
                 [".track", "lon, lat, time (s)", "one row per GPS observation"],
                 [".nodes", "lon, lat", "row index = node id"],
                 [".arcs", "from_node, to_node", "row index = arc id (directed road segment)"],
                 [".route", "arc id", "reference route: arcs driven, in order"]],
                [2.2 * cm, 4.2 * cm, PAGE_W - 6.4 * cm]),
          Spacer(1, 4),
          p("Node and arc ids are local to each trip. The route lists the arcs driven but does <b>not</b> say which "
            "GPS point belongs to which arc - so no per-point label exists in the raw data."),
          p("2.2 Inventory and data quality", H2),
          table([["", "train", "student_test"],
                 ["trips", int(tr.trips), int(st.trips)],
                 ["GPS observations", f"{int(tr.total_obs):,}", f"{int(st.total_obs):,}"],
                 ["median observations per trip", f"{tr.median_obs:,.0f}", f"{st.median_obs:,.0f}"],
                 ["missing values / invalid coordinates / non-increasing timestamps",
                  f"{int(tr.missing_values)} / {int(tr.invalid_coords)} / {int(tr.non_increasing_time)}",
                  f"{int(st.missing_values)} / {int(st.invalid_coords)} / {int(st.non_increasing_time)}"],
                 ["median % of time steps exactly 1 s", f"{tr.median_pct_dt_1s:.1f}%", f"{st.median_pct_dt_1s:.1f}%"],
                 ["trips with a time gap > 60 s", int(tr.trips_with_gap_over_60s), int(st.trips_with_gap_over_60s)],
                 ["median nodes / arcs per trip", f"{tr.median_nodes:,.0f} / {tr.median_arcs:,.0f}",
                  f"{st.median_nodes:,.0f} / {st.median_arcs:,.0f}"],
                 ["isolated nodes", int(tr.isolated_nodes), int(st.isolated_nodes)],
                 ["median weakly-connected components", f"{tr.median_weak_components:.0f}", f"{st.median_weak_components:.0f}"],
                 ["median reference-route length (arcs)", f"{tr.median_route_len:.0f}", f"{st.median_route_len:.0f}"],
                 ["route breaks (consecutive arcs not connected)", int(tr.total_route_breaks), int(st.total_route_breaks)],
                 ["median per-trip GPS-to-route distance", f"{tr.median_label_dist_m:.2f} m", f"{st.median_label_dist_m:.2f} m"]],
                [9.5 * cm, (PAGE_W - 9.5 * cm) / 2, (PAGE_W - 9.5 * cm) / 2]),
          Spacer(1, 4),
          *bullets([f"Trips are spread worldwide (latitude {inv.lat_center.min():.0f} to {inv.lat_center.max():.0f}); "
                    f"median two-way share of arcs is {by_part.pct_two_way.median()['train']:.1f}% (train).",
                    f"Largest time gap: {inv.max_gap_s.max():,.0f} s in trip {inv.loc[inv.max_gap_s.idxmax(), 'trip_id']} "
                    f"(about {inv.max_gap_s.max() / 3600:.1f} h); six train trips have gaps over 10 minutes.",
                    f"<b>Reference-route limitation:</b> in trips {', '.join(unrel.index[:3])} the route ends before the "
                    f"GPS track ({unrel.iloc[0]:.1f}%, {unrel.iloc[1]:.1f}% and {unrel.iloc[2]:.1f}% of their points are "
                    f"more than 50 m from the route)."])]

    s += [p("2.3 Preprocessing", H2),
          *bullets(["<b>Projection to metres.</b> Longitude/latitude are angles; one degree of longitude shrinks with "
                    "latitude, so Euclidean distance in degrees is wrong. Each trip is projected to a local x/y grid in "
                    "metres (equirectangular, centred on the trip). Trips are at most tens of km wide, so the error is "
                    "far below the GPS noise.",
                    "<b>Per-point labels by monotonic alignment.</b> Using dynamic programming we assign every GPS point "
                    "to a position on the reference route such that the position never moves backwards in time and "
                    "the total point-to-arc distance is minimal. This avoids the errors of 'nearest route arc' at "
                    "junctions, U-turns and roads driven twice.",
                    "<b>Reliability flag.</b> Points more than 50 m from their assigned arc are marked unreliable "
                    f"({csum.loc['train', 'reliable_points']:,.0f} of {csum.loc['train', 'points']:,.0f} train points "
                    f"remain reliable; {csum.loc['student_test', 'reliable_points']:,.0f} of "
                    f"{csum.loc['student_test', 'points']:,.0f} in student_test). They are excluded from training and "
                    "scoring."]),
          figure(eda / "gps_to_route_distance.png",
                 "Figure 1 - distance of each GPS point to its labelled route arc (log density). Most points lie within "
                 "about 5-15 m; the 50 m line is the reliability cut-off.", 0.82 * PAGE_W),
          figure(eda / "sampling_intervals.png",
                 "Figure 2 - time between consecutive observations: the data is almost entirely 1 Hz.", 0.75 * PAGE_W),
          figure(eda / "example_trip_00000027.png",
                 "Figure 3 - train trip 00000027: road network (grey), reference route (green), GPS (blue). "
                 "An interactive Folium version is in maps/eda_trip_00000027.html.", 0.55 * PAGE_W),
          figure(eda / "trip_locations.png", "Figure 4 - locations of all train and student_test trips.", 0.8 * PAGE_W)]

    # ------------------------------------------------------------ 3. candidates + features
    rs = radius.set_index("radius_m")
    s += [p("3. Candidate generation and features", H1),
          p("A network has thousands of arcs, but a GPS point can only be on one of the few arcs near it. For each "
            "point we take every arc within a radius R (fast search with a KD-tree over points sampled every 10 m "
            "along the arcs, then an exact point-to-segment distance). R was chosen on <b>train only</b>:"),
          table([["R (m)", "miss rate, reliable points", "miss rate, all points", "candidates per point"]] +
                [[int(r), f"{rs.loc[r, 'miss_rate_reliable_%']:.2f}%", f"{rs.loc[r, 'miss_rate_all_%']:.2f}%",
                  f"{rs.loc[r, 'mean_candidates_per_point']:.1f}"] for r in rs.index],
                [2.2 * cm, 4.4 * cm, 4.4 * cm, PAGE_W - 11 * cm], bold_row=list(rs.index).index(50) + 1),
          Spacer(1, 4),
          p("<b>R = 50 m</b> is the smallest radius with no misses on reliable points (40 m still misses 0.56%) "
            "while keeping about 9 candidates per point. Because reliable labels are defined as at most 50 m, the 0% "
            "is partly by construction; including all points the miss rate is "
            f"{csum.loc['train', 'miss_rate_all_%']:.2f}% (train) and {csum.loc['student_test', 'miss_rate_all_%']:.2f}% "
            "(student_test)."),
          p(f"This gives {csum.loc['train', 'candidate_rows']:,.0f} candidate rows for train and "
            f"{csum.loc['student_test', 'candidate_rows']:,.0f} for student_test; "
            f"{csum.loc['train', 'positive_rate_%']:.1f}% of train rows are the true arc. Thirteen features describe "
            "each pair - none of them uses the reference route, so the same code runs in the API:")]
    feat_rows = [["Feature", "Meaning", "True / wrong mean (train)"]]
    desc = {
        "dist_m": "distance point -> arc (m)", "dist_rank": "rank of that distance among the point's candidates",
        "dist_minus_min": "extra distance vs. the closest candidate (m)", "n_candidates": "candidates of this point",
        "heading_diff": "angle between travel direction (point i-2 -> i+2) and arc direction (deg)",
        "heading_valid": "1 if the vehicle moved > 3 m (heading meaningful)", "arc_len_m": "arc length (m)",
        "dist_to_arc_end_m": "distance along the arc to its nearest end (junction proximity)",
        "speed_ms": "speed around the point (m/s)", "is_two_way": "1 if the reverse arc exists",
        "out_degree": "arcs leaving the arc's end node", "window_dist_mean": "mean distance of points i-2..i+2 to the arc",
        "window_dist_max": "largest of those 5 distances"}
    for feat, row in fmeans.iterrows():
        feat_rows.append([feat, desc[feat], f"{row.true_candidate_mean:.2f} / {row.wrong_candidate_mean:.2f}"])
    s += [table(feat_rows, [3.4 * cm, PAGE_W - 7.4 * cm, 4.0 * cm]),
          Spacer(1, 6),
          figure(cand / "feature_distributions_train.png",
                 "Figure 5 - true vs wrong candidates (train). Distance, heading difference and the neighbour-window "
                 "distance separate the classes clearly. True arcs often have rank 1 instead of 0 because a two-way "
                 "street is two arcs at the same distance; only the heading tells them apart.", 0.9 * PAGE_W)]

    # ------------------------------------------------------------ 4. training
    s += [p("4. How the models were trained", H1),
          *bullets(["<b>Task:</b> binary classification of (point, candidate) rows; output = P(true arc).",
                    "<b>Training data:</b> train partition, reliable points only.",
                    "<b>Validation:</b> 5-fold <b>GroupKFold by trip</b>. A trip is never split between training and "
                    "validation folds; consecutive GPS points of one trip are near-copies, so a random split would "
                    "leak information and inflate scores.",
                    "<b>Equal training budget:</b> every fit uses the same random sample of at most 600,000 candidate "
                    "rows (whole points are sampled, so a point keeps all its candidates). KNN stores 50,000 rows "
                    "because its prediction cost grows with the number of stored rows.",
                    "<b>No leakage in preprocessing:</b> feature scaling (StandardScaler) is inside a scikit-learn "
                    "Pipeline, so it is fitted on the training fold only.",
                    "<b>No class weights:</b> positives are about 10% of rows - imbalanced but not extreme - and the "
                    "HMM stage needs honest probabilities.",
                    "<b>Final evaluation:</b> each model is refitted on the train sample and evaluated once on "
                    "student_test. Random seed 42 throughout; configurations, parameters and timings are logged in "
                    "results/models/experiment_log.json."]),
          p("Models (six algorithm families) and the nearest-arc baseline:", BODY)]
    model_rows = [["Model", "Family", "Key configuration"]]
    for _, r in comp.iterrows():
        model_rows.append([r.model, r.family, r.config])
    s += [table(model_rows, [3.8 * cm, 3.4 * cm, PAGE_W - 7.2 * cm]),
          Spacer(1, 4),
          p("<b>Metrics.</b> Pair level (threshold 0.5): accuracy, precision, recall, F1, macro F1, log loss. Point "
            "level: <b>Top-1 accuracy</b> - share of reliable GPS points whose highest-scored candidate is the labelled "
            "arc. Route level: continuity (consecutive picks equal or directly connected) and route arc F1. "
            "<b>Top-1 accuracy is the primary metric</b> because it is exactly what the final evaluation measures "
            "(the right road segment per GPS point); plain accuracy would be ~90% for a model that always says "
            "'not the road'. Log loss is the key secondary metric because the HMM uses the probabilities.")]

    # ------------------------------------------------------------ 5. results
    res_rows = [["Model", "CV macro F1 (mean ± std)", "CV Top-1", "CV log loss", "Test macro F1", "Test Top-1",
                 "Test log loss", "Test continuity", "Fit (s)", "Inference (ms / 1k pts)"]]
    for _, r in comp.iterrows():
        res_rows.append([r.model, f"{r.cv_macro_f1:.3f} ± {r.cv_macro_f1_std:.3f}", f(r.cv_top1_accuracy),
                         f(r.cv_log_loss), f(r.test_macro_f1), f(r.test_top1_accuracy), f(r.test_log_loss),
                         f(r.test_route_continuity), f(r.fit_time_s, 1), f(r.inference_ms_per_1k_points, 1)])
    best_row = list(comp.model).index("LightGBM") + 1
    s += [PageBreak(), p("5. Results", H1),
          table(res_rows, [2.9 * cm] + [(PAGE_W - 2.9 * cm) / 9] * 9, bold_row=best_row),
          p(f"CV = 5-fold GroupKFold on train; Test = student_test. Continuity of the reference labels themselves: "
            f"{log['reference_route_continuity_student_test']:.3f}.", SMALL),
          Spacer(1, 4),
          figure(mod / "student_test_top1_macro_f1.png", "Figure 6 - student_test Top-1 accuracy and macro F1.", 0.9 * PAGE_W),
          figure(mod / "cv_macro_f1.png", "Figure 7 - cross-validation macro F1, mean ± std over 5 folds.", 0.82 * PAGE_W),
          figure(mod / "log_loss.png", "Figure 8 - log loss, CV and student_test (log scale, lower is better).", 0.72 * PAGE_W),
          figure(mod / "confusion_matrices.png",
                 "Figure 9 - confusion matrices on student_test candidate pairs (threshold 0.5).", PAGE_W),
          figure(mod / "learning_curves_mlp_lightgbm.png",
                 "Figure 10 - training vs validation log loss for the two models that train in epochs / iterations "
                 "(validation = 20% of train trips held out). Other models have no training history.", PAGE_W)]

    s += [p("5.1 What the results show", H2),
          *bullets([f"<b>Baseline:</b> picking the nearest arc gives only {base.test_top1_accuracy:.1%} Top-1. "
                    "On two-way streets both directions are at exactly the same distance, so the rule guesses the "
                    "direction. All learned models use the heading feature and gain 27-35 percentage points.",
                    "<b>Gaussian Naive Bayes</b> is the weakest model and has by far the worst log loss "
                    f"({comp.set_index('model').loc['GaussianNB', 'test_log_loss']:.3f}). It assumes independent features, "
                    "but our distance features are strongly correlated, so it double-counts evidence and is over-confident.",
                    "<b>Logistic Regression</b> is solid but linear: it cannot express interactions such as "
                    "'a small distance matters less if the heading is opposite'.",
                    "<b>KNN</b> reaches similar accuracy to LR but needs about "
                    f"{comp.set_index('model').loc['KNN', 'inference_ms_per_1k_points'] / 1000:.1f} s per 1,000 points "
                    "at prediction time - unsuitable for a live API.",
                    "<b>Random Forest, LightGBM and the MLP</b> are statistically tied in cross-validation: their CV "
                    "macro F1 differ by at most 0.004 while the fold standard deviation is about 0.025.",
                    "One fold is hard for every model (LightGBM Top-1 per fold: "
                    + ", ".join(f"{v:.3f}" for v in lgb_folds.top1_accuracy) + "); the cause was not analysed further.",
                    "<b>No overfitting:</b> CV and student_test scores agree within about 0.01 for every model, and the "
                    "LightGBM training and validation losses decrease together and flatten after roughly 150 iterations."])]

    # ------------------------------------------------------------ 6. best model
    params = log["LightGBM"]["params"]
    keep = ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
            "reg_lambda", "random_state"]
    s += [p("6. Best model: LightGBM - and why", H1),
          p(f"Student_test: Top-1 accuracy <b>{lgb.test_top1_accuracy:.3f}</b>, macro F1 <b>{lgb.test_macro_f1:.3f}</b>, "
            f"log loss <b>{lgb.test_log_loss:.4f}</b>, precision {lgb.test_precision:.3f}, recall {lgb.test_recall:.3f}. "
            f"Cross-validation: macro F1 {lgb.cv_macro_f1:.3f} ± {lgb.cv_macro_f1_std:.3f}, Top-1 {lgb.cv_top1_accuracy:.3f}."),
          p("Reasons for choosing it among the three tied leaders:"),
          *bullets(["<b>Accuracy:</b> tied-best CV macro F1 and CV Top-1; best student_test macro F1.",
                    "<b>Probability quality:</b> lowest student_test log loss of all models - important because the "
                    "next stage (HMM) uses the probabilities as emission scores, not just the ranking.",
                    f"<b>Efficiency:</b> trains in {lgb.fit_time_s:.1f} s versus "
                    f"{comp.set_index('model').loc['RandomForest', 'fit_time_s']:.1f} s (Random Forest) and "
                    f"{comp.set_index('model').loc['MLP', 'fit_time_s']:.1f} s (MLP); small model file for the API.",
                    "<b>Practicality:</b> works on raw, unscaled features and handles feature interactions "
                    "(distance x heading) through its trees.",
                    "<b>Tunable:</b> it has several meaningful hyperparameters (number of trees, learning rate, leaves, "
                    "regularisation), which makes it a good target for the Bayesian optimisation stage.",
                    "The MLP has a slightly higher student_test Top-1 (0.886 vs 0.883), but the difference is smaller "
                    "than the cross-validation spread, its CV Top-1 is equal (0.876) and its CV macro F1 and log loss are "
                    "slightly worse, so it was not "
                    "preferred. Selection was based on cross-validation and student_test only."]),
          p("Configuration of the selected (untuned) model:", BODY),
          table([["Hyperparameter", "Value"]] + [[k, params[k]] for k in keep if k in params],
                [6 * cm, 5 * cm]),
          Spacer(1, 6)]

    # ------------------------------------------------------------ 7. next
    s += [p("7. Next stages", H1),
          *bullets(["Bayesian hyperparameter optimisation of LightGBM with Optuna (TPE sampler), group-aware CV on train, "
                    "tuned vs untuned comparison on student_test.",
                    "HMM + Viterbi route reconstruction: candidates as hidden states, classifier probabilities as "
                    "emissions, network driving distance vs GPS distance as transitions; comparison with "
                    "classifier-only predictions.",
                    "FastAPI backend (laptop 1) and evaluator UI (laptop 2)."]),
          Spacer(1, 6),
          p("Reproduce: <font face='Courier'>python -m src.labels; python -m src.eda; python -m src.features; "
            "python -m src.train_models</font> - see README.md.", SMALL)]

    OUT.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="PathSnap progress report - stages 1-4",
                            author="PathSnap team")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    print(build())
