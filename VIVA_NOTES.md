# PathSnap - Viva Notes

Revision notes, one section per stage. All numbers come from the files in
`results/` produced by the code in `src/`.

---

## Stage 1 - Dataset exploration (EDA) and per-point labels

**Partitions used:** `train` (80 trips) and `student_test` (10 trips).
The 10 evaluator trips are not in our folder and were never touched.

### What we did and why
1. **Loader** (`src/data_io.py`) - reads the 4 files of a trip into one `Trip`
   object. Node and arc ids are local to each trip.
2. **Projection to metres** (`src/geo.py`) - lon/lat are angles, so Euclidean
   distance in degrees is wrong (1 degree of longitude is ~111 km at the equator but
   only ~56 km at 60 N, and trips span 35 S to 62 N). Each trip is converted to a
   flat local x/y grid in metres (equirectangular projection centred on the
   trip). Trips are at most tens of km wide, so the error is well under 1% -
   much smaller than GPS noise.
3. **Labels** (`src/labels.py`) - the `.route` file gives the arcs driven, in
   order, but not which GPS point belongs to which arc. We align points to
   route arcs with dynamic programming: the route position can only stay or
   move forward in time, and we minimise the total point-to-arc distance.
   This avoids the mistakes of "nearest arc" at junctions, U-turns and roads
   driven twice. A label is marked *unreliable* if the point is > 50 m from
   its arc.
4. **EDA** (`src/eda.py`) - per-trip inventory, partition summary,
   data-quality flags and plots in `results/eda/`.
5. **Folium maps** (`src/maps.py`) - `maps/eda_trip_00000027.html` (clean
   example) and `maps/eda_trip_00000054.html` (route ends early), with layers
   for road network, reference route and GPS points, a legend and layer
   control.

### Key code in plain words
- `LocalProjection.to_xy`: x = R x delta-lon x cos(lat0), y = R x delta-lat.
- `point_segment_distance`: projects a point onto a segment, clamps to the
  segment ends, returns the distance - used everywhere later.
- `align_points_to_route`: fills a table `D[i, j]` = best total distance with
  point *i* on route arc *j*; each row only looks at the running minimum of
  the previous row up to *j* (that is the "never go backwards" rule). Then it
  traces back the best path.

### Results (real numbers)
| | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| median obs per trip | 1,529 | 1,420 |
| missing values / invalid coords / non-increasing timestamps | 0 / 0 / 0 | 0 / 0 / 0 |
| steps exactly 1 s apart (all steps) | 96.5% | 98.6% |
| trips with a gap > 60 s | 22 | 1 |
| median nodes / arcs per trip | 9,608 / 19,526 | 6,597 / 13,145 |
| isolated nodes | 0 | 0 |
| median weakly-connected components | 20 | 11 |
| median route length (arcs) | 280 | 250 |
| route breaks (consecutive arcs not touching) | 13 (in 10 trips) | 1 (trip 76) |
| median per-trip GPS-to-route distance | 4.63 m | 4.74 m |
| points with unreliable label (> 50 m) | 11.16% | 0.14% |

Other findings:
- Every trip folder has all 4 files; all route arc ids are valid.
- Trips are spread worldwide (Europe, Russia, Japan, Americas, Australia).
- ~95% of arcs have a reverse twin (two-way streets); the rest are one-way.
  The largest strongly-connected component holds ~96% of nodes (median).
- Median arc length 24.5 m (train); ~5 GPS points per route arc.
- Median trip speed 12.7 m/s (~46 km/h).
- 6 train trips have gaps > 10 min (11, 21, 42, 69, 78, 90); the largest is
  153,064 s (~42.5 h) in trip 90.
- **Reference-route limitation:** in trips 54 and 33 the route ends while the
  GPS keeps recording (86% / 68% of their points pile onto the last route
  arc; 89.4% / 58.9% unreliable). Trip 2 has a similar tail (12%). These three
  trips cause almost all unreliable labels; the other 77 train trips
  contribute only 0.33% of train points as unreliable.

**Decision:** points with unreliable labels (> 50 m) are excluded from training
and from accuracy calculations; the flag is kept so we can report it.

### Viva questions
1. **Why not use Euclidean distance on lon/lat?** Degrees of longitude shrink
   with latitude (cos(lat)), and our trips range from 35 S to 62 N, so degree
   distances are distorted differently in each trip. We project each trip to
   metres first.
2. **How did you get a label for each GPS point?** The route is an ordered list
   of arcs with no timing. We use dynamic programming to assign each point to
   a route position that never moves backwards, minimising total distance.
3. **Why not simply take the nearest route arc?** At junctions, U-turns and
   roads driven twice the nearest arc may be one the car only reaches later
   (or already left). The monotonic constraint uses the order of the route.
4. **Is the reference route perfect ground truth?** No. 11 trips have small
   breaks, and in trips 54/33/2 the route stops before the GPS track ends.
   We flag labels > 50 m away as unreliable and leave them out.
5. **What data-quality issues did you find?** No missing values, but: large
   time gaps (up to ~42 h), some tracks not at 1 Hz, routes that end early,
   disconnected network pieces, a few duplicate arcs and self-loops.
6. **Did you use student_test or evaluator data for any decision?** Only
   train was used to decide the 50 m rule and the method; student_test was
   only summarised. Evaluator data was never available to us.

---

## Stage 2 - Candidate generation and feature engineering

### What we did and why
- **Candidates** (`src/candidates.py`): a road network has ~20,000 arcs, but a
  GPS point can only be on one of the few arcs near it. For each point we
  collect all arcs within **R = 50 m**. This turns map-matching into a small
  ranking problem: "which of these ~9 candidates is the true arc?"
- **Fast search:** arcs near the track are sampled every 10 m, the samples go
  into a KD-tree, and each GPS point asks for samples within R + 5 m; then the
  exact point-to-arc distance is computed and only arcs <= R are kept. (+5 m
  because the nearest sample is at most half a spacing from the true closest
  point, so nothing within R is missed.)
- **Choosing R** - radius study on TRAIN only (`results/candidates/radius_study_train.csv`):

| R (m) | miss rate, reliable points | candidates / point |
|---|---|---|
| 20 | 3.70% | 3.6 |
| 30 | 1.52% | 5.3 |
| 40 | 0.56% | 7.0 |
| **50** | **0.00%** | **9.0** |
| 75 | 0.00% | 14.9 |
| 100 | 0.00% | 22.4 |

  50 m is the smallest radius with no misses on reliable points while keeping
  the candidate set small. Honest caveat: reliable labels are defined as
  <= 50 m, so 0% at 50 m is partly by construction; counting all points
  (including routes that stop early) the miss rate is 11.16% on train and
  0.14% on student_test.
- **Features** (`src/features.py`) - 13 per (point, candidate) pair, none of
  which use the reference route, so the API can compute them for new trips:
  distance, distance rank, extra distance vs closest, number of candidates,
  heading difference (+ validity flag), arc length, distance to arc end,
  speed, two-way flag, out-degree of end node, mean and max distance of the 5
  neighbouring GPS points to the arc.
- **Label:** `is_true = 1` if the candidate equals the point's labelled arc.

### Key code in plain words
- `arc_sample_points`: cuts every arc into 10 m pieces and returns the cut points.
- `generate_candidates`: KD-tree lookup, then exact distance filter.
- `point_motion`: heading = direction from point i-2 to i+2; speed = that
  distance / time. If the car moved < 3 m the heading is marked invalid.
- `build_features`: groups candidates by point to get ranks and relative
  distances, then adds arc attributes and the "window" distances.

### Results (real numbers, `results/candidates/`)
| | train | student_test |
|---|---|---|
| candidate rows | 1,869,591 | 209,581 |
| candidates per point (mean) | 8.97 | 10.94 |
| positive (true) rows | 9.90% | 9.13% |
| points with no candidate within 50 m | 8.75% | 0.00% |
| miss rate (reliable points) | 0.00% | 0.00% |

Feature means on train (reliable points), true vs wrong candidate:
distance 6.5 vs 27.0 m; heading difference 23.9 vs 95.3 deg; window mean
distance 8.8 vs 29.8 m; distance rank 0.73 vs 10.41.
The plot `feature_distributions_train.png` shows the separation.

Observation: true candidates often have rank 1 instead of 0, because a
two-way street is two arcs with the same geometry (same distance). Only the
heading feature can tell the two directions apart.

The 8.75% of train points with no candidates are almost all in trips 54 / 33,
where the GPS continues after the route ends - far from the clipped network.

### Viva questions
1. **Why generate candidates instead of classifying among all arcs?** Arc ids
   are local to each trip and there are thousands of them. Restricting to
   nearby arcs gives a small, fixed-meaning binary problem (is this candidate
   the true one?) that transfers across trips.
2. **How did you choose the 50 m radius?** Radius study on train: miss rate
   vs candidate count; 50 m was the smallest radius with 0% miss on reliable
   points (40 m missed 0.56%).
3. **What is the candidate miss rate and why does it matter?** The share of
   points whose true arc is not in the candidate set. Those points can never
   be predicted correctly, so it caps the achievable Top-1 accuracy.
4. **Why the heading feature?** Two-way streets are two arcs with identical
   geometry; distance can't separate them, but the travel direction can
   (true: 24 deg mean difference, wrong: 95 deg).
5. **Why the window distance features?** A single GPS point is noisy; the true
   arc should also be close to the neighbouring points, a wrong parallel road
   usually is not. It is a cheap way to use local context before the HMM.
6. **Do features leak the label?** No. They use only GPS, timestamps and the
   network. The route is used only to make the `is_true` target.

---

## Stages 3-4 - Baseline and six-model comparison

### What we did and why
- **Task for every model:** binary classification of (GPS point, candidate arc)
  rows: is this candidate the true arc? The predicted probability is used to
  pick the best candidate per point (Top-1) and later as the HMM emission.
- **Baseline (no learning):** pick the nearest candidate arc.
- **Six models, six algorithm families** (`src/models.py`):

| Model | Family | Key configuration |
|---|---|---|
| Logistic Regression | Linear | StandardScaler + L2, C=1.0 |
| Gaussian Naive Bayes | Probabilistic | var_smoothing=1e-9 |
| KNN | Instance-based | StandardScaler, k=25, distance-weighted, 50k stored rows |
| Random Forest | Bagging ensemble | 100 trees, max_depth=16, min_samples_leaf=5 |
| LightGBM | Gradient boosting | 300 trees, learning_rate=0.05, num_leaves=31 |
| MLP | Neural network | StandardScaler, 64-32 ReLU, Adam lr=1e-3, alpha=1e-4, 20 epochs |

- **Protocol** (`src/train_models.py`):
  - 5-fold **GroupKFold by trip** on TRAIN reliable points - no trip is in
    both a training and a validation fold (prevents leakage between
    neighbouring GPS points of the same trip).
  - Each fit uses the same random sample of at most 600,000 candidate rows
    (whole points sampled, so a point keeps all its candidates); KNN keeps
    50,000 because its prediction cost grows with the stored rows.
  - Scaling is inside a Pipeline, so it is fitted on the training fold only.
  - No class weights: positives are ~10%, not extreme, and we want honest
    probabilities for the HMM.
  - Final model per algorithm fitted on the TRAIN sample, evaluated once on
    STUDENT_TEST. Seed 42 everywhere. Logs: `results/models/experiment_log.json`.
- **Metrics:** pair-level accuracy, precision, recall, F1, macro F1 and log loss
  (threshold 0.5); point-level **Top-1 accuracy** (best candidate per point =
  labelled arc); route continuity (consecutive picks equal or directly
  connected); route arc F1 (set of picked arcs vs labelled arcs, per trip).

**Primary metric: Top-1 accuracy** - it is exactly what the evaluator measures
(was each GPS point matched to the right road segment). Macro F1 and log loss
are secondary: log loss matters because the HMM uses the probabilities.

### Results (real numbers, `results/models/model_comparison.csv`)
| Model | CV macro F1 (mean ± std) | CV Top-1 | CV log loss | Test macro F1 | Test Top-1 | Test log loss | Test continuity | Fit (s) | Inference (ms / 1k points) |
|---|---|---|---|---|---|---|---|---|---|
| Nearest arc (baseline) | 0.752 | 0.555 | n/a | 0.745 | 0.537 | n/a | 0.821 | 0 | n/a |
| Logistic Regression | 0.903 ± 0.021 | 0.865 | 0.105 | 0.908 | 0.864 | 0.086 | 0.924 | 1.1 | 1.0 |
| Gaussian NB | 0.781 ± 0.011 | 0.815 | 0.505 | 0.783 | 0.805 | 0.476 | 0.926 | 0.1 | 2.8 |
| KNN | 0.898 ± 0.023 | 0.850 | 0.115 | 0.904 | 0.857 | 0.094 | 0.930 | 0.2 | 4,786 |
| Random Forest | 0.929 ± 0.026 | 0.876 | 0.067 | 0.932 | 0.879 | 0.057 | 0.933 | 24.7 | 19.3 |
| **LightGBM** | 0.929 ± 0.025 | 0.876 | 0.068 | **0.932** | 0.883 | **0.056** | 0.934 | **5.5** | 25.2 |
| MLP | 0.925 ± 0.026 | 0.876 | 0.071 | 0.931 | 0.886 | 0.057 | 0.939 | 28.9 | 2.2 |

Continuity of the labels themselves (reference) on student-test: 0.985.

### Observations
- **Baseline only 53.7% Top-1.** Two-way streets are two arcs at exactly the
  same distance, so "nearest" is a coin-flip between directions. Every
  learned model uses the heading feature and gains 27-35 points.
- **Naive Bayes** has the worst log loss (0.48): it assumes independent
  features, but distance, distance rank, extra distance and window distance
  are strongly correlated, so it double-counts evidence -> over-confident.
- **Logistic Regression** is good (0.864) but linear: it cannot express rules
  like "distance matters less when the heading is wrong".
- **KNN** is close to LR but needs ~4.8 s per 1,000 points to predict - too
  slow for a live API, and it needs scaling.
- **Random Forest, LightGBM, MLP** are tied in CV (macro F1 0.925-0.929,
  Top-1 0.876). Differences between them are smaller than the fold std
  (~0.025), so CV cannot separate them.
- Fold 2 is hard for every model (e.g. LightGBM Top-1 0.794 vs ~0.90-0.91 on
  the other folds); the cause was not investigated in detail - it is likely
  that fold holds trips with harder road layouts or noisier GPS.
- **Overfitting check:** CV and student-test scores are very close for every
  model (e.g. LightGBM CV macro F1 0.929, test 0.932), so no sign of
  overfitting. In the LightGBM learning curve train and validation log loss
  fall together and flatten after ~150 iterations (train 0.051 vs
  validation 0.054 at 300 - only a tiny gap). The MLP's validation loss is
  below its training loss because the held-out trips happen to be easier.
- **Model selection: LightGBM.** Tied-best CV macro F1 and Top-1, best
  student-test log loss and macro F1, 5x faster to train than RF and MLP,
  small model file, handles raw (unscaled) features, gives good
  probabilities for the HMM, and has many hyperparameters for Bayesian
  tuning to improve.

### Visualisations (`results/models/`)
`cv_macro_f1.png`, `student_test_top1_macro_f1.png`, `log_loss.png`,
`confusion_matrices.png`, `inference_time.png`,
`learning_curves_mlp_lightgbm.png` (only these two models train in
epochs / iterations, so only they have loss curves).

### Viva questions
1. **Why GroupKFold by trip?** Consecutive GPS points of one trip are almost
   identical; if a trip were split across folds the model would be tested on
   near-copies of its training data and the score would be too optimistic.
2. **Why is Top-1 accuracy the primary metric and not accuracy?** Pair accuracy
   is ~90% just by predicting "not the road" for everything (only ~10% of
   rows are positive). Top-1 measures what we actually need: the right
   segment per GPS point.
3. **Why did Naive Bayes do worst?** Its independence assumption is broken by
   our correlated distance features, so its probabilities are over-confident
   (log loss 0.48 vs 0.06).
4. **Why not KNN in production?** Similar accuracy to LR but ~5 s per 1,000
   points at prediction time - it compares each row with all stored rows.
5. **Why LightGBM over RF and MLP when they tie?** Same accuracy within one
   standard deviation, but LightGBM is 5x faster to train, smaller, needs no
   scaling, gives the best log loss (probabilities for the HMM) and has
   meaningful hyperparameters to tune.
6. **Is there overfitting?** No: CV and student-test results agree within
   ~0.01 for every model, and the LightGBM train/validation loss curves
   flatten together.
7. **Why sample 600k rows?** Equal training size for every model makes the
   comparison fair and keeps run time manageable; the 600k rows still cover
   all 80 training trips.

---

## Stage 5 - Bayesian hyperparameter optimisation (LightGBM)

### What we did and why
- **Tool:** Optuna with the **TPE sampler** (Tree-structured Parzen Estimator).
  TPE is Bayesian: it keeps two probability models - one of parameter values
  that gave good scores, one of values that gave bad scores - and proposes
  the next trial where "good" is most likely relative to "bad". So it learns
  from earlier trials, unlike grid or random search.
- **Objective:** maximise mean **Top-1 accuracy** over **3-fold GroupKFold by
  trip** on TRAIN reliable points (the same primary metric as model
  selection). Macro F1 and log loss were recorded for every trial.
- **Search space** (`src/tune.py`):

| Hyperparameter | Range | Why |
|---|---|---|
| n_estimators | 100-800 (step 50) | more trees = more capacity |
| learning_rate | 0.01-0.2 (log) | step size; small needs more trees |
| num_leaves | 15-255 (log) | complexity of each tree |
| min_child_samples | 10-300 (log) | minimum rows per leaf - regularisation |
| subsample | 0.5-1.0 | row sampling per tree - reduces variance |
| colsample_bytree | 0.5-1.0 | feature sampling per tree |
| reg_lambda | 0.001-10 (log) | L2 penalty on leaf values |

- **Trials:** 25 (first trial = the default-like configuration so the search
  starts from the untuned point). Search time 1,423 s (~24 min).
- Same 600k-row training budget per fit as Stage 4, so tuned vs untuned is a
  fair comparison. STUDENT_TEST was used only after the search to report.
- After tuning, the model was refitted on the train sample and exported to
  `artifacts/classifier.joblib` (+ `classifier_config.json`). We also saved
  out-of-fold (OOF) probabilities on train for tuning the HMM.

### Results (`results/tuning/`)
Best parameters: n_estimators **700**, learning_rate **0.0111**, num_leaves
**127**, min_child_samples **24**, subsample **0.592**, colsample_bytree
**0.696**, reg_lambda **1.134**. Best 3-fold CV Top-1: 0.8766 (trial 18),
starting point (trial 0): 0.8728.

Tuned vs untuned (same 5-fold CV and the same student-test data):

| | CV macro F1 | CV Top-1 | CV log loss | Test macro F1 | Test Top-1 | Test log loss | Fit (s) | Inference (ms / 1k pts) |
|---|---|---|---|---|---|---|---|---|
| Untuned | 0.9288 | 0.8760 | 0.0679 | 0.9322 | 0.8831 | 0.0556 | 6.8 | 26.8 |
| **Tuned** | **0.9299** | **0.8778** | **0.0649** | **0.9338** | **0.8849** | **0.0545** | 48.0 | 88.2 |

- The tuned model is better on **every** quality metric, but the gain is
  small (+0.18 points Top-1 on both CV and student-test). Interpretation: the
  features already carry most of the information; the defaults were close
  to optimal.
- The improvement is in the same direction on CV and on the unseen
  student-test trips, so it is not an artefact of the search.
- Cost: ~7x longer training, ~3x slower prediction (still < 0.1 s per 1,000
  points - fine for the API).
- What the search preferred: many small steps (low learning rate, many
  trees), bigger trees (127 leaves) and row/feature sub-sampling (~0.6-0.7)
  as regularisation. Trial 1 (0.855) shows that a bad combination can cost
  2 points.

Plots: `optimisation_history.png` (score per trial + best so far),
`params_vs_score.png` (each hyperparameter vs score). All trials:
`optuna_trials.csv`.

### Viva questions
1. **What makes this "Bayesian"?** TPE builds probability models of good and
   bad parameter regions from past trials and samples where good is more
   likely; each trial uses the evidence from previous ones.
2. **Why optimise Top-1 accuracy?** It is the primary metric (correct road per
   GPS point) and what the evaluator measures.
3. **Why group-aware CV in tuning?** Tuning on a random split would reward
   parameters that memorise individual trips; grouping by trip measures
   generalisation to new trips.
4. **Did tuning help?** Yes, slightly: CV Top-1 0.8760 -> 0.8778, test Top-1
   0.8831 -> 0.8849, test log loss 0.0556 -> 0.0545. It is a consistent but
   small gain.
5. **Why only 25 trials and 3 folds?** Time budget (~24 min). The best-so-far
   curve flattens after ~18 trials, suggesting more trials would add little.
6. **Was student-test used during tuning?** No. Only TRAIN (3-fold CV) drove
   the search; student-test was evaluated once afterwards.

---

## Stage 6 - HMM and Viterbi route reconstruction

### What we did and why
The classifier scores each GPS point on its own, so consecutive points can be
matched to roads that are not connected (a "jump"). The HMM adds the rule
that the vehicle must drive along the network (`src/hmm.py`).

- **Hidden states:** at GPS point *i*, "the vehicle is on candidate arc *c*",
  for the top-K candidates by classifier probability. Each state keeps its
  arc id and projected position on the arc, so every decoded state maps back
  to exactly one road segment (candidate-to-segment correspondence preserved).
- **Emission:** `w * log P(candidate is true)` from the tuned LightGBM
  (`w` = emission weight).
- **Transition:** for candidate *a* at point *i-1* and *b* at point *i*:
  `d_route` = driving distance on the network from *a*'s projected position
  to *b*'s (Dijkstra on arc lengths, limited search radius), `d_gps` =
  straight-line distance between the two GPS points;
  `log P(a -> b) = -|d_route - d_gps| / beta`. On the right path the driven
  distance is close to the straight-line distance (Newson & Krumm idea).
  Same arc moving slightly backwards (<= 10 m) is treated as GPS jitter.
  No path within the search limit = impossible transition.
- **Viterbi recurrence:**
  `V_i(b) = w log e_i(b) + max_a [ V_{i-1}(a) - |d_route(a,b) - d_gps| / beta ]`,
  with a back-pointer to the best *a*; the best final state is traced back
  to get one arc per point.
- **Gaps and missing candidates:** the chain is cut and restarted when the time
  gap is > 60 s, or when no transition from any previous state is possible
  (disconnected roads). Points without candidates are skipped (no prediction,
  counted as wrong).
- **Parameter choice without leakage:** beta, w and top-K were chosen by a grid
  search (3 x 3 x 2 = 18 settings) on 30 train trips using **out-of-fold**
  classifier probabilities (5-fold GroupKFold from Stage 5), so the decoder
  sees probabilities as noisy as on unseen trips. Student-test was used only
  to report the result.

### Results (real numbers, `results/hmm/`)
Grid (30 train trips, OOF probabilities): classifier only 0.8975 Top-1; HMM
settings 0.9110-0.9208. **Best: beta = 3, emission weight = 2.0, top-K = 8**
(0.9208). Saved in `artifacts/hmm_config.json`.

| | Top-1 accuracy | Route continuity | Route arc F1 |
|---|---|---|---|
| Student-test, classifier only (tuned LightGBM) | 0.8849 | 0.9362 | 0.9191 |
| **Student-test, classifier + HMM** | **0.9123** | **0.9834** | **0.9657** |
| Train (all 80 trips, OOF), classifier only | 0.8792 | 0.9404 | 0.9053 |
| Train (all 80 trips, OOF), classifier + HMM | 0.9019 | 0.9841 | 0.9590 |

Reference continuity of the labels themselves: 0.985 - the HMM output is now
almost as continuous as the reference.

- Student-test: the HMM improved **all 10 trips** (+0.2 to +5.7 points Top-1;
  biggest gains on trips 76, 65, 9, 56). Decoding all 10 trips took 6.2 s.
- Train (OOF): improved 72 of 80 trips, median +2.3 points.
- **Where the HMM did not help:** 8 train trips got worse, two badly -
  trip 82 (0.839 -> 0.625) and trip 10 (0.630 -> 0.446). In both the decoder
  stays on a wrong road for long stretches (578 and 708 consecutive wrong
  points); only 1.8% / 4.6% of their errors are the opposite direction of the
  true arc, so it is not a direction mix-up. Trip 82 is split into 49 chains,
  i.e. transitions were often impossible - a sign that the network around
  the true path is poorly connected or that the reference path uses
  connections the search limit does not reach. Once the decoder commits to a
  well-connected wrong road, the transition cost of switching back keeps it
  there. Both are slow trips (~8.5 m/s median speed).
- Continuity is not exactly 1.0 because our continuity metric only accepts
  *directly* connected consecutive arcs; at 1 Hz a vehicle can pass a very
  short arc between two GPS fixes, which the HMM allows.

Map: `maps/hmm_trip_00000076.html` (biggest change on student-test) - GPS,
road network, reference route (green), HMM route (red), classifier-only
picks (orange, toggle in layer control).
Plots: `per_trip_student_test.png`, `hmm_gain_per_trip_train.png`.

### Viva questions
1. **What are the hidden states?** The candidate arcs of each GPS point (top 8
   by classifier probability), with the projected position on the arc.
2. **How are emission probabilities computed?** From the tuned LightGBM:
   log P(candidate is true), weighted by w = 2.
3. **How are transition probabilities computed?** Exponential in the
   difference between network driving distance and straight-line GPS
   distance: exp(-|d_route - d_gps| / beta), beta = 3 m.
4. **What happens with large gaps or disconnected roads?** Gap > 60 s or no
   possible transition -> the chain is cut and a new one starts, so one bad
   step cannot corrupt the whole trip.
5. **Did the HMM improve the route?** Yes: student-test Top-1 0.885 -> 0.912,
   continuity 0.936 -> 0.983, route arc F1 0.919 -> 0.966; all 10 trips
   improved. On train it helped 72/80 trips but hurt 8 (see above).
6. **Why tune the HMM on out-of-fold probabilities?** In-sample probabilities
   on train trips are over-confident; OOF probabilities behave like those on
   new trips, so the chosen beta / w transfer.
7. **What does beta mean?** The tolerance (in metres) for the mismatch between
   driven and straight-line distance; small beta = strict "must follow the
   network". With 1 Hz data consecutive fixes are only ~13 m apart, so a
   strict beta (3 m) worked best.

---

## Stage 7 - Exported artifacts and FastAPI backend (laptop 1)

### What we did and why
- **Artifacts** (`artifacts/`): `classifier.joblib` (tuned LightGBM),
  `classifier_config.json` (feature list, 50 m radius, parameters, seed),
  `hmm_config.json` (beta 3, emission weight 2, top-K 8, gap 60 s).
- **Prediction pipeline** (`src/pipeline.py`, class `PathSnapModel`): loads the
  artifacts once; for a trip it runs candidates -> 13 features -> LightGBM
  probabilities -> Viterbi, and returns one arc per GPS point (-1 if no
  candidate within 50 m). The trip is loaded with `with_route=False`, so the
  reference route cannot reach the model.
- **Scoring** (`score_prediction`): only after prediction, the server loads the
  reference route, labels the points the same way as in development
  (monotonic alignment, points > 50 m not scored) and computes point
  accuracy, route continuity and route arc F1.
- **API** (`api/main.py`):
  - `GET /health` - status and loaded model
  - `GET /trips` - trip ids found in `PATHSNAP_EVAL_DIR`
  - `POST /predict {"trip_id": ...}` - runs the pipeline, returns GPS
    points (max 2,000 for the map), the predicted route geometry, the
    metrics and timings. It does **not** return the reference route.
  - Unknown ids are rejected (404), which also blocks path tricks like `../`.
  - CORS is enabled because the UI runs on another laptop (different origin).
  - Every request is logged to the console (trip, points, accuracy, time) -
    these are the logs shown to the evaluator.
- Start: `uvicorn api.main:app --host 0.0.0.0 --port 8000` (`0.0.0.0` so the
  other laptop can reach it).

### Results (real numbers, `results/api_student_test_check.json`)
All 10 student-test trips through the running API:

| Trip | Points | Accuracy | Continuity | Route arc F1 | Time (s) |
|---|---|---|---|---|---|
| 00000000 | 2,503 | 0.9664 | 0.9972 | 0.9609 | 0.95 |
| 00000007 | 1,354 | 0.9638 | 1.0000 | 0.9878 | 0.40 |
| 00000009 | 1,049 | 0.9152 | 0.9532 | 0.9675 | 0.76 |
| 00000040 | 2,995 | 0.9466 | 0.9957 | 0.9817 | 1.16 |
| 00000052 | 262 | 0.9924 | 0.9579 | 0.9939 | 0.17 |
| 00000056 | 3,958 | 0.9504 | 0.9682 | 0.9808 | 3.58 |
| 00000065 | 768 | 0.8919 | 0.9844 | 0.9588 | 1.11 |
| 00000076 | 1,486 | 0.9704 | 0.9744 | 0.9619 | 2.19 |
| 00000077 | 3,911 | 0.7712 | 0.9916 | 0.9261 | 1.92 |
| 00000091 | 874 | 0.9119 | 0.9679 | 0.9373 | 0.58 |

Point-weighted accuracy 0.9124, mean request time 1.28 s. The per-trip
accuracies are identical to the offline Stage 6 results, so the API runs
exactly the same pipeline as the experiments.

### Viva questions
1. **How do you make sure ground truth does not leak into prediction?** The
   trip is loaded without the route for prediction; the route is read only
   afterwards inside `score_prediction`, and only numbers go back to the UI.
2. **Why `--host 0.0.0.0`?** The default 127.0.0.1 only accepts connections
   from the same machine; 0.0.0.0 listens on the network so laptop 2 can
   connect.
3. **What is CORS and why enable it?** Browsers block a page from one origin
   (laptop 2) calling an API on another origin (laptop 1) unless the API
   allows it.
4. **How would evaluator trips be served?** Set `PATHSNAP_EVAL_DIR` to their
   folder; nothing in the code or model changes, and nothing is hardcoded.
5. **How long does a prediction take?** 0.2-3.6 s per trip on student-test
   (mean 1.3 s), mostly feature building and the HMM.

---

## Stage 8 - Evaluator UI (laptop 2)

### What we did and why
- **Plain HTML + JavaScript + Leaflet** (`ui/index.html`, `ui/app.js`,
  `ui/style.css`, `ui/config.js`). Nothing to install on laptop 2: it is
  served with `python -m http.server 8080 --directory ui` and opened in a
  browser. Leaflet (map library) and OpenStreetMap tiles come from the
  internet.
- **Flow:**
  1. The API address is typed in the "API connection" box (or set once in
     `ui/config.js`); it is remembered in the browser.
  2. **Connect** calls `GET /health` and `GET /trips` and shows one button
     per trip (at the demo: the 10 evaluator trips).
  3. Clicking a trip sends **only the trip id** to `POST /predict`.
  4. The UI draws the GPS observations (blue dots) and the predicted route
     (red line) with a legend and a layer switch, zooms to the trip, and
     shows: point accuracy, points matched / scored, route continuity,
     route segment F1, number of predicted segments, HMM chains, and server
     time split into features / classifier / HMM.
- The UI never receives or draws the reference route - ground truth stays on
  the server; the UI only shows the server's accuracy numbers.

### Results
Tested in the browser against the final API (tuned LightGBM + HMM) on the
10 student-test trips: connection status "Connected · LightGBM
(Optuna-tuned)", 10 trip buttons, e.g. trip 00000076 -> 97.04% point
accuracy, 1,486 points, map drawn and zoomed, no browser console errors.
The API console logged every request.

### Viva questions
1. **Why a static HTML page and not Streamlit/React?** Laptop 2 then needs no
   Python packages or build step - only a browser - so the demo setup has
   fewer things that can go wrong.
2. **How does the UI talk to the API?** `fetch()` calls over HTTP to
   `http://<laptop-1-ip>:8000`; CORS is enabled on the API for this.
3. **Can the UI see the ground truth?** No. The API response contains only
   the GPS preview, the predicted route and the metrics.
4. **What if the API address changes?** Type the new address in the box and
   press Connect - no code change.
