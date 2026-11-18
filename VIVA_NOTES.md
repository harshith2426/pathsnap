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
