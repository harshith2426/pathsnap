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
