"""Feature engineering for (GPS point, candidate arc) pairs.

Every row of the table is one candidate arc for one GPS point. The model
predicts P(this candidate is the arc the vehicle was on).

Features (all computed WITHOUT the reference route, so the API can use them):
  dist_m              distance point -> arc (m). Main signal: GPS noise is ~5 m.
  dist_rank           0 = closest candidate of this point, 1 = second, ...
  dist_minus_min      how much further than the closest candidate (m)
  n_candidates        how crowded the area is (junctions, parallel roads)
  heading_diff        angle between the vehicle's direction of travel and the
                      arc's direction (0 = same way, 180 = opposite way).
                      Separates the two directions of a two-way street.
  heading_valid       1 if the vehicle moved enough (> 3 m over ~4 s) for
                      the heading to be meaningful, else 0 (heading_diff = 90)
  arc_len_m           length of the arc
  dist_to_arc_end_m   distance along the arc from the projected point to the
                      nearest end; small values = near a junction (ambiguous)
  speed_ms            vehicle speed around this point
  is_two_way          1 if the reverse arc exists
  out_degree          number of arcs leaving the arc's end node (junction size)
  window_dist_mean    mean distance of the 5 surrounding GPS points (i-2..i+2)
                      to this arc: a true arc fits the neighbours too
  window_dist_max     worst of those 5 distances
"""
import numpy as np
import pandas as pd

from .candidates import generate_candidates, pair_distance
from .config import DERIVED_DIR
from .data_io import list_trips, load_trip
from .geo import angle_diff_deg, bearing_deg
from .labels import load_labels

RADIUS_M = 50.0          # chosen from the radius study on train (see VIVA_NOTES)
HEADING_SPAN = 2         # use points i-2 .. i+2 for heading and speed
MIN_MOVE_M = 3.0         # below this the heading is noise

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min", "n_candidates",
    "heading_diff", "heading_valid", "arc_len_m", "dist_to_arc_end_m",
    "speed_ms", "is_two_way", "out_degree", "window_dist_mean", "window_dist_max",
]


def point_motion(trip):
    """Heading (deg), heading validity and speed (m/s) for every GPS point."""
    n = len(trip.gps_xy)
    idx = np.arange(n)
    before = np.clip(idx - HEADING_SPAN, 0, n - 1)
    after = np.clip(idx + HEADING_SPAN, 0, n - 1)
    move = trip.gps_xy[after] - trip.gps_xy[before]
    move_len = np.linalg.norm(move, axis=1)
    dt = trip.times[after] - trip.times[before]
    heading = bearing_deg(move[:, 0], move[:, 1])
    valid = move_len > MIN_MOVE_M
    speed = np.where(dt > 0, move_len / np.maximum(dt, 1e-9), 0.0)
    return heading, valid, speed


def arc_attributes(trip):
    """Per-arc length, bearing, two-way flag and out-degree of the end node."""
    a, b = trip.arc_start_xy, trip.arc_end_xy
    vec = b - a
    length = np.linalg.norm(vec, axis=1)
    bearing = bearing_deg(vec[:, 0], vec[:, 1])
    arc_set = set(map(tuple, trip.arcs))
    two_way = np.array([(v, u) in arc_set for u, v in trip.arcs], dtype=int)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes_xy))
    return length, bearing, two_way, out_deg[trip.arcs[:, 1]]


def build_features(trip, radius_m=RADIUS_M):
    """Candidate table with features for one trip (no labels used)."""
    cand = generate_candidates(trip, radius_m)
    if cand.empty:
        return cand
    p = cand.point_idx.to_numpy()
    arc = cand.arc_id.to_numpy()

    # distance-based features within each point's candidate group
    g = cand.groupby("point_idx").dist_m
    cand["dist_rank"] = g.rank(method="first").to_numpy() - 1
    cand["dist_minus_min"] = cand.dist_m - g.transform("min")
    cand["n_candidates"] = g.transform("size")

    heading, valid, speed = point_motion(trip)
    length, bearing, two_way, out_deg = arc_attributes(trip)
    cand["heading_valid"] = valid[p].astype(int)
    cand["heading_diff"] = np.where(valid[p], angle_diff_deg(heading[p], bearing[arc]), 90.0)
    cand["arc_len_m"] = length[arc]
    cand["dist_to_arc_end_m"] = np.minimum(cand.proj_t, 1 - cand.proj_t) * length[arc]
    cand["speed_ms"] = speed[p]
    cand["is_two_way"] = two_way[arc]
    cand["out_degree"] = out_deg[arc]

    # how well does this arc fit the neighbouring GPS points?
    n = len(trip.gps_xy)
    a, b = trip.arc_start_xy[arc], trip.arc_end_xy[arc]
    window = []
    for off in range(-HEADING_SPAN, HEADING_SPAN + 1):
        q = np.clip(p + off, 0, n - 1)
        d, _ = pair_distance(trip.gps_xy[q], a, b)
        window.append(d)
    window = np.column_stack(window)
    cand["window_dist_mean"] = window.mean(axis=1)
    cand["window_dist_max"] = window.max(axis=1)

    cand.insert(0, "trip_id", trip.trip_id)
    return cand


def build_table(partition, radius_m=RADIUS_M):
    """Candidate + feature table for a whole partition, with labels attached.

    Added label columns:
      is_true         1 if the candidate is the point's labelled arc
      label_reliable  point label is within 50 m (unreliable points are not
                      used for training or accuracy)
    Also returns a point table (one row per GPS point) for miss-rate analysis.
    """
    labels = load_labels(partition)
    frames, points = [], []
    for tid in list_trips(partition):
        trip = load_trip(partition, tid)
        lab = labels[labels.trip_id == tid].reset_index(drop=True)
        cand = build_features(trip, radius_m)
        cand["is_true"] = (cand.arc_id.to_numpy() == lab.arc_id.to_numpy()[cand.point_idx]).astype(int)
        cand["label_reliable"] = lab.label_reliable.to_numpy()[cand.point_idx]
        frames.append(cand)

        n_cand = np.bincount(cand.point_idx, minlength=len(lab))
        has_true = np.zeros(len(lab), bool)
        has_true[cand.point_idx[cand.is_true == 1].to_numpy()] = True
        points.append(pd.DataFrame({
            "trip_id": tid, "point_idx": np.arange(len(lab)),
            "true_arc": lab.arc_id, "label_reliable": lab.label_reliable,
            "n_candidates": n_cand, "true_in_candidates": has_true}))
    table = pd.concat(frames, ignore_index=True)
    point_table = pd.concat(points, ignore_index=True)
    DERIVED_DIR.mkdir(exist_ok=True)
    table.to_pickle(DERIVED_DIR / f"cands_{partition}.pkl")
    point_table.to_pickle(DERIVED_DIR / f"points_{partition}.pkl")
    return table, point_table


def load_table(partition):
    return (pd.read_pickle(DERIVED_DIR / f"cands_{partition}.pkl"),
            pd.read_pickle(DERIVED_DIR / f"points_{partition}.pkl"))


if __name__ == "__main__":
    import time
    for part in ["train", "student_test"]:
        t0 = time.time()
        table, pts = build_table(part)
        rel = pts[pts.label_reliable]
        print(f"{part}: {len(table):,} candidate rows for {len(pts):,} points "
              f"({time.time() - t0:.0f}s); positives {table.is_true.mean() * 100:.1f}%; "
              f"miss rate (reliable pts) {100 * (~rel.true_in_candidates).mean():.2f}%")
