"""Candidate generation: which road segments could each GPS point be on?

For every GPS point we collect all arcs whose closest point is within a
search radius R (metres). The model later only has to choose among these
few candidates instead of among all ~20,000 arcs of the network.

How we find nearby arcs quickly:
  1. keep only arcs near the GPS track's bounding box,
  2. place sample points every 10 m along each of those arcs,
  3. put the samples in a KD-tree (fast "what is near this point" search),
  4. for each GPS point, ask the tree for samples within R + 5 m,
  5. compute the exact point-to-arc distance and keep arcs with distance <= R.
Step 4 uses R + 5 m because the nearest sample on a segment is at most 5 m
(half the spacing) from the true closest point, so no arc within R is missed.

Run:  python -m src.candidates   -> radius study on TRAIN only
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import RESULTS_DIR, TRAIN
from .data_io import list_trips, load_trip
from .labels import load_labels

SAMPLE_SPACING_M = 10.0


def arc_sample_points(seg_a, seg_b, spacing=SAMPLE_SPACING_M):
    """Evenly spaced points along each segment. Returns (points, arc_index)."""
    length = np.linalg.norm(seg_b - seg_a, axis=1)
    n_pieces = np.maximum(1, np.ceil(length / spacing).astype(int))
    arc_idx = np.repeat(np.arange(len(seg_a)), n_pieces + 1)
    # fraction along the segment: 0, 1/k, ..., 1 for each arc
    starts = np.repeat(np.cumsum(n_pieces + 1) - (n_pieces + 1), n_pieces + 1)
    step = np.arange(len(arc_idx)) - starts
    frac = step / np.repeat(n_pieces, n_pieces + 1)
    pts = seg_a[arc_idx] + frac[:, None] * (seg_b - seg_a)[arc_idx]
    return pts, arc_idx


def pair_distance(points, seg_a, seg_b):
    """Row-wise distance from point k to segment k, plus position t in [0, 1]."""
    ab = seg_b - seg_a
    length_sq = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    t = np.clip(((points - seg_a) * ab).sum(axis=1) / length_sq, 0.0, 1.0)
    closest = seg_a + t[:, None] * ab
    return np.sqrt(((points - closest) ** 2).sum(axis=1)), t


def generate_candidates(trip, radius_m):
    """All (point, arc) pairs with distance <= radius_m.

    Returns a DataFrame with columns point_idx, arc_id, dist_m, proj_t.
    Points with no arc within the radius simply have no rows.
    """
    gps = trip.gps_xy
    a_all, b_all = trip.arc_start_xy, trip.arc_end_xy

    # 1. arcs near the track (bounding box + radius)
    lo, hi = gps.min(axis=0) - radius_m, gps.max(axis=0) + radius_m
    near = ~((np.maximum(a_all, b_all) < lo).any(axis=1) | (np.minimum(a_all, b_all) > hi).any(axis=1))
    arc_ids = np.where(near)[0]
    if len(arc_ids) == 0:
        return pd.DataFrame(columns=["point_idx", "arc_id", "dist_m", "proj_t"])

    # 2-4. KD-tree over sample points along the arcs
    samples, sample_arc = arc_sample_points(a_all[arc_ids], b_all[arc_ids])
    tree = cKDTree(samples)
    hits = tree.query_ball_point(gps, r=radius_m + SAMPLE_SPACING_M / 2)

    point_idx, cand_arc = [], []
    for i, h in enumerate(hits):
        if h:
            arcs_i = np.unique(arc_ids[sample_arc[h]])
            point_idx.append(np.full(len(arcs_i), i))
            cand_arc.append(arcs_i)
    if not point_idx:
        return pd.DataFrame(columns=["point_idx", "arc_id", "dist_m", "proj_t"])
    point_idx = np.concatenate(point_idx)
    cand_arc = np.concatenate(cand_arc)

    # 5. exact distance, keep only within radius
    dist, t = pair_distance(gps[point_idx], a_all[cand_arc], b_all[cand_arc])
    keep = dist <= radius_m
    return pd.DataFrame({"point_idx": point_idx[keep], "arc_id": cand_arc[keep],
                         "dist_m": dist[keep], "proj_t": t[keep]})


def radius_study(radii=(20, 30, 40, 50, 75, 100)):
    """For each radius: candidate miss rate and candidates per point (TRAIN only).

    Miss = the labelled (true) arc is not among the point's candidates.
    Reported for reliable-label points (what we train/evaluate on) and all points.
    """
    labels = load_labels(TRAIN)
    max_r = max(radii)
    rows = []
    for tid in list_trips(TRAIN):
        trip = load_trip(TRAIN, tid)
        lab = labels[labels.trip_id == tid].reset_index(drop=True)
        cand = generate_candidates(trip, max_r)
        true_arc = lab.arc_id.values[cand.point_idx.values]
        cand["is_true"] = cand.arc_id.values == true_arc
        for r in radii:
            c = cand[cand.dist_m <= r]
            n_cand = np.bincount(c.point_idx, minlength=len(lab))
            hit = np.zeros(len(lab), bool)
            hit[c.point_idx[c.is_true].values] = True
            rows.append({"trip_id": tid, "radius_m": r, "n_points": len(lab),
                         "n_reliable": int(lab.label_reliable.sum()),
                         "miss_all": int((~hit).sum()),
                         "miss_reliable": int((~hit & lab.label_reliable.values).sum()),
                         "no_candidates": int((n_cand == 0).sum()),
                         "sum_candidates": int(n_cand.sum())})
    df = pd.DataFrame(rows)
    g = df.groupby("radius_m").sum(numeric_only=True)
    out = pd.DataFrame({
        "miss_rate_reliable_%": 100 * g.miss_reliable / g.n_reliable,
        "miss_rate_all_%": 100 * g.miss_all / g.n_points,
        "points_without_candidates_%": 100 * g.no_candidates / g.n_points,
        "mean_candidates_per_point": g.sum_candidates / g.n_points,
    })
    path = RESULTS_DIR / "candidates"
    path.mkdir(parents=True, exist_ok=True)
    out.to_csv(path / "radius_study_train.csv")
    return out


if __name__ == "__main__":
    print(radius_study().round(3).to_string())
