"""Turn the reference route into one label per GPS observation.

Problem: the .route file lists the arcs the vehicle drove, in order, but does
NOT say which GPS point belongs to which arc (about 8 points per arc on
average). We need a per-point "true arc" to train and evaluate classifiers.

Naive idea: give each point the nearest route arc. This fails at junctions,
on roads driven twice, and where a road is driven in both directions,
because the nearest arc may be one the car only reaches later.

Our method: monotonic alignment (dynamic programming).
Walk through the GPS points in time order and the route arcs in route order
together. Each point is assigned to one route position, and the position can
only stay the same or move forward (the car never drives the route
backwards). Among all such assignments we pick the one with the smallest total
point-to-arc distance.

    D[i, j] = dist(point i, route arc j) + min_{k <= j} D[i-1, k]

The answer is found by tracing back from the best final cell. Route arcs that
no point falls on (very short arcs, or arcs crossed during a GPS gap) are
simply skipped, which the min over k <= j allows.
"""
import numpy as np
import pandas as pd

from .config import DERIVED_DIR
from .data_io import list_trips, load_trip
from .geo import point_segment_distance

# A label further than this from its GPS point is marked "unreliable" (used to
# flag suspicious trips/points in the EDA; it does not change the label).
RELIABLE_DIST_M = 50.0


def align_points_to_route(trip, chunk=2000):
    """Return (route_pos, dist_m) for every GPS point of a trip."""
    route = trip.route
    seg_a = trip.nodes_xy[trip.arcs[route, 0]]
    seg_b = trip.nodes_xy[trip.arcs[route, 1]]
    n, m = len(trip.gps_xy), len(route)

    back = np.zeros((n, m), dtype=np.int16)   # best previous route position
    prev = None
    for start in range(0, n, chunk):
        cost, _ = point_segment_distance(trip.gps_xy[start:start + chunk], seg_a, seg_b)
        for r in range(cost.shape[0]):
            i = start + r
            if prev is None:                   # first point: may start anywhere
                prev = cost[r].copy()
                back[i] = np.arange(m)
                continue
            # running minimum of previous row = best way to reach "position <= j"
            best_prev = np.minimum.accumulate(prev)
            # index where that running minimum came from
            idx = np.arange(m)
            is_new_min = prev <= best_prev
            arg = np.maximum.accumulate(np.where(is_new_min, idx, 0))
            back[i] = arg
            prev = cost[r] + best_prev

    # Trace back from the cheapest final position
    pos = np.zeros(n, dtype=int)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    # Distance of each point to its assigned arc
    dist = point_segment_distance_pairs(trip.gps_xy, seg_a[pos], seg_b[pos])
    return pos, dist


def point_segment_distance_pairs(points, seg_a, seg_b):
    """Distance from point k to segment k (row-wise, not all pairs)."""
    ab = seg_b - seg_a
    length_sq = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    t = np.clip(((points - seg_a) * ab).sum(axis=1) / length_sq, 0, 1)
    closest = seg_a + t[:, None] * ab
    return np.sqrt(((points - closest) ** 2).sum(axis=1))


def label_trip(trip):
    pos, dist = align_points_to_route(trip)
    return pd.DataFrame({
        "trip_id": trip.trip_id,
        "point_idx": np.arange(len(pos)),
        "route_pos": pos,
        "arc_id": trip.route[pos],
        "label_dist_m": dist,
        "label_reliable": dist <= RELIABLE_DIST_M,
    })


def build_labels(partition):
    """Label every trip of a partition and save to data_derived/."""
    frames = []
    for tid in list_trips(partition):
        frames.append(label_trip(load_trip(partition, tid)))
    df = pd.concat(frames, ignore_index=True)
    DERIVED_DIR.mkdir(exist_ok=True)
    df.to_csv(DERIVED_DIR / f"labels_{partition}.csv", index=False)
    return df


def load_labels(partition):
    return pd.read_csv(DERIVED_DIR / f"labels_{partition}.csv", dtype={"trip_id": str})


if __name__ == "__main__":
    import time
    for part in ["train", "student_test"]:
        t0 = time.time()
        df = build_labels(part)
        print(f"{part}: {len(df)} points labelled in {time.time() - t0:.1f}s, "
              f"median dist {df.label_dist_m.median():.2f} m, "
              f"unreliable {100 * (~df.label_reliable).mean():.2f}%")
