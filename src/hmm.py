"""Stage 6: HMM + Viterbi route reconstruction.

Hidden states   : at GPS point i, the state is "the vehicle is on candidate
                  arc c" (c from the point's candidate set, the top-K by
                  classifier probability). A state remembers its arc id and the
                  projected position on the arc (proj_t), so every decoded
                  state maps back to exactly one road segment.
Emission        : log P(candidate is true) from the trained classifier.
Transition      : the vehicle must drive along the network. For candidate a
                  at point i-1 and candidate b at point i we compute
                    d_route = network driving distance from a's projected
                              position to b's projected position
                    d_gps   = straight-line distance between the two GPS points
                  and use  log P(a -> b) = -|d_route - d_gps| / beta
                  (Newson & Krumm 2009 idea: on the right path the driven
                  distance is close to the straight-line distance).
                  If no path exists within the search limit the transition
                  is impossible (-inf).
Decoding        : Viterbi = dynamic programming that keeps, for every state,
                  the best score of any path ending there:
                    V_i(b) = w * log_emit_i(b) + max_a [ V_{i-1}(a) + log_trans(a, b) ]
                  and a back-pointer to the a that achieved the max; the best
                  final state is traced back to recover the whole route.
Breaks          : the chain is cut (a new independent segment starts) when
                  - the time gap to the previous point is > GAP_S seconds, or
                  - no transition from any previous state is possible.
                  Points with no candidates are skipped (no prediction).
"""
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd

GAP_S = 60.0                 # time gap that splits the HMM chain
DEFAULT_PARAMS = {"beta": 10.0, "emission_weight": 1.0, "top_k": 8}
BASE_SEARCH_M = 300.0        # Dijkstra search limit beyond the GPS step


class RoadGraph:
    """Directed road network with cached, distance-limited Dijkstra."""

    def __init__(self, trip):
        self.arcs = trip.arcs
        self.length = np.linalg.norm(trip.arc_end_xy - trip.arc_start_xy, axis=1)
        self.out = defaultdict(list)                  # node -> [(next_node, length)]
        for (u, v), L in zip(self.arcs, self.length):
            self.out[u].append((v, L))
        self.cache = {}                               # (node, limit) -> {node: dist}

    def distances_from(self, source, limit):
        """Shortest driving distance from source to every node within limit metres."""
        key = (source, limit)
        if key in self.cache:
            return self.cache[key]
        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for v, L in self.out[u]:
                nd = d + L
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self.cache[key] = dist
        return dist

    def route_distance(self, arc_a, t_a, arc_b, t_b, limit):
        """Driving distance from position t_a on arc_a to t_b on arc_b (inf if none)."""
        if arc_a == arc_b:
            forward = (t_b - t_a) * self.length[arc_a]
            if forward >= -10.0:                       # small backward jitter = GPS noise
                return abs(forward)
        rest_a = (1.0 - t_a) * self.length[arc_a]      # drive to the end of arc a
        into_b = t_b * self.length[arc_b]              # then from start of arc b
        reach = self.distances_from(self.arcs[arc_a, 1], limit)
        mid = reach.get(self.arcs[arc_b, 0])
        return np.inf if mid is None else rest_a + mid + into_b


def _limit_for(d_gps):
    """Search limit rounded up to 100 m buckets so the Dijkstra cache is reused."""
    return float(np.ceil((BASE_SEARCH_M + 2.0 * d_gps) / 100.0) * 100.0)


def viterbi_trip(trip, cand, prob, params=None, graph=None):
    """Decode one trip.

    cand : candidate table of this trip (point_idx, arc_id, proj_t, ...)
    prob : classifier probability for every row of cand
    Returns a DataFrame (point_idx, arc_id, prob, segment) - one row per point
    that has candidates - plus the number of chain breaks.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    graph = graph or RoadGraph(trip)
    c = cand[["point_idx", "arc_id", "proj_t"]].copy()
    c["prob"] = prob
    c = (c.sort_values(["point_idx", "prob"], ascending=[True, False])
          .groupby("point_idx").head(int(p["top_k"])))
    groups = [(i, g) for i, g in c.groupby("point_idx", sort=True)]

    out_points, out_arcs, out_probs, out_seg = [], [], [], []
    segment = 0

    def flush(states, back):
        """Trace back the best path of the current chain and store it."""
        best = int(np.argmax(states[-1][3]))
        path = [best]
        for k in range(len(states) - 1, 0, -1):
            path.append(back[k][path[-1]])
        path.reverse()
        for (pi, arcs, probs, _, _), s in zip(states, path):
            out_points.append(pi); out_arcs.append(arcs[s]); out_probs.append(probs[s])
            out_seg.append(segment)

    states, back = [], []
    prev_pi = None
    for pi, g in groups:
        arcs = g.arc_id.to_numpy()
        ts = g.proj_t.to_numpy()
        probs = g.prob.to_numpy()
        emit = p["emission_weight"] * np.log(np.clip(probs, 1e-6, 1.0))
        new_chain = prev_pi is None or (trip.times[pi] - trip.times[prev_pi]) > GAP_S
        if not new_chain:
            d_gps = float(np.linalg.norm(trip.gps_xy[pi] - trip.gps_xy[prev_pi]))
            limit = _limit_for(d_gps)
            _, prev_arcs, _, prev_score, prev_ts = states[-1]
            trans = np.full((len(prev_arcs), len(arcs)), -np.inf)
            for a_i, (a, ta) in enumerate(zip(prev_arcs, prev_ts)):
                if not np.isfinite(prev_score[a_i]):
                    continue
                for b_i, (b, tb) in enumerate(zip(arcs, ts)):
                    d_route = graph.route_distance(a, ta, b, tb, limit)
                    if np.isfinite(d_route):
                        trans[a_i, b_i] = -abs(d_route - d_gps) / p["beta"]
            total = prev_score[:, None] + trans
            best_prev = total.argmax(axis=0)
            best_val = total.max(axis=0)
            if not np.isfinite(best_val).any():
                new_chain = True                       # disconnected: start again
        if new_chain:
            if states:
                flush(states, back)
                segment += 1
            states, back = [], []
            score = emit
            back.append(np.zeros(len(arcs), dtype=int))
        else:
            score = best_val + emit
            back.append(best_prev)
        states.append((pi, arcs, probs, score, ts))
        prev_pi = pi
    if states:
        flush(states, back)
    result = pd.DataFrame({"point_idx": out_points, "arc_id": out_arcs,
                           "prob": out_probs, "segment": out_seg})
    return result, segment + 1 if len(result) else 0


def classifier_only(cand, prob):
    """Top-1 per point without the HMM (for comparison)."""
    c = cand[["point_idx", "arc_id"]].copy()
    c["prob"] = prob
    c = c.sort_values(["point_idx", "prob"], ascending=[True, False])
    return c.drop_duplicates("point_idx").reset_index(drop=True)
