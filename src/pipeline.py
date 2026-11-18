"""End-to-end prediction for one trip (used by the API).

    GPS track + road network  ->  candidates + features  ->  classifier
    probabilities  ->  HMM/Viterbi  ->  one road segment per GPS point.

The reference route is NOT an input. Scoring against the reference route is a
separate function (`score_prediction`) that only the API server calls.
"""
import json
import time

import joblib
import numpy as np

from .config import ARTIFACTS_DIR
from .hmm import RoadGraph, viterbi_trip
from .features import build_features
from .labels import label_trip


class PathSnapModel:
    """Loads the exported artifacts once and predicts trips."""

    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        self.classifier = joblib.load(artifacts_dir / "classifier.joblib")
        with open(artifacts_dir / "classifier_config.json") as fh:
            self.clf_config = json.load(fh)
        with open(artifacts_dir / "hmm_config.json") as fh:
            self.hmm_config = json.load(fh)
        self.features = self.clf_config["features"]
        self.radius = self.clf_config["candidate_radius_m"]

    def predict(self, trip):
        """trip must be loaded with with_route=False. Returns dict of results."""
        t0 = time.time()
        cand = build_features(trip, self.radius)
        t1 = time.time()
        if cand.empty:
            prob = np.array([])
            decoded = None
        else:
            prob = self.classifier.predict_proba(cand[self.features].to_numpy(dtype=np.float32))[:, 1]
            t2 = time.time()
            params = {k: self.hmm_config[k] for k in ("beta", "emission_weight", "top_k")}
            decoded, n_segments = viterbi_trip(trip, cand, prob, params, RoadGraph(trip))
        t3 = time.time()
        arc_per_point = np.full(len(trip.times), -1)
        if decoded is not None:
            arc_per_point[decoded.point_idx.to_numpy()] = decoded.arc_id.to_numpy()
        return {
            "arc_per_point": arc_per_point,               # -1 = no candidate within radius
            "n_candidates": len(cand),
            "n_segments": int(n_segments) if decoded is not None else 0,
            "timing_s": {"features": t1 - t0,
                         "classifier": (t2 - t1) if decoded is not None else 0.0,
                         "hmm": (t3 - t2) if decoded is not None else 0.0,
                         "total": t3 - t0},
        }


def route_continuity_of(arcs, seq):
    """Share of consecutive predicted arcs that are equal or connected."""
    seq = seq[seq >= 0]
    if len(seq) < 2:
        return 1.0
    prev, cur = seq[:-1], seq[1:]
    return float(np.mean((prev == cur) | (arcs[prev, 1] == arcs[cur, 0])))


def score_prediction(trip_with_route, arc_per_point):
    """Compare predictions with the reference route (server side only).

    Labels are made the same way as in development (monotonic alignment,
    points > 50 m from the route are not scored)."""
    labels = label_trip(trip_with_route)
    rel = labels.label_reliable.to_numpy()
    correct = arc_per_point[rel] == labels.arc_id.to_numpy()[rel]
    true_set = set(labels.arc_id[rel])
    pred_set = set(arc_per_point[rel][arc_per_point[rel] >= 0])
    tp = len(true_set & pred_set)
    arc_f1 = 0.0 if tp == 0 else 2 * tp / (len(true_set) + len(pred_set))
    return {
        "point_accuracy": float(correct.mean()) if rel.any() else None,
        "scored_points": int(rel.sum()),
        "unscored_points": int((~rel).sum()),
        "route_arc_f1": arc_f1,
        "route_continuity": route_continuity_of(trip_with_route.arcs, arc_per_point),
    }
