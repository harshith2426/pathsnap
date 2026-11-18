"""Evaluation metrics.

Two levels:
  1. Pair level  - each (point, candidate) row is a binary example
                   (accuracy, precision, recall, F1, macro F1, log loss, confusion).
  2. Point/route level - for each GPS point pick the candidate with the
                   highest score (Top-1), then:
       top1_accuracy     share of reliable points whose pick is the labelled arc.
                         Points with no candidates count as wrong.
       route_continuity  share of consecutive picks that are the same arc or
                         directly connected (end node of one = start node of next).
       route_arc_f1      per trip: F1 between the set of picked arcs and the
                         set of labelled arcs, averaged over trips.
Only points with reliable labels (<= 50 m) are scored.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from .data_io import list_trips, load_trip

_ARCS_CACHE = {}


def arcs_of(partition):
    """{trip_id: arcs array} for a partition (cached)."""
    if partition not in _ARCS_CACHE:
        _ARCS_CACHE[partition] = {t: load_trip(partition, t, with_route=False).arcs
                                  for t in list_trips(partition)}
    return _ARCS_CACHE[partition]


def pair_metrics(y_true, prob, threshold=0.5):
    y_pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def top1_picks(table, score):
    """One row per point: the candidate with the highest score."""
    df = table[["trip_id", "point_idx", "arc_id", "is_true"]].copy()
    df["score"] = score
    df = df.sort_values(["trip_id", "point_idx", "score"], ascending=[True, True, False])
    return df.drop_duplicates(["trip_id", "point_idx"]).reset_index(drop=True)


def top1_accuracy(picks, points):
    """Share of reliable points whose top pick is the labelled arc."""
    rel = points[points.label_reliable][["trip_id", "point_idx"]]
    merged = rel.merge(picks, on=["trip_id", "point_idx"], how="left")
    return float(merged.is_true.fillna(0).mean())


def route_continuity(picks, partition):
    """Share of consecutive picked arcs that are equal or directly connected."""
    arcs = arcs_of(partition)
    ok = total = 0
    for tid, g in picks.groupby("trip_id"):
        seq = g.sort_values("point_idx").arc_id.to_numpy()
        a = arcs[tid]
        prev, cur = seq[:-1], seq[1:]
        good = (prev == cur) | (a[prev, 1] == a[cur, 0])
        ok += good.sum()
        total += len(good)
    return ok / max(total, 1)


def route_arc_f1(picks, points):
    """Mean over trips of F1(set of picked arcs, set of labelled arcs), reliable points."""
    rel = points[points.label_reliable]
    merged = rel.merge(picks, on=["trip_id", "point_idx"], how="left")
    scores = []
    for _, g in merged.groupby("trip_id"):
        true_set = set(g.true_arc)
        pred_set = set(g.arc_id.dropna().astype(int))
        tp = len(true_set & pred_set)
        if tp == 0:
            scores.append(0.0)
            continue
        p, r = tp / len(pred_set), tp / len(true_set)
        scores.append(2 * p * r / (p + r))
    return float(np.mean(scores))


def route_metrics(picks, points, partition):
    return {
        "top1_accuracy": top1_accuracy(picks, points),
        "route_continuity": route_continuity(picks, partition),
        "route_arc_f1": route_arc_f1(picks, points),
    }


def reference_continuity(points, partition):
    """Continuity of the labels themselves (upper reference for the metric)."""
    rel = points[points.label_reliable].rename(columns={"true_arc": "arc_id"})
    return route_continuity(rel[["trip_id", "point_idx", "arc_id"]], partition)
