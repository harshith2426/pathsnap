"""Exploratory data analysis for the train and student_test partitions.

Outputs (results/eda/):
  trip_inventory.csv        one row per trip: sizes, gaps, network, route, label quality
  partition_summary.csv     totals / medians per partition
  data_quality_flags.csv    trips with anomalies and why
  *.png                     plots used in the report

Run:  python -m src.eda      (needs labels: python -m src.labels first)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .config import DATA_DIR, RESULTS_DIR, STUDENT_TEST, TRAIN
from .data_io import list_trips, load_trip
from .labels import load_labels

OUT = RESULTS_DIR / "eda"
BIG_GAP_S = 60          # a time step longer than this is a "large gap"


def file_sizes_kb(partition, tid):
    base = DATA_DIR / partition / tid / tid
    return {f"kb_{ext}": os.path.getsize(f"{base}.{ext}") / 1024
            for ext in ["track", "nodes", "arcs", "route"]}


def network_stats(trip):
    """Size and connectivity of the trip's road network."""
    n_nodes, arcs = len(trip.nodes_xy), trip.arcs
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(map(tuple, arcs))
    degree = np.bincount(arcs.ravel(), minlength=n_nodes)
    arc_set = set(map(tuple, arcs))
    weak = list(nx.weakly_connected_components(g))
    strong = max(nx.strongly_connected_components(g), key=len)
    seg_len = np.linalg.norm(trip.arc_end_xy - trip.arc_start_xy, axis=1)
    return {
        "n_nodes": n_nodes,
        "n_arcs": len(arcs),
        "isolated_nodes": int((degree == 0).sum()),
        "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "duplicate_arcs": len(arcs) - len(arc_set),
        # share of arcs whose reverse also exists (two-way streets)
        "pct_two_way": 100 * np.mean([(b, a) in arc_set for a, b in arcs]),
        "n_weak_components": len(weak),
        "pct_nodes_largest_weak": 100 * max(map(len, weak)) / n_nodes,
        "pct_nodes_largest_strong": 100 * len(strong) / n_nodes,
        "arc_len_median_m": np.median(seg_len),
    }


def track_stats(trip):
    """Sampling, gaps, missing/invalid values and extent of the GPS track."""
    t, ll, xy = trip.times, trip.gps_lonlat, trip.gps_xy
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = step / np.where(dt > 0, dt, np.nan)
    extent = xy.max(axis=0) - xy.min(axis=0)
    return {
        "n_obs": len(t),
        "n_missing_values": int(np.isnan(ll).sum() + np.isnan(t).sum()),
        "n_invalid_coords": int(((np.abs(ll[:, 0]) > 180) | (np.abs(ll[:, 1]) > 90)).sum()),
        "n_non_increasing_time": int((dt <= 0).sum()),
        "t_start": t[0],
        "duration_s": t[-1] - t[0],
        "pct_dt_1s": 100 * np.mean(np.isclose(dt, 1.0)),
        "max_gap_s": dt.max(),
        "n_gaps_over_60s": int((dt > BIG_GAP_S).sum()),
        "pct_stationary_steps": 100 * np.mean(step < 0.5),
        "speed_median_ms": np.nanmedian(speed),
        "speed_p99_ms": np.nanpercentile(speed, 99),
        "path_length_km": step.sum() / 1000,
        "extent_x_km": extent[0] / 1000,
        "extent_y_km": extent[1] / 1000,
        "lon_center": ll[:, 0].mean(),
        "lat_center": ll[:, 1].mean(),
    }


def route_stats(trip, labels):
    """Structure of the reference route and how well GPS fits it."""
    r, arcs = trip.route, trip.arcs
    connected = arcs[r[1:], 0] == arcs[r[:-1], 1]
    return {
        "route_len": len(r),
        "route_unique_arcs": len(np.unique(r)),
        "route_ids_valid": bool(((r >= 0) & (r < len(arcs))).all()),
        "route_breaks": int((~connected).sum()),   # consecutive arcs not touching
        "obs_per_route_arc": len(trip.times) / len(r),
        "label_dist_median_m": labels.label_dist_m.median(),
        "label_dist_p95_m": labels.label_dist_m.quantile(0.95),
        "pct_unreliable_labels": 100 * (~labels.label_reliable).mean(),
    }


def build_inventory():
    rows = []
    for part in [TRAIN, STUDENT_TEST]:
        labels = load_labels(part)
        for tid in list_trips(part):
            trip = load_trip(part, tid)
            lab = labels[labels.trip_id == tid]
            rows.append({"partition": part, "trip_id": tid,
                         **file_sizes_kb(part, tid), **track_stats(trip),
                         **network_stats(trip), **route_stats(trip, lab)})
    return pd.DataFrame(rows)


def quality_flags(inv):
    """List trips with anomalies worth knowing about."""
    rules = {
        "large time gap (>10 min)": inv.max_gap_s > 600,
        "route ends before GPS track (>20% unreliable labels)": inv.pct_unreliable_labels > 20,
        "some GPS far from route (5-20% unreliable labels)": inv.pct_unreliable_labels.between(5, 20),
        "route not fully connected": inv.route_breaks > 0,
        "track does not start at t=1": inv.t_start > 1,
        "low 1 Hz share (<80%)": inv.pct_dt_1s < 80,
    }
    rows = []
    for reason, mask in rules.items():
        for _, r in inv[mask].iterrows():
            rows.append({"partition": r.partition, "trip_id": r.trip_id, "issue": reason})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plot_all(inv):
    colors = {TRAIN: "tab:blue", STUDENT_TEST: "tab:orange"}

    # 1. observations per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for part, c in colors.items():
        ax.hist(inv[inv.partition == part].n_obs, bins=np.logspace(2, 4.5, 25),
                alpha=0.6, color=c, label=part)
    ax.set_xscale("log"); ax.set_xlabel("GPS observations per trip (log)")
    ax.set_ylabel("trips"); ax.set_title("Trip size"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "obs_per_trip.png", dpi=130); plt.close(fig)

    # 2. sampling intervals (all steps, both partitions)
    bins = [0, 1.01, 2.01, 5.01, 10.01, 60.01, np.inf]
    names = ["1 s", "2 s", "3-5 s", "6-10 s", "11-60 s", ">60 s"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    for k, (part, c) in enumerate(colors.items()):
        dts = np.concatenate([np.diff(load_trip(part, t, with_route=False).times)
                              for t in inv[inv.partition == part].trip_id])
        share = 100 * np.histogram(dts, bins=bins)[0] / len(dts)
        ax.bar(np.arange(len(names)) + (k - 0.5) * width, share, width, color=c, label=part)
        for x, s in zip(np.arange(len(names)) + (k - 0.5) * width, share):
            ax.text(x, s + 1, f"{s:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(names)), names); ax.set_ylabel("% of time steps")
    ax.set_title("Time between consecutive GPS observations"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "sampling_intervals.png", dpi=130); plt.close(fig)

    # 3. GPS noise: distance from each point to its labelled route arc
    fig, ax = plt.subplots(figsize=(7, 4))
    for part, c in colors.items():
        d = load_labels(part).label_dist_m
        ax.hist(d.clip(upper=100), bins=100, alpha=0.6, color=c, label=part, density=True)
    ax.axvline(50, color="red", ls="--", lw=1, label="50 m reliability cut-off")
    ax.set_yscale("log"); ax.set_xlabel("distance GPS point -> labelled route arc (m, clipped at 100)")
    ax.set_ylabel("density (log)"); ax.set_title("GPS noise relative to reference route"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "gps_to_route_distance.png", dpi=130); plt.close(fig)

    # 4. road network size
    fig, ax = plt.subplots(figsize=(6, 5))
    for part, c in colors.items():
        s = inv[inv.partition == part]
        ax.scatter(s.n_nodes, s.n_arcs, color=c, label=part, s=18)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("nodes"); ax.set_ylabel("arcs (directed road segments)")
    ax.set_title("Road network size per trip"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "network_size.png", dpi=130); plt.close(fig)

    # 5. where the trips are
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for part, c in colors.items():
        s = inv[inv.partition == part]
        ax.scatter(s.lon_center, s.lat_center, color=c, label=part, s=18)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.grid(alpha=0.3)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Trip locations (centre of each GPS track)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "trip_locations.png", dpi=130); plt.close(fig)

    # 6. speed distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(inv.speed_median_ms * 3.6, bins=25, color="tab:blue")
    ax.set_xlabel("median speed per trip (km/h)"); ax.set_ylabel("trips")
    ax.set_title("Typical trip speed (train + student_test)")
    fig.tight_layout(); fig.savefig(OUT / "trip_speed.png", dpi=130); plt.close(fig)

    # 7. one example trajectory with network and route (compact train trip)
    plot_example_trip(TRAIN, "00000027", OUT / "example_trip_00000027.png")


def plot_example_trip(part, tid, path, margin_m=300):
    trip = load_trip(part, tid)
    xy = trip.gps_xy
    lo, hi = xy.min(axis=0) - margin_m, xy.max(axis=0) + margin_m
    a, b = trip.arc_start_xy, trip.arc_end_xy
    inside = ((a >= lo) & (a <= hi)).all(axis=1) | ((b >= lo) & (b <= hi)).all(axis=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    for s, e in zip(a[inside], b[inside]):
        ax.plot([s[0], e[0]], [s[1], e[1]], color="lightgray", lw=0.6)
    for arc in trip.route:
        s, e = trip.nodes_xy[trip.arcs[arc]]
        ax.plot([s[0], e[0]], [s[1], e[1]], color="green", lw=2.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=3, color="blue", alpha=0.5, zorder=3)
    ax.plot([], [], color="lightgray", label="road network")
    ax.plot([], [], color="green", lw=2.5, label="reference route")
    ax.scatter([], [], color="blue", s=8, label="GPS observations")
    ax.set_aspect("equal"); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x (m, local projection)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Trip {tid} ({part})"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = build_inventory()
    inv.to_csv(OUT / "trip_inventory.csv", index=False)

    summary = inv.groupby("partition").agg(
        trips=("trip_id", "size"),
        total_obs=("n_obs", "sum"),
        median_obs=("n_obs", "median"),
        missing_values=("n_missing_values", "sum"),
        invalid_coords=("n_invalid_coords", "sum"),
        non_increasing_time=("n_non_increasing_time", "sum"),
        median_pct_dt_1s=("pct_dt_1s", "median"),
        trips_with_gap_over_60s=("n_gaps_over_60s", lambda s: int((s > 0).sum())),
        median_nodes=("n_nodes", "median"),
        median_arcs=("n_arcs", "median"),
        isolated_nodes=("isolated_nodes", "sum"),
        median_weak_components=("n_weak_components", "median"),
        median_route_len=("route_len", "median"),
        total_route_breaks=("route_breaks", "sum"),
        median_label_dist_m=("label_dist_median_m", "median"),
    ).T
    summary.to_csv(OUT / "partition_summary.csv")

    flags = quality_flags(inv)
    flags.to_csv(OUT / "data_quality_flags.csv", index=False)

    plot_all(inv)
    print(summary.round(2).to_string())
    print("\nData-quality flags:\n", flags.to_string(index=False))


if __name__ == "__main__":
    main()
