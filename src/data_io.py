"""Reading the raw trip files.

Each trip folder <id>/ holds four header-less, whitespace-separated files:
  <id>.track  lon  lat  time_s      one row per GPS observation
  <id>.nodes  lon  lat              row index = node_id
  <id>.arcs   from_node  to_node    row index = arc_id (directed road segment)
  <id>.route  arc_id                reference route, arcs in driving order

Node/arc ids are LOCAL to a trip: arc 5 in trip A has nothing to do with
arc 5 in trip B.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import DATA_DIR
from .geo import LocalProjection


def list_trips(partition):
    """Sorted trip ids (folder names) of a partition, e.g. 'train'."""
    folder = DATA_DIR / partition
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


@dataclass
class Trip:
    trip_id: str
    partition: str
    gps_lonlat: np.ndarray      # (n_obs, 2)
    times: np.ndarray           # (n_obs,) seconds
    nodes_lonlat: np.ndarray    # (n_nodes, 2)
    arcs: np.ndarray            # (n_arcs, 2) int: from_node, to_node
    route: np.ndarray = None    # (n_route,) int arc ids, None if not supplied
    proj: LocalProjection = field(init=False)
    gps_xy: np.ndarray = field(init=False)
    nodes_xy: np.ndarray = field(init=False)

    def __post_init__(self):
        # Projection origin = centre of the GPS track, so the trip's own area
        # has the smallest distortion.
        lon0, lat0 = self.gps_lonlat.mean(axis=0)
        self.proj = LocalProjection(lon0, lat0)
        self.gps_xy = self.proj.to_xy(self.gps_lonlat)
        self.nodes_xy = self.proj.to_xy(self.nodes_lonlat)

    # Start / end coordinates (metres) of every arc
    @property
    def arc_start_xy(self):
        return self.nodes_xy[self.arcs[:, 0]]

    @property
    def arc_end_xy(self):
        return self.nodes_xy[self.arcs[:, 1]]


def _read(path, dtype=float, ncols=None):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if ncols is not None and arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def load_trip(partition, trip_id, with_route=True):
    """Load one trip. with_route=False never reads the reference route
    (this is what the prediction pipeline uses)."""
    base = DATA_DIR / partition / trip_id / trip_id
    track = _read(f"{base}.track", float, 3)
    nodes = _read(f"{base}.nodes", float, 2)
    arcs = _read(f"{base}.arcs", int, 2)
    route = None
    if with_route:
        route = _read(f"{base}.route", int, 1)[:, 0]
    return Trip(trip_id, partition, track[:, :2], track[:, 2], nodes, arcs, route)
