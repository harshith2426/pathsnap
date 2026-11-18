"""Small geometry helpers.

Coordinates in the dataset are longitude/latitude in degrees. Degrees are not
metres (1 degree of longitude shrinks towards the poles), so every trip is
converted to a flat local x/y grid in metres before any distance is computed.

We use an equirectangular projection centred on the trip:
    x = R * (lon - lon0) * cos(lat0)
    y = R * (lat - lat0)
Each trip covers at most a few tens of km, so the error of this flat
approximation is well under 1% - far smaller than GPS noise (~5 m).
"""
import numpy as np

EARTH_RADIUS_M = 6_371_000.0


class LocalProjection:
    """Converts lon/lat degrees <-> local metres around a fixed origin."""

    def __init__(self, lon0, lat0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.cos_lat0 = np.cos(np.radians(self.lat0))

    def to_xy(self, lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        x = np.radians(lonlat[:, 0] - self.lon0) * self.cos_lat0 * EARTH_RADIUS_M
        y = np.radians(lonlat[:, 1] - self.lat0) * EARTH_RADIUS_M
        return np.column_stack([x, y])

    def to_lonlat(self, xy):
        xy = np.asarray(xy, dtype=float)
        lon = self.lon0 + np.degrees(xy[:, 0] / (EARTH_RADIUS_M * self.cos_lat0))
        lat = self.lat0 + np.degrees(xy[:, 1] / EARTH_RADIUS_M)
        return np.column_stack([lon, lat])


def point_segment_distance(points, seg_a, seg_b):
    """Distance from every point to every segment (all in metres).

    points: (n, 2), seg_a / seg_b: (m, 2) segment start and end points.
    Returns (dist, t), both shape (n, m). t in [0, 1] is how far along the
    segment the closest point lies (0 = start, 1 = end).
    """
    points = np.atleast_2d(points)
    ab = seg_b - seg_a                                   # (m, 2)
    length_sq = (ab ** 2).sum(axis=1)
    length_sq[length_sq == 0] = 1e-9                     # zero-length segments
    ap = points[:, None, :] - seg_a[None, :, :]          # (n, m, 2)
    t = np.clip((ap * ab[None]).sum(axis=2) / length_sq[None], 0.0, 1.0)
    closest = seg_a[None] + t[..., None] * ab[None]
    dist = np.sqrt(((points[:, None, :] - closest) ** 2).sum(axis=2))
    return dist, t


def bearing_deg(dx, dy):
    """Compass bearing in degrees (0 = north, 90 = east) of a direction vector."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)
