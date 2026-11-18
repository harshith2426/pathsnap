"""Interactive Folium maps of a trip.

Layers (each can be switched on/off with the layer control):
  - road network near the track (grey)
  - reference route (green)            - only if given
  - predicted route (red)              - only if given (used after modelling)
  - raw GPS observations (blue dots)

Run:  python -m src.maps      -> writes maps/eda_trip_<id>.html
"""
import folium
import numpy as np

from .config import MAPS_DIR, TRAIN
from .data_io import load_trip

LEGEND_HTML = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
     background: white; padding: 8px 12px; border: 1px solid #999;
     border-radius: 6px; font-size: 13px; line-height: 1.6;">
  <b>{title}</b><br>
  {rows}
</div>
"""


def _legend_row(color, text, dot=False):
    shape = "border-radius:50%;width:10px;height:10px" if dot else "width:22px;height:4px"
    return (f'<span style="display:inline-block;{shape};background:{color};'
            f'margin-right:6px;vertical-align:middle"></span>{text}<br>')


def _arc_lines(trip, arc_ids):
    """List of [[lat, lon], [lat, lon]] lines for the given arcs."""
    ll = trip.nodes_lonlat
    return [[[ll[a, 1], ll[a, 0]], [ll[b, 1], ll[b, 0]]] for a, b in trip.arcs[arc_ids]]


def nearby_arcs(trip, margin_m=200):
    """Arcs with an end inside the GPS bounding box (+ margin). Two-way
    streets are drawn once."""
    lo = trip.gps_xy.min(axis=0) - margin_m
    hi = trip.gps_xy.max(axis=0) + margin_m
    a, b = trip.arc_start_xy, trip.arc_end_xy
    inside = ((a >= lo) & (a <= hi)).all(axis=1) | ((b >= lo) & (b <= hi)).all(axis=1)
    ids = np.where(inside)[0]
    pairs = np.sort(trip.arcs[ids], axis=1)
    _, first = np.unique(pairs, axis=0, return_index=True)
    return ids[np.sort(first)]


def trip_map(trip, route_arcs=None, predicted_arcs=None, title=None, max_points=3000):
    """Build a folium.Map for a trip. route_arcs / predicted_arcs are arc ids."""
    lat_lon = trip.gps_lonlat[:, ::-1]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([lat_lon.min(axis=0).tolist(), lat_lon.max(axis=0).tolist()])

    net = folium.FeatureGroup(name="Road network (nearby)", show=True)
    folium.PolyLine(_arc_lines(trip, nearby_arcs(trip)), color="#2c3e50",
                    weight=2, opacity=0.55).add_to(net)
    net.add_to(m)

    legend = [_legend_row("#2c3e50", "Road network")]
    if route_arcs is not None:
        layer = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_arc_lines(trip, np.unique(route_arcs)), color="green",
                        weight=6, opacity=0.7).add_to(layer)
        layer.add_to(m)
        legend.append(_legend_row("green", "Reference route"))
    if predicted_arcs is not None:
        layer = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_arc_lines(trip, np.unique(predicted_arcs)), color="red",
                        weight=3, opacity=0.9).add_to(layer)
        layer.add_to(m)
        legend.append(_legend_row("red", "Predicted route"))

    # GPS points (subsampled evenly if the track is very long, to keep HTML small)
    step = max(1, len(lat_lon) // max_points)
    gps = folium.FeatureGroup(name="GPS observations", show=True)
    for i in range(0, len(lat_lon), step):
        folium.CircleMarker(lat_lon[i].tolist(), radius=2, color="blue", fill=True,
                            fill_opacity=0.8, weight=0,
                            tooltip=f"obs {i}, t={trip.times[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    legend.append(_legend_row("blue", "GPS observation", dot=True))

    folium.LayerControl(collapsed=False).add_to(m)
    title = title or f"Trip {trip.trip_id} ({trip.partition})"
    m.get_root().html.add_child(folium.Element(
        LEGEND_HTML.format(title=title, rows="".join(legend))))
    return m


def save_eda_map(partition, trip_id, note=""):
    trip = load_trip(partition, trip_id)
    title = f"Trip {trip_id} ({partition}){note}"
    m = trip_map(trip, route_arcs=trip.route, title=title)
    MAPS_DIR.mkdir(exist_ok=True)
    path = MAPS_DIR / f"eda_trip_{trip_id}.html"
    m.save(str(path))
    return path


if __name__ == "__main__":
    # A clean example, and the trip whose reference route stops early.
    print(save_eda_map(TRAIN, "00000027"))
    print(save_eda_map(TRAIN, "00000054", " - route ends before GPS track"))
