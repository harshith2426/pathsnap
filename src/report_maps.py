"""Folium maps for the report: student-test trip 00000076.

  maps/report_trip_00000076_overview.html  whole trip
  maps/report_trip_00000076_zoom.html      zoomed on the stretch where the
                                           HMM changed the classifier's picks most

GPS dots are drawn first so the route lines stay visible on top of them.
Run:  python -m src.report_maps
"""
import folium
import numpy as np

from .config import MAPS_DIR
from .data_io import load_trip
from .features import build_features
from .hmm import classifier_only
from .maps import LEGEND_HTML, _arc_lines, _legend_row, nearby_arcs
from .pipeline import PathSnapModel

TRIP = "00000076"
PART = "student_test"


def build(trip, ref_arcs, hmm_arcs, cls_arcs, bounds, title, path):
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds(bounds)
    layers = [("Road network (nearby)", nearby_arcs(trip), "#2c3e50", 1.5, 0.45, None),
              ]
    for name, arcs, color, weight, opacity, dash in layers:
        g = folium.FeatureGroup(name=name)
        folium.PolyLine(_arc_lines(trip, arcs), color=color, weight=weight, opacity=opacity,
                        dash_array=dash).add_to(g)
        g.add_to(m)
    gps = folium.FeatureGroup(name="GPS observations")
    for i, (lon, lat) in enumerate(trip.gps_lonlat):
        folium.CircleMarker([lat, lon], radius=1.8, color="#1f6feb", weight=0, fill=True,
                            fill_opacity=0.7).add_to(gps)
    gps.add_to(m)
    for name, arcs, color, weight, opacity, dash in [
            ("Reference route", np.unique(ref_arcs), "green", 9, 0.45, None),
            ("Classifier-only picks", np.unique(cls_arcs), "orange", 4, 0.95, "6"),
            ("Predicted route (classifier + HMM)", np.unique(hmm_arcs), "red", 3, 0.95, None)]:
        g = folium.FeatureGroup(name=name)
        folium.PolyLine(_arc_lines(trip, arcs), color=color, weight=weight, opacity=opacity,
                        dash_array=dash).add_to(g)
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    rows = (_legend_row("#2c3e50", "Road network") + _legend_row("green", "Reference route")
            + _legend_row("orange", "Classifier-only picks") + _legend_row("red", "Classifier + HMM route")
            + _legend_row("#1f6feb", "GPS observation", dot=True))
    m.get_root().html.add_child(folium.Element(LEGEND_HTML.format(title=title, rows=rows)))
    m.save(str(path))
    return path


def main():
    model = PathSnapModel()
    trip = load_trip(PART, TRIP, with_route=True)
    pred = model.predict(load_trip(PART, TRIP, with_route=False))
    hmm_arcs = pred["arc_per_point"]
    cand = build_features(trip, model.radius)
    prob = model.classifier.predict_proba(cand[model.features].to_numpy(dtype=np.float32))[:, 1]
    cls = classifier_only(cand, prob).set_index("point_idx").arc_id
    cls_arcs = cls.reindex(range(len(trip.times))).fillna(-1).astype(int).to_numpy()

    ll = trip.gps_lonlat[:, ::-1]
    overview = [ll.min(axis=0).tolist(), ll.max(axis=0).tolist()]
    build(trip, trip.route, hmm_arcs[hmm_arcs >= 0], cls_arcs[cls_arcs >= 0], overview,
          f"Trip {TRIP} (student_test) - whole trip", MAPS_DIR / f"report_trip_{TRIP}_overview.html")

    # window of 60 points where classifier-only and HMM disagree most
    diff = (hmm_arcs != cls_arcs).astype(int)
    win = 60
    counts = np.convolve(diff, np.ones(win, dtype=int), mode="valid")
    start = int(np.argmax(counts))
    seg = ll[start:start + win]
    pad = 0.0015
    zoom = [(seg.min(axis=0) - pad).tolist(), (seg.max(axis=0) + pad).tolist()]
    print(f"zoom on points {start}-{start + win}: {counts[start]} of {win} picks changed by the HMM")
    build(trip, trip.route, hmm_arcs[hmm_arcs >= 0], cls_arcs[cls_arcs >= 0], zoom,
          f"Trip {TRIP} (student_test) - where the HMM corrects the classifier",
          MAPS_DIR / f"report_trip_{TRIP}_zoom.html")


if __name__ == "__main__":
    main()
