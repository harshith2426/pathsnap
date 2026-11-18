"""PathSnap FastAPI backend (runs on laptop 1).

Start (from the repo root):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_EVAL_DIR  folder holding the evaluation trips, one sub-folder per
                       trip (<id>/<id>.track, .nodes, .arcs, .route).
                       Default: the student_test partition (for development).

Endpoints:
    GET  /health           is the server up, which model is loaded
    GET  /trips            ids of the trips available for evaluation
    POST /predict          {"trip_id": "..."} -> predicted route + accuracy

The reference route (.route) is read ONLY inside `score_prediction`, after the
prediction is finished. It is never given to the model and never sent to the UI.
"""
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # make `src` importable
from src.config import ARTIFACTS_DIR, DATA_DIR, STUDENT_TEST        # noqa: E402
from src.data_io import load_trip_from_dir                           # noqa: E402
from src.pipeline import PathSnapModel, score_prediction             # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pathsnap")

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", DATA_DIR / STUDENT_TEST))
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ARTIFACTS_DIR))
MAX_GPS_POINTS_RETURNED = 2000          # keep responses small; the map only needs a preview

app = FastAPI(title="PathSnap API", version="1.0")
# The UI runs on another laptop (different origin), so the browser needs CORS.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = PathSnapModel(ARTIFACTS)
log.info("Model loaded: %s | HMM %s", model.clf_config["model"],
         {k: model.hmm_config[k] for k in ("beta", "emission_weight", "top_k")})
log.info("Evaluation trips folder: %s", EVAL_DIR)


def available_trips():
    if not EVAL_DIR.is_dir():
        return []
    return sorted(p.name for p in EVAL_DIR.iterdir()
                  if p.is_dir() and (p / f"{p.name}.track").exists())


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": model.clf_config["model"],
            "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    ids = available_trips()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in available_trips():          # also blocks path tricks like "../"
        raise HTTPException(status_code=404, detail=f"Unknown trip id {req.trip_id!r}")
    t0 = time.time()
    log.info("POST /predict trip=%s : running pipeline", req.trip_id)

    # 1. Prediction - the route file is NOT read here
    trip = load_trip_from_dir(EVAL_DIR, req.trip_id, with_route=False)
    result = model.predict(trip)
    arcs = result["arc_per_point"]

    # 2. Scoring against the server-side reference route
    reference = load_trip_from_dir(EVAL_DIR, req.trip_id, with_route=True)
    metrics = score_prediction(reference, arcs)

    # 3. Geometry for the map (lat, lon order for Leaflet)
    step = max(1, len(trip.times) // MAX_GPS_POINTS_RETURNED)
    gps = trip.gps_lonlat[::step, ::-1].round(6).tolist()
    ordered_arcs = list(dict.fromkeys(int(a) for a in arcs if a >= 0))   # unique, in driving order
    ll = trip.nodes_lonlat
    route_lines = [[[ll[u, 1], ll[u, 0]], [ll[v, 1], ll[v, 0]]] for u, v in trip.arcs[ordered_arcs]]

    elapsed = time.time() - t0
    log.info("POST /predict trip=%s : %d points, accuracy=%.4f, continuity=%.4f, %.2fs",
             req.trip_id, len(arcs), metrics["point_accuracy"] or float("nan"),
             metrics["route_continuity"], elapsed)
    return {
        "trip_id": req.trip_id,
        "n_gps_points": int(len(arcs)),
        "n_points_matched": int((arcs >= 0).sum()),
        "n_route_segments": len(ordered_arcs),
        "n_hmm_chains": result["n_segments"],
        "metrics": metrics,
        "timing_s": {**{k: round(v, 3) for k, v in result["timing_s"].items()},
                     "request_total": round(elapsed, 3)},
        "gps_points": gps,
        "predicted_route": route_lines,
    }
