# PathSnap - Map-matching and Route Reconstruction

Matches noisy GPS trajectories to a road network: a classifier scores
candidate road segments for every GPS point and an HMM/Viterbi decoder turns
those scores into a connected route. Served by a FastAPI backend (laptop 1)
and a browser UI (laptop 2).

## Setup
```bash
pip install -r requirements.txt
```
The raw dataset is **not** in the repo. By default the code expects it next to
the repo folder at `../Student folder/Student folder/` (containing `train/` and
`student_test/`). To use another location, set `PATHSNAP_DATA`:
```bash
set PATHSNAP_DATA=D:\path\to\data          # Windows cmd
export PATHSNAP_DATA=/path/to/data          # bash
```

## Pipeline (run from the repo root)
| Step | Command | Output |
|---|---|---|
| 1. Per-point labels | `python -m src.labels` | `data_derived/labels_*.csv` |
| 2. EDA tables + plots | `python -m src.eda` | `results/eda/` |
| 3. Folium maps | `python -m src.maps` | `maps/` |

## Repository layout
```
src/        data loading, geometry, labels, EDA, maps (more stages to come)
api/        FastAPI backend
ui/         evaluator interface
artifacts/  trained model and configuration
results/    CSV results and plots
maps/       interactive Folium maps
VIVA_NOTES.md   stage-by-stage explanation of the work
```
