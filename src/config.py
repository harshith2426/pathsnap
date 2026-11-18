"""Central paths and constants used by every script."""
import os
from pathlib import Path

# Repo root = folder that contains src/
ROOT = Path(__file__).resolve().parent.parent

# Raw dataset folder (contains train/ and student_test/).
# Default: the dataset sits next to the repo. Override with env var PATHSNAP_DATA.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent / "Student folder" / "Student folder"))

# Derived data (labels, candidate tables) - reproducible, so not committed.
DERIVED_DIR = ROOT / "data_derived"

RESULTS_DIR = ROOT / "results"
MAPS_DIR = ROOT / "maps"
ARTIFACTS_DIR = ROOT / "artifacts"

# Partitions we are allowed to use during development.
TRAIN = "train"
STUDENT_TEST = "student_test"

RANDOM_SEED = 42
