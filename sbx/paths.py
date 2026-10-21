"""Locate the starter kit (or Kaggle competition data) and make it importable.

Search order: $SMART_BUILDINGS_DATA_ROOT, Kaggle input folders, then a
`starter-kit` folder next to or inside this repository.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_CANDIDATES = [
    os.getenv("SMART_BUILDINGS_DATA_ROOT", ""),
    "/kaggle/input/competitions/neurips-2026-smart-buildings",
    "/kaggle/input/neurips-2026-smart-buildings",
    str(REPO_ROOT.parent / "starter-kit"),
    str(REPO_ROOT / "starter-kit"),
]


def find_data_root() -> Path:
    for p in _CANDIDATES:
        if p and (Path(p) / "smart_buildings").exists():
            return Path(p).resolve()
    raise FileNotFoundError(
        "Starter kit not found. Clone https://github.com/smart-buildings-2026/starter-kit "
        "next to this repo, or set SMART_BUILDINGS_DATA_ROOT.")


def setup(single_thread: bool = True) -> Path:
    """Put the data root and this repo on sys.path; export them for child processes."""
    if single_thread:
        for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(k, "1")
    root = find_data_root()
    os.environ["SMART_BUILDINGS_DATA_ROOT"] = str(root)
    for p in (str(REPO_ROOT), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def scenario_dir(scenario) -> Path:
    """'building_train_1/atlanta_2' -> training scenario folder; an existing path is returned as is."""
    p = Path(scenario)
    if p.is_absolute() and (p / "config.gin").exists():
        return p
    return find_data_root() / "smart_buildings" / "training_scenarios" / scenario


def training_scenarios() -> list[str]:
    root = find_data_root() / "smart_buildings" / "training_scenarios"
    return sorted(f"{d.parent.name}/{d.name}" for d in root.glob("*/*") if (d / "config.gin").exists())
