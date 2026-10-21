"""Locate the starter kit (or Kaggle competition data) and make it importable.

Search order: $SMART_BUILDINGS_DATA_ROOT, Kaggle input folders, then a
`starter-kit` folder in, next to, or above the current/code directory.
"""
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent


def _candidates():
    return [
        os.getenv("SMART_BUILDINGS_DATA_ROOT", ""),
        "/kaggle/input/competitions/neurips-2026-smart-buildings",
        "/kaggle/input/neurips-2026-smart-buildings",
        str(Path.cwd() / "starter-kit"),
        str(Path.cwd().parent / "starter-kit"),
        str(Path.cwd().parent.parent / "starter-kit"),
        str(REPO_ROOT / "starter-kit"),
        str(REPO_ROOT.parent / "starter-kit"),
    ]


def find_data_root() -> Path:
    for p in _candidates():
        if p and (Path(p) / "smart_buildings").exists():
            return Path(p).resolve()
    raise FileNotFoundError(
        "Starter kit not found. Attach the competition data on Kaggle, clone "
        "https://github.com/smart-buildings-2026/starter-kit next to this folder, "
        "or set SMART_BUILDINGS_DATA_ROOT.")


def setup(single_thread: bool = True) -> Path:
    """Make the data root and this code importable here and in child processes."""
    if single_thread:
        for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(k, "1")
    root = find_data_root()
    os.environ["SMART_BUILDINGS_DATA_ROOT"] = str(root)
    # Absolute paths: workers chdir into the data root, so '' would stop working.
    for p in (str(REPO_ROOT), str(CODE_DIR), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def scenario_dir(scenario) -> Path:
    """'building_train_1/atlanta_2' -> training scenario folder; an existing path is returned as is."""
    p = Path(scenario)
    if p.is_absolute() and (p / "config.gin").exists():
        return p
    return find_data_root() / "smart_buildings" / "training_scenarios" / scenario


def training_scenarios() -> list:
    root = find_data_root() / "smart_buildings" / "training_scenarios"
    return sorted(f"{d.parent.name}/{d.name}" for d in root.glob("*/*") if (d / "config.gin").exists())
