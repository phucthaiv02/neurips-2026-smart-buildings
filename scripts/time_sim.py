"""Measure steady-state simulator step time.

Run from the starter-kit root (config.gin files use relative paths):
    cd starter-kit && python /path/to/scripts/time_sim.py building_train_1/atlanta_2 building_train_2/atlanta_3
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from smart_buildings import evaluate as ev  # noqa: E402

STEPS, WARMUP = 300, 20


def main(scenarios):
    for sc in scenarios or ["building_train_1/atlanta_2"]:
        cfg = Path("smart_buildings/training_scenarios") / sc / "config.gin"
        t0 = time.time()
        env = ev.load_hybrid_environment(cfg)
        env.reset()
        load = time.time() - t0
        spec = env.action_spec()
        action = {
            "continuous_action": np.zeros(spec["continuous_action"].shape, np.float32),
            "discrete_action": np.ones(spec["discrete_action"].shape, np.int32),
        }
        times = []
        for _ in range(STEPS):
            t = time.time()
            env.step(action)
            times.append(time.time() - t)
        med = float(np.median(times[WARMUP:]))
        print(f"{sc}: load {load:.1f}s, median {med:.4f} s/step, "
              f"1 episode (8640) ~ {med * 8640 / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
