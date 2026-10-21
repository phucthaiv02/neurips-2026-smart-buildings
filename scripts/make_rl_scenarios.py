"""Generate extra scenarios for RL training: shipped buildings x unused weather strips.

The 30 training scenarios use one strip per station per building. The other
strips (8 per station, plus minneapolis) give new 30-day weather windows for
the same buildings. New buildings can be added with the starter kit's
run_floor_generator (see its README).

    python scripts/make_rl_scenarios.py --per-building 6          # 36 new scenarios
    python scripts/train_rl.py ppo --extra-train $(cat runs/extra_scenarios.txt)
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbx import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-building", type=int, default=6, help="new weather strips per building")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-out", default="runs/extra_scenarios.txt")
    args = ap.parse_args()
    root = paths.setup()
    used = set(paths.training_scenarios())
    buildings = sorted({s.split("/")[0] for s in used})
    strips = sorted(p.stem for p in (root / "smart_buildings" / "weather").glob("*.yaml"))
    rng = random.Random(args.seed)
    made = []
    for b in buildings:
        free = [w for w in strips if f"{b}/{w}" not in used]
        pick = rng.sample(free, min(args.per_building, len(free)))
        subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator",
                        "--buildings", b, "--weathers", *pick], cwd=root, check=True)
        made += [f"{b}/{w}" for w in pick]
    out = Path(args.list_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(" ".join(made) + "\n")
    print(f"{len(made)} scenarios -> {out}")


if __name__ == "__main__":
    main()
