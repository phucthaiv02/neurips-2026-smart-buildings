"""Roll out a policy on scenarios, in parallel, with cached results.

A *policy spec* names a policy in `sbx.candidates.REGISTRY`, optionally with
class-attribute overrides:  "ref", "fan_tr", "ref:KP=0.12,KI=0.006".
Results go to <out>/<spec-tag>/<building>__<weather>.json and are reused when
the same spec, scenario and step count were already run.

The rollout loop mirrors the starter kit's evaluate.py (SimulationTracker,
occupancy constant 125, validate_action), so returns are comparable with
baseline_returns.json.
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sbx import paths

STEPS_PER_DAY = 288
FULL_EPISODE = 8640


def spec_tag(spec: str) -> str:
    """Filesystem-safe folder name for a policy spec."""
    safe = "".join(c if c.isalnum() or c in "._-=," else "_" for c in spec)
    if len(safe) <= 80:
        return safe
    return safe[:60] + "_" + hashlib.sha1(spec.encode()).hexdigest()[:10]


def scenario_key(scenario: str) -> str:
    p = Path(scenario)
    return f"{p.parent.name}__{p.name}"


def result_path(out_dir, spec, scenario) -> Path:
    return Path(out_dir) / spec_tag(spec) / (scenario_key(scenario) + ".json")


def load_env(scenario):
    """Load one scenario's environment exactly as evaluate.py does."""
    root = paths.setup()
    os.chdir(root)  # gin files use paths relative to the data root
    from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
    sdir = paths.scenario_dir(scenario)
    env = load_hybrid_environment(sdir / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = create_building_meta_data(env, sdir)
    return env, meta


def run_scenario(scenario: str, spec: str, n_steps: int = FULL_EPISODE, out_dir=None,
                 save_steps: bool = False) -> dict:
    """Run one policy spec on one scenario; return (and optionally save) the summary."""
    paths.setup()
    from smart_buildings.evaluate import validate_action
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    from sbx import candidates

    t_start = time.perf_counter()
    env, meta = load_env(scenario)
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    spec_obj = env.action_spec()
    policy = candidates.make(spec, meta, spec_obj)

    tracker.reset()
    obs = tracker.last_obs
    t_policy = t_env = 0.0
    for _ in range(n_steps):
        t0 = time.perf_counter()
        action = validate_action(policy.get_action(obs), spec_obj)
        t1 = time.perf_counter()
        obs, _, terminated, truncated, _ = tracker.step(action)
        t_policy += t1 - t0
        t_env += time.perf_counter() - t1
        if terminated or truncated:
            break

    summary, steps = _summarize(tracker, scenario, spec)
    summary.update(seconds_policy=t_policy, seconds_env=t_env,
                   seconds_total=time.perf_counter() - t_start)
    if out_dir is not None:
        path = result_path(out_dir, spec, scenario)
        path.parent.mkdir(parents=True, exist_ok=True)
        if save_steps:
            import pandas as pd
            pd.DataFrame(steps).to_csv(path.with_suffix(".steps.csv.gz"), index=False, float_format="%.6g")
        path.write_text(json.dumps(summary, indent=2))
    return summary


def _summarize(tracker, scenario, spec):
    rewards = np.asarray(tracker.rewards, dtype=float)
    rc = tracker.reward_components
    regret = np.array([r["normalized_productivity_regret"] for r in rc], dtype=float)
    energy = np.array([r["normalized_energy_cost"] for r in rc], dtype=float)
    carbon = np.array([r["normalized_carbon_emission"] for r in rc], dtype=float)
    occ = np.asarray(tracker.occupancy, dtype=float)
    wp, we, wc = tracker.productivity_weight, tracker.energy_weight, tracker.carbon_weight
    comfort_s, energy_s, carbon_s = wp * np.abs(regret), we * energy, wc * carbon
    run = np.array([d["ahu_run_cmds"] for d in tracker.ahu_log], dtype=int)
    outside = np.array([d["outside_air"] for d in tracker.ahu_log], dtype=float)
    temps = np.asarray(tracker.zone_temps, dtype=float)
    T = len(rewards)
    occupied = occ > 0
    p = Path(scenario)
    summary = {
        "scenario": f"{p.parent.name}/{p.name}",
        "building": p.parent.name,
        "weather": p.name,
        "station": p.name.rsplit("_", 1)[0],
        "spec": spec,
        "steps": T,
        "total_reward": float(rewards.sum()),
        "score": float(abs(rewards.mean())) if T else float("nan"),
        "comfort_loss": float(comfort_s.mean()) if T else 0.0,
        "energy_loss": float(energy_s.mean()) if T else 0.0,
        "carbon_loss": float(carbon_s.mean()) if T else 0.0,
        "comfort_violation_frac": float((regret[occupied] != 0).mean()) if occupied.any() else 0.0,
        "zone_temp_min_occ_C": float(temps[occupied].min() - 273.15) if occupied.any() else None,
        "zone_temp_max_occ_C": float(temps[occupied].max() - 273.15) if occupied.any() else None,
        "n_zones": int(temps.shape[1]) if temps.ndim == 2 else None,
        "n_ahus": int(run.shape[1]) if run.ndim == 2 else None,
        "ahu_switches_per_ahu_day": (float(np.abs(np.diff(run, axis=0)).sum(0).mean() / (T / STEPS_PER_DAY))
                                     if run.ndim == 2 and T > 1 else None),
        "ahu_on_frac": float(run.mean()) if run.size else None,
    }
    steps = {
        "reward": rewards, "comfort": comfort_s, "energy": energy_s, "carbon": carbon_s,
        "occupancy": occ, "outside_C": outside - 273.15,
        "ahus_on": run.sum(1) if run.ndim == 2 else np.zeros(T),
        "zone_temp_min_C": temps.min(1) - 273.15 if temps.ndim == 2 else np.zeros(T),
        "zone_temp_max_C": temps.max(1) - 273.15 if temps.ndim == 2 else np.zeros(T),
    }
    return summary, steps


def _cached(out_dir, spec, scenario, steps) -> bool:
    p = result_path(out_dir, spec, scenario)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("steps") == steps
    except json.JSONDecodeError:
        return False


def _size(scenario) -> int:
    try:
        return int(np.load(paths.scenario_dir(scenario) / "floor_plan.npy", mmap_mode="r").size)
    except OSError:
        return 0


def run_batch(jobs, out_dir, workers=None, save_steps=False, verbose=True):
    """jobs: iterable of (scenario, spec, steps). Cached jobs are skipped; bigger buildings start first."""
    paths.setup()
    jobs = list(jobs)
    todo = [j for j in jobs if not _cached(out_dir, j[1], j[0], j[2])]
    size = {s: _size(s) for s in {j[0] for j in todo}}
    todo.sort(key=lambda j: -size[j[0]])
    workers = workers or int(os.getenv("SB_WORKERS", 0)) or os.cpu_count() or 1
    if verbose:
        print(f"{len(jobs) - len(todo)} cached, {len(todo)} to run on {workers} workers", flush=True)
    if not todo:
        return
    kw = {"max_workers": min(workers, len(todo)), "mp_context": mp.get_context("spawn")}
    if sys.version_info >= (3, 11):
        kw["max_tasks_per_child"] = 1
    t0 = time.time()
    with ProcessPoolExecutor(**kw) as ex:
        futs = {ex.submit(run_scenario, sc, spec, steps, str(out_dir), save_steps): (sc, spec)
                for sc, spec, steps in todo}
        for i, f in enumerate(as_completed(futs), 1):
            sc, spec = futs[f]
            try:
                s = f.result()
                if verbose:
                    print(f"[{i:3d}/{len(todo)}] {time.time() - t0:6.0f}s  {spec[:28]:28s} {sc:32s} "
                          f"score {s['score']:.4f}  ({s['seconds_total'] / 60:.1f} min)", flush=True)
            except Exception as e:  # keep the batch going; the job is retried next run
                print(f"[{i:3d}/{len(todo)}] FAILED {spec} {sc}: {e!r}", flush=True)


def load_results(out_dir, specs=None, steps=None):
    """All cached summaries under out_dir (optionally filtered) as a DataFrame."""
    import pandas as pd
    rows = []
    for p in Path(out_dir).glob("*/*.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "spec" not in r:
            continue
        if specs is not None and r["spec"] not in specs:
            continue
        if steps is not None and r["steps"] != steps:
            continue
        rows.append(r)
    return pd.DataFrame(rows)


def compare(df, ref_spec="ref", noise=None):
    """Per spec: mean score and change vs ref on the scenarios both ran.

    noise: DataFrame indexed by scenario with 'std' (from a null group). When
    given, adds per-scenario z = (score - ref) / std and counts |z| >= 2.
    """
    import pandas as pd
    piv = df.pivot_table(index="scenario", columns="spec", values="score")
    if ref_spec not in piv:
        raise KeyError(f"{ref_spec!r} has no results")
    rows = []
    for spec in piv.columns:
        both = pd.DataFrame({"ref": piv[ref_spec], "cand": piv[spec]}).dropna()
        if both.empty:
            continue
        d = both["cand"] - both["ref"]
        row = {"spec": spec, "scenarios": len(both), "score": both["cand"].mean(),
               "ref_score": both["ref"].mean(),
               "change_pct": 100 * d.sum() / both["ref"].sum(),
               "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
               "worst_change_pct": float((100 * d / both["ref"]).max())}
        if noise is not None:
            sd = noise.reindex(both.index)["std"].replace(0, np.nan)
            z = (d / sd).dropna()
            row.update(z_better=int((z <= -2).sum()), z_worse=int((z >= 2).sum()))
        rows.append(row)
    return pd.DataFrame(rows).set_index("spec").sort_values("score")
