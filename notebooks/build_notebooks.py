"""Regenerate notebooks 01 and 02 from the cell sources below.

    python notebooks/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = '''import os, sys, json, time
from pathlib import Path

# Find the repo (the folder holding sbx/) and the starter kit.
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "sbx").is_dir())
sys.path.insert(0, str(REPO))
# os.environ["SMART_BUILDINGS_DATA_ROOT"] = "/path/to/starter-kit"   # if not next to the repo
from sbx import paths
DATA_ROOT = paths.setup()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sbx import evaluation as ev, candidates as cand

WORKERS = int(os.getenv("SB_WORKERS", 0)) or os.cpu_count()   # set to your physical core count
print("repo:", REPO, "\\ndata:", DATA_ROOT, "\\nworkers:", WORKERS)'''

SCREEN_SET = '''def default_screen_set(phase0_csv=None, n=10):
    """Top contributors from the phase-0 run if available, else two scenarios per station."""
    for p in [phase0_csv, REPO / "notebooks" / "eval_ref" / "scores_ref.csv", REPO / "eval_ref" / "scores_ref.csv"]:
        if p and Path(p).exists():
            df = pd.read_csv(p).sort_values("score", ascending=False)
            print("screen set: top", n, "contributors from", p)
            return df.scenario.head(n).tolist()
    allsc = paths.training_scenarios()
    by_station = {}
    for s in allsc:
        by_station.setdefault(s.split("/")[1].rsplit("_", 1)[0], []).append(s)
    pick = [s for v in by_station.values() for s in v[::3][:2]]
    print("screen set: 2 per station (no phase-0 results found)")
    return pick'''

nb01 = [
md("""# Sàng lọc policy ứng viên

So từng ứng viên trong `sbx/candidates.py` với policy tham chiếu (public LB 0,01582), có nhóm null để biết đâu là nhiễu.

| Bước | Kịch bản | Độ dài | Mục đích |
|---|---|---|---|
| 1 · Sàng nhanh | 10 kịch bản đóng góp nhiều nhất | 7 ngày | Loại nhanh ứng viên kém |
| 2 · Xác nhận | 30 kịch bản | 30 ngày | Con số thật cho 3 ứng viên tốt nhất |

Cần: repo này + starter kit (Python 3.12). Kết quả lưu ở `results/screen/`, chạy lại sẽ bỏ qua phần đã xong."""),
code(SETUP),
code('''QUICK_STEPS = 2016          # 7 days. Short runs skip some weekends; use only to rank.
FULL_STEPS = 8640
CONFIRM_TOP = 3
OUT = REPO / "results" / "screen"
NULLS = ["null_1", "null_2", "null_3"]
CANDS = cand.CANDIDATES      # or a subset, e.g. ["fan_tr", "optimal_start", "night_flush"]
print("candidates:", CANDS)'''),
code(SCREEN_SET + '''

SCREEN = default_screen_set()
SCREEN'''),
md("## Bước 1 · Sàng nhanh"),
code('''specs = ["ref"] + NULLS + CANDS
n_jobs = len(specs) * len(SCREEN)
print(f"{n_jobs} runs of {QUICK_STEPS} steps")
ev.run_batch([(s, sp, QUICK_STEPS) for sp in specs for s in SCREEN], OUT, workers=WORKERS)'''),
code('''quick = ev.load_results(OUT, steps=QUICK_STEPS)
quick = quick[quick.scenario.isin(SCREEN)]

# Noise per scenario from ref + null runs.
grp = quick[quick.spec.isin(["ref"] + NULLS)]
noise = grp.groupby("scenario").score.agg(["mean", "std"])
print(f"noise: median sd {100 * (noise['std'] / noise['mean']).median():.1f}% of the score")

table = ev.compare(quick[~quick.spec.isin(NULLS)], "ref", noise=noise)
display(table.round(5))'''),
code('''piv = quick.pivot_table(index="spec", columns="scenario", values="score")
chg = 100 * (piv - piv.loc["ref"]) / piv.loc["ref"]
chg = chg.drop(index=["ref"] + [n for n in NULLS if n in chg.index]).loc[table.index.drop("ref", errors="ignore")]
fig, ax = plt.subplots(figsize=(1.0 + 0.9 * chg.shape[1], 0.45 * chg.shape[0] + 1.5))
lim = max(5, np.nanpercentile(np.abs(chg.to_numpy()), 95))
im = ax.imshow(chg.to_numpy(), cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(chg.shape[1]), [c.replace("building_train_", "B").replace("/", "\\n") for c in chg.columns], fontsize=8)
ax.set_yticks(range(chg.shape[0]), chg.index, fontsize=9)
for i in range(chg.shape[0]):
    for j in range(chg.shape[1]):
        v = chg.iat[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="% thay đổi so với ref (xanh = tốt hơn)")
ax.set_title(f"Sàng nhanh: {QUICK_STEPS} bước")
fig.tight_layout(); plt.show()'''),
md("## Bước 2 · Xác nhận trên 30 kịch bản đầy đủ"),
code('''ok = table.drop(index="ref", errors="ignore")
ok = ok[ok.change_pct < 0]
if "z_worse" in ok:
    ok = ok[ok.z_worse == 0]           # drop anything that clearly hurts a scenario
best = ok.sort_values("score").index[:CONFIRM_TOP].tolist()
print("confirming:", best)
ALL = paths.training_scenarios()
ev.run_batch([(s, sp, FULL_STEPS) for sp in ["ref"] + best for s in ALL], OUT, workers=WORKERS)'''),
code('''full = ev.load_results(OUT, steps=FULL_STEPS)
full = full[full.spec.isin(["ref"] + best)]
final = ev.compare(full, "ref")
display(final.round(5))
by_station = full.pivot_table(index="station", columns="spec", values="score")
display(by_station.round(4))
final.to_csv(OUT / "confirm_summary.csv")
table.to_csv(OUT / "quick_summary.csv")'''),
md("""**Đọc kết quả:** một ứng viên đáng giữ khi `change_pct` âm trên 30 kịch bản, không làm tệ hơn rõ rệt kịch bản nào (`worst_change_pct` nhỏ),
và cải thiện lớn hơn mức nhiễu. Ứng viên tốt có thể kết hợp với nhau (vd. `combo`) rồi sàng lại."""),
]

nb02 = [
md("""# Tìm tham số controller (giai đoạn 1)

Tối ưu ~10 tham số của policy tham chiếu bằng CMA-ES (gói `cma`, nếu thiếu thì random search).

* **Chung**: một bộ cho mọi tòa nhà → nộp được ngay.
* **Riêng** (`MODE = "per_scenario"`): bộ tốt nhất cho từng kịch bản → "thầy" cho giai đoạn 2. Đắt hơn nhiều.

Đánh giá nhanh bằng episode ngắn trên một nhóm kịch bản, rồi kiểm tra bộ tốt nhất trên các kịch bản **không dùng để tune**.
Mọi lần chạy được cache trong `results/search/`, ngắt giữa chừng chạy lại sẽ tiếp tục."""),
code(SETUP),
code('''# name: (low, high, kind)   kind: lin | log | int
SPACE = {
    "KP":                  (0.04, 0.20, "log"),
    "KI":                  (0.0015, 0.012, "log"),
    "PRESTART_H":          (1, 4, "int"),
    "HOLD_STEPS":          (48, 576, "log"),
    "NIGHT_HEAT_DROP_K":   (0.0, 3.0, "lin"),
    "MILD_SHIFT_K":        (0.0, 1.0, "lin"),
    "MILD_BELOW_C":        (5.0, 16.0, "lin"),
    "EXTREME_BELOW_C":     (-25.0, -5.0, "lin"),
    "EXTREME_PRESTART_H":  (2, 6, "int"),
    "COLD_SAT":            (-1.0, -0.3, "lin"),
    "MIN_DAMPER":          (0.1, 0.3, "lin"),
}
REF_VALUES = {k: getattr(cand.REGISTRY["ref"][0], k) for k in SPACE}
print(REF_VALUES)

def decode(x):
    out = {}
    for (k, (lo, hi, kind)), v in zip(SPACE.items(), np.clip(x, 0, 1)):
        if kind == "log":
            val = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))))
        else:
            val = lo + v * (hi - lo)
        out[k] = int(round(val)) if kind == "int" or k == "HOLD_STEPS" else round(val, 5)
    return out

def encode(values):
    x = []
    for k, (lo, hi, kind) in SPACE.items():
        v = values[k]
        x.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if kind == "log" else (v - lo) / (hi - lo))
    return np.clip(np.array(x, dtype=float), 0, 1)

def spec_of(values):
    return "ref:" + ",".join(f"{k}={v}" for k, v in values.items())

print(spec_of(decode(encode(REF_VALUES))))'''),
code(SCREEN_SET + '''

MODE = "shared"               # "shared" | "per_scenario"
SEARCH_STEPS = 2016           # 7 days per evaluation
GENERATIONS = 10
POPSIZE = 6
OUT = REPO / "results" / "search"

TUNE = default_screen_set(n=6)
HOLDOUT = [s for s in paths.training_scenarios() if s not in TUNE]
print("tune on:", TUNE)
cost = GENERATIONS * POPSIZE * len(TUNE)
print(f"budget: {cost} runs of {SEARCH_STEPS} steps")'''),
code('''def evaluate(values_list, scenarios, steps=SEARCH_STEPS):
    """Mean score of each parameter set over the scenarios (runs in parallel, cached)."""
    specs = [spec_of(v) for v in values_list]
    ev.run_batch([(s, sp, steps) for sp in specs for s in scenarios], OUT, workers=WORKERS, verbose=False)
    df = ev.load_results(OUT, specs=set(specs), steps=steps)
    df = df[df.scenario.isin(scenarios)]
    means = df.groupby("spec").score.mean()
    return [float(means.get(sp, np.inf)) for sp in specs]


def search(scenarios, generations=GENERATIONS, popsize=POPSIZE, seed=0):
    """CMA-ES from the reference parameters; returns the log of every evaluation."""
    log = []
    x0 = encode(REF_VALUES)
    ref_score = evaluate([REF_VALUES], scenarios)[0]
    log.append({"gen": -1, **REF_VALUES, "score": ref_score})
    print(f"ref: {ref_score:.5f}")
    try:
        import cma
        es = cma.CMAEvolutionStrategy(x0, 0.2, {"popsize": popsize, "bounds": [0, 1], "seed": seed + 1, "verbose": -9})
        for g in range(generations):
            xs = es.ask()
            vals = [decode(x) for x in xs]
            scores = evaluate(vals, scenarios)
            es.tell(xs, scores)
            log += [{"gen": g, **v, "score": s} for v, s in zip(vals, scores)]
            print(f"gen {g:2d}: best {min(scores):.5f}  mean {np.mean(scores):.5f}  (ref {ref_score:.5f})", flush=True)
    except ImportError:
        print("cma not installed (pip install cma): random search around ref")
        rng = np.random.default_rng(seed)
        for g in range(generations):
            xs = [np.clip(x0 + rng.normal(0, 0.2, len(x0)), 0, 1) for _ in range(popsize)]
            vals = [decode(x) for x in xs]
            scores = evaluate(vals, scenarios)
            log += [{"gen": g, **v, "score": s} for v, s in zip(vals, scores)]
            print(f"gen {g:2d}: best {min(scores):.5f}  (ref {ref_score:.5f})", flush=True)
    return pd.DataFrame(log)'''),
md("## Tìm"),
code('''if MODE == "shared":
    logs = {"shared": search(TUNE)}
else:
    logs = {s: search([s]) for s in TUNE}
for key, log in logs.items():
    log.to_csv(OUT / f"log_{key.replace('/', '__')}.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
for key, log in logs.items():
    g = log[log.gen >= 0].groupby("gen").score.min().cummin()
    ax.plot(g.index, g.values, marker="o", label=key)
    ax.axhline(log.loc[log.gen == -1, "score"].iloc[0], ls="--", lw=1, color=ax.lines[-1].get_color())
ax.set_xlabel("thế hệ"); ax.set_ylabel("điểm tốt nhất (thấp = tốt)"); ax.set_title("Hội tụ (nét đứt = ref)")
ax.legend(fontsize=8); plt.show()'''),
md("## Kiểm tra trên kịch bản chưa dùng để tune\n\nTune trên 7 ngày dễ khớp quá mức; chỉ tin bộ tham số nếu nó thắng ref trên các kịch bản giữ lại, chạy đủ 30 ngày."),
code('''CHECK_TOP = 3
FULL_STEPS = 8640
log = logs["shared"] if MODE == "shared" else pd.concat(logs.values())
top = (log[log.gen >= 0].sort_values("score").drop_duplicates(subset=list(SPACE)).head(CHECK_TOP))
top_vals = [{k: (int(r[k]) if SPACE[k][2] == "int" or k == "HOLD_STEPS" else r[k]) for k in SPACE} for _, r in top.iterrows()]
specs = ["ref"] + [spec_of(v) for v in top_vals]
ev.run_batch([(s, sp, FULL_STEPS) for sp in specs for s in HOLDOUT], OUT, workers=WORKERS)
full = ev.load_results(OUT, specs=set(specs), steps=FULL_STEPS)
full = full[full.scenario.isin(HOLDOUT)]
res = ev.compare(full, "ref")
display(res.round(5))
res.to_csv(OUT / "holdout_check.csv")
best_spec = res.index[0]
print("best on hold-out:", best_spec)'''),
md("""**Đọc kết quả:** nếu bộ tốt nhất không thắng `ref` trên kịch bản giữ lại, tham số không phải nút thắt → chuyển sang sửa logic (notebook 01) hoặc RL.
Với `MODE = "per_scenario"`, khoảng cách giữa điểm tune riêng và tune chung chính là "cổng" của giai đoạn 1 trong kế hoạch."""),
]

for name, cells in [("01_candidates_screen.ipynb", nb01), ("02_param_search.ipynb", nb02)]:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, HERE / name)
    print("wrote", HERE / name)
