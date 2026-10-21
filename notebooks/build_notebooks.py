"""Generate the self-contained notebooks from the code in sbx/.

    python notebooks/build_notebooks.py

Every notebook writes its own copy of the code (sb_paths.py, sb_policy.py,
sb_eval.py) with %%writefile, so it runs on its own: locally next to a
starter-kit checkout, or on Kaggle with the competition data attached.
sbx/ stays the single source; edit there and rebuild.
"""
import ast
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SBX = HERE.parent / "sbx"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

DESCRIPTIONS = {
    "fan_tr": "Áp suất quạt của mỗi AHU theo damper mở nhiều nhất trong các zone của nó (trim & respond, kiểu ASHRAE G36), thay vì giữ ở mức tối thiểu.",
    "optimal_start": "Giờ bắt đầu chạy buổi sáng tính từ tốc độ nóng/nguội học được trên chính tòa nhà, thay vì cố định 2 h trước giờ làm.",
    "night_flush": "Trước giờ làm, nếu ngoài trời mát hơn trong nhà ≥ 3 K, chạy AHU 100% gió ngoài, chỉ quạt, không chiller, để làm mát sẵn.",
    "evening_coast": "Tắt vòng PI 1 h trước khi hết giờ làm, để khối nhiệt của tòa nhà tự giữ nhiệt độ.",
    "occ_gate": "Bỏ chạy giờ PI vào ngày tòa nhà không có người (ngày lễ).",
    "heat_seq": "Khi sưởi: mở van reheat trước ở lưu lượng gió tối thiểu, chỉ thêm gió khi van đã mở hết.",
    "sat_ramp": "Nhiệt độ gió cấp giảm dần theo nhu cầu làm lạnh, thay vì bật/tắt.",
    "pi_leak": "Thêm rò rỉ nhỏ cho khâu tích phân của PI theo zone.",
    "water_max": "Nước nóng ở nhiệt độ tối đa bất cứ khi nào có zone cần sưởi trong giờ PI.",
    "combo": "Kết hợp optimal_start + fan_tr + night_flush.",
}
NEEDS = {  # extra classes each candidate needs from sbx/candidates.py
    "fan_tr": ["FanTrimRespondPolicy"],
    "optimal_start": ["OptimalStartPolicy"],
    "night_flush": ["NightFlushPolicy"],
    "combo": ["FanTrimRespondPolicy", "OptimalStartPolicy", "NightFlushPolicy", "ComboPolicy"],
}


# --------------------------------------------------------------------------- code generation
def _registry():
    src = (SBX / "candidates.py").read_text()
    tree = ast.parse(src)
    reg = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "REGISTRY")
    out = {}
    for k, v in zip(reg.value.keys, reg.value.values):
        out[ast.literal_eval(k)] = (v.elts[0].id, ast.literal_eval(v.elts[1]))
    return src, tree, out


def policy_module(names, candidate=None) -> str:
    """sb_policy.py: reference code + the candidate classes in `names` + REGISTRY/make.

    candidate: if given, also define `CandidatePolicy` for it (used for Kaggle submission).
    """
    ref = (SBX / "ref_policy.py").read_text()
    ref = ref[:ref.index("class NudgedRefPolicy")].rstrip() + "\n"
    src, tree, registry = _registry()
    needed = {"_TimeMixin", "NudgedPolicy"}
    for n in names:
        needed |= set(NEEDS.get(n, []))
    classes = [ast.get_source_segment(src, n) for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name in needed]
    funcs = [ast.get_source_segment(src, n) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in ("_baseline", "parse_spec", "make")]
    reg = {k: v for k, v in registry.items() if k == "ref" or k in names}
    lines = ["REGISTRY = {"] + [f"    {k!r}: ({c}, {o!r})," for k, (c, o) in reg.items()]
    if candidate:
        lines.append("    'cand': (CandidatePolicy, {}),")
    lines.append("}")
    cand_cls = ""
    if candidate:
        base, over = registry[candidate]
        body = "\n".join(f"    {k} = {v!r}" for k, v in over.items()) or "    pass"
        cand_cls = (f'class CandidatePolicy({base}):\n    """Policy version `{candidate}`: the one this notebook '
                    f'scores and submits."""\n{body}\n')
    doc = ('"""Policies: the reference controller (public LB 0.01582, copied verbatim from the Kaggle\n'
           'notebook "Smart Buildings Probe DS K07", Apache-2.0) plus candidate versions.\n'
           'Generated from sbx/ref_policy.py and sbx/candidates.py."""\n')
    ref_body = ref.split('"""', 2)[2].lstrip("\n")
    parts = [doc, "import ast\n", ref_body, "\n# --- candidate versions " + "-" * 50 + "\n"]
    parts += [c + "\n\n" for c in classes]
    parts += [cand_cls + "\n" if cand_cls else ""]
    parts += [f + "\n\n" for f in funcs]
    parts += ["\n".join(lines) + "\n", f"CANDIDATES = {[k for k in reg if k != 'ref']!r}\n"]
    out = "".join(parts)
    compile(out, "sb_policy.py", "exec")
    return out


def eval_module() -> str:
    src = (SBX / "evaluation.py").read_text()
    src = src.replace("from sbx import paths\n", "import sb_paths as paths\n")
    src = src.replace("    from sbx import candidates\n", "    import sb_policy as candidates\n")
    assert "sbx" not in src.split('"""', 2)[2], "unconverted sbx import in evaluation.py"
    compile(src, "sb_eval.py", "exec")
    return src


def paths_module() -> str:
    return (SBX / "paths.py").read_text()


def code_cells(policy_src):
    return [
        md("## Code\n\nBa file được ghi ra cạnh notebook, để các tiến trình chạy song song import được."),
        code("%%writefile sb_paths.py\n" + paths_module()),
        code("%%writefile sb_policy.py\n" + policy_src),
        code("%%writefile sb_eval.py\n" + eval_module()),
        code('''import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sb_paths, sb_policy, sb_eval as ev
for m in (sb_paths, sb_policy, ev):
    importlib.reload(m)
DATA_ROOT = sb_paths.setup()
SCENARIOS = [s for s in sb_paths.training_scenarios() if not ONLY or s in ONLY]
print(f"data: {DATA_ROOT}\\nscenarios: {len(SCENARIOS)}\\nworkers: {WORKERS}")'''),
    ]


CONFIG_HEAD = '''import os, sys, json, time
from pathlib import Path

for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"   # one thread per simulator; parallelism = scenarios side by side
# os.environ["SMART_BUILDINGS_DATA_ROOT"] = "/path/to/starter-kit"   # if not found automatically

IS_RERUN = os.getenv("KAGGLE_IS_COMPETITION_RERUN") is not None
ON_KAGGLE = Path("/kaggle/working").exists()
WORKERS = int(os.getenv("SB_WORKERS", 0)) or os.cpu_count()   # best: number of physical cores
'''

# --------------------------------------------------------------------------- per-version notebooks
def version_notebook(name):
    desc = DESCRIPTIONS[name]
    cells = [
        md(f"""# Policy `{name}`

{desc}

Xây trên policy tham chiếu (public LB 0,01582). Notebook tự chứa toàn bộ code.

| Chạy ở | Làm gì | Ra |
|---|---|---|
| Máy local | Chạy 30 kịch bản train, 8.640 bước mỗi kịch bản | Điểm train, so với `pid_water` và bản tham chiếu |
| Kaggle (Save & Run All) | Như trên + chạy inference server | Như trên, và **Submit** được để lấy điểm LB |

Khi Kaggle chạy lại để chấm (submit), phần đánh giá train tự bỏ qua, chỉ còn server."""),
        code(CONFIG_HEAD + f'''
POLICY = "{name}"
N_STEPS = 8640            # full 30-day episode; shorter runs are not comparable
ONLY = []                 # e.g. ["building_train_8/phoenix_4"] for a quick check; [] = all 30
RUN_EVAL = not IS_RERUN   # score on the training scenarios
RUN_REF = False           # also run the reference on the same scenarios (doubles the time)
REF_SCORES_CSV = ""       # or: scores_ref.csv from notebook 00, to compare without re-running
RUN_SERVER = ON_KAGGLE    # Kaggle: serve this policy so the version can be submitted
OUT = Path.cwd().resolve() / "results" / POLICY
OUT.mkdir(parents=True, exist_ok=True)'''),
        *code_cells(policy_module([name], candidate=name)),
        md("## Chạy 30 kịch bản"),
        code('''if RUN_EVAL:
    specs = ["cand"] + (["ref"] if RUN_REF else [])
    t0 = time.time()
    ev.run_batch([(s, sp, N_STEPS) for sp in specs for s in SCENARIOS], OUT, workers=WORKERS, save_steps=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")'''),
        md("## Điểm"),
        code('''if RUN_EVAL:
    df = ev.load_results(OUT, steps=N_STEPS)
    df = df[df.scenario.isin(SCENARIOS)]
    cand = df[df.spec == "cand"].set_index("scenario").sort_values("score", ascending=False)
    table = cand[["score", "comfort_loss", "energy_loss", "carbon_loss"]].copy()

    base_file = DATA_ROOT / "baseline_returns.json"
    if base_file.exists() and N_STEPS == 8640:
        base = json.loads(base_file.read_text())
        table["pid_water"] = [abs(base[s]["pid_water"]) / 8640 if s in base else np.nan for s in table.index]
    ref = None
    if RUN_REF:
        ref = df[df.spec == "ref"].set_index("scenario").score
    elif REF_SCORES_CSV and Path(REF_SCORES_CSV).exists():
        ref = pd.read_csv(REF_SCORES_CSV).set_index("scenario").score
    if ref is not None:
        table["ref"] = ref.reindex(table.index)
        table["vs_ref_pct"] = 100 * (table.score - table.ref) / table.ref

    print(f"POLICY {POLICY}: score {table.score.mean():.5f} on {len(table)} scenarios")
    print(f"  comfort {table.comfort_loss.mean():.5f}  energy {table.energy_loss.mean():.5f}  carbon {table.carbon_loss.mean():.5f}")
    if "pid_water" in table:
        print(f"  pid_water on the same scenarios: {table.pid_water.mean():.5f}")
    if "ref" in table and table.ref.notna().all():
        print(f"  reference on the same scenarios: {table.ref.mean():.5f}  ({100 * (table.score.sum() / table.ref.sum() - 1):+.1f}%)")
    display(table.round(4))
    display(cand.groupby("station")[["score", "comfort_loss", "energy_loss", "carbon_loss"]].mean().round(4))'''),
        code('''if RUN_EVAL:
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(table) + 1.2))
    y = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for col, c in (("comfort_loss", "#2f6fb3"), ("energy_loss", "#d9551c"), ("carbon_loss", "#677689")):
        ax.barh(y, table[col], left=left, color=c, label=col.replace("_loss", ""))
        left += table[col].to_numpy()
    if "ref" in table:
        ax.scatter(table.ref, y, color="black", marker="|", s=120, label="reference", zorder=3)
    ax.set_yticks(y, table.index, fontsize=8)
    ax.set_xlabel("điểm (thấp hơn = tốt hơn)")
    ax.set_title(f"{POLICY}: {table.score.mean():.5f}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "scores.png", dpi=120); plt.show()

    table.to_csv(OUT / f"scores_{POLICY}.csv")
    summary = {"policy": POLICY, "steps": N_STEPS, "scenarios": len(table), "score": float(table.score.mean()),
               "comfort_loss": float(table.comfort_loss.mean()), "energy_loss": float(table.energy_loss.mean()),
               "carbon_loss": float(table.carbon_loss.mean())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))'''),
        md("## Nộp bài (Kaggle)\n\nChạy policy này qua inference server chính thức. Trên Kaggle, sau khi *Save Version* xong, bấm **Submit** để lấy điểm LB."),
        code('''if RUN_SERVER:
    for p in (str(DATA_ROOT), str(DATA_ROOT / "kaggle_evaluation")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from kaggle_evaluation import smart_buildings_inference_server

    class SubmissionPolicy(sb_policy.CandidatePolicy, smart_buildings_inference_server.Policy):
        pass

    smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(
        competition_data_folder=str(DATA_ROOT))'''),
    ]
    return cells


# --------------------------------------------------------------------------- 01 screen
SCREEN_SET = '''def default_screen_set(n=10, phase0_csv=""):
    """Top contributors from notebook 00 if its scores are found, else two scenarios per station."""
    for p in [phase0_csv, "eval_ref/scores_ref.csv", "../eval_ref/scores_ref.csv"]:
        if p and Path(p).exists():
            df = pd.read_csv(p).sort_values("score", ascending=False)
            print(f"screen set: top {n} contributors from {p}")
            return [s for s in df.scenario if s in SCENARIOS][:n]
    by_station = {}
    for s in SCENARIOS:
        by_station.setdefault(s.split("/")[1].rsplit("_", 1)[0], []).append(s)
    print("screen set: 2 per station (no notebook-00 results found)")
    return [s for v in by_station.values() for s in v[::3][:2]]'''

ALL_NAMES = list(DESCRIPTIONS)

nb01 = [
    md("""# Sàng lọc các version policy

So mọi version với policy tham chiếu (public LB 0,01582), có nhóm null để biết đâu là nhiễu. Notebook tự chứa toàn bộ code.

| Bước | Kịch bản | Độ dài | Ra |
|---|---|---|---|
| 1 · Sàng nhanh | 10 kịch bản đóng góp nhiều nhất | 7 ngày | Bảng xếp hạng + z-score |
| 2 · Xác nhận | 30 kịch bản | 30 ngày | **Điểm thật** của 3 version tốt nhất và của ref |

Kết quả lưu ở `results/screen/`; chạy lại sẽ bỏ qua phần đã xong.

| Version | Thay đổi |
|---|---|
""" + "\n".join(f"| `{k}` | {v} |" for k, v in DESCRIPTIONS.items())),
    code(CONFIG_HEAD + '''
QUICK_STEPS = 2016        # 7 days, for ranking only
FULL_STEPS = 8640
CONFIRM_TOP = 3
ONLY = []
OUT = Path.cwd().resolve() / "results" / "screen"
OUT.mkdir(parents=True, exist_ok=True)
NULLS = ["null_1", "null_2", "null_3"]'''),
    *code_cells(policy_module(ALL_NAMES)),
    code(SCREEN_SET + '''

CANDS = sb_policy.CANDIDATES      # or a subset, e.g. ["fan_tr", "optimal_start"]
SCREEN = default_screen_set()
print("versions:", CANDS)
SCREEN'''),
    md("## Bước 1 · Sàng nhanh"),
    code('''specs = ["ref"] + NULLS + CANDS
print(f"{len(specs) * len(SCREEN)} runs of {QUICK_STEPS} steps")
ev.run_batch([(s, sp, QUICK_STEPS) for sp in specs for s in SCREEN], OUT, workers=WORKERS)

quick = ev.load_results(OUT, steps=QUICK_STEPS)
quick = quick[quick.scenario.isin(SCREEN)]
noise = quick[quick.spec.isin(["ref"] + NULLS)].groupby("scenario").score.agg(["mean", "std"])
print(f"noise: median sd {100 * (noise['std'] / noise['mean']).median():.1f}% of the score")
table = ev.compare(quick[~quick.spec.isin(NULLS)], "ref", noise=noise)
display(table.round(5))'''),
    code('''piv = quick.pivot_table(index="spec", columns="scenario", values="score")
chg = 100 * (piv - piv.loc["ref"]) / piv.loc["ref"]
chg = chg.loc[[s for s in table.index if s != "ref"]]
fig, ax = plt.subplots(figsize=(1.2 + 0.9 * chg.shape[1], 0.45 * chg.shape[0] + 1.6))
lim = max(5, float(np.nanpercentile(np.abs(chg.to_numpy()), 95)))
im = ax.imshow(chg.to_numpy(), cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(chg.shape[1]), [c.replace("building_train_", "B").replace("/", "\\n") for c in chg.columns], fontsize=8)
ax.set_yticks(range(chg.shape[0]), chg.index, fontsize=9)
for i in range(chg.shape[0]):
    for j in range(chg.shape[1]):
        if np.isfinite(chg.iat[i, j]):
            ax.text(j, i, f"{chg.iat[i, j]:+.0f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="% so với ref (xanh = tốt hơn)")
ax.set_title(f"Sàng nhanh, {QUICK_STEPS} bước")
fig.tight_layout(); plt.show()'''),
    md("## Bước 2 · Xác nhận trên 30 kịch bản đầy đủ"),
    code('''ok = table.drop(index="ref", errors="ignore")
ok = ok[(ok.change_pct < 0) & (ok.get("z_worse", 0) == 0)]
best = ok.sort_values("score").index[:CONFIRM_TOP].tolist()
print("confirming:", best or "nothing beat ref in the quick screen")
ev.run_batch([(s, sp, FULL_STEPS) for sp in ["ref"] + best for s in SCENARIOS], OUT, workers=WORKERS)

full = ev.load_results(OUT, steps=FULL_STEPS)
full = full[full.spec.isin(["ref"] + best) & full.scenario.isin(SCENARIOS)]
final = ev.compare(full, "ref")
print("SCORES on", final.scenarios.max(), "scenarios x", FULL_STEPS, "steps:")
for spec, r in final.iterrows():
    print(f"  {spec:16s} {r.score:.5f}  ({r.change_pct:+.1f}% vs ref)")
display(final.round(5))
display(full.pivot_table(index="station", columns="spec", values="score").round(4))
final.to_csv(OUT / "confirm_summary.csv"); table.to_csv(OUT / "quick_summary.csv")'''),
    md("**Đọc kết quả:** giữ version có `change_pct` âm trên 30 kịch bản và `worst_change_pct` nhỏ. Mở notebook riêng của version đó trong `policies/` để nộp lên Kaggle."),
]

# --------------------------------------------------------------------------- 02 param search
nb02 = [
    md("""# Tìm tham số controller (giai đoạn 1)

Tối ưu 11 tham số của policy tham chiếu bằng CMA-ES (gói `cma`; thiếu thì random search). Notebook tự chứa toàn bộ code.

* `MODE = "shared"`: một bộ cho mọi tòa nhà → nộp được ngay.
* `MODE = "per_scenario"`: bộ tốt nhất cho từng kịch bản → "thầy" cho giai đoạn 2.

Tune bằng episode ngắn trên 6 kịch bản, rồi **chấm bộ tốt nhất trên 24 kịch bản còn lại, đủ 30 ngày**. Kết quả trong `results/search/`."""),
    code(CONFIG_HEAD + '''
MODE = "shared"           # "shared" | "per_scenario"
SEARCH_STEPS = 2016       # 7 days per evaluation
GENERATIONS = 10
POPSIZE = 6
FULL_STEPS = 8640
CHECK_TOP = 3
ONLY = []
OUT = Path.cwd().resolve() / "results" / "search"
OUT.mkdir(parents=True, exist_ok=True)
try:
    import cma  # noqa: F401
except ImportError:
    print("tip: pip install cma  (falls back to random search)")'''),
    *code_cells(policy_module([])),
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
INTS = {k for k, (_, _, kind) in SPACE.items() if kind == "int"} | {"HOLD_STEPS"}
REF_VALUES = {k: getattr(sb_policy.RefPolicy, k) for k in SPACE}
print("reference:", REF_VALUES)

def decode(x):
    out = {}
    for (k, (lo, hi, kind)), v in zip(SPACE.items(), np.clip(x, 0, 1)):
        val = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))) if kind == "log" else lo + v * (hi - lo)
        out[k] = int(round(val)) if k in INTS else round(float(val), 5)
    return out

def encode(values):
    return np.clip([(np.log(values[k]) - np.log(lo)) / (np.log(hi) - np.log(lo)) if kind == "log" else (values[k] - lo) / (hi - lo)
                    for k, (lo, hi, kind) in SPACE.items()], 0, 1)

def spec_of(values):
    return "ref:" + ",".join(f"{k}={v}" for k, v in values.items())'''),
    code(SCREEN_SET + '''

TUNE = default_screen_set(n=6)
HOLDOUT = [s for s in SCENARIOS if s not in TUNE]
print("tune on:", TUNE, "\\nhold-out:", len(HOLDOUT), "scenarios")
print(f"budget: {GENERATIONS * POPSIZE * len(TUNE)} runs of {SEARCH_STEPS} steps")'''),
    code('''def evaluate(values_list, scenarios, steps=SEARCH_STEPS):
    """Mean score of each parameter set over the scenarios (parallel, cached)."""
    specs = [spec_of(v) for v in values_list]
    ev.run_batch([(s, sp, steps) for sp in specs for s in scenarios], OUT, workers=WORKERS, verbose=False)
    df = ev.load_results(OUT, specs=set(specs), steps=steps)
    means = df[df.scenario.isin(scenarios)].groupby("spec").score.mean()
    return [float(means.get(sp, np.inf)) for sp in specs]


def search(scenarios, generations=GENERATIONS, popsize=POPSIZE, seed=0):
    """CMA-ES (or random search) starting from the reference parameters."""
    x0 = encode(REF_VALUES)
    ref_score = evaluate([REF_VALUES], scenarios)[0]
    log = [{"gen": -1, **REF_VALUES, "score": ref_score}]
    print(f"ref: {ref_score:.5f}", flush=True)
    try:
        import cma
        es = cma.CMAEvolutionStrategy(x0, 0.2, {"popsize": popsize, "bounds": [0, 1], "seed": seed + 1, "verbose": -9})
        ask, tell = es.ask, es.tell
    except ImportError:
        rng = np.random.default_rng(seed)
        ask = lambda: [np.clip(x0 + rng.normal(0, 0.2, len(x0)), 0, 1) for _ in range(popsize)]
        tell = lambda xs, ys: None
    for g in range(generations):
        xs = ask()
        vals = [decode(x) for x in xs]
        scores = evaluate(vals, scenarios)
        tell(xs, scores)
        log += [{"gen": g, **v, "score": s} for v, s in zip(vals, scores)]
        print(f"gen {g:2d}: best {min(scores):.5f}  mean {np.mean(scores):.5f}  (ref {ref_score:.5f})", flush=True)
    return pd.DataFrame(log)'''),
    md("## Tìm"),
    code('''logs = {"shared": search(TUNE)} if MODE == "shared" else {s: search([s]) for s in TUNE}
for key, log in logs.items():
    log.to_csv(OUT / f"log_{key.replace('/', '__')}.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
for key, log in logs.items():
    g = log[log.gen >= 0].groupby("gen").score.min().cummin()
    line, = ax.plot(g.index, g.values, marker="o", label=key)
    ax.axhline(log.loc[log.gen == -1, "score"].iloc[0], ls="--", lw=1, color=line.get_color())
ax.set_xlabel("thế hệ"); ax.set_ylabel("điểm tốt nhất (thấp = tốt)"); ax.set_title("Hội tụ (nét đứt = ref)")
ax.legend(fontsize=8); plt.show()
if MODE == "per_scenario":
    gap = pd.DataFrame({k: {"ref": l.score.iloc[0], "best": l.score.min()} for k, l in logs.items()}).T
    gap["gain_pct"] = 100 * (gap.best / gap.ref - 1)
    display(gap.round(5))
    print(f"per-scenario tuning gain: {gap.gain_pct.mean():+.1f}% (the gate of phase 1)")'''),
    md("## Điểm trên kịch bản giữ lại (30 ngày)"),
    code('''log = pd.concat(logs.values())
top = log[log.gen >= 0].sort_values("score").drop_duplicates(subset=list(SPACE)).head(CHECK_TOP)
top_vals = [{k: (int(r[k]) if k in INTS else float(r[k])) for k in SPACE} for _, r in top.iterrows()]
specs = ["ref"] + [spec_of(v) for v in top_vals]
ev.run_batch([(s, sp, FULL_STEPS) for sp in specs for s in HOLDOUT], OUT, workers=WORKERS)
full = ev.load_results(OUT, specs=set(specs), steps=FULL_STEPS)
res = ev.compare(full[full.scenario.isin(HOLDOUT)], "ref")
print(f"SCORES on {res.scenarios.max()} hold-out scenarios x {FULL_STEPS} steps:")
for spec, r in res.iterrows():
    print(f"  {r.score:.5f}  ({r.change_pct:+.1f}% vs ref)  {spec}")
display(res.round(5))
res.to_csv(OUT / "holdout_check.csv")
print("\\nbest:", res.index[0])'''),
    md("**Đọc kết quả:** chỉ tin bộ tham số nếu thắng `ref` trên kịch bản giữ lại. Dán dòng `best` (`ref:KP=...`) vào `ONLY`/spec của notebook 01 hoặc tạo một version mới trong `sbx/candidates.py`."),
]


def write(path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path.relative_to(HERE.parent))


if __name__ == "__main__":
    write(HERE / "01_candidates_screen.ipynb", nb01)
    write(HERE / "02_param_search.ipynb", nb02)
    for name in DESCRIPTIONS:
        write(HERE / "policies" / f"{name}.ipynb", version_notebook(name))
