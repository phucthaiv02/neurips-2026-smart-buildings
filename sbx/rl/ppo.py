"""PPO for building control, residual on the reference controller by default.

Each iteration: N workers each roll out one episode (a random training
scenario, cfg.steps steps), then PPO epochs over those steps. Minibatches are
taken within one episode, so all steps in a batch share a building and no
padding is needed.

Rewards are divided by a running std of discounted returns (as in common PPO
implementations), so the same settings work across climates.
"""
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sbx import paths


@dataclass
class PPOConfig:
    mode: str = "residual"              # "residual" (controller + correction) | "direct"
    controller: str = "ref"             # controller spec (sbx.candidates) used as base / layout
    hidden: int = 64
    steps: int = 2016                   # steps per episode (7 days)
    iters: int = 200
    workers: int = 4
    envs_per_iter: int = 0              # 0 = workers
    gamma: float = 0.995
    lam: float = 0.95
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 512
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03             # stop the epoch loop early above this
    residual_scale: float = 0.3         # max correction (normalized action units) per unit of output
    action_penalty: float = 0.0         # reward -= penalty * mean squared correction
    init_log_std: float = -1.5              # exploration std 0.22 x residual_scale
    init: str = ""                      # checkpoint to start from (e.g. a BC model for direct mode)
    train_scenarios: list = field(default_factory=list)
    val_scenarios: list = field(default_factory=list)
    val_every: int = 10
    val_steps: int = 0                  # 0 = steps
    out: str = "runs/ppo"
    seed: int = 0


class RunningStd:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        for v in np.asarray(x, dtype=np.float64).ravel():
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.m2 += d * (v - self.mean)

    @property
    def std(self):
        return float(np.sqrt(self.m2 / max(self.n - 1, 1))) if self.n > 1 else 1.0


def _gae(rewards, values, last_value, gamma, lam):
    adv = np.zeros_like(rewards)
    gae = 0.0
    nxt = last_value
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * nxt - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        nxt = values[t]
    return adv, adv + values


def _discounted(rewards, gamma):
    out, run = np.zeros_like(rewards), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        run = rewards[t] + gamma * run
        out[t] = run
    return out


def _pool(workers):
    paths.setup()
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=paths.setup)


def save(path, model, cfg, extra=None):
    import torch
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), **(extra or {})}, path)


def load(path):
    import torch
    from sbx.rl.model import EntityActorCritic
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = EntityActorCritic(ck["config"]["hidden"])
    m.load_state_dict(ck["state_dict"])
    return m, ck


def validate(model, cfg, pool, scenarios, steps, controller_scores):
    """Deterministic rollouts; returns mean score and mean change vs the controller."""
    from sbx.rl import worker
    rcfg = {"controller": cfg.controller, "mode": cfg.mode, "hidden": cfg.hidden,
            "residual_scale": cfg.residual_scale, "steps": steps}
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    futs = {s: pool.submit(worker.rollout, s, sd, rcfg, 0, True) for s in scenarios}
    for s in scenarios:
        if s not in controller_scores:
            controller_scores[s] = pool.submit(worker.controller_score, s, cfg.controller, steps)
    scores = {s: f.result()["score"] for s, f in futs.items()}
    for s in scenarios:
        if not isinstance(controller_scores[s], float):
            controller_scores[s] = controller_scores[s].result()
    model_mean = float(np.mean(list(scores.values())))
    ctrl_mean = float(np.mean([controller_scores[s] for s in scenarios]))
    return model_mean, ctrl_mean, scores


def train(cfg: PPOConfig):
    import torch
    from sbx.rl import worker
    from sbx.rl.model import EntityActorCritic

    paths.setup()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if not cfg.train_scenarios:
        cfg.train_scenarios = paths.training_scenarios()

    model = EntityActorCritic(cfg.hidden, cfg.init_log_std)
    if cfg.init:
        m0, _ = load(cfg.init)
        model.load_state_dict(m0.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ret_std = RunningStd()
    rcfg = {"controller": cfg.controller, "mode": cfg.mode, "hidden": cfg.hidden,
            "residual_scale": cfg.residual_scale, "steps": cfg.steps}
    n_envs = cfg.envs_per_iter or cfg.workers
    ctrl_scores, best_val = {}, np.inf
    log_path = out / "log.jsonl"

    with _pool(cfg.workers) as pool:
        for it in range(cfg.iters):
            t0 = time.time()
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            scs = rng.choice(cfg.train_scenarios, size=n_envs, replace=len(cfg.train_scenarios) < n_envs)
            trajs = list(pool.map(worker.rollout, scs, [sd] * n_envs, [rcfg] * n_envs,
                                  [int(rng.integers(1 << 30)) for _ in range(n_envs)]))
            t_roll = time.time() - t0

            # rewards -> shaped, normalized; advantages
            for tr in trajs:
                r = tr["reward"].astype(np.float64)
                if cfg.action_penalty and cfg.mode == "residual":
                    sq = np.concatenate([tr["az"].reshape(len(r), -1), tr["aa"].reshape(len(r), -1),
                                         tr["aw"].reshape(len(r), -1)], 1) ** 2
                    r = r - cfg.action_penalty * sq.mean(1)
                tr["shaped"] = r
                ret_std.update(_discounted(r, cfg.gamma))
            for tr in trajs:
                r = tr["shaped"] / (ret_std.std + 1e-8)
                adv, ret = _gae(r, tr["value"].astype(np.float64), float(tr["last_value"]), cfg.gamma, cfg.lam)
                tr["adv"], tr["ret"] = adv.astype(np.float32), ret.astype(np.float32)
            all_adv = np.concatenate([tr["adv"] for tr in trajs])
            a_mean, a_std = all_adv.mean(), all_adv.std() + 1e-8

            # PPO epochs
            stats = {"pg": [], "vf": [], "kl": [], "clipfrac": []}
            stop = False
            for epoch in range(cfg.epochs):
                batches = []
                for ti, tr in enumerate(trajs):
                    idx = rng.permutation(len(tr["reward"]))
                    batches += [(ti, idx[i:i + cfg.minibatch]) for i in range(0, len(idx), cfg.minibatch)]
                rng.shuffle(batches)
                kls = []
                for ti, b in batches:
                    tr = trajs[ti]
                    T = lambda k: torch.as_tensor(tr[k][b])
                    o = model(T("zone"), T("ahu"), T("glob"), torch.as_tensor(tr["zone_ahu"]))
                    logp = model.log_prob(o, T("az"), T("aa"), T("aw"))
                    ratio = torch.exp(logp - T("logp"))
                    adv = (T("adv") - a_mean) / a_std
                    pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                    vf = ((o["value"] - T("ret")) ** 2).mean()
                    V, K = tr["zone"].shape[1], tr["ahu"].shape[1]
                    ent = model.entropy(V, K) / (V + K + 1)
                    loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        kl = float((T("logp") - logp).mean())
                    kls.append(kl)
                    stats["pg"].append(pg.item()); stats["vf"].append(vf.item()); stats["kl"].append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                if np.mean(kls) > cfg.target_kl:
                    stop = True
                    break

            rec = {"iter": it, "rollout_score": float(np.mean([tr["score"] for tr in trajs])),
                   "scenarios": [str(s) for s in scs], "t_rollout": round(t_roll, 1),
                   "t_total": round(time.time() - t0, 1), "early_stop": stop,
                   "std_z": float(model.log_std_z.exp().mean()), "std_a": float(model.log_std_a.exp().mean()),
                   **{k: float(np.mean(v)) for k, v in stats.items() if v}}
            if cfg.val_scenarios and (it % cfg.val_every == 0 or it == cfg.iters - 1):
                vm, cm, per = validate(model, cfg, pool, cfg.val_scenarios, cfg.val_steps or cfg.steps, ctrl_scores)
                rec.update(val_score=vm, val_controller=cm, val_change_pct=100 * (vm / cm - 1))
                if vm < best_val:
                    best_val = vm
                    save(out / "best.pt", model, cfg, {"iter": it, "val_score": vm, "ret_std": ret_std.std})
            save(out / "last.pt", model, cfg, {"iter": it, "ret_std": ret_std.std})
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            msg = (f"it {it:4d}  rollout {rec['rollout_score']:.5f}  kl {rec.get('kl', 0):.4f}  "
                   f"vf {rec.get('vf', 0):.3f}  std {rec['std_z']:.3f}  {rec['t_total']:.0f}s")
            if "val_score" in rec:
                msg += f"  | val {rec['val_score']:.5f} vs controller {rec['val_controller']:.5f} ({rec['val_change_pct']:+.2f}%)"
            print(msg, flush=True)
    return model
