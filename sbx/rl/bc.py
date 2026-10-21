"""Behaviour cloning + DAgger: a standalone network that imitates a controller.

Round 0 records the controller acting (beta = 1). Later rounds let the
student act with probability 1 - beta while the controller still labels every
state, so the student learns to recover from its own mistakes. The dataset
keeps all rounds. The result runs without the controller (mode "direct") and
can be fine-tuned with PPO (`PPOConfig(mode="direct", init=...)`).
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sbx import paths
from sbx.rl.ppo import PPOConfig, _pool, save


@dataclass
class BCConfig:
    controller: str = "ref"
    hidden: int = 64
    steps: int = 2016
    rounds: int = 5                     # DAgger rounds (round 0 = pure behaviour cloning)
    episodes_per_round: int = 8
    beta0: float = 1.0
    beta_decay: float = 0.5             # beta for round i = beta0 * decay**i (round 0 forced to 1)
    workers: int = 4
    epochs: int = 3
    minibatch: int = 512
    lr: float = 1e-3
    run_weight: float = 0.2             # weight of the on/off (BCE) terms
    train_scenarios: list = field(default_factory=list)
    val_scenarios: list = field(default_factory=list)
    val_steps: int = 0
    out: str = "runs/bc"
    seed: int = 0


def _loss(model, tr, b, run_weight):
    import torch
    import torch.nn.functional as F
    T = lambda k: torch.as_tensor(tr[k][b])
    o = model(T("zone"), T("ahu"), T("glob"), torch.as_tensor(tr["zone_ahu"]))
    mse = F.mse_loss(o["mu_z"], T("tz")) + F.mse_loss(o["mu_a"], T("ta")) + F.mse_loss(o["mu_w"], T("tw"))
    bce = F.binary_cross_entropy_with_logits(o["run_a"], T("tra")) + \
        F.binary_cross_entropy_with_logits(o["run_w"], T("trw").reshape(-1))
    return mse + run_weight * bce, mse.item(), bce.item()


def train_bc(cfg: BCConfig):
    import torch
    from sbx.rl import worker
    from sbx.rl.model import EntityActorCritic

    paths.setup()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if not cfg.train_scenarios:
        cfg.train_scenarios = paths.training_scenarios()
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    model = EntityActorCritic(cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rcfg = {"controller": cfg.controller, "hidden": cfg.hidden, "steps": cfg.steps,
            "mode": "direct", "residual_scale": 0.0}
    # Saved checkpoints look like a direct-mode PPO config so policy.py / ppo.py can load them.
    ppo_like = PPOConfig(mode="direct", controller=cfg.controller, hidden=cfg.hidden, out=cfg.out)
    data, best = [], np.inf
    with _pool(cfg.workers) as pool:
        ctrl_scores = {}
        for rnd in range(cfg.rounds):
            t0 = time.time()
            beta = 1.0 if rnd == 0 else cfg.beta0 * cfg.beta_decay ** rnd
            sd = None if rnd == 0 else {k: v.detach().clone() for k, v in model.state_dict().items()}
            scs = rng.choice(cfg.train_scenarios, size=cfg.episodes_per_round,
                             replace=len(cfg.train_scenarios) < cfg.episodes_per_round)
            new = list(pool.map(worker.rollout_imitation, scs, [sd] * len(scs), [rcfg] * len(scs),
                                [int(rng.integers(1 << 30)) for _ in scs], [beta] * len(scs)))
            data += new
            mses, bces = [], []
            for _ in range(cfg.epochs):
                batches = [(ti, idx[i:i + cfg.minibatch])
                           for ti, tr in enumerate(data)
                           for idx in [rng.permutation(len(tr["reward"]))]
                           for i in range(0, len(idx), cfg.minibatch)]
                rng.shuffle(batches)
                for ti, b in batches:
                    loss, mse, bce = _loss(model, data[ti], b, cfg.run_weight)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    mses.append(mse); bces.append(bce)
            rec = {"round": rnd, "beta": beta, "episodes": len(data), "mse": float(np.mean(mses)),
                   "bce": float(np.mean(bces)), "rollout_score": float(np.mean([t["score"] for t in new])),
                   "t": round(time.time() - t0, 1)}
            if cfg.val_scenarios:
                from sbx.rl.ppo import validate
                vm, cm, _ = validate(model, ppo_like, pool, cfg.val_scenarios, cfg.val_steps or cfg.steps, ctrl_scores)
                rec.update(val_score=vm, val_controller=cm, val_change_pct=100 * (vm / cm - 1))
                if vm < best:
                    best = vm
                    save(out / "best.pt", model, ppo_like, {"round": rnd, "val_score": vm})
            save(out / "last.pt", model, ppo_like, {"round": rnd})
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            msg = f"round {rnd}  beta {beta:.2f}  mse {rec['mse']:.4f}  bce {rec['bce']:.3f}  {rec['t']:.0f}s"
            if "val_score" in rec:
                msg += f"  | student {rec['val_score']:.5f} vs controller {rec['val_controller']:.5f} ({rec['val_change_pct']:+.1f}%)"
            print(msg, flush=True)
    return model
