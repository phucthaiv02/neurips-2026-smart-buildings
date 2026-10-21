"""Train RL policies.

    # residual PPO on top of the reference controller (main path)
    python scripts/train_rl.py ppo --workers 4 --iters 200 --out runs/ppo_residual

    # standalone network: behaviour cloning + DAgger, then PPO fine-tuning
    python scripts/train_rl.py bc  --workers 4 --rounds 5 --out runs/bc
    python scripts/train_rl.py ppo --mode direct --init runs/bc/best.pt --out runs/ppo_direct

    # score a checkpoint on all 30 training scenarios (full 30-day episodes)
    python scripts/train_rl.py eval runs/ppo_residual/best.pt --workers 4

Validation uses scenarios held out from training (--val, default: 5 scenarios,
one per climate). Needs the starter kit (see README) and torch.
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sbx import paths  # noqa: E402

DEFAULT_VAL = ["building_train_1/atlanta_2", "building_train_2/fairbanks_2", "building_train_4/phoenix_8",
               "building_train_7/singapore_6", "building_train_8/denver_6"]


def _add_dataclass_args(p, cls, skip=("train_scenarios", "val_scenarios")):
    for f in fields(cls):
        if f.name in skip:
            continue
        default = f.default
        p.add_argument("--" + f.name.replace("_", "-"), type=type(default), default=default)


def main(argv=None):
    from sbx.rl.bc import BCConfig
    from sbx.rl.ppo import PPOConfig
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, cls in (("ppo", PPOConfig), ("bc", BCConfig)):
        p = sub.add_parser(name)
        _add_dataclass_args(p, cls)
        p.add_argument("--val", nargs="*", default=DEFAULT_VAL, help="held-out validation scenarios")
        p.add_argument("--train", nargs="*", default=None, help="training scenarios (default: all others)")
        p.add_argument("--extra-train", nargs="*", default=[], help="generated scenarios to add to training")
    pe = sub.add_parser("eval")
    pe.add_argument("checkpoint")
    pe.add_argument("--workers", type=int, default=4)
    pe.add_argument("--steps", type=int, default=8640)
    pe.add_argument("--out", default="results/rl_eval")
    pe.add_argument("--scenarios", nargs="*", default=None)
    args = ap.parse_args(argv)
    paths.setup()

    if args.cmd == "eval":
        from sbx import evaluation as ev
        scen = args.scenarios or paths.training_scenarios()
        spec = f"rl:WEIGHTS={str(Path(args.checkpoint).resolve())!r}"
        ev.run_batch([(s, sp, args.steps) for sp in ["ref", spec] for s in scen], args.out, workers=args.workers)
        df = ev.load_results(args.out, specs={"ref", spec}, steps=args.steps)
        res = ev.compare(df[df.scenario.isin(scen)], "ref")
        print(res.to_string())
        return

    cls = PPOConfig if args.cmd == "ppo" else BCConfig
    kw = {f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)}
    val = list(args.val or [])
    train = args.train if args.train is not None else [s for s in paths.training_scenarios() if s not in val]
    cfg = cls(**kw, train_scenarios=train + list(args.extra_train), val_scenarios=val)
    print(f"{args.cmd}: {len(cfg.train_scenarios)} training scenarios, {len(val)} validation -> {cfg.out}")
    if args.cmd == "ppo":
        from sbx.rl.ppo import train
        train(cfg)
    else:
        from sbx.rl.bc import train_bc
        train_bc(cfg)


if __name__ == "__main__":
    main()
