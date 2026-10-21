"""Compute the leaderboard metric for the shipped baselines.

Usage:
    python scripts/baseline_scores.py path/to/starter-kit/baseline_returns.json

Metric: mean over scenarios of |total_reward / 8640| (lower is better).
"""
import json
import statistics
import sys
from collections import Counter

T = 8640
POLICIES = ["bang_bang", "fixed_setpoint", "pid", "pid_water"]


def main(path: str) -> None:
    with open(path) as f:
        returns = json.load(f)
    print(f"{len(returns)} scenarios")
    for p in POLICIES:
        print(f"{p:16s} {statistics.mean(-v[p] / T for v in returns.values()):.4f}")
    oracle = statistics.mean(min(-v[p] / T for p in POLICIES) for v in returns.values())
    print(f"{'best per scenario':16s} {oracle:.4f}")
    wins = Counter(min(POLICIES, key=lambda p: -v[p]) for v in returns.values())
    print("wins:", dict(wins))
    print()
    for k, v in returns.items():
        print(k, {p: round(-v[p] / T, 3) for p in POLICIES})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline_returns.json")
