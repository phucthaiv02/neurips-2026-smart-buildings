"""Reinforcement learning for building control.

    features.py  observation -> per-zone / per-AHU / building tensors (any building size)
    model.py     entity-based actor-critic shared across zones and AHUs
    worker.py    rollouts in separate processes (residual RL and imitation)
    ppo.py       residual PPO on top of the reference controller
    bc.py        behaviour cloning / DAgger: a standalone network that imitates a controller
    policy.py    competition-API policy that loads a trained network

Two modes share the same network:
  residual  output = controller action + small learned correction (PPO)
  direct    output = full action, no controller at run time (BC / DAgger, then PPO)
"""
