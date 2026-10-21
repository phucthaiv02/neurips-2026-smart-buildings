"""Competition-API policy that runs a trained network.

    NeuralPolicy(building_meta_data, action_spec).get_action(obs) -> action dict

Weights: class attribute WEIGHTS, or $SB_RL_WEIGHTS. In sbx.candidates the
spec  "rl:WEIGHTS='runs/ppo/best.pt'"  evaluates a checkpoint like any other policy.
"""
import os

import numpy as np


class NeuralPolicy:
    WEIGHTS = ""

    def __init__(self, building_meta_data, action_spec):
        import torch
        from sbx import candidates
        from sbx.rl.features import Layout
        from sbx.rl.ppo import load
        torch.set_num_threads(1)
        path = self.WEIGHTS or os.getenv("SB_RL_WEIGHTS", "rl_policy.pt")
        self.model, ck = load(path)
        self.model.eval()
        cfg = ck["config"]
        self.mode = cfg["mode"]
        self.residual_scale = cfg.get("residual_scale", 0.0)
        # The controller gives the device layout in both modes; it only acts in residual mode.
        self.ctrl = candidates.make(cfg["controller"], building_meta_data, action_spec)
        self.layout = Layout(self.ctrl)
        self.action_spec = action_spec

    def get_action(self, obs):
        L = self.layout
        if self.mode == "residual":
            ca = self.ctrl.get_action(obs)
            z, a, g = L.featurize(obs, ca)
            out = self.model.act(z, a, g, L.zone_ahu, deterministic=True)
            action = L.apply_residual(ca, out["az"], out["aa"], out["aw"], self.residual_scale)
        else:
            z, a, g = L.featurize(obs, None)
            out = self.model.act(z, a, g, L.zone_ahu, deterministic=True)
            action = L.assemble(out["az"], out["aa"], out["aw"], out["run_a"], out["run_w"])
        for key, dtype in (("continuous_action", np.float32), ("discrete_action", np.int32)):
            space = self.action_spec[key]
            action[key] = np.clip(action[key], space.low, space.high).astype(dtype)
        return action
