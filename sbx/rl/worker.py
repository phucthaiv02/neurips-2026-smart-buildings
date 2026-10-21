"""Rollouts, run in worker processes. Everything returned is plain numpy.

rollout(...)            one episode with the network acting (residual or direct mode)
rollout_imitation(...)  one episode for BC/DAgger: records the controller's action as label
"""
import numpy as np

from sbx import paths


class BuildingEpisode:
    """One scenario: environment + a fresh controller + feature layout per episode."""

    def __init__(self, scenario, controller="ref"):
        paths.setup()
        from smart_buildings.evaluate import validate_action
        from sbx import evaluation
        self._validate = validate_action
        self.env, self.meta = evaluation.load_env(scenario)
        self.spec = self.env.action_spec()
        self.controller = controller
        self.scenario = scenario

    def reset(self):
        from sbx import candidates
        from sbx.rl.features import Layout
        obs, _ = self.env.reset()
        self.ctrl = candidates.make(self.controller, self.meta, self.spec)
        self.layout = Layout(self.ctrl)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(self._validate(action, self.spec))
        return obs, float(reward), bool(terminated or truncated)


def _model(state_dict, hidden):
    import torch
    from sbx.rl.model import EntityActorCritic
    torch.set_num_threads(1)
    m = EntityActorCritic(hidden)
    if state_dict is not None:
        m.load_state_dict(state_dict)
    m.eval()
    return m


def _stack(rows):
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def rollout(scenario, state_dict, cfg, seed=0, deterministic=False):
    """Network acts for cfg['steps'] steps. cfg: controller, mode, hidden, residual_scale."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    ep = BuildingEpisode(scenario, cfg["controller"])
    model = _model(state_dict, cfg["hidden"])
    residual = cfg["mode"] == "residual"
    obs = ep.reset()
    L = ep.layout
    rows, rewards = [], []
    for _ in range(cfg["steps"]):
        ca = ep.ctrl.get_action(obs) if residual else None
        z, a, g = L.featurize(obs, ca)
        out = model.act(z, a, g, L.zone_ahu, deterministic=deterministic)
        if residual:
            action = L.apply_residual(ca, out["az"], out["aa"], out["aw"], cfg["residual_scale"])
        else:
            action = L.assemble(out["az"], out["aa"], out["aw"], out["run_a"], out["run_w"])
        rows.append({"zone": z, "ahu": a, "glob": g, "az": out["az"].astype(np.float32),
                     "aa": out["aa"].astype(np.float32), "aw": out["aw"].astype(np.float32),
                     "logp": np.float32(out["logp"]), "value": np.float32(out["value"])})
        obs, r, done = ep.step(action)
        rewards.append(r)
        if done:
            break
    ca = ep.ctrl.get_action(obs) if residual else None
    z, a, g = L.featurize(obs, ca)
    last_value = model.act(z, a, g, L.zone_ahu, deterministic=True)["value"]
    traj = _stack(rows)
    traj.update(reward=np.asarray(rewards, dtype=np.float32), last_value=np.float32(last_value),
                zone_ahu=L.zone_ahu.copy(), scenario=scenario, done=bool(done),
                score=float(abs(np.mean(rewards))))
    return traj


def rollout_imitation(scenario, state_dict, cfg, seed=0, beta=1.0):
    """DAgger episode: with prob. beta the controller acts, else the student (direct mode).

    Labels are always the controller's action for the observed state; the
    controller keeps its own internal state by seeing every observation.
    """
    rng = np.random.default_rng(seed)
    ep = BuildingEpisode(scenario, cfg["controller"])
    model = _model(state_dict, cfg["hidden"]) if state_dict is not None and beta < 1.0 else None
    obs = ep.reset()
    L = ep.layout
    rows, rewards = [], []
    for _ in range(cfg["steps"]):
        teacher = ep.ctrl.get_action(obs)
        z, a, g = L.featurize(obs, None)
        tz, ta, tw, tra, trw = L.controller_targets(teacher)
        rows.append({"zone": z, "ahu": a, "glob": g, "tz": tz, "ta": ta, "tw": tw, "tra": tra, "trw": trw})
        if model is None or rng.random() < beta:
            action = teacher
        else:
            out = model.act(z, a, g, L.zone_ahu, deterministic=True)
            action = L.assemble(out["az"], out["aa"], out["aw"], out["run_a"], out["run_w"])
        obs, r, done = ep.step(action)
        rewards.append(r)
        if done:
            break
    traj = _stack(rows)
    traj.update(reward=np.asarray(rewards, dtype=np.float32), zone_ahu=L.zone_ahu.copy(),
                scenario=scenario, score=float(abs(np.mean(rewards))))
    return traj


def controller_score(scenario, controller, steps):
    """Score of the controller alone over the same horizon (reference for RL)."""
    ep = BuildingEpisode(scenario, controller)
    obs = ep.reset()
    rewards = []
    for _ in range(steps):
        obs, r, done = ep.step(ep.ctrl.get_action(obs))
        rewards.append(r)
        if done:
            break
    return float(abs(np.mean(rewards)))
