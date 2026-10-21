"""Turn one building's flat observation into entity features of fixed width.

Every building has the same *kinds* of devices with the same observation
fields, only different counts. So each zone becomes a row of ZONE_DIM numbers,
each AHU a row of AHU_DIM, and the building a vector of GLOBAL_DIM; a network
shared across rows works on any building.

Actions are mapped back the same way: per zone (damper, reheat), per AHU
(supply temperature, static pressure, recirculation, run), and building-level
(hot-water temperature, pump pressure, hot-water run).
"""
import numpy as np

ZONE_FIELDS = [
    "zone_air_temperature_sensor",          # normalized (decoded below)
    "zone_air_heating_temperature_setpoint_sensor",
    "zone_air_cooling_temperature_setpoint_sensor",
    "supply_air_damper_percentage_command",
    "reheat_valve_percentage_command",
    "supply_air_flowrate_setpoint",
]
AHU_FIELDS = [
    "cooling_request_count", "discharge_fan_speed_percentage_command", "exhaust_fan_run_command",
    "outside_air_flowrate_sensor", "outside_air_temperature_sensor", "recirculation_percentage_command",
    "supervisor_run_command", "supply_air_flowrate_sensor", "supply_air_flowrate_setpoint",
    "supply_air_static_pressure_sensor", "supply_air_static_pressure_setpoint",
    "supply_air_temperature_sensor", "supply_air_temperature_setpoint", "supply_fan_run_command",
    "supply_fan_speed_percentage_command",
]
HWS_FIELDS = [
    "hws_differential_pressure_sensor", "hws_differential_pressure_setpoint", "hws_heating_request_count",
    "hws_run_status", "hws_supervisor_run_command", "hws_supply_water_temperature_sensor",
    "hws_supply_water_temperature_setpoint",
]
GLOBAL_FIELDS = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000",
                 "comfort_mode_now", "comfort_mode_soon", "num_occupants"]

ZONE_ACT, AHU_ACT, WATER_ACT = 2, 3, 2          # continuous outputs per entity
# zone: 10 obs-derived + 2 error EMAs + 2 controller actions
ZONE_DIM = 14
# ahu: 15 fields + mean zone error + 4 controller actions (sat, pressure, recirc, run)
AHU_DIM = 20
# global: 7 fields + 7 hws + frac-hour sin/cos + 3 controller water actions + 2 size + 3 band stats
GLOBAL_DIM = 24

EMA_FAST, EMA_SLOW = 1 / 12, 1 / 72             # ~1 h and ~6 h at 5-minute steps


def _scale(name: str, v: float) -> float:
    if "temperature" in name and v > 150:        # raw kelvin
        return (v - 293.15) / 10.0
    if "static_pressure" in name:
        return v / 1000.0
    if "differential_pressure" in name:
        return v / 100.0
    if "flowrate" in name:
        return float(np.log1p(max(v, 0.0)))
    if "request_count" in name:
        return v / 10.0
    if name == "num_occupants":
        return float(np.log1p(max(v, 0.0))) / 4.0
    return v


class Layout:
    """Index maps for one building, built from a PID-family policy instance."""

    def __init__(self, ctrl_policy):
        p = ctrl_policy
        names = list(p.field_names)
        idx = {n: i for i, n in enumerate(names)}
        self._denorm = p._denormalize_zone_temp
        self.comfort = tuple(p.comfort_temp_window)
        self.zone_obs, self.zone_act, self.zone_ahu = [], [], []
        for k, ctrl in enumerate(p.ahu_controllers):
            for vav_id, _, ai in ctrl["zones"]:
                self.zone_obs.append([idx.get(f"{vav_id}_{f}", -1) for f in ZONE_FIELDS])
                self.zone_act.append([ai.get("supply_air_damper_percentage_command", -1),
                                      ai.get("reheat_valve_percentage_command", -1)])
                self.zone_ahu.append(k)
        self.zone_obs = np.array(self.zone_obs, dtype=int).reshape(-1, len(ZONE_FIELDS))
        self.zone_act = np.array(self.zone_act, dtype=int).reshape(-1, ZONE_ACT)
        self.zone_ahu = np.array(self.zone_ahu, dtype=int)
        self.ahu_obs = np.array([[idx.get(f"ahs_{c['ahu_id']}_{f}", -1) for f in AHU_FIELDS]
                                 for c in p.ahu_controllers], dtype=int).reshape(-1, len(AHU_FIELDS))
        self.ahu_act = np.array([[c["temp_idx"], c["pressure_idx"] if c["pressure_idx"] is not None else -1,
                                  c.get("recirc_idx") if c.get("recirc_idx") is not None else -1]
                                 for c in p.ahu_controllers], dtype=int).reshape(-1, AHU_ACT)
        self.ahu_run = np.array([c["run_discrete_idx"] for c in p.ahu_controllers], dtype=int)
        self.hws_obs = np.array([idx.get(f, -1) for f in HWS_FIELDS], dtype=int)
        self.glob_obs = np.array([idx.get(f, -1) for f in GLOBAL_FIELDS], dtype=int)
        self.water_act = np.array([p.water_temp_idx if p.water_temp_idx is not None else -1,
                                   p.water_pressure_idx if p.water_pressure_idx is not None else -1], dtype=int)
        self.water_run = p.water_discrete_idx if 0 <= p.water_discrete_idx < p.n_discrete else -1
        self.n_cont, self.n_disc = p.n_continuous, p.n_discrete
        self.V, self.K = len(self.zone_ahu), len(self.ahu_obs)
        self._zone_names = [ZONE_FIELDS] * self.V
        self.reset()

    def reset(self):
        self.ema_fast = np.zeros(self.V)
        self.ema_slow = np.zeros(self.V)

    @staticmethod
    def _take(obs, ix):
        out = np.where(ix >= 0, obs[np.clip(ix, 0, None)], 0.0)
        return out.astype(np.float64)

    def featurize(self, obs, ctrl_action=None):
        """obs -> (zone [V,ZONE_DIM], ahu [K,AHU_DIM], glob [GLOBAL_DIM]) float32.

        ctrl_action: the controller's action dict (residual mode), or None
        (direct mode: those columns are zero). Updates the error EMAs, so call
        exactly once per step.
        """
        obs = np.asarray(obs, dtype=np.float64)
        z = self._take(obs, self.zone_obs)                        # [V, 6]
        t = np.array([self._denorm(v) for v in z[:, 0]])
        heat = np.where(z[:, 1] > 150, z[:, 1], self.comfort[0])
        cool = np.where(z[:, 2] > 150, z[:, 2], self.comfort[1])
        err = np.where(t < heat, t - heat, np.where(t > cool, t - cool, 0.0))
        self.ema_fast += EMA_FAST * (err - self.ema_fast)
        self.ema_slow += EMA_SLOW * (err - self.ema_slow)
        zone = np.zeros((self.V, ZONE_DIM))
        zone[:, 0] = (t - 295.15) / 3
        zone[:, 1] = (heat - 295.15) / 3
        zone[:, 2] = (cool - 295.15) / 3
        zone[:, 3] = np.clip((t - heat) / 2, -5, 5)
        zone[:, 4] = np.clip((t - cool) / 2, -5, 5)
        zone[:, 5] = z[:, 3]
        zone[:, 6] = z[:, 4]
        zone[:, 7] = np.log1p(np.maximum(z[:, 5], 0))
        zone[:, 8] = (cool - heat) / 5
        zone[:, 9] = np.clip(err / 2, -5, 5)
        zone[:, 10] = np.clip(self.ema_fast / 2, -5, 5)
        zone[:, 11] = np.clip(self.ema_slow / 2, -5, 5)

        a_raw = self._take(obs, self.ahu_obs)
        ahu = np.zeros((self.K, AHU_DIM))
        for j, f in enumerate(AHU_FIELDS):
            ahu[:, j] = [_scale(f, v) for v in a_raw[:, j]]
        for k in range(self.K):
            m = self.zone_ahu == k
            ahu[k, 15] = np.clip(err[m].mean() / 2, -5, 5) if m.any() else 0.0

        g_raw = self._take(obs, self.glob_obs)
        h_raw = self._take(obs, self.hws_obs)
        glob = np.zeros(GLOBAL_DIM)
        glob[:7] = [_scale(f, v) for f, v in zip(GLOBAL_FIELDS, g_raw)]
        glob[7:14] = [_scale(f, v) for f, v in zip(HWS_FIELDS, h_raw)]
        angle = np.arctan2(g_raw[0], g_raw[1])
        glob[14:16] = np.sin(angle), np.cos(angle)
        glob[19] = self.V / 50
        glob[20] = self.K / 5
        glob[21] = np.clip(err.mean() / 2, -5, 5)
        glob[22] = np.clip(err.min() / 2, -5, 5)
        glob[23] = np.clip(err.max() / 2, -5, 5)

        if ctrl_action is not None:
            c = np.asarray(ctrl_action["continuous_action"], dtype=np.float64)
            d = np.asarray(ctrl_action["discrete_action"], dtype=np.float64)
            zone[:, 12:14] = self._take(c, self.zone_act)
            ahu[:, 16:19] = self._take(c, self.ahu_act)
            ahu[:, 19] = d[self.ahu_run]
            glob[16:18] = self._take(c, self.water_act)
            glob[18] = d[self.water_run] if self.water_run >= 0 else 0.0
        clip = lambda x: np.clip(x, -10, 10).astype(np.float32)
        return clip(zone), clip(ahu), clip(glob)

    # ---------------------------------------------------------------- actions
    def controller_targets(self, ctrl_action):
        """Controller action as per-entity arrays (targets for imitation)."""
        c = np.asarray(ctrl_action["continuous_action"], dtype=np.float64)
        d = np.asarray(ctrl_action["discrete_action"], dtype=np.float64)
        za = self._take(c, self.zone_act)
        aa = self._take(c, self.ahu_act)
        wa = self._take(c, self.water_act)
        ar = d[self.ahu_run]
        wr = np.array([d[self.water_run] if self.water_run >= 0 else 0.0])
        f = lambda x: x.astype(np.float32)
        return f(za), f(aa), f(wa), f(ar), f(wr)

    def assemble(self, za, aa, wa, ar, wr):
        """Per-entity actions -> environment action dict (continuous clipped to [-1, 1])."""
        cont = np.zeros(self.n_cont, dtype=np.float32)
        disc = np.zeros(self.n_disc, dtype=np.int32)
        for ix, val in ((self.zone_act, za), (self.ahu_act, aa), (self.water_act[None, :], np.asarray(wa)[None, :])):
            m = ix >= 0
            cont[ix[m]] = np.clip(np.asarray(val)[m], -1, 1)
        disc[self.ahu_run] = (np.asarray(ar) > 0.5).astype(np.int32)
        if self.water_run >= 0:
            disc[self.water_run] = int(np.asarray(wr).reshape(-1)[0] > 0.5)
        return {"continuous_action": cont, "discrete_action": disc}

    def apply_residual(self, ctrl_action, dz, da, dw, scale):
        """Controller action + scale * correction (continuous only; on/off kept)."""
        cont = np.array(ctrl_action["continuous_action"], dtype=np.float32, copy=True)
        for ix, val in ((self.zone_act, dz), (self.ahu_act, da), (self.water_act[None, :], np.asarray(dw)[None, :])):
            m = ix >= 0
            cont[ix[m]] = np.clip(cont[ix[m]] + scale * np.asarray(val)[m], -1, 1)
        return {"continuous_action": cont,
                "discrete_action": np.array(ctrl_action["discrete_action"], dtype=np.int32, copy=True)}
