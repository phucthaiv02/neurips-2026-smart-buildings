"""Candidate controllers built on the reference policy (public LB 0.01582).

Each candidate changes one thing, so a screen can tell what helps. The spec
syntax used everywhere is  "name"  or  "name:ATTR=value,ATTR2=value".

    ref             the reference SubmissionPolicy, unchanged
    evening_coast   stop the zone PI loops 1 h before the end of the work day
    occ_gate        skip PI hours on days the building reports no occupants
    heat_seq        heating: open reheat first at minimum air, add air after
    sat_ramp        supply-air temperature ramps with cooling demand (not on/off)
    pi_leak         small integral leak in the zone PI loops
    water_max       full hot-water temperature whenever a zone heats in PI hours
    fan_tr          trim-and-respond static pressure from zone damper demand
    optimal_start   start the morning recovery from a learned warm-up rate
    night_flush     pre-cool with outside air before the work day (fan only)
    combo           optimal_start + fan_tr + night_flush
    null_<k>        ref with a ~0.01 Pa nudge on every pressure command (noise)
    pid_water, bang_bang   organizer baselines, for reference

Note: AdaptiveColdPolicy resets PRESTART_H, WATER_MAX_PI, HEAT_DAMPER_GAIN,
TARGET_SHIFT_K and COLD_SHIFT_ONLY on every step from its weather regime.
Tune those through the regime constants (EXTREME_PRESTART_H, MILD_SHIFT_K,
EXTREME_HEAT_DAMPER, ...) or the base PRESTART_H, not directly.
"""
import ast

import numpy as np

from sbx.ref_policy import RefPolicy, _outside_kelvin


class _TimeMixin:
    def _frac_hour(self, obs) -> float:
        angle = float(np.arctan2(obs[self.hod_sin_index], obs[self.hod_cos_index]))
        if angle < 0:
            angle += 2 * np.pi
        return 24.0 * angle / (2 * np.pi)

    def _zone_temps(self, ctrl, obs):
        return [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]


class FanTrimRespondPolicy(_TimeMixin, RefPolicy):
    """Static pressure follows the most open damper on each AHU (ASHRAE G36 style).

    The reference runs forced AHUs at minimum pressure and PID AHUs at a
    pressure tied to the AHU loop effort. Neither looks at whether the zones
    actually get the air they ask for. Here, in PI hours, each running AHU
    raises pressure while any of its zones asks for near-full air and trims it
    otherwise: zones starved of air get it, and fan energy is not spent on
    zones that are satisfied.
    """
    TR_HIGH = 0.85      # max zone |u| above which pressure rises
    TR_LOW = 0.55       # max zone |u| below which pressure trims
    TR_UP = 0.08        # normalized pressure step up per 5 min
    TR_DOWN = 0.04      # step down
    TR_MIN = -1.0
    KEY = "fan_tr"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._tr_p = [self.TR_MIN] * len(self.ahu_controllers)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._pi_active:
            self._tr_p = [self.TR_MIN] * len(self.ahu_controllers)
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            j = ctrl["pressure_idx"]
            if j is None or not disc[ctrl["run_discrete_idx"]]:
                continue
            us = [abs(self._zone_u[v]) for v, _, _ in ctrl["zones"] if v in self._zone_u]
            if not us:
                continue
            demand = max(us)
            if demand > self.TR_HIGH:
                self._tr_p[i] = min(1.0, self._tr_p[i] + self.TR_UP)
            elif demand < self.TR_LOW:
                self._tr_p[i] = max(self.TR_MIN, self._tr_p[i] - self.TR_DOWN)
            cont[j] = np.float32(self._tr_p[i])
        return action


class OptimalStartPolicy(_TimeMixin, RefPolicy):
    """Begin the morning recovery when the building needs it, not at a fixed hour.

    The reference starts PI hours a fixed 2 h (4 h in deep cold) before
    occupancy. A mild morning wastes most of that; a Monday after a cold
    weekend may need more. Here the lead time is the distance of the worst
    zones from the comfort band divided by a recovery rate learned on this
    building from previous mornings (separately for heating and cooling).
    """
    OS_RATE_HEAT = 1.0      # initial guess, K per hour
    OS_RATE_COOL = 1.5
    OS_RATE_MIN = 0.3
    OS_LEARN = 0.3          # EMA weight of each new observed morning
    OS_MARGIN_H = 0.25
    OS_MIN_LEAD_H = 0.5
    OS_MAX_LEAD_H = 5.0
    OS_QUANTILE = 0.9       # which zone sets the distance (0.9 = near-worst)
    KEY = "optimal_start"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._rate = {"heat": self.OS_RATE_HEAT, "cool": self.OS_RATE_COOL}
        self._recovery = None   # (mode, start_hour, start_distance) while recovering

    def _distance(self, obs):
        lo, hi = self.comfort_temp_window
        temps = np.array([self._denormalize_zone_temp(obs[j])
                          for c in self.ahu_controllers for j in c["zone_obs_indices"]])
        if temps.size == 0:
            return "heat", 0.0
        below = np.quantile(np.maximum(0.0, lo - temps), self.OS_QUANTILE)
        above = np.quantile(np.maximum(0.0, temps - hi), self.OS_QUANTILE)
        return ("heat", float(below)) if below >= above else ("cool", float(above))

    def _pi_hours(self, obs):
        hour, day = self._get_hour_and_day(obs)
        h = self._frac_hour(obs)
        start, end = self.morning_start_hour, self.evening_start_hour - self.EVENING_STOP_H
        weekday = day < 5
        if weekday and start <= hour < end:
            if self._recovery is not None:      # learn from the recovery that just ended
                mode, h0, d0 = self._recovery
                _, d1 = self._distance(obs)
                took = (h - h0) % 24
                if took > 0.25 and d0 > 0.5:
                    observed = max(self.OS_RATE_MIN, (d0 - d1) / took)
                    self._rate[mode] += self.OS_LEARN * (observed - self._rate[mode])
                self._recovery = None
            return True
        if not weekday or not (start - self.OS_MAX_LEAD_H <= h < start):
            self._recovery = None
            return False
        if self._recovery is not None:          # latched until occupancy
            return True
        mode, d = self._distance(obs)
        lead = min(self.OS_MAX_LEAD_H, max(self.OS_MIN_LEAD_H, d / self._rate[mode] + self.OS_MARGIN_H))
        if start - h <= lead:
            self._recovery = (mode, h, d)
            return True
        return False


class NightFlushPolicy(_TimeMixin, RefPolicy):
    """Pre-cool a warm building with outside air before a work day.

    In hot climates the building ends the night warm and the morning cooling
    runs on the chiller. When the outside air is clearly colder than the zones
    in the hours before occupancy, run each warm AHU on 100% outside air with
    the supply setpoint at its maximum (no mechanical cooling, fan only).
    """
    NF_HOURS = 6.0          # window before morning start
    NF_DELTA_K = 3.0        # zones at least this much warmer than outside
    NF_ABOVE_K = 0.5        # and this much above the comfort-band centre
    NF_PRESSURE = -0.5
    NF_DAMPER = 0.0         # normalized damper command (0 = mid)
    KEY = "night_flush"

    def _next_morning_is_workday(self, hour, day):
        d = day if hour < self.morning_start_hour else (day + 1) % 7
        return d < 5

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self._pi_active:
            return action
        hour, day = self._get_hour_and_day(obs)
        h = self._frac_hour(obs)
        lead = (self.morning_start_hour - h) % 24
        if lead > self.NF_HOURS or not self._next_morning_is_workday(hour, day):
            return action
        lo, hi = self.comfort_temp_window
        centre = 0.5 * (lo + hi)
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            idx = ctrl.get("oat_idx")
            if idx is None or not ctrl["zone_obs_indices"]:
                continue
            outside = _outside_kelvin(self, obs[idx])
            zone_mean = float(np.mean(self._zone_temps(ctrl, obs)))
            if zone_mean - outside < self.NF_DELTA_K or zone_mean < centre + self.NF_ABOVE_K:
                continue
            if disc[ctrl["run_discrete_idx"]] and cont[ctrl["temp_idx"]] < 0:
                continue  # already cooling mechanically under the base loop
            disc[ctrl["run_discrete_idx"]] = 1
            cont[ctrl["temp_idx"]] = np.float32(1.0)
            if ctrl["pressure_idx"] is not None:
                cont[ctrl["pressure_idx"]] = np.float32(self.NF_PRESSURE)
            if ctrl.get("recirc_idx") is not None:
                cont[ctrl["recirc_idx"]] = np.float32(-1.0)
            for _, _, ai in ctrl["zones"]:
                if ai.get("supply_air_damper_percentage_command") is not None:
                    cont[ai["supply_air_damper_percentage_command"]] = np.float32(self.NF_DAMPER)
                if ai.get("reheat_valve_percentage_command") is not None:
                    cont[ai["reheat_valve_percentage_command"]] = np.float32(-1.0)
        return action


class ComboPolicy(NightFlushPolicy, FanTrimRespondPolicy, OptimalStartPolicy):
    KEY = "combo"


class NudgedPolicy(RefPolicy):
    """Reference plus a physically negligible pressure offset: a noise probe."""
    NUDGE = 1e-5

    def get_action(self, obs):
        action = super().get_action(obs)
        cont = action["continuous_action"]
        for ctrl in self.ahu_controllers:
            j = ctrl.get("pressure_idx")
            if j is not None:
                cont[j] = np.float32(np.clip(cont[j] + self.NUDGE, -1.0, 1.0 - 1e-6))
        return action


def _baseline(name):
    from smart_buildings.baselines.bang_bang import BangBangPolicy
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
    return {"bang_bang": BangBangPolicy, "pid_water": PIDWaterResetPolicy}[name]


# name -> (class, default overrides)
REGISTRY = {
    "ref": (RefPolicy, {}),
    "evening_coast": (RefPolicy, {"EVENING_STOP_H": 1}),
    "occ_gate": (RefPolicy, {"OCC_GATE": True}),
    "heat_seq": (RefPolicy, {"HEAT_SEQUENCE_SPLIT": 0.5}),
    "sat_ramp": (RefPolicy, {"RAMP": 0.3}),
    "pi_leak": (RefPolicy, {"INTEGRAL_LEAK": 0.002}),
    "water_max": (RefPolicy, {"ALWAYS_WATER_MAX": True}),
    "fan_tr": (FanTrimRespondPolicy, {}),
    "optimal_start": (OptimalStartPolicy, {}),
    "night_flush": (NightFlushPolicy, {}),
    "combo": (ComboPolicy, {}),
}
CANDIDATES = [k for k in REGISTRY if k != "ref"]


def parse_spec(spec: str):
    name, _, rest = spec.partition(":")
    overrides = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        overrides[k.strip()] = ast.literal_eval(v.strip())
    return name.strip(), overrides


def make(spec: str, building_meta_data, action_spec):
    """Instantiate a policy from a spec string (see module docstring)."""
    name, overrides = parse_spec(spec)
    if name in ("bang_bang", "pid_water"):
        return _baseline(name)(building_meta_data, action_spec)
    if name.startswith("null_"):
        cls, base = NudgedPolicy, {"NUDGE": int(name.split("_")[1]) * 1e-5}
    elif name in REGISTRY:
        cls, base = REGISTRY[name]
    else:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(REGISTRY)}")
    attrs = {**base, **overrides}
    for k in attrs:
        if not hasattr(cls, k):
            raise AttributeError(f"{cls.__name__} has no attribute {k!r}")
    if attrs:
        cls = type(f"{cls.__name__}_custom", (cls,), attrs)
    return cls(building_meta_data, action_spec)
