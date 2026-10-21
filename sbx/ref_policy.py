"""Reference controller (public LB 0.01582) for offline evaluation.

Code copied verbatim from the Kaggle notebook "Smart Buildings Probe DS K07" by
Malyshev Danil (Apache-2.0), version run on 2026-10-21:
https://www.kaggle.com/code/malyshevdanil/smart-buildings-probe-ds-k07
It builds on the organizers' PIDWaterResetPolicy (Apache-2.0).

Only additions, below the copied code: `RefPolicy` carries the submitted
constants without the Kaggle inference-server base class, and `make_policy`
adds nudged "null" variants and the bang-bang baseline for the noise and
runtime checks.
"""
import numpy as np
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


class BaselinePolicy(PIDWaterResetPolicy):
    USE_SENSOR_BAND = True
    # Worst-zone aggregation increased loss in both full-length pilot scenarios.
    USE_WORST_ZONE = False

    @property
    def name(self):
        return "transfer"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        stats = self.observation_normalization_metadata.get(
            "outside_air_temperature_sensor", {}
        )
        self.outside_mean = float(stats.get("sample_mean", 0.0))
        self.outside_std = float(stats.get("sample_std", 1.0))

    def _recirculation(self, obs, ctrl, u, low, high):
        # The parent's economizer expects kelvin; the observation is standardized.
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        decoded = obs.copy()
        decoded[idx] = obs[idx] * self.outside_std + self.outside_mean
        return super()._recirculation(decoded, ctrl, u, low, high)

    def get_action(self, obs):
        action = super().get_action(obs)
        # Enforce the gateway's dtype, shape and bounds contract.
        for key, dtype in (("continuous_action", np.float32),
                           ("discrete_action", np.int32)):
            space = self.action_spec[key]
            action[key] = np.clip(action[key], space.low, space.high).astype(dtype)
        return action


class CoolingHoldPolicy(BaselinePolicy):
    HOLD_STEPS = 288
    # Outside work hours use (eco low, comfort high): no overnight drift up to
    # the eco ceiling in hot climates, full heating setback kept in cold ones.
    # Idea from the public "Neurips 2026 Smart Buildings Smartpolicy" notebook.
    NIGHT_COOL_CEILING = False

    @property
    def name(self):
        return f"cooling_hold_{self.HOLD_STEPS}"

    def _base_action(self, obs):
        if not self.NIGHT_COOL_CEILING:
            return super().get_action(obs)
        hour, day = self._get_hour_and_day(obs)
        if day < 5 and self.morning_start_hour <= hour < self.evening_start_hour:
            return super().get_action(obs)
        saved = self.eco_temp_window
        self.eco_temp_window = (saved[0], self.comfort_temp_window[1])
        try:
            return super().get_action(obs)
        finally:
            self.eco_temp_window = saved

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._on_steps = [0] * len(self.ahu_controllers)
        self._last_on = [None] * len(self.ahu_controllers)

    def get_action(self, obs):
        action = self._base_action(obs)
        cont, disc = action["continuous_action"], action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue
            run = ctrl["run_discrete_idx"]
            # Supply temperature first: its sign tells cooling from heating.
            idx = [j for j in (ctrl["temp_idx"], ctrl["pressure_idx"], ctrl.get("recirc_idx"))
                   if j is not None]
            if (not disc[run] and self._on_steps[i] > 0
                    and self._on_steps[i] < self.HOLD_STEPS and self._last_on[i][0] < 0):
                disc[run] = 1
                cont[idx] = self._last_on[i]
            if disc[run]:
                self._on_steps[i] += 1
                self._last_on[i] = cont[idx].copy()
            else:
                self._on_steps[i] = 0
        return action


class ZonePIPolicy(CoolingHoldPolicy):
    """v5 controller with per-zone PI control to the band centre in work hours.

    A soft proportional zone loop stops the 2-4 K oscillation but leaves zones
    sitting steadily outside the band (no integral action). Here each zone runs
    a PI loop on (temperature - band centre) from one hour before occupancy to
    the end of the work day. One signal u in [-1, 1] drives both actuators:
    the damper opens with |u| (air is needed to cool and to carry reheat) and
    the reheat valve opens with -u. AHUs are kept running in those hours so the
    zones have air to work with. Outside them, v5 logic is unchanged.
    """
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    KP = 0.2
    KI = 0.02
    PRESTART_H = 1
    # Hours before the end of the work day at which the PI loops stop (coast).
    EVENING_STOP_H = 0
    FORCED_SAT = -0.7       # normalized supply temp for a forced-on AHU (~15 C)
    FORCED_PRESSURE = -0.6  # normalized static pressure for a forced-on AHU
    KEY = "zone_pi"

    @property
    def name(self):
        return self.KEY

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._zone_integral = {}
        self._zone_u = {}
        self._pi_active = False

    def _pi_hours(self, obs):
        hour, day = self._get_hour_and_day(obs)
        return (day < 5 and self.morning_start_hour - self.PRESTART_H <= hour
                < self.evening_start_hour - self.EVENING_STOP_H)

    def _command_zones(self, continuous, zones, obs, low, high):
        if not self._pi_active:
            for vav_id, _, _ in zones:
                self._zone_integral[vav_id] = 0.0
            return super()._command_zones(continuous, zones, obs, low, high)
        lo_c, hi_c = self.comfort_temp_window
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, lo_c, hi_c)
            # Before occupancy the sensor band is still the eco band; aim at comfort.
            if hi - lo > (hi_c - lo_c) + 1e-6:
                lo, hi = lo_c, hi_c
            self._cur_vav = vav_id
            err = self._denormalize_zone_temp(obs[obs_idx]) - self._zone_target(obs, lo, hi)
            integral = self._zone_integral.get(vav_id, 0.0)
            u_raw = self.KP * err + self.KI * integral
            # Conditional integration: stop winding up while saturated.
            integral *= 1.0 - self.INTEGRAL_LEAK
            if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                limit = self.INTEGRAL_LIMIT_FRAC / self.KI
                integral = float(np.clip(integral + err, -limit, limit))
            self._zone_integral[vav_id] = integral
            u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
            if u < 0:
                u *= self.HEAT_U_SCALE
            self._max_heat = max(self._max_heat, -u)
            self._zone_u[vav_id] = u
            gain = self.HEAT_DAMPER_GAIN if u < 0 else 1.0
            damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u) * gain
            reheat = max(0.0, -u)
            if self.REHEAT_FULL_AT is not None and u < 0:
                # Hottest possible discharge for the air that is moved; the
                # damper alone sets how much heat reaches the zone.
                reheat = min(1.0, -u / self.REHEAT_FULL_AT)
            split = self.HEAT_SEQUENCE_SPLIT
            if split is not None and u < 0:
                # Reheat first at minimum air; add air only once reheat is full.
                reheat = min(1.0, -u / split)
                damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * max(0.0, (-u - split) / (1.0 - split))
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 2.0 * reheat - 1.0

    MIN_DAMPER = 0.1
    # Damper opening per unit of heating demand (reheated air costs gas).
    HEAT_DAMPER_GAIN = 1.0
    # Anti-windup: integral term limited to this share of full output, and a
    # per-step leak towards zero (both neutral by default).
    INTEGRAL_LIMIT_FRAC = 1.0
    INTEGRAL_LEAK = 0.0
    # Heating sequence: None = damper and reheat both follow |u|; a number s =
    # reheat opens fully by |u| = s at minimum air, then the damper opens.
    HEAT_SEQUENCE_SPLIT = None
    # Reheat valve fully open once heating demand reaches this |u| (None =
    # valve follows |u|): each unit of cold mixed air arrives as hot as allowed.
    REHEAT_FULL_AT = None
    # Scale of the PI output when heating (softer loop for hot-discharge heating).
    HEAT_U_SCALE = 1.0

    def _zone_target(self, obs, lo, hi):
        """Temperature each zone's PI loop aims at; the band centre by default."""
        return 0.5 * (lo + hi)

    def _base_action(self, obs):
        self._pi_active = self._pi_hours(obs)
        self._max_heat = 0.0
        self._forced = []
        self._zone_u = {}
        action = super()._base_action(obs)
        if self._pi_active:
            cont, disc = action["continuous_action"], action["discrete_action"]
            # Zone reheat needs hot water even when no AHU loop asks for heat.
            if self.water_temp_idx is not None and self._max_heat > 0:
                cont[self.water_temp_idx] = max(cont[self.water_temp_idx],
                                                np.float32(2.0 * self._max_heat - 1.0))
                if self.water_pressure_idx is not None:
                    cont[self.water_pressure_idx] = 1.0
            for i, ctrl in enumerate(self.ahu_controllers):
                if ctrl["zone_obs_indices"] and not disc[ctrl["run_discrete_idx"]]:
                    self._forced.append(i)
                    disc[ctrl["run_discrete_idx"]] = 1
                    cont[ctrl["temp_idx"]] = self.FORCED_SAT
                    if ctrl["pressure_idx"] is not None:
                        cont[ctrl["pressure_idx"]] = self.FORCED_PRESSURE
            if 0 <= self.water_discrete_idx < self.n_discrete:
                disc[self.water_discrete_idx] = 1
        return action

    def get_action(self, obs):
        action = super().get_action(obs)
        # A forced-on unit is not a cooling run; do not let the hold extend it.
        for i in self._forced:
            if self._last_on[i] is not None:
                self._last_on[i][0] = 0.0
        return action


def _outside_kelvin(policy, value):
    """AHU outside-air sensors arrive in raw kelvin; decode only if normalized."""
    value = float(value)
    return value if value > 150.0 else value * policy.outside_std + policy.outside_mean


class ZonePIEconFixPolicy(ZonePIPolicy):
    """pe_lowp with a working economizer (raw-kelvin outside air temperature).

    TransferPolicy decoded the AHU outside-air sensor as if normalized, but it
    arrives in kelvin, so the economizer saw ~3000 K outside and never opened.
    """
    KP = 0.10
    KI = 0.005
    FORCED_PRESSURE = -1.0
    WARM_SAT = None
    WARM_BELOW_K = 283.15
    KEY = "econ_fix"

    def _recirculation(self, obs, ctrl, u, low, high):
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        fixed = obs.copy()
        fixed[idx] = _outside_kelvin(self, obs[idx])
        # Skip TransferPolicy's decoding; call the organizer economizer directly.
        return super(BaselinePolicy, self)._recirculation(fixed, ctrl, u, low, high)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.WARM_SAT is None:
            return action
        cont = action["continuous_action"]
        for i in self._forced:
            ctrl = self.ahu_controllers[i]
            idx = ctrl.get("oat_idx")
            if idx is not None and _outside_kelvin(self, obs[idx]) < self.WARM_BELOW_K:
                cont[ctrl["temp_idx"]] = self.WARM_SAT
        return action


class DemandSatPolicy(ZonePIEconFixPolicy):
    """Supply-air temperature follows the zones' own PI demand in work hours.

    Forced-on AHUs used a fixed ~15 C supply. In cool weather that chills an
    already cool mix, then zone reheat burns gas to warm it back. Here each
    running AHU looks at the mean PI output of its zones (positive = cooling):
    with net heating demand the supply setpoint goes to maximum (no chiller);
    with cooling demand it ramps down to ~15 C, reached at mean demand RAMP
    (RAMP 0 = switch at zero demand).
    """
    RAMP = 0.0
    COLD_SAT = -0.7
    KEY = "dsat_bin"

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._pi_active:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            us = [self._zone_u[v] for v, _, _ in ctrl["zones"] if v in self._zone_u]
            if not us or not disc[ctrl["run_discrete_idx"]]:
                continue
            d = float(np.mean(us))
            frac = (float(np.clip(d / self.RAMP, 0.0, 1.0)) if self.RAMP > 0
                    else (1.0 if d > 0 else 0.0))
            cont[ctrl["temp_idx"]] = np.float32(1.0 + frac * (self.COLD_SAT - 1.0))
        return action


class TunedPIPolicy(ZonePIEconFixPolicy):
    """econ_fix_warmcold plus switchable energy measures, all off by default.

    NIGHT_HEAT_DROP_K: lower the heating setpoint outside PI hours.
    WATER_SCALE: scale the hot-water temperature command (0..1 of range).
    TARGET_SHIFT_K: aim zones this far from the band centre towards the side
        the weather pushes them to (cheaper to hold).
    OCC_GATE: after the arrival hour, run PI hours only while the building
        reports occupants (skips holidays and empty days).
    """
    WARM_SAT = -0.4
    NIGHT_HEAT_DROP_K = 0.0
    WATER_SCALE = 1.0
    TARGET_SHIFT_K = 0.0
    OCC_GATE = False
    # Shift the target towards the low edge only in heating weather.
    COLD_SHIFT_ONLY = False
    # In PI hours, run the hot-water loop at full temperature whenever any
    # zone calls for heat (more heat per unit of cold mixed air).
    WATER_MAX_PI = False
    KEY = "tuned"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        names = list(self.field_names)
        self._occ_idx = names.index("num_occupants") if "num_occupants" in names else None
        self._oat_any = next((c["oat_idx"] for c in self.ahu_controllers
                              if c.get("oat_idx") is not None), None)

    def _pi_hours(self, obs):
        active = super()._pi_hours(obs)
        if active and self.OCC_GATE and self._occ_idx is not None:
            hour, _ = self._get_hour_and_day(obs)
            if hour >= self.morning_start_hour + 2 and float(obs[self._occ_idx]) < 0.5:
                return False
        return active

    def _zone_target(self, obs, lo, hi):
        centre = 0.5 * (lo + hi)
        if self.TARGET_SHIFT_K <= 0 or self._oat_any is None:
            return centre
        outside = _outside_kelvin(self, obs[self._oat_any])
        shift = min(self.TARGET_SHIFT_K, 0.4 * (hi - lo))
        if outside < centre:
            return centre - shift
        return centre if self.COLD_SHIFT_ONLY else centre + shift

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        if self.NIGHT_HEAT_DROP_K and not self._pi_active:
            lo -= self.NIGHT_HEAT_DROP_K
        return lo, hi

    def _ahu_error(self, obs, ctrl, low, high):
        if self.NIGHT_HEAT_DROP_K and not self._pi_active:
            low -= self.NIGHT_HEAT_DROP_K
        return super()._ahu_error(obs, ctrl, low, high)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if (self.WATER_MAX_PI and self._pi_active and self.water_temp_idx is not None
                and self._max_heat > 0):
            action["continuous_action"][self.water_temp_idx] = np.float32(1.0)
        if self.WATER_SCALE != 1.0 and self.water_temp_idx is not None:
            cont = action["continuous_action"]
            frac = (float(cont[self.water_temp_idx]) + 1.0) * 0.5 * self.WATER_SCALE
            cont[self.water_temp_idx] = np.float32(2.0 * frac - 1.0)
        return action


class TunedDemandSatPolicy(TunedPIPolicy, DemandSatPolicy):
    """dsat_bin with the TunedPIPolicy switches (all off by default)."""
    WARM_SAT = None  # dsat_bin did not use the cold-weather warm supply
    KEY = "tuned_dsat"


class AdaptiveColdPolicy(TunedDemandSatPolicy):
    """Pick the heating measures from a smoothed outside temperature.

    Screening showed each cold-weather measure helps one regime and hurts
    another: an early start and full-temperature water with less heating air
    pay off only in deep cold (mean -21 C), while aiming 0.5 K below the band
    centre pays off in moderate cold but hurts deep cold. The regime follows an
    exponential average of outside air temperature over about a day.
    """
    NIGHT_HEAT_DROP_K = 2.0
    PRESTART_H = 2
    NIGHT_COOL_CEILING = False
    EXTREME_BELOW_C = -15.0
    MILD_BELOW_C = 12.0
    EXTREME_PRESTART_H = 4
    EXTREME_WATER_MAX = True
    EXTREME_HEAT_DAMPER = 0.5
    MILD_SHIFT_K = 0.5
    EMA_STEPS = 288
    # Full hot-water temperature in PI hours whenever a zone heats, any weather.
    ALWAYS_WATER_MAX = False
    KEY = "adaptive_cold"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._oat_ema = None
        self._base_prestart = self.PRESTART_H

    def _update_regime(self, obs):
        if self._oat_any is not None:
            outside = _outside_kelvin(self, obs[self._oat_any]) - 273.15
            if self._oat_ema is None:
                self._oat_ema = outside
            else:
                self._oat_ema += (outside - self._oat_ema) / self.EMA_STEPS
        ema = self._oat_ema
        extreme = ema is not None and ema < self.EXTREME_BELOW_C
        mild = ema is not None and not extreme and ema < self.MILD_BELOW_C
        self.PRESTART_H = self.EXTREME_PRESTART_H if extreme else self._base_prestart
        self.WATER_MAX_PI = bool((extreme and self.EXTREME_WATER_MAX) or self.ALWAYS_WATER_MAX)
        self.HEAT_DAMPER_GAIN = self.EXTREME_HEAT_DAMPER if extreme else 1.0
        self.TARGET_SHIFT_K = self.MILD_SHIFT_K if mild else 0.0
        self.COLD_SHIFT_ONLY = True

    def _base_action(self, obs):
        self._update_regime(obs)
        return super()._base_action(obs)


# ---------------------------------------------------------------------------
# Additions for offline evaluation (not part of the reference notebook).
# ---------------------------------------------------------------------------

class RefPolicy(AdaptiveColdPolicy):
    """SubmissionPolicy of the reference notebook, minus the server base class."""
    KP = 0.1
    KI = 0.005
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = False
    EXTREME_WATER_MAX = False
    EXTREME_HEAT_DAMPER = 1.0
    KEY = "ref"


class NudgedRefPolicy(RefPolicy):
    """RefPolicy with a negligible offset on every AHU static-pressure command.

    1e-5 in normalized units is about 0.01 Pa: physically nothing, but enough
    to flip a later switching decision. The spread of these runs is the noise
    floor a real change has to clear.
    """
    NUDGE = 0.0

    def get_action(self, obs):
        action = super().get_action(obs)
        cont = action["continuous_action"]
        for ctrl in self.ahu_controllers:
            j = ctrl.get("pressure_idx")
            if j is not None:
                cont[j] = np.float32(np.clip(cont[j] + self.NUDGE, -1.0, 1.0 - 1e-6))
        return action


def make_policy(variant, building_meta_data, action_spec):
    """'ref', 'null_<k>' (pressure nudge k * 1e-5) or 'bang_bang'."""
    if variant == "ref":
        return RefPolicy(building_meta_data, action_spec)
    if variant.startswith("null_"):
        k = int(variant.split("_")[1])
        cls = type(f"Null{k}Policy", (NudgedRefPolicy,), {"NUDGE": k * 1e-5})
        return cls(building_meta_data, action_spec)
    if variant == "bang_bang":
        from smart_buildings.baselines.bang_bang import BangBangPolicy
        return BangBangPolicy(building_meta_data, action_spec)
    raise ValueError(f"unknown variant: {variant}")
