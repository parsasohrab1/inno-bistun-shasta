"""Advanced Process Control (APC) -- economic MPC for energy/quality optimisation.

This is the *energy* half of the Bistun "Integrated Quality & Energy Control"
product.  It closes the loop opened by the machine-vision detector:

    camera frame  ->  detector (defect_rate, quality)  ->  APC controller
        -> optimal setpoints that hold quality within spec at minimum cost
        ->  every defective tonne NOT produced = energy + emissions NOT spent

Process model (linear; coefficients illustrative of a kraft/printing-paper
machine, see ``DEFAULTS``).  Controllable setpoints:

    speed          (m/min)     -- wire/drive speed  -> throughput
    steam_pressure (bar)       -- dryer-section steam -> drying capacity (moisture)
    chem_dosage    (kg/t)      -- retention/formation aid dosage

Measured disturbance:
    consistency    (%)          -- stock consistency (from stock prep)

Quality model (predicted defect fraction):
    dr = k0 + k_speed*(speed-nom) - k_steam*(steam-nom) - k_chem*(chem-nom)
           - k_consist*(consist-nom)
    => faster = more defects; more drying / chemical / consistency = fewer.
    Steam is an *effective* quality lever (moisture control) and, in energy
    terms, a *cheap* one (pressure is a setpoint, not proportional fuel use),
    so the controller genuinely dries-to-spec rather than over-drying.

Energy model (GJ per tonne of gross production):
    E = e0 + c_steam*steam + c_chem*chem
    e0 is the (constant) base drying + electrical load; steam pressure and
    chemical dosage add small controllable terms.  Speed is not in E because
    specific energy per tonne is roughly speed-independent.

Emissions (kgCO2/t) = e0*ef_base + c_steam*steam*ef_steam + c_chem*chem*ef_chem.

Controller (economic MPC, per step):  solve the LP
    minimise   c_steam*steam + c_chem*chem  -  v_thr*speed     (energy - throughput value)
    s.t.       dr(x) <= dr_target                            (quality spec)
               x_lo <= x <= x_hi                             (actuator limits)
               |x - x_prev| <= rate_max                      (move suppression, soft)
The negative throughput term makes the controller prefer higher speed
(production) while the quality constraint forces it to spend steam/chemical
only as much as needed -- i.e. operate at the quality boundary, maximising
good-tonne production at minimum specific energy.

Savings headline (always >= 0, matches the product logic):
    defective tonnes avoided/h = (dr_cur - dr_opt) * throughput
    energy avoided/h           = tonnes_avoided * E_cur
    CO2 avoided/h              = tonnes_avoided * CO2_cur
A secondary specific-energy optimisation term (E_cur - E_opt) is reported too.

This is a deliberate, auditable linear MPC -- fast (ms), deterministic, no
neural network in the control path (appropriate for QA sign-off at TRL 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Calibrated model parameters (kraft/printing-paper machine, illustrative)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    # --- quality model (defect fraction) ---
    "k0":        0.020,
    "k_speed":   0.00070,   # +defect per m/min above nominal
    "k_steam":   0.01800,   # -defect per bar  (drying -> moisture -> quality)
    "k_chem":    0.00800,   # -defect per kg/t  (retention/formation aid)
    "k_consist": 0.01000,   # -defect per % consistency
    # --- energy model (GJ/t gross) ---
    "e0":        1.600,     # base drying + electrical load (constant per tonne)
    "c_steam":   0.060,     # small controllable term per bar (pressure setpoint)
    "c_chem":    0.030,     # embodied energy per kg/t
    # --- economic objective weight ---
    "v_thr":     0.0035,    # throughput value per m/min (production proxy)
    # --- emission factors (kgCO2 per GJ of each carrier) ---
    "ef_base":   42.0,      # grid+gas blend for the base load
    "ef_steam":  56.0,      # natural-gas boiler
    "ef_chem":   40.0,      # embodied emissions of retention aid
    # --- nominal operating point ---
    "nom_speed": 700.0, "nom_steam": 4.5, "nom_chem": 2.0, "nom_consist": 4.5,
    # --- actuator bounds ---
    "speed_lo": 500.0,  "speed_hi": 900.0,
    "steam_lo": 2.5,    "steam_hi": 7.0,
    "chem_lo":  0.5,    "chem_hi": 4.0,
    # --- move suppression (max change per step) ---
    "rate_speed": 60.0, "rate_steam": 0.8, "rate_chem": 0.6,
    # --- quality spec ---
    "dr_target": 0.030,   # max acceptable defect fraction (3%)
    "dr_min":    0.002,   # floor -- don't over-control
    # --- production ---
    "throughput_tonnes_per_h": 18.0,
}


@dataclass
class ProcessState:
    """Snapshot of the process + the latest vision result."""
    speed: float
    steam_pressure: float
    chem_dosage: float
    consistency: float
    defect_rate: float          # from detector (defect_area_ratio or classifier)
    quality_score: float        # from detector (0..1)
    moisture: float = 0.0       # optional measured disturbance (%)

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class ControlAction:
    setpoints: dict[str, float]            # new recommended setpoints
    deltas: dict[str, float]               # change vs current
    predicted_defect_rate: float
    predicted_energy_gj_per_t: float
    predicted_co2_kg_per_t: float
    current_energy_gj_per_t: float
    current_co2_kg_per_t: float
    energy_saving_gj_per_t: float          # specific-energy optimisation (can be +/-)
    co2_saving_kg_per_t: float
    defect_rate_reduction: float
    tonnes_saved_per_h: float              # defective tonnes avoided / h (headline)
    energy_saved_gj_per_h: float           # energy avoided / h (headline)
    co2_saved_kg_per_h: float              # CO2 avoided / h (headline)
    in_spec: bool
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, (int, float)):
                out[k] = round(float(v), 5)
            else:
                out[k] = v
        return out


class APCController:
    """Economic linear MPC for the paper-machine quality/energy loop."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.p = {**DEFAULTS, **(params or {})}

    # -- model primitives ----------------------------------------------------
    def defect_rate(self, speed: float, steam: float, chem: float, consist: float) -> float:
        p = self.p
        dr = (p["k0"]
              + p["k_speed"] * (speed - p["nom_speed"])
              - p["k_steam"] * (steam - p["nom_steam"])
              - p["k_chem"] * (chem - p["nom_chem"])
              - p["k_consist"] * (consist - p["nom_consist"]))
        return float(max(p["dr_min"], dr))

    def energy(self, speed: float, steam: float, chem: float) -> float:
        p = self.p
        return float(p["e0"] + p["c_steam"] * steam + p["c_chem"] * chem)

    def co2(self, speed: float, steam: float, chem: float) -> float:
        p = self.p
        return float(p["e0"] * p["ef_base"]
                     + p["c_steam"] * steam * p["ef_steam"]
                     + p["c_chem"] * chem * p["ef_chem"])

    # -- control step (LP) ---------------------------------------------------
    def step(self, state: ProcessState, suppress_moves: bool = True) -> ControlAction:
        p = self.p
        x_prev = np.array([state.speed, state.steam_pressure, state.chem_dosage], dtype=float)

        # objective: minimise controllable energy - throughput value
        #   c . x = -v_thr*speed + c_steam*steam + c_chem*chem
        c_obj = np.array([-p["v_thr"], p["c_steam"], p["c_chem"]])

        # quality constraint:  k_speed*speed - k_steam*steam - k_chem*chem <= rhs
        rhs = (p["dr_target"] - p["k0"]
               + p["k_speed"] * p["nom_speed"]
               - p["k_steam"] * p["nom_steam"]
               - p["k_chem"] * p["nom_chem"]
               - p["k_consist"] * (state.consistency - p["nom_consist"]))
        A_ub = np.array([[p["k_speed"], -p["k_steam"], -p["k_chem"]]])
        b_ub = np.array([rhs])

        bounds = [
            (p["speed_lo"], p["speed_hi"]),
            (p["steam_lo"], p["steam_hi"]),
            (p["chem_lo"], p["chem_hi"]),
        ]
        if suppress_moves:
            bounds = [
                (max(p["speed_lo"], x_prev[0] - p["rate_speed"]),
                 min(p["speed_hi"], x_prev[0] + p["rate_speed"])),
                (max(p["steam_lo"], x_prev[1] - p["rate_steam"]),
                 min(p["steam_hi"], x_prev[1] + p["rate_steam"])),
                (max(p["chem_lo"], x_prev[2] - p["rate_chem"]),
                 min(p["chem_hi"], x_prev[2] + p["rate_chem"])),
            ]

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success and suppress_moves:
            # move suppression made the quality spec unreachable in one step --
            # it is a soft preference, not a hard limit. Retry over the full
            # actuator range so the controller always finds a feasible setpoint.
            full_bounds = [
                (p["speed_lo"], p["speed_hi"]),
                (p["steam_lo"], p["steam_hi"]),
                (p["chem_lo"], p["chem_hi"]),
            ]
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=full_bounds, method="highs")
        x_opt = res.x if res.success else x_prev.copy()
        speed_o, steam_o, chem_o = float(x_opt[0]), float(x_opt[1]), float(x_opt[2])

        dr_cur = self.defect_rate(state.speed, state.steam_pressure, state.chem_dosage, state.consistency)
        dr_opt = self.defect_rate(speed_o, steam_o, chem_o, state.consistency)
        e_cur = self.energy(state.speed, state.steam_pressure, state.chem_dosage)
        e_opt = self.energy(speed_o, steam_o, chem_o)
        co2_cur = self.co2(state.speed, state.steam_pressure, state.chem_dosage)
        co2_opt = self.co2(speed_o, steam_o, chem_o)

        # --- savings (headline = defective tonnes avoided) ---
        tph = p["throughput_tonnes_per_h"]
        dr_red = max(0.0, dr_cur - dr_opt)
        tonnes_saved_per_h = dr_red * tph
        # energy/CO2 embodied in the defective tonnes that will no longer be produced
        energy_saved_gj_per_h = tonnes_saved_per_h * e_cur
        co2_saved_kg_per_h = tonnes_saved_per_h * co2_cur

        rationale = self._rationale(state, speed_o, steam_o, chem_o, dr_cur, dr_opt, res.success)

        return ControlAction(
            setpoints={"speed": round(speed_o, 2), "steam_pressure": round(steam_o, 3),
                       "chem_dosage": round(chem_o, 3)},
            deltas={"speed": round(speed_o - state.speed, 2),
                    "steam_pressure": round(steam_o - state.steam_pressure, 3),
                    "chem_dosage": round(chem_o - state.chem_dosage, 3)},
            predicted_defect_rate=round(dr_opt, 5),
            predicted_energy_gj_per_t=round(e_opt, 4),
            predicted_co2_kg_per_t=round(co2_opt, 2),
            current_energy_gj_per_t=round(e_cur, 4),
            current_co2_kg_per_t=round(co2_cur, 2),
            energy_saving_gj_per_t=round(e_cur - e_opt, 4),
            co2_saving_kg_per_t=round(co2_cur - co2_opt, 2),
            defect_rate_reduction=round(dr_red, 5),
            tonnes_saved_per_h=round(tonnes_saved_per_h, 4),
            energy_saved_gj_per_h=round(energy_saved_gj_per_h, 3),
            co2_saved_kg_per_h=round(co2_saved_kg_per_h, 2),
            in_spec=bool(dr_opt <= p["dr_target"]),
            rationale=rationale,
        )

    # -- helpers -------------------------------------------------------------
    def _rationale(self, state, sp, st, ch, dr_cur, dr_opt, ok) -> list[str]:
        msgs: list[str] = []
        p = self.p
        if dr_cur > p["dr_target"]:
            msgs.append(f"Defect rate {dr_cur:.1%} exceeds target {p['dr_target']:.1%}; APC retuning setpoints.")
        if sp < state.speed - 1:
            msgs.append(f"Reduce speed {state.speed:.0f}->{sp:.0f} m/min (trade throughput for quality).")
        elif sp > state.speed + 1:
            msgs.append(f"Increase speed {state.speed:.0f}->{sp:.0f} m/min (quality headroom -> more production).")
        if st > state.steam_pressure + 0.05:
            msgs.append(f"Raise steam {state.steam_pressure:.2f}->{st:.2f} bar (dry to spec, cut defects).")
        elif st < state.steam_pressure - 0.05:
            msgs.append(f"Cut steam {state.steam_pressure:.2f}->{st:.2f} bar (quality met -> save drying energy).")
        if ch > state.chem_dosage + 0.02:
            msgs.append(f"Increase retention aid {state.chem_dosage:.2f}->{ch:.2f} kg/t (better formation).")
        elif ch < state.chem_dosage - 0.02:
            msgs.append(f"Cut retention aid {state.chem_dosage:.2f}->{ch:.2f} kg/t (quality met -> save chemical).")
        if not ok:
            msgs.append("LP infeasible within move limits; holding current setpoints.")
        if dr_opt <= p["dr_target"] and dr_cur <= p["dr_target"]:
            msgs.append("In spec; maximising throughput at minimum specific energy.")
        msgs.append(f"Predicted: defect {dr_opt:.2%}, energy {self.energy(sp,st,ch):.2f} GJ/t, CO2 {self.co2(sp,st,ch):.0f} kg/t.")
        return msgs
