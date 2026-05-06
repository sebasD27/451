from __future__ import annotations

import math
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
G = 9.80665
ULTIMATE_LOAD_FACTOR_MARGIN = 1.5
TURN_LOAD_FACTOR_MARGIN = 0.95
RHO0 = 1.225
HP_TO_W = 745.7
MPS_TO_KT = 1.94384
KT_TO_MPS = 1.0 / MPS_TO_KT
M_TO_NM = 1.0 / 1852.0
KG_TO_LB = 2.20462262185
LB_TO_KG = 1.0 / KG_TO_LB
M2_TO_FT2 = 10.7639104167
M_TO_FT = 3.280839895
PA_TO_PSF = 0.020885434273
AVGAS_DENSITY_KG_M3 = 720.0


# -----------------------------------------------------------------------------
# Aircraft definition
# -----------------------------------------------------------------------------
@dataclass
class PiperCherokeeLike:
    """Simple Cherokee-like single engine piston concept.

    Geometry is kept close to PA-28/Archer territory, but the engine can be
    increased above stock to meet the 175 kt dash requirement.
    """

    name: str = "Piper Archer / Cherokee-like PA-28-181 concept"

    # Wing / aero
    S: float = 12.8
    b: float = 10.8
    taper: float = 0.45
    e: float = 0.80
    t_c_wing: float = 0.12
    CD0: float = 0.026
    CL0: float = 0.20
    CLa: float = 5.0
    CLmax_clean: float = 1.55
    CLmax_takeoff: float = 2.20
    CLmax_landing: float = 2.60

    CLmin_clean: float = -1.10
    CLmin_takeoff: float = -0.85
    CLmin_landing: float = -0.65

    # Empennage / fuselage geometry for Raymer weight build-up
    htail_area: float = 2.6
    htail_AR: float = 4.0
    htail_taper: float = 0.60
    htail_t_c: float = 0.12

    vtail_area: float = 1.5
    vtail_AR: float = 1.8
    vtail_taper: float = 0.70
    vtail_t_c: float = 0.10

    fuselage_length: float = 6.5
    fuselage_width: float = 1.2
    fuselage_height: float = 1.4
    tail_arm: float = 4.4
    wing_x: float = 2.2  # m from nose, approximate wing aerodynamic center location

    # Landing gear lengths (height)
    main_gear_length: float = 0.5
    nose_gear_length: float = 0.5


    # Meters
    tail_clearance_height: float = 1.25
    x_main_gear: float           = 3
    x_nose_gear: float           = 0.2
    gear_track: float            = 2.8

    @property
    def cg_height(self) -> float:
        return 0.33 * self.fuselage_height + self.main_gear_length

    @property
    def mean_chord(self) -> float:
        return self.S / self.b

    wing_box_depth: float = 0.18
    wing_spar_thickness: float = 0.035
    wing_spar_caps: int = 2

    # Payload / propulsion
    n_engines: int = 1
    n_crew: int = 1
    n_pax: int = 3
    m_payload: float = 250.0
    m_fuel_max: float = 131.0
    engine_hp: float = 260.0
    prop_eff: float = 0.84
    static_thrust_max: float = 2600.0
    bsfc_kg_per_kwh: float = 0.285

    @property
    def flap_settings(self) -> Dict[str, Dict[str, float]]:
        return {
            "clean": {
                "delta_CL0": 0.00,
                "delta_CD0": 0.000,
                "CLmax": self.CLmax_clean,
                "CLmin": self.CLmin_clean,
            },
            "takeoff": {
                "delta_CL0": 0.35,
                "delta_CD0": 0.015,
                "CLmax": self.CLmax_takeoff,
                "CLmin": self.CLmin_takeoff,
            },
            "landing": {
                "delta_CL0": 0.70,
                "delta_CD0": 0.060,
                "CLmax": self.CLmax_landing,
                "CLmin": self.CLmin_landing,
            },
        }
    

    # Raymer systems inputs
    avionics_uninstalled: float = 25.0
    n_tanks: int = 2
    integral_tank_fraction: float = 0.80
    L_D_cruise: float = 11.5
    n_ult: float = 5.7

    
    
    # Mission settings
    cruise_alt_m: float = 2500.0
    cruise_speed_kt: float = 128.0
    dash_speed_req_kt: float = 175.0
    dash_check_alt_m: float = 0.0  # use sea level unless your professor says otherwise

    @property
    def AR(self) -> float:
        return self.b ** 2 / self.S

    @property
    def k(self) -> float:
        return 1.0 / (math.pi * self.e * self.AR)

    @property
    def n_limit(self) -> float:
        return self.n_ult / ULTIMATE_LOAD_FACTOR_MARGIN

    @property
    def max_turn_n(self) -> float:
        return TURN_LOAD_FACTOR_MARGIN * self.n_limit

    @property
    def max_turn_bank_deg(self) -> float:
        return math.degrees(math.acos(1.0 / self.max_turn_n))

    @property
    def power_max_w(self) -> float:
        return self.engine_hp * HP_TO_W

    @property
    def engine_dry_mass_kg(self) -> float:
        # Simple piston-engine estimate used as the input to Raymer's installed
        # engine correlation. It keeps the engine mass increasing with hp instead
        # of pretending the heavier engine weighs nothing, which would be comedy.
        return 0.62 * self.engine_hp + 18.0

    @property
    def fuselage_wetted_area_m2(self) -> float:
        return math.pi * 0.5 * (self.fuselage_width + self.fuselage_height) * self.fuselage_length * 0.92


# -----------------------------------------------------------------------------
# Atmosphere / aero / propulsion
# -----------------------------------------------------------------------------
def atmosphere(h_m: float) -> Tuple[float, float, float]:
    h = max(0.0, float(h_m))
    if h <= 11000.0:
        T = 288.15 - 0.0065 * h
        p = 101325.0 * (T / 288.15) ** (G / (287.05 * 0.0065))
        rho = p / (287.05 * T)
    else:
        T = 216.65
        p11 = 22632.1
        p = p11 * math.exp(-G * (h - 11000.0) / (287.05 * T))
        rho = p / (287.05 * T)
    a = math.sqrt(1.4 * 287.05 * T)
    sigma = rho / RHO0
    return rho, sigma, a


def _section_lift_coefficient(ac: PiperCherokeeLike, alpha_eff_rad: float, flap_data: Dict[str, float]) -> float:
    CL_linear = ac.CL0 + flap_data["delta_CL0"] + ac.CLa * alpha_eff_rad
    CLmax = flap_data["CLmax"]
    CLmin = flap_data["CLmin"]

    if CL_linear > CLmax:
        excess = CL_linear - CLmax
        return CLmax - 0.12 * (1.0 - math.exp(-excess / 0.35))
    if CL_linear < CLmin:
        excess = CLmin - CL_linear
        return CLmin + 0.10 * (1.0 - math.exp(-excess / 0.35))
    return CL_linear


def _profile_drag_coefficient(ac: PiperCherokeeLike, CL: float, alpha_eff_rad: float, flap_data: Dict[str, float]) -> float:
    CL_linear = ac.CL0 + flap_data["delta_CL0"] + ac.CLa * alpha_eff_rad
    stall_excess = max(CL_linear - flap_data["CLmax"], flap_data["CLmin"] - CL_linear, 0.0)
    lift_offset = CL - (ac.CL0 + flap_data["delta_CL0"])
    return ac.CD0 + flap_data["delta_CD0"] + 0.002 * lift_offset ** 2 + 0.080 * stall_excess ** 2


def aero_coeffs(ac: PiperCherokeeLike, alpha_rad: float, flap: str = "clean") -> Tuple[float, float]:
    if flap not in ac.flap_settings:
        raise ValueError(f"Unknown flap setting '{flap}'. Use one of {list(ac.flap_settings)}.")

    flap_data = ac.flap_settings[flap]

    # Finite-wing correction using induced angle of attack. The section lift
    # curve is nonlinear near stall, so use a relaxed fixed-point solve.
    CL = (ac.CL0 + flap_data["delta_CL0"] + ac.CLa * alpha_rad) / (1.0 + ac.CLa * ac.k)
    for _ in range(16):
        alpha_eff = alpha_rad - ac.k * CL
        CL_new = _section_lift_coefficient(ac, alpha_eff, flap_data)
        CL = 0.55 * CL + 0.45 * CL_new

    alpha_eff = alpha_rad - ac.k * CL
    CD_profile = _profile_drag_coefficient(ac, CL, alpha_eff, flap_data)
    CD = CD_profile + ac.k * CL ** 2
    return CL, CD


def alpha_for_lift_coefficient(ac: PiperCherokeeLike, CL: float, flap: str = "clean") -> float:
    if flap not in ac.flap_settings:
        raise ValueError(f"Unknown flap setting '{flap}'. Use one of {list(ac.flap_settings)}.")

    flap_data = ac.flap_settings[flap]
    alpha_stall = (flap_data["CLmax"] - ac.CL0 - flap_data["delta_CL0"]) / ac.CLa + ac.k * flap_data["CLmax"]
    lo = math.radians(-12.0)
    hi = max(alpha_stall, lo + math.radians(1.0))

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        CL_mid, _ = aero_coeffs(ac, mid, flap)
        if CL_mid < CL:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def lift_drag(ac: PiperCherokeeLike, V_mps: float, alpha_rad: float, h_m: float, flap: str = "clean") -> Tuple[float, float, float, float]:
    rho, _, _ = atmosphere(h_m)
    q = 0.5 * rho * V_mps ** 2
    CL, CD = aero_coeffs(ac, alpha_rad, flap)
    L = q * ac.S * CL
    D = q * ac.S * CD
    return L, D, CL, CD


def power_available_w(ac: PiperCherokeeLike, throttle: float, h_m: float) -> float:
    _, sigma, _ = atmosphere(h_m)
    throttle = float(np.clip(throttle, 0.0, 1.0))
    return throttle * ac.power_max_w * sigma ** 0.90


def thrust_available_n(ac: PiperCherokeeLike, throttle: float, V_mps: float, h_m: float) -> float:
    _, sigma, _ = atmosphere(h_m)
    V_eff = max(20.0, V_mps)
    thrust_from_power = ac.prop_eff * power_available_w(ac, throttle, h_m) / V_eff
    thrust_static_cap = ac.static_thrust_max * sigma ** 0.80
    return min(thrust_from_power, thrust_static_cap)


def fuel_burn_rate_kg_s(ac: PiperCherokeeLike, throttle: float, h_m: float) -> float:
    power_kw = power_available_w(ac, throttle, h_m) / 1000.0
    return ac.bsfc_kg_per_kwh * power_kw / 3600.0


# -----------------------------------------------------------------------------
# Raymer weight build-up
#   Main structure and systems use Raymer-style empirical correlations.
#   The landing-gear terms are taken from the Raymer equations the user supplied
#   in the screenshots because they behave more sensibly here than the odd GA
#   port that divides the gear length by 12.
# -----------------------------------------------------------------------------
def dynamic_pressure_psf(V_mps: float, h_m: float) -> float:
    rho, _, _ = atmosphere(h_m)
    return 0.5 * rho * V_mps ** 2 * PA_TO_PSF


def raymer_empty_weight_breakdown(ac: PiperCherokeeLike, mtow_kg: float) -> Dict[str, float]:
    W0_lb = mtow_kg * KG_TO_LB
    V_cruise = ac.cruise_speed_kt * KT_TO_MPS
    q_psf = dynamic_pressure_psf(V_cruise, ac.cruise_alt_m)

    fuel_in_wing_factor = max((ac.m_fuel_max * KG_TO_LB) ** 0.0035, 1.0)

    wing_lb = (
        0.036
        * (ac.S * M2_TO_FT2) ** 0.758
        * fuel_in_wing_factor
        * (ac.AR) ** 0.60
        * q_psf ** 0.006
        * ac.taper ** 0.04
        * (100.0 * ac.t_c_wing) ** -0.30
        * (W0_lb * ac.n_ult) ** 0.49
    )

    htail_lb = (
        0.016
        * (W0_lb * ac.n_ult) ** 0.414
        * q_psf ** 0.168
        * (ac.htail_area * M2_TO_FT2) ** 0.896
        * (100.0 * ac.htail_t_c) ** -0.12
        * ac.htail_AR ** 0.043
        * ac.htail_taper ** -0.02
    )

    vtail_lb = (
        0.073
        * (W0_lb * ac.n_ult) ** 0.376
        * q_psf ** 0.122
        * (ac.vtail_area * M2_TO_FT2) ** 0.876
        * (100.0 * ac.vtail_t_c) ** -0.49
        * ac.vtail_AR ** 0.357
        * ac.vtail_taper ** 0.039
    )

    fuselage_lb = (
        0.052
        * (ac.fuselage_wetted_area_m2 * M2_TO_FT2) ** 1.086
        * (W0_lb * ac.n_ult) ** 0.177
        * (ac.tail_arm * M_TO_FT) ** -0.051
        * ac.L_D_cruise ** -0.072
        * q_psf ** 0.241
    )

    # Raymer landing-gear equations from the screenshot the user provided.
    W_l_lb = W0_lb
    N_l = 3.0
    L_m_in = ac.main_gear_length * M_TO_FT * 12.0
    L_n_in = ac.nose_gear_length * M_TO_FT * 12.0
    N_mw = 2.0
    N_mss = 2.0
    V_stall_kt = 61.0
    main_gear_lb = (
        0.0106
        * W_l_lb ** 0.888
        * N_l ** 0.25
        * L_m_in ** 0.40
        * N_mw ** 0.321
        * N_mss ** -0.50
        * V_stall_kt ** 0.10
    )
    nose_gear_lb = (
        0.032
        * W_l_lb ** 0.646
        * N_l ** 0.20
        * L_n_in ** 0.50
        * 1.0 ** 0.45
    )

    installed_engine_lb = 2.575 * (ac.engine_dry_mass_kg * KG_TO_LB) ** 0.922 * ac.n_engines

    fuel_volume_gal = (ac.m_fuel_max / AVGAS_DENSITY_KG_M3) * 264.172052
    fuel_system_lb = (
        2.49
        * fuel_volume_gal ** 0.726
        * (1.0 + ac.integral_tank_fraction) ** -0.363
        * ac.n_tanks ** 0.242
        * ac.n_engines ** 0.157
    )

    flight_controls_lb = (
        0.053
        * (ac.fuselage_length * M_TO_FT) ** 1.536
        * (ac.b * M_TO_FT) ** 0.371
        * (W0_lb * ac.n_ult * 1e-4) ** 0.80
    )

    _, _, a_cruise = atmosphere(ac.cruise_alt_m)
    mach = V_cruise / a_cruise
    K_h = 0.16472092991402892 * mach ** 0.8327375101470056
    hydraulics_lb = K_h * (ac.fuselage_width * M_TO_FT) ** 0.80 * mach ** 0.50

    avionics_lb = 2.117 * (ac.avionics_uninstalled * KG_TO_LB) ** 0.933
    electrical_lb = 12.57 * (fuel_system_lb + avionics_lb) ** 0.51
    aircon_antiice_lb = 0.265 * W0_lb ** 0.52 * (ac.n_crew + ac.n_pax) ** 0.68 * avionics_lb ** 0.17 * mach ** 0.08
    furnishings_lb = max(0.0, 0.0582 * W0_lb - 65.0)

    breakdown_kg = {
        "wing": wing_lb * LB_TO_KG,
        "horizontal_tail": htail_lb * LB_TO_KG,
        "vertical_tail": vtail_lb * LB_TO_KG,
        "fuselage": fuselage_lb * LB_TO_KG,
        "main_gear": main_gear_lb * LB_TO_KG,
        "nose_gear": nose_gear_lb * LB_TO_KG,
        "installed_engine": installed_engine_lb * LB_TO_KG,
        "fuel_system": fuel_system_lb * LB_TO_KG,
        "flight_controls": flight_controls_lb * LB_TO_KG,
        "hydraulics": hydraulics_lb * LB_TO_KG,
        "avionics": avionics_lb * LB_TO_KG,
        "electrical": electrical_lb * LB_TO_KG,
        "aircon_antiice": aircon_antiice_lb * LB_TO_KG,
        "furnishings": furnishings_lb * LB_TO_KG,
    }
    breakdown_kg["paint_misc"] = 0.03 * sum(breakdown_kg.values())
    return breakdown_kg


def size_aircraft_weights(ac: PiperCherokeeLike) -> Tuple[float, float, Dict[str, float]]:
    mtow_kg = 1200.0
    breakdown = {}
    for _ in range(60):
        breakdown = raymer_empty_weight_breakdown(ac, mtow_kg)
        empty_kg = sum(breakdown.values())
        mtow_new = empty_kg + ac.m_payload + ac.m_fuel_max
        if abs(mtow_new - mtow_kg) < 0.1:
            mtow_kg = mtow_new
            break
        mtow_kg = 0.55 * mtow_kg + 0.45 * mtow_new
    empty_kg = sum(breakdown.values())
    return mtow_kg, empty_kg, breakdown


items = {
    "empty": {"weight": 850, "x": 2.4},
    "pilot": {"weight": 80, "x": 2.2},
    "front_pax": {"weight": 80, "x": 2.2},
    "rear_pax1": {"weight": 80, "x": 3.2},
    "rear_pax2": {"weight": 80, "x": 3.2},
    "baggage": {"weight": 50, "x": 3.5},
    "fuel": {"weight": 131, "x": 2.5},
}



def compute_cg(items):
    total_weight = sum(item["weight"] for item in items.values())
    total_moment = sum(item["weight"] * item["x"] for item in items.values())
    x_cg = total_moment / total_weight
    return total_weight, x_cg

def compute_cg_with_wing(ac: PiperCherokeeLike):
    mtow_kg, empty_kg, breakdown = size_aircraft_weights(ac)
    wing_weight = breakdown.get("wing", 0.0)
    fuselage_empty_weight = max(empty_kg - wing_weight, 0.0)

    cg_items = {
        "empty_fuselage": {"weight": fuselage_empty_weight, "x": 2.4},
        "wing": {"weight": wing_weight, "x": ac.wing_x},
        "pilot": {"weight": 80, "x": 1.45},
        "front_pax": {"weight": 80, "x": 1.45},
        "rear_pax1": {"weight": 80, "x": 2},
        "rear_pax2": {"weight": 80, "x": 2},
        "baggage": {"weight": 50, "x": 3.5},
        "fuel": {"weight": ac.m_fuel_max, "x": 2.5},
    }

    total_weight, x_cg = compute_cg(cg_items)
    return total_weight, x_cg, cg_items

ac_for_cg = PiperCherokeeLike()
W, xcg, cg_items = compute_cg_with_wing(ac_for_cg)
print(f"W: {W}, xcg: {xcg}")

# -----------------------------------------------------------------------------
# Trim solves
# -----------------------------------------------------------------------------
def solve_trim_climb(ac: PiperCherokeeLike, h_m: float, mass_kg: float, throttle: float, z_dot_mps: float, guess=None):
    W = mass_kg * G
    if guess is None:
        guess = np.array([np.deg2rad(4.0), 45.0, np.deg2rad(3.0)])

    lb = np.array([np.deg2rad(-4.0), 30.0, np.deg2rad(-8.0)])
    ub = np.array([np.deg2rad(14.0), 95.0, np.deg2rad(12.0)])
    x0 = np.clip(np.asarray(guess, dtype=float), lb + 1e-6, ub - 1e-6)

    def residuals(x):
        alpha, V, gamma = x
        L, D, CL, CD = lift_drag(ac, V, alpha, h_m)
        T = thrust_available_n(ac, throttle, V, h_m)
        return np.array([
            (T * math.cos(alpha) - D - W * math.sin(gamma)) / W,
            (L + T * math.sin(alpha) - W * math.cos(gamma)) / W,
            (z_dot_mps - V * math.sin(gamma)) / max(3.0, abs(z_dot_mps) + 3.0),
        ])

    res = least_squares(residuals, x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=150)
    alpha, V, gamma = res.x
    L, D, CL, CD = lift_drag(ac, V, alpha, h_m)
    T = thrust_available_n(ac, throttle, V, h_m)

    good = np.max(np.abs(res.fun)) < 6e-4
    return {
        "success": bool(good),
        "alpha": alpha,
        "V": V,
        "gamma": gamma,
        "throttle": float(throttle),
        "T": T,
        "L": L,
        "D": D,
        "CL": CL,
        "CD": CD,
        "residuals": res.fun,
    }


def solve_trim_v_gamma(ac: PiperCherokeeLike, h_m: float, mass_kg: float, V_mps: float, gamma_rad: float, guess=None, flap: str = "clean"):
    W = mass_kg * G
    if guess is None:
        guess = np.array([np.deg2rad(2.0), 0.60])

    lb = np.array([np.deg2rad(-4.0), 0.0])
    ub = np.array([np.deg2rad(14.0), 1.0])
    x0 = np.clip(np.asarray(guess, dtype=float), lb + 1e-6, ub - 1e-6)

    def residuals(x):
        alpha, throttle = x
        L, D, CL, CD = lift_drag(ac, V_mps, alpha, h_m, flap=flap)
        T = thrust_available_n(ac, throttle, V_mps, h_m)
        return np.array([
            (T * math.cos(alpha) - D - W * math.sin(gamma_rad)) / W,
            (L + T * math.sin(alpha) - W * math.cos(gamma_rad)) / W,
        ])

    res = least_squares(residuals, x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=150)
    alpha, throttle = res.x
    L, D, CL, CD = lift_drag(ac, V_mps, alpha, h_m, flap=flap)
    T = thrust_available_n(ac, throttle, V_mps, h_m)

    good = np.max(np.abs(res.fun)) < 6e-4 and throttle <= 1.0 + 1e-6
    return {
        "success": bool(good),
        "alpha": alpha,
        "V": V_mps,
        "gamma": gamma_rad,
        "throttle": throttle,
        "T": T,
        "L": L,
        "D": D,
        "CL": CL,
        "CD": CD,
        "residuals": res.fun,
    }


# -----------------------------------------------------------------------------
# Performance helpers
# -----------------------------------------------------------------------------
def estimate_takeoff_landing(ac: PiperCherokeeLike, takeoff_mass_kg: float, landing_mass_kg: float) -> Dict[str, float]:
    rho, _, _ = atmosphere(0.0)

    W_to = takeoff_mass_kg * G
    Vs_to = math.sqrt(2.0 * W_to / (rho * ac.S * ac.CLmax_takeoff))
    Vlof = 1.2 * Vs_to
    alpha_to = math.radians(6.0)
    V_avg = 0.7 * Vlof
    L_avg, D_avg, _, _ = lift_drag(ac, V_avg, alpha_to, 0.0, flap="takeoff")
    T_avg = thrust_available_n(ac, 1.0, V_avg, 0.0)
    mu_roll = 0.03
    a_avg = (T_avg - D_avg - mu_roll * max(W_to - L_avg, 0.0)) / takeoff_mass_kg
    a_avg = max(a_avg, 0.8)
    s_to = Vlof ** 2 / (2.0 * a_avg)

    W_land = landing_mass_kg * G
    Vs_land = math.sqrt(2.0 * W_land / (rho * ac.S * ac.CLmax_landing))
    Vtd = 1.3 * Vs_land
    alpha_land = math.radians(7.0)
    L_td, D_td, _, _ = lift_drag(ac, 0.7 * Vtd, alpha_land, 0.0, flap="landing")
    mu_brake = 0.15
    decel = mu_brake * G + 0.5 * D_td / landing_mass_kg + 0.02 * max(W_land - L_td, 0.0) / landing_mass_kg
    s_land = Vtd ** 2 / (2.0 * max(decel, 1.0))

    return {
        "Vs_to": Vs_to,
        "Vlof": Vlof,
        "takeoff_ground_roll_m": s_to,
        "Vs_land": Vs_land,
        "Vtd": Vtd,
        "landing_ground_roll_m": s_land,
    }


def estimate_max_level_speed(ac: PiperCherokeeLike, mass_kg: float, h_m: float):
    # First do a coarse scan to find a feasible/infeasible bracket near Vmax.
    # Then refine with bisection. This avoids the "one speed bin short" nonsense
    # from the earlier script and also avoids relying on low-speed trim points.
    rho, _, _ = atmosphere(h_m)
    Vs_clean = math.sqrt(2.0 * mass_kg * G / (rho * ac.S * ac.CLmax_clean))
    V_grid = np.linspace(max(1.15 * Vs_clean, 50.0), 140.0, 181)

    guess = np.array([np.deg2rad(2.0), 0.70])
    last_feasible = None
    first_infeasible_after = None

    for V in V_grid:
        sol = solve_trim_v_gamma(ac, h_m, mass_kg, float(V), 0.0, guess)
        if sol["success"]:
            last_feasible = sol
            guess = np.array([sol["alpha"], min(sol["throttle"], 0.99)])
        elif last_feasible is not None:
            first_infeasible_after = float(V)
            break

    if last_feasible is None:
        return None

    lo = last_feasible["V"]
    hi = first_infeasible_after if first_infeasible_after is not None else min(160.0, lo * 1.10)
    guess = np.array([last_feasible["alpha"], min(last_feasible["throttle"], 0.99)])
    best = last_feasible

    for _ in range(35):
        mid = 0.5 * (lo + hi)
        sol_mid = solve_trim_v_gamma(ac, h_m, mass_kg, mid, 0.0, guess)
        if sol_mid["success"]:
            best = sol_mid
            lo = mid
            guess = np.array([sol_mid["alpha"], min(sol_mid["throttle"], 0.99)])
        else:
            hi = mid

    return best


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def make_constraint_diagram(ac: PiperCherokeeLike, takeoff_mass_kg: float, outpath: Path):
    W = takeoff_mass_kg * G
    rho_dash, _, _ = atmosphere(ac.dash_check_alt_m)
    rho_sl, _, _ = atmosphere(0.0)
    V_dash = ac.dash_speed_req_kt * KT_TO_MPS

    WS_range = np.linspace(30.0, 125.0, 240)  # kg/m^2
    TW_dash = []
    TW_to = []

    for ws in WS_range:
        S_trial = takeoff_mass_kg / ws
        AR = ac.b ** 2 / S_trial
        k_trial = 1.0 / (math.pi * ac.e * AR)

        q_dash = 0.5 * rho_dash * V_dash ** 2
        CL_dash = ws * G / q_dash
        CD_dash = ac.CD0 + k_trial * CL_dash ** 2
        D_dash = q_dash * S_trial * CD_dash
        TW_dash.append(D_dash / W)

        Vs = math.sqrt(ws * G / (0.5 * rho_sl * ac.CLmax_takeoff))
        Vlof = 1.2 * Vs
        s_req = 1000.0 * 0.3048
        TW_to.append(Vlof ** 2 / (2.0 * G * s_req) + 0.04)

    V_stall_max = 61.0 * KT_TO_MPS
    WS_stall = 0.5 * rho_sl * V_stall_max ** 2 * ac.CLmax_takeoff / G

    WS_design = takeoff_mass_kg / ac.S
    T_dash = thrust_available_n(ac, 1.0, V_dash, ac.dash_check_alt_m)
    TW_design = T_dash / W

    plt.figure(figsize=(9, 6))
    plt.plot(WS_range, TW_dash, "b-", lw=2, label=f"Dash {ac.dash_speed_req_kt:.0f} kts")
    plt.plot(WS_range, TW_to, "r-", lw=2, label="Takeoff < 1000 ft")
    plt.axvline(WS_stall, color="g", lw=2, label="Stall < 61 kts")
    plt.plot(WS_design, TW_design, "k*", ms=16, label="Your design")
    plt.annotate(f"({WS_design:.1f}, {TW_design:.3f})", (WS_design, TW_design), textcoords="offset points", xytext=(10, -12))
    plt.xlabel("W/S  (kg/m²)")
    plt.ylabel("T/W  (-)")
    plt.title("Constraint Diagram")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -----------------------------------------------------------------------------
# Aerodynamic Polar Report
# -----------------------------------------------------------------------------
def report_aero_polar(ac, mass_kg, h_m, outpath):
    rho, _, _ = atmosphere(h_m)
    W = mass_kg * G
    aero_data = {}

    for flap, flap_data in ac.flap_settings.items():
        Vs = math.sqrt(2.0 * W / (rho * ac.S * flap_data["CLmax"]))
        V_range = np.linspace(Vs * 1.05, 175 * KT_TO_MPS * 1.1, 120)
        V_kt=[]; CL_a=[]; CD_a=[]; CDi_a=[]; CDp_a=[]; D_a=[]; LD_a=[]

        for V in V_range:
            q = 0.5 * rho * V ** 2
            CL_req = W / (q * ac.S)
            if CL_req > flap_data["CLmax"] or CL_req < 0:
                continue

            alpha = alpha_for_lift_coefficient(ac, CL_req, flap)
            CL, CD = aero_coeffs(ac, alpha, flap)
            CDi = ac.k * CL ** 2
            CDp = CD - CDi
            D = q * ac.S * CD

            V_kt.append(V * MPS_TO_KT); CL_a.append(CL); CD_a.append(CD)
            CDi_a.append(CDi); CDp_a.append(CDp); D_a.append(D); LD_a.append(CL / CD)

        aero_data[flap] = {
            "V_kt": np.array(V_kt),
            "CL": np.array(CL_a),
            "CD": np.array(CD_a),
            "CDi": np.array(CDi_a),
            "CDp": np.array(CDp_a),
            "D": np.array(D_a),
            "LD": np.array(LD_a),
        }

        if len(LD_a) > 0:
            LD_np = aero_data[flap]["LD"]
            V_np = aero_data[flap]["V_kt"]
            D_np = aero_data[flap]["D"]
            bi = np.argmax(LD_np)
            print(f"\n  {flap.title()} best L/D: {LD_np[bi]:.2f} at {V_np[bi]:.1f} kts")
            print(f"  {flap.title()} min drag: {D_np.min():.1f} N at {V_np[np.argmin(D_np)]:.1f} kts")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Aerodynamic Polar and Flap Settings  h={h_m:.0f} m", fontsize=11)
    colors = {"clean": "b", "takeoff": "g", "landing": "r"}

    for flap, data in aero_data.items():
        if len(data["V_kt"]) == 0:
            continue
        c = colors.get(flap, None)
        label = flap.title()
        qS = 0.5 * rho * (data["V_kt"] * KT_TO_MPS) ** 2 * ac.S

        axs[0,0].plot(data["V_kt"], data["CL"], color=c, lw=2, label=label)
        axs[0,1].plot(data["V_kt"], data["D"] / 1000.0, color=c, lw=2, label=f"{label} total")
        axs[0,1].plot(data["V_kt"], qS * data["CDi"] / 1000.0, color=c, lw=1.2, ls="--", label=f"{label} induced")
        axs[0,1].plot(data["V_kt"], qS * data["CDp"] / 1000.0, color=c, lw=1.2, ls=":", label=f"{label} profile")
        axs[1,0].plot(data["CL"], data["CD"], color=c, lw=2, label=label)
        axs[1,1].plot(data["V_kt"], data["LD"], color=c, lw=2, label=label)

    axs[0,0].set(xlabel='Speed (kts)', ylabel='CL', title='Lift Coefficient'); axs[0,0].legend(); axs[0,0].grid(alpha=0.3)
    axs[0,1].set(xlabel='Speed (kts)', ylabel='Drag (kN)', title='Drag Breakdown'); axs[0,1].legend(fontsize=7); axs[0,1].grid(alpha=0.3)
    axs[1,0].set(xlabel='CL', ylabel='CD', title='Drag Polar'); axs[1,0].legend(); axs[1,0].grid(alpha=0.3)
    axs[1,1].set(xlabel='Speed (kts)', ylabel='L/D', title='L/D Ratio'); axs[1,1].legend(); axs[1,1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

def report_performance(ac, mass_kg, outpath):
    alts = np.linspace(0, 5000, 60); RC_list=[]
    for h in alts:
        rho,_,_ = atmosphere(h); W=mass_kg*G
        Vs = math.sqrt(2*W/(rho*ac.S*ac.CLmax_clean)); V=Vs*1.3
        T = thrust_available_n(ac, 1.0, V, h)
        q = 0.5*rho*V**2; CL_req=W/(q*ac.S)
        alpha = alpha_for_lift_coefficient(ac, CL_req, "clean")
        CL, CD = aero_coeffs(ac, alpha, "clean")
        D=q*ac.S*CD
        RC_list.append(max(0.0,(T-D)*V/W))
    RC=np.array(RC_list)
    ci = np.where(RC < 0.5)[0]
    service_ceiling = alts[ci[0]] if len(ci)>0 else alts[-1]
    rho_cr,_,_ = atmosphere(ac.cruise_alt_m); W=mass_kg*G; V_cr=ac.cruise_speed_kt*KT_TO_MPS
    bank_a=np.linspace(15,ac.max_turn_bank_deg,50)
    turn_n=1.0 / np.cos(np.radians(bank_a))
    TR=[V_cr**2/(G*math.sqrt(n**2-1)) for n in turn_n]
    print(f"\n  Max climb rate SL:  {RC[0]*196.85:.0f} ft/min")
    print(f"  Service ceiling:    {service_ceiling*M_TO_FT:.0f} ft")
    print(f"  Turn radius 30 deg: {TR[np.argmin(abs(bank_a-30))]:.0f} m")
    print(f"  Max turn n:         {ac.max_turn_n:.2f} g at {ac.max_turn_bank_deg:.1f} deg bank")
    print(f"  Structural n limit: {ac.n_limit:.2f} g  (turn margin {ac.n_limit - ac.max_turn_n:.2f} g)")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(RC*196.85, alts*M_TO_FT, 'b-', lw=2)
    axs[0].axvline(100, color='r', ls='--', label='100 fpm service ceiling')
    axs[0].set(xlabel='ROC (ft/min)', ylabel='Altitude (ft)', title='Climb Performance'); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[1].plot(bank_a, TR, 'g-', lw=2)
    axs[1].set(xlabel='Bank Angle (deg)', ylabel='Turn Radius (m)', title='Level Turn Radius'); axs[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()
    return service_ceiling

def report_vn_diagram(ac, mass_kg, h_m, outpath):
    rho,_,_ = atmosphere(h_m); W=mass_kg*G
    Vs = math.sqrt(2*W/(rho*ac.S*ac.CLmax_clean))
    n_pos=ac.n_limit; n_neg=-0.40 * n_pos
    Va=Vs*math.sqrt(n_pos); Vd=175*KT_TO_MPS*1.25
    V_range=np.linspace(Vs*0.5, Vd*1.05, 300); V_kt=V_range*MPS_TO_KT
    def ns_p(V): return min(0.5*rho*V**2*ac.S*ac.CLmax_clean/W, n_pos)
    def ns_n(V): return max(-0.5*rho*V**2*ac.S*ac.CLmax_clean/W, n_neg)
    nsp=np.array([ns_p(V) for V in V_range]); nsn=np.array([ns_n(V) for V in V_range])
    V_g=np.linspace(Vs,Vd,200)
    mu_g=2*mass_kg/(rho*ac.S*ac.CLa*(ac.cruise_speed_kt*KT_TO_MPS))
    Kg=0.88*mu_g/(5.3+mu_g)
    n_gc=np.array([1+Kg*rho*15.24*V*ac.CLa*ac.S/(2*W) for V in V_g])
    print(f"\n  Vs={Vs*MPS_TO_KT:.1f} kts, Va={Va*MPS_TO_KT:.1f} kts, Vd={Vd*MPS_TO_KT:.1f} kts")
    print(f"  n_pos_limit={n_pos:.2f}, n_neg_limit={n_neg:.2f}, n_ultimate={ac.n_ult:.2f}")
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(V_kt, nsp, 'b-', lw=2, label='Pos stall')
    ax.plot(V_kt, nsn, 'b--', lw=2, label='Neg stall')
    ax.axhline(n_pos, color='r', lw=1.5, label=f'+{n_pos}g limit')
    ax.axhline(n_neg, color='r', lw=1.5, ls='--', label=f'{n_neg}g limit')
    ax.axhline(ac.n_ult, color='darkred', lw=1, ls=':', label=f'+{ac.n_ult:.1f}g ultimate')
    ax.axvline(Va*MPS_TO_KT, color='g', lw=1.5, ls='--', label=f'Va={Va*MPS_TO_KT:.0f} kts')
    ax.axvline(Vd*MPS_TO_KT, color='purple', lw=1.5, ls='--', label=f'Vd={Vd*MPS_TO_KT:.0f} kts')
    ax.plot(V_g*MPS_TO_KT, n_gc, 'orange', lw=1.5, ls='-.', label='Gust 50fps')
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='Speed (kts)', ylabel='Load Factor (g)', title=f'V-n Diagram h={h_m:.0f}m')
    ax.legend(fontsize=8); ax.set_ylim(n_neg*1.8, n_pos*1.8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()


def compute_wing_bending(ac: PiperCherokeeLike, mass_kg: float, n_load: Optional[float] = None):
    if n_load is None:
        n_load = ac.n_limit

    W = mass_kg * G
    total_lift = n_load * W
    semi_span = ac.b / 2.0
    # Per-wing root bending for an elliptical lift distribution. total_lift is
    # the full aircraft lift, so each wing carries half of it.
    M_root = 2.0 / (3.0 * math.pi) * total_lift * semi_span
    shear_root = total_lift / 2.0

    chord = ac.mean_chord
    cap_area = ac.wing_spar_caps * ac.wing_spar_thickness * chord
    section_modulus = cap_area * (ac.wing_box_depth / 2.0)
    stress_mpa = (M_root / section_modulus) / 1e6 if section_modulus > 0 else float('inf')

    y = np.linspace(0.0, semi_span, 120)
    q0 = 2.0 * total_lift / (math.pi * semi_span)
    q = q0 * np.sqrt(np.clip(1.0 - (y / semi_span) ** 2, 0.0, 1.0))
    M = np.zeros_like(y)
    for i in range(len(y)):
        s = y[i:]
        q_s = q[i:]
        M[i] = np.trapezoid(q_s * (s - y[i]), s)

    return {
        "n_load": n_load,
        "M_root_Nm": M_root,
        "V_root_N": shear_root,
        "section_modulus_m3": section_modulus,
        "stress_mpa": stress_mpa,
        "y_m": y,
        "M_distribution_Nm": M,
        "total_lift_N": total_lift,
        "n_limit": ac.n_limit,
        "n_ult": ac.n_ult,
        "max_turn_n": ac.max_turn_n,
        "turn_margin_n": ac.n_limit - ac.max_turn_n,
    }


def report_wing_bending(ac, mass_kg, outpath):
    bending = compute_wing_bending(ac, mass_kg)
    print("\nWing Bending Summary")
    print("---------------------")
    print(f"  Limit load factor:     {bending['n_load']:.2f} g")
    print(f"  Ultimate load factor:  {bending['n_ult']:.2f} g")
    print(f"  Max turn n:            {bending['max_turn_n']:.2f} g")
    print(f"  Turn margin:           {bending['turn_margin_n']:.2f} g "
          f"({'PASS' if bending['turn_margin_n'] >= 0 else 'FAIL'})")
    print(f"  Root bending moment:   {bending['M_root_Nm']:.0f} Nm")
    print(f"  Root shear:            {bending['V_root_N']:.0f} N")
    print(f"  Section modulus:       {bending['section_modulus_m3']:.4e} m^3")
    print(f"  Estimated stress:      {bending['stress_mpa']:.1f} MPa")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bending['y_m'] * M_TO_FT, bending['M_distribution_Nm'] / 1000.0, 'b-', lw=2)
    ax.set(xlabel='Spanwise station (ft)', ylabel='Bending moment (kNm)',
           title='Wing Bending Moment Distribution')
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()


def report_trim(ac, mass_kg, h_m, x_cg, x_ac, outpath):
    rho,_,_ = atmosphere(h_m); W=mass_kg*G
    Vs=math.sqrt(2*W/(rho*ac.S*ac.CLmax_clean))
    V_range=np.linspace(Vs*1.05, 175*KT_TO_MPS, 100)
    c_bar=ac.S/ac.b; Cm0=-0.05
    lh=ac.tail_arm; Sh=ac.htail_area
    static_margin=(x_ac-x_cg)/c_bar
    Vm=[]; Mm=[]; CLtm=[]; Dtm=[]
    for V in V_range:
        q=0.5*rho*V**2; CL=W/(q*ac.S)
        if CL>ac.CLmax_clean: continue
        Cm_cg=Cm0+CL*(x_cg-x_ac)/c_bar
        M_wing=q*ac.S*c_bar*Cm_cg
        CL_tail=-M_wing/max(q*Sh*lh,1e-6)
        D_trim=q*Sh*(0.009+0.05*CL_tail**2)
        Vm.append(V*MPS_TO_KT); Mm.append(M_wing); CLtm.append(CL_tail); Dtm.append(D_trim)
    Vm=np.array(Vm); Mm=np.array(Mm); CLtm=np.array(CLtm); Dtm=np.array(Dtm)
    print(f"\n  Static margin: {static_margin*100:.1f}% MAC")
    print(f"  Max trim drag: {Dtm.max():.1f} N at {Vm[np.argmax(Dtm)]:.1f} kts")
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    axs[0].plot(Vm,Mm,'b-',lw=2); axs[0].axhline(0,color='k',lw=0.8); axs[0].set(xlabel='Speed (kts)',ylabel='Moment (Nm)',title='Wing Pitching Moment'); axs[0].grid(alpha=0.3)
    axs[1].plot(Vm,CLtm,'g-',lw=2); axs[1].axhline(0,color='k',lw=0.8); axs[1].set(xlabel='Speed (kts)',ylabel='Tail CL',title='Tail Lift for Trim'); axs[1].grid(alpha=0.3)
    axs[2].plot(Vm,Dtm,'r-',lw=2); axs[2].set(xlabel='Speed (kts)',ylabel='Trim Drag (N)',title='Trim Drag'); axs[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()

def report_cg_envelope(items, outpath):
    cum_W=0; cum_M=0; cgs=[]; Ws=[]; labs=[]
    for name, item in items.items():
        cum_W+=item['weight']; cum_M+=item['weight']*item['x']
        cgs.append(cum_M/cum_W); Ws.append(cum_W); labs.append(name)
    print("\nCG Envelope")
    for lab,W,cg in zip(labs,Ws,cgs):
        print(f"  {lab:20s}: W={W:.0f} kg  CG={cg:.3f} m")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(cgs,Ws,'bo-',lw=2,ms=8)
    for i,lab in enumerate(labs):
        ax.annotate(lab,(cgs[i],Ws[i]),textcoords='offset points',xytext=(6,4),fontsize=8)
    ax.set(xlabel='CG from Nose (m)',ylabel='Gross Weight (kg)',title='CG Envelope — Loading Sequence')
    ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(outpath,dpi=160); plt.close()

def report_landing_gear(ac, x_cg_fwd, x_cg_aft,
                        x_main, x_nose, track, outpath):
    """
    x_cg_fwd: most forward CG (m from nose)
    x_cg_aft: most aft CG (m from nose)
    x_main:   main gear x from nose (m)
    x_nose:   nose gear x from nose (m)
    track:    lateral track width (m)
    """
    wheelbase = x_main - x_nose

    # 1. Forward gear margin — angle from main gear to aft CG >= 15 deg
    forward_gear_angle = math.degrees(math.atan(
        (x_main - x_cg_aft) / ac.cg_height
    ))

    # 2. Touchdown clearance — tail clearance at rotation >= 15 deg
    dist_main_to_tail = ac.fuselage_length - x_main
    tail_clearance_height = getattr(ac, "tail_clearance_height", ac.main_gear_length)
    touchdown_angle   = math.degrees(math.atan(
        tail_clearance_height / dist_main_to_tail
    ))

    # 3. Overturn criteria — use CG height and lateral track for rollover
    
    overturn_angle = math.degrees(math.atan(
        (track / 2) / ac.cg_height
    ))

    # 4. Nose gear load fraction
    F_nose = (x_main - x_cg_aft) / wheelbase      # aft CG = minimum nose load
    F_nose_fwd = (x_main - x_cg_fwd) / wheelbase  # forward CG = maximum nose load
    nose_aft_pass = 0.08 <= F_nose <= 0.15
    nose_fwd_pass = 0.08 <= F_nose_fwd <= 0.15

    print("\nLanding Gear Metrics")
    print("-" * 45)
    print(f"  Forward gear margin:   {forward_gear_angle:.1f} deg  "
          f"({'PASS' if forward_gear_angle >= 15 else 'FAIL'}, need >= 15)")
    print(f"  Touchdown clearance:   {touchdown_angle:.1f} deg  "
          f"({'PASS' if touchdown_angle >= 15 else 'FAIL'}, need >= 15)")
    print(f"  Overturn angle:        {overturn_angle:.1f} deg  "
          f"({'PASS' if 55 <= overturn_angle <= 65 else 'FAIL'}, need 55-65)")
    print(f"  Nose load (aft CG):    {F_nose*100:.1f}%  "
          f"({'PASS' if nose_aft_pass else 'FAIL'}, want 8-15%)")
    print(f"  Nose load (fwd CG):    {F_nose_fwd*100:.1f}%  "
          f"({'PASS' if nose_fwd_pass else 'FAIL'}, want 8-15%)")

def report_trade_studies(ac, outpath):
    base_S=ac.S; base_b=ac.b; base_CD0=ac.CD0; base_AR=ac.AR
    base_CLa=ac.CLa
    base_CLmax_clean=ac.CLmax_clean
    base_CLmax_takeoff=ac.CLmax_takeoff
    base_CLmax_landing=ac.CLmax_landing
    def breguet_range(ac, mass_kg):
        rho,_,_=atmosphere(ac.cruise_alt_m); V_cr=ac.cruise_speed_kt*KT_TO_MPS
        q=0.5*rho*V_cr**2; W=mass_kg*G; CL_req=W/(q*ac.S)
        if CL_req >= ac.CLmax_clean:
            return 0.0
        alpha=alpha_for_lift_coefficient(ac,CL_req,"clean")
        CL,CD=aero_coeffs(ac,alpha,"clean"); LD=CL/CD
        mtow,empty,_=size_aircraft_weights(ac); W1=(empty+ac.m_payload)*G
        return (V_cr/G)*LD*math.log(W/W1)*M_TO_NM
    def dash_margin(ac, mass_kg):
        rho,_,_=atmosphere(ac.dash_check_alt_m); W=mass_kg*G
        V_d=175*KT_TO_MPS; q=0.5*rho*V_d**2
        CL_req=W/(q*ac.S)
        alpha=alpha_for_lift_coefficient(ac,CL_req,"clean")
        CL,CD=aero_coeffs(ac,alpha,"clean")
        D=q*ac.S*CD
        return thrust_available_n(ac,1.0,V_d,ac.dash_check_alt_m)-D
    S_vals=np.linspace(9,20,15); rS=[]; dS=[]
    for S in S_vals:
        ac.S=S; ac.b=math.sqrt(S*base_AR)
        mtow,empty,_=size_aircraft_weights(ac); mass=empty+ac.m_payload+ac.m_fuel_max
        rS.append(breguet_range(ac,mass)); dS.append(dash_margin(ac,mass))
    ac.S=base_S; ac.b=base_b
    AR_vals=np.linspace(5,12,15); rAR=[]; dAR=[]
    for AR in AR_vals:
        ac.b=math.sqrt(base_S*AR); ac.S=base_S
        mtow,empty,_=size_aircraft_weights(ac); mass=empty+ac.m_payload+ac.m_fuel_max
        rAR.append(breguet_range(ac,mass)); dAR.append(dash_margin(ac,mass))
    ac.b=base_b
    airfoils={
        'NACA 2412': {'CD0':0.028, 'CLa':5.0, 'CLmax_clean':1.50, 'CLmax_takeoff':1.80, 'CLmax_landing':2.00},
        'NACA 65-415': {'CD0':0.023, 'CLa':5.4, 'CLmax_clean':1.40, 'CLmax_takeoff':1.70, 'CLmax_landing':1.90},
        'Clark Y': {'CD0':0.030, 'CLa':4.8, 'CLmax_clean':1.55, 'CLmax_takeoff':1.85, 'CLmax_landing':2.05},
        'Custom low-drag': {'CD0':0.020, 'CLa':5.2, 'CLmax_clean':1.45, 'CLmax_takeoff':1.75, 'CLmax_landing':1.95},
    }
    af_names=list(airfoils.keys()); rAF=[]; dAF=[]
    for name,params in airfoils.items():
        ac.CD0=params['CD0']
        ac.CLa=params['CLa']
        ac.CLmax_clean=params['CLmax_clean']
        ac.CLmax_takeoff=params['CLmax_takeoff']
        ac.CLmax_landing=params['CLmax_landing']
        mtow,empty,_=size_aircraft_weights(ac); mass=empty+ac.m_payload+ac.m_fuel_max
        rAF.append(breguet_range(ac,mass)); dAF.append(dash_margin(ac,mass))
    ac.CD0=base_CD0
    ac.CLa=base_CLa
    ac.CLmax_clean=base_CLmax_clean
    ac.CLmax_takeoff=base_CLmax_takeoff
    ac.CLmax_landing=base_CLmax_landing
    print("\nTrade Studies")
    print(f"  Best S for range:    {S_vals[np.argmax(rS)]:.1f} m²")
    print(f"  Best AR for range:   {AR_vals[np.argmax(rAR)]:.1f}")
    print(f"  Best airfoil:        {af_names[np.argmax(rAF)]}")
    fig,axs=plt.subplots(2,3,figsize=(15,8)); fig.suptitle("Trade Studies",fontsize=12)
    axs[0,0].plot(S_vals,rS,'b-o',ms=5); axs[0,0].axvline(base_S,color='k',ls='--',label=f'Current S={base_S:.1f}'); axs[0,0].set(xlabel='S (m²)',ylabel='Range (nmi)',title='Range vs Wing Area'); axs[0,0].legend(); axs[0,0].grid(alpha=0.3)
    axs[0,1].plot(AR_vals,rAR,'g-o',ms=5); axs[0,1].axvline(base_AR,color='k',ls='--',label=f'Current AR={base_AR:.1f}'); axs[0,1].set(xlabel='AR',ylabel='Range (nmi)',title='Range vs AR'); axs[0,1].legend(); axs[0,1].grid(alpha=0.3)
    axs[0,2].bar(range(len(af_names)),rAF,color='purple',alpha=0.7); axs[0,2].set_xticks(range(len(af_names))); axs[0,2].set_xticklabels(af_names,rotation=15,fontsize=8); axs[0,2].set(ylabel='Range (nmi)',title='Range vs Airfoil'); axs[0,2].grid(alpha=0.3,axis='y')
    axs[1,0].plot(S_vals,dS,'b-o',ms=5); axs[1,0].axhline(0,color='r',ls='--'); axs[1,0].axvline(base_S,color='k',ls='--'); axs[1,0].set(xlabel='S (m²)',ylabel='T-D at 175kts (N)',title='Dash vs Wing Area'); axs[1,0].grid(alpha=0.3)
    axs[1,1].plot(AR_vals,dAR,'g-o',ms=5); axs[1,1].axhline(0,color='r',ls='--'); axs[1,1].axvline(base_AR,color='k',ls='--'); axs[1,1].set(xlabel='AR',ylabel='T-D at 175kts (N)',title='Dash vs AR'); axs[1,1].grid(alpha=0.3)
    axs[1,2].bar(range(len(af_names)),dAF,color=['g' if d>0 else 'r' for d in dAF],alpha=0.7); axs[1,2].axhline(0,color='k',ls='--'); axs[1,2].set_xticks(range(len(af_names))); axs[1,2].set_xticklabels(af_names,rotation=15,fontsize=8); axs[1,2].set(ylabel='T-D at 175kts (N)',title='Dash vs Airfoil'); axs[1,2].grid(alpha=0.3,axis='y')
    plt.tight_layout(); plt.savefig(outpath,dpi=160); plt.close()



# -----------------------------------------------------------------------------
# Mission simulation
# -----------------------------------------------------------------------------
def run_mission(ac=None) -> Dict[str, float]:
    if ac is None:
        ac = PiperCherokeeLike()
    mtow_kg, empty_kg, breakdown = size_aircraft_weights(ac)

    range_req_nm = 500.0
    target_total_range_m = range_req_nm * 1852.0
    cruise_alt_m = ac.cruise_alt_m
    climb_rate_mps = 3.0
    cruise_speed_mps = ac.cruise_speed_kt * KT_TO_MPS
    descent_speed_mps = 75.0 * KT_TO_MPS
    descent_gamma_rad = math.radians(-3.0)
    dt = 15.0

    fuel_kg = ac.m_fuel_max
    takeoff_mass_kg = empty_kg + ac.m_payload + fuel_kg

    outdir = Path(__file__).resolve().parent
    mission_plot = outdir / "piper_mission_profile_fixed.png"
    dash_plot = outdir / "dash_fixed.png"

    dash_sl = estimate_max_level_speed(ac, takeoff_mass_kg, ac.dash_check_alt_m)
    dash_cruise_alt = estimate_max_level_speed(ac, takeoff_mass_kg, cruise_alt_m)
    dash_sl_kt = np.nan if dash_sl is None else dash_sl["V"] * MPS_TO_KT
    dash_cruise_kt = np.nan if dash_cruise_alt is None else dash_cruise_alt["V"] * MPS_TO_KT

    print(f"Aircraft: {ac.name}")
    print(f"Raymer-sized empty mass: {empty_kg:.1f} kg")
    print(f"Initial gross mass:      {takeoff_mass_kg:.1f} kg")
    print(f"Initial fuel load:       {fuel_kg:.1f} kg")
    print(f"Wing area:               {ac.S:.2f} m^2")
    print(f"Power:                   {ac.engine_hp:.0f} hp")
    print(f"Design W/S:              {takeoff_mass_kg / ac.S:.1f} kg/m^2")
    print()
    print("Raymer empty-weight breakdown")
    print("----------------------------")
    for key, value in breakdown.items():
        print(f"{key:18s}: {value:7.1f} kg")
    print()
    print("Dash speed check")
    print("----------------")
    print(f"Sea-level max level speed:     {dash_sl_kt:7.2f} kt")
    print(f"Cruise-alt max level speed:    {dash_cruise_kt:7.2f} kt at {cruise_alt_m:.0f} m")
    print(f"Dash requirement ({ac.dash_speed_req_kt:.0f} kt) met at sea level: {dash_sl_kt >= ac.dash_speed_req_kt}")
    print()

    history = []
    t_s = 0.0
    x_m = 0.0
    h_m = 0.0

    def log_state(seg_name: str, trim: Dict[str, float], mass_now_kg: float, fuel_now_kg: float):
        history.append({
            "segment": seg_name,
            "t_s": t_s,
            "x_m": x_m,
            "h_m": h_m,
            "fuel_kg": fuel_now_kg,
            "mass_kg": mass_now_kg,
            "V_kt": trim["V"] * MPS_TO_KT,
            "alpha_deg": math.degrees(trim["alpha"]),
            "gamma_deg": math.degrees(trim["gamma"]),
            "throttle": trim["throttle"],
        })

    # --- climb ---
    climb_guess = None
    climb_steps = 0
    while h_m < cruise_alt_m - 1.0:
        mass_kg = empty_kg + ac.m_payload + fuel_kg
        trim = solve_trim_climb(ac, h_m, mass_kg, throttle=0.60, z_dot_mps=climb_rate_mps, guess=climb_guess)
        if not trim["success"]:
            raise RuntimeError(f"Climb trim failed at h = {h_m:.1f} m, residuals = {trim['residuals']}")

        dx = trim["V"] * math.cos(trim["gamma"]) * dt
        dz = trim["V"] * math.sin(trim["gamma"]) * dt
        if h_m + dz > cruise_alt_m:
            frac = (cruise_alt_m - h_m) / max(dz, 1e-9)
            dx *= frac
            dz *= frac
            fuel_dt = dt * frac
        else:
            fuel_dt = dt

        x_m += dx
        h_m += dz
        fuel_kg -= fuel_burn_rate_kg_s(ac, trim["throttle"], h_m) * fuel_dt
        fuel_kg = max(fuel_kg, 0.0)
        t_s += fuel_dt
        mass_kg = empty_kg + ac.m_payload + fuel_kg
        log_state("climb", trim, mass_kg, fuel_kg)

        climb_guess = np.array([trim["alpha"], trim["V"], trim["gamma"]])
        climb_steps += 1
        if fuel_kg <= 0.0:
            raise RuntimeError("Fuel exhausted during climb.")
        if climb_steps > 5000:
            raise RuntimeError("Climb loop exceeded safety limit.")

    climb_range_nm = x_m * M_TO_NM

    # Predict descent distance from cruise altitude.
    descent_distance_m = cruise_alt_m / math.tan(abs(descent_gamma_rad))
    cruise_target_end_x_m = target_total_range_m - descent_distance_m
    if cruise_target_end_x_m <= x_m:
        raise RuntimeError("Target range is too short for chosen climb and descent profile.")

    # --- cruise ---
    cruise_guess = np.array([math.radians(2.0), 0.65])
    cruise_steps = 0
    while x_m < cruise_target_end_x_m:
        mass_kg = empty_kg + ac.m_payload + fuel_kg
        trim = solve_trim_v_gamma(ac, h_m, mass_kg, cruise_speed_mps, 0.0, guess=cruise_guess)
        if not trim["success"]:
            raise RuntimeError(f"Cruise trim failed at x = {x_m * M_TO_NM:.1f} nm, residuals = {trim['residuals']}")

        dx = trim["V"] * dt
        if x_m + dx > cruise_target_end_x_m:
            frac = (cruise_target_end_x_m - x_m) / max(dx, 1e-9)
            dx *= frac
            fuel_dt = dt * frac
        else:
            fuel_dt = dt

        x_m += dx
        fuel_kg -= fuel_burn_rate_kg_s(ac, trim["throttle"], h_m) * fuel_dt
        fuel_kg = max(fuel_kg, 0.0)
        t_s += fuel_dt
        mass_kg = empty_kg + ac.m_payload + fuel_kg
        log_state("cruise", trim, mass_kg, fuel_kg)

        cruise_guess = np.array([trim["alpha"], trim["throttle"]])
        cruise_steps += 1
        if fuel_kg <= 0.0:
            raise RuntimeError("Fuel exhausted during cruise.")
        if cruise_steps > 20000:
            raise RuntimeError("Cruise loop exceeded safety limit.")

    cruise_range_nm = x_m * M_TO_NM - climb_range_nm

    # --- descent ---
    descent_guess = np.array([math.radians(1.0), 0.15])
    descent_steps = 0
    while h_m > 1.0:
        mass_kg = empty_kg + ac.m_payload + fuel_kg
        trim = solve_trim_v_gamma(ac, h_m, mass_kg, descent_speed_mps, descent_gamma_rad, guess=descent_guess, flap="landing")
        if not trim["success"]:
            raise RuntimeError(f"Descent trim failed at h = {h_m:.1f} m, residuals = {trim['residuals']}")

        dx = trim["V"] * math.cos(trim["gamma"]) * dt
        dz = trim["V"] * math.sin(trim["gamma"]) * dt
        if h_m + dz < 0.0:
            frac = h_m / max(-dz, 1e-9)
            dx *= frac
            dz *= frac
            fuel_dt = dt * frac
        else:
            fuel_dt = dt

        x_m += dx
        h_m += dz
        h_m = max(h_m, 0.0)
        fuel_kg -= fuel_burn_rate_kg_s(ac, trim["throttle"], h_m) * fuel_dt
        fuel_kg = max(fuel_kg, 0.0)
        t_s += fuel_dt
        mass_kg = empty_kg + ac.m_payload + fuel_kg
        log_state("descent", trim, mass_kg, fuel_kg)

        descent_guess = np.array([trim["alpha"], trim["throttle"]])
        descent_steps += 1
        if fuel_kg <= 0.0:
            raise RuntimeError("Fuel exhausted during descent.")
        if descent_steps > 5000:
            raise RuntimeError("Descent loop exceeded safety limit.")

    arr = {k: np.array([row[k] for row in history]) for k in history[0].keys() if k != "segment"}
    segments = np.array([row["segment"] for row in history])

    landing_mass_kg = empty_kg + ac.m_payload + fuel_kg
    runway = estimate_takeoff_landing(ac, takeoff_mass_kg, landing_mass_kg)

    total_range_nm = x_m * M_TO_NM
    total_time_hr = t_s / 3600.0
    fuel_used_kg = ac.m_fuel_max - fuel_kg
    avg_cruise_throttle = np.mean(arr["throttle"][segments == "cruise"])

    print("Mission summary")
    print("---------------")
    print(f"Climb range:             {climb_range_nm:7.1f} nm")
    print(f"Cruise range:            {cruise_range_nm:7.1f} nm")
    print(f"Total range:             {total_range_nm:7.1f} nm")
    print(f"Total time:              {total_time_hr:7.2f} hr")
    print(f"Fuel used:               {fuel_used_kg:7.1f} kg")
    print(f"Fuel remaining:          {fuel_kg:7.1f} kg")
    print(f"Cruise altitude:         {cruise_alt_m:7.0f} m")
    print(f"Cruise speed target:     {cruise_speed_mps * MPS_TO_KT:7.1f} kt")
    print(f"Average cruise throttle: {avg_cruise_throttle:7.2f}")
    print()
    print("Simple runway estimates")
    print("-----------------------")
    print(f"Takeoff ground roll:     {runway['takeoff_ground_roll_m']:7.0f} m")
    print(f"Landing ground roll:     {runway['landing_ground_roll_m']:7.0f} m")
    print()
    print("Requirements check")
    print("------------------")
    print(f"500 nm range met:        {total_range_nm >= 500.0}")
    print(f"175 kt dash met:         {False if dash_sl is None else dash_sl_kt >= ac.dash_speed_req_kt}")
    print(f"1000 ft takeoff met:     {runway['takeoff_ground_roll_m'] <= 1000.0 * 0.3048}")
    print(f"1000 ft landing met:     {runway['landing_ground_roll_m'] <= 1000.0 * 0.3048}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(arr["x_m"] * M_TO_NM, arr["h_m"])
    axs[0].set_ylabel("Altitude (m)")
    axs[0].set_title("Piper Cherokee-like Mission Profile")
    axs[0].grid(True)

    axs[1].plot(arr["x_m"] * M_TO_NM, arr["V_kt"])
    axs[1].set_ylabel("Speed (kt)")
    axs[1].grid(True)

    axs[2].plot(arr["x_m"] * M_TO_NM, arr["fuel_kg"])
    axs[2].set_ylabel("Fuel (kg)")
    axs[2].set_xlabel("Range (nm)")
    axs[2].grid(True)

    fig.tight_layout()
    fig.savefig(mission_plot, dpi=160)
    plt.close(fig)
    print(f"\nSaved mission plot to: {mission_plot}")

    make_constraint_diagram(ac, takeoff_mass_kg, dash_plot)
    print(f"Saved dash/constraint plot to: {dash_plot}")

    # ── Additional reporting ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  FULL AERODYNAMIC REPORT")
    print("="*50)
    report_aero_polar(ac, takeoff_mass_kg, ac.cruise_alt_m,
                      outdir / "aero_polar.png")

    print("\n" + "="*50)
    print("  PERFORMANCE METRICS")
    print("="*50)
    service_ceiling = report_performance(ac, takeoff_mass_kg,
                                         outdir / "performance.png")

    print("\n" + "="*50)
    print("  V-n DIAGRAM")
    print("="*50)
    report_vn_diagram(ac, takeoff_mass_kg, ac.cruise_alt_m,
                      outdir / "vn_diagram.png")

    print("\n" + "="*50)
    print("  WING BENDING")
    print("="*50)
    report_wing_bending(ac, takeoff_mass_kg, outdir / "wing_bending.png")

    # CG and trim — use the items dict and reasonable CG/AC locations
    x_cg = 2.45   # m from nose — update from your CG calculation
    x_ac  = 2.60  # m from nose — approx 0.25 MAC position + wing LE
    print("\n" + "="*50)
    print("  TRIM ANALYSIS")
    print("="*50)
    _, x_cg, _ = compute_cg_with_wing(ac)
    report_trim(ac, takeoff_mass_kg, ac.cruise_alt_m,
                x_cg, x_ac, outdir / "trim_analysis.png")

    print("\n" + "="*50)
    print("  CG ENVELOPE")
    print("="*50)
    report_cg_envelope(items, outdir / "cg_envelope.png")

    # Landing gear — update x_main, x_nose, track to match your design
    print("\n" + "="*50)
    print("  LANDING GEAR METRICS")
    print("="*50)
    # Compute CG bounds based on the wing-aware calculation
    _, x_cg_computed, cg_items = compute_cg_with_wing(ac)
    # Use ±0.05 m as CG envelope bounds around computed CG
    x_cg_fwd = x_cg_computed - 0.05
    x_cg_aft = x_cg_computed + 0.05
    report_landing_gear(ac,
                    x_cg_fwd = x_cg_fwd,
                    x_cg_aft = x_cg_aft,
                    x_main   = ac.x_main_gear,
                    x_nose   = ac.x_nose_gear,
                    track    = ac.gear_track,
                    outpath  = outdir / "landing_gear.png")

    print("\n" + "="*50)
    print("  TRADE STUDIES")
    print("="*50)
    report_trade_studies(ac, outdir / "trade_studies.png")

    print("\nAll plots saved to:", outdir)

    return {
        "empty_mass_kg": empty_kg,
        "takeoff_mass_kg": takeoff_mass_kg,
        "dash_sl_kt": dash_sl_kt,
        "dash_cruise_kt": dash_cruise_kt,
        "range_nm": total_range_nm,
        "fuel_remaining_kg": fuel_kg,
        "takeoff_roll_m": runway["takeoff_ground_roll_m"],
        "landing_roll_m": runway["landing_ground_roll_m"],
    }


if __name__ == "__main__":


    fuel_guess = 90

    for _ in range(20):
        ac = PiperCherokeeLike()
        ac.m_fuel_max = fuel_guess
        reserve_fuel_kg = 0.1 * fuel_guess

        with redirect_stdout(StringIO()):
            mission_result = run_mission(ac)
        fuel_used = ac.m_fuel_max - mission_result["fuel_remaining_kg"]
        fuel_required = fuel_used + reserve_fuel_kg

        if abs(fuel_required - fuel_guess) < 0.1:
            break

        fuel_guess = 0.5 * fuel_guess + 0.5 * fuel_required

    ac = PiperCherokeeLike()
    ac.m_fuel_max = fuel_guess
    run_mission(ac)
    print(f"Final fuel guess: {fuel_guess:.1f} kg")
