"""
2DOF Aircraft Mission Simulator
================================
Solves the nonlinear equations of motion in the x-z plane:

  ΣFx = mẍ:  T cos(γ+α) - D cosγ - L sinγ = mẍ
  ΣFz = mz̈:  L cosγ + T sin(γ+α) - D sinγ - W = mz̈

Supports five segment types (as described in MAE 451):
  1. Constant Throttle & Rate of Climb  → fixed: δt, ż   | solve: α, V, γ
  2. Constant Airspeed & Throttle       → fixed: V, δt   | solve: α, ż, γ
  3. Constant Throttle & Flight Path Angle → fixed: δt, γ | solve: α, V, ż
  4. Constant Airspeed & Flight Path Angle → fixed: V, γ  | solve: α, ż, δt
  5. Fixed L/D Cruise (Breguet)         → fixed: α, V   | solve: ż, δt, γ

Usage:
  python mission_sim_2dof.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


# ──────────────────────────────────────────────────────────────────────────────
# Atmosphere  (ISA, troposphere + lower stratosphere)
# ──────────────────────────────────────────────────────────────────────────────
def isa_atmosphere(h: float):
    """
    International Standard Atmosphere.
    Returns: rho [kg/m³], P [Pa], T [K], a [m/s] (speed of sound)
    """
    T0, P0 = 288.15, 101325.0
    L_rate = 0.0065   # temperature lapse rate [K/m]
    g, R   = 9.80665, 287.058

    h = max(h, 0.0)   # clamp to sea level

    if h <= 11000.0:
        T = T0 - L_rate * h
        P = P0 * (T / T0) ** (g / (R * L_rate))
    else:
        # isothermal stratosphere above 11 km
        T11 = T0 - L_rate * 11000.0
        P11 = P0 * (T11 / T0) ** (g / (R * L_rate))
        T   = T11
        P   = P11 * np.exp(-g * (h - 11000.0) / (R * T11))

    rho = P / (R * T)
    a   = np.sqrt(1.4 * R * T)
    return rho, P, T, a


# ──────────────────────────────────────────────────────────────────────────────
# Aircraft Parameters
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AircraftParams:
    # ── Geometry ──────────────────────────────────────────────────────────────
    S:  float = 20.0     # wing reference area  [m²]
    AR: float = 8.0      # aspect ratio         [-]
    e:  float = 0.85     # Oswald efficiency    [-]

    # ── Aerodynamics ──────────────────────────────────────────────────────────
    CL0:  float = 0.20   # lift at zero AoA     [-]
    CLa:  float = 5.50   # lift-curve slope     [1/rad]
    CD0:  float = 0.020  # parasite drag        [-]
    # k = 1/(π AR e) computed as property

    # ── Mass ──────────────────────────────────────────────────────────────────
    m: float = 5000.0    # total mass  [kg]

    # ── Propulsion ────────────────────────────────────────────────────────────
    T_max_sl:   float = 40000.0  # max thrust at sea level [N]
    T_alt_exp:  float = 0.75     # density-ratio exponent  [-]

    @property
    def k(self) -> float:
        """Induced drag factor  k = 1/(π AR e)"""
        return 1.0 / (np.pi * self.AR * self.e)

    @property
    def W(self) -> float:
        """Weight [N]"""
        return self.m * 9.80665


# ──────────────────────────────────────────────────────────────────────────────
# Aircraft Model
# ──────────────────────────────────────────────────────────────────────────────
class Aircraft:
    def __init__(self, params: AircraftParams = None):
        self.p = params or AircraftParams()

    # ── Aerodynamics ──────────────────────────────────────────────────────────
    def aero(self, V: float, alpha: float, h: float):
        """
        Returns: L [N], D [N], CL [-], CD [-]
        Parabolic drag polar: CD = CD0 + k·CL²
        """
        rho = isa_atmosphere(h)[0]
        q   = 0.5 * rho * V**2
        CL  = self.p.CL0 + self.p.CLa * alpha
        CD  = self.p.CD0 + self.p.k * CL**2
        return q * self.p.S * CL, q * self.p.S * CD, CL, CD

    # ── Propulsion ────────────────────────────────────────────────────────────
    def thrust(self, delta_t: float, h: float) -> float:
        """Available thrust with altitude lapse"""
        rho    = isa_atmosphere(h)[0]
        sigma  = rho / 1.225          # density ratio
        return delta_t * self.p.T_max_sl * sigma ** self.p.T_alt_exp

    # ── Equations of Motion ───────────────────────────────────────────────────
    def derivatives(self, t: float, state: np.ndarray,
                    delta_t: float, alpha: float) -> list:
        """
        State: [x, z, Vx, Vz]
        ODE right-hand side for RK45 integration.
        """
        x, z, Vx, Vz = state
        V     = np.sqrt(Vx**2 + Vz**2)
        gamma = np.arctan2(Vz, Vx)

        L, D, _, _ = self.aero(V, alpha, z)
        T          = self.thrust(delta_t, z)
        m          = self.p.m

        # Full nonlinear EoM (Image 3)
        ax = (T * np.cos(gamma + alpha)
              - D * np.cos(gamma)
              - L * np.sin(gamma)) / m

        az = (L * np.cos(gamma)
              + T * np.sin(gamma + alpha)
              - D * np.sin(gamma)
              - m * 9.80665) / m

        return [Vx, Vz, ax, az]


# ──────────────────────────────────────────────────────────────────────────────
# Mission Segment Definitions
# ──────────────────────────────────────────────────────────────────────────────
class SegmentType(Enum):
    CONST_THROTTLE_ROC  = "Constant Throttle & Rate of Climb"
    CONST_AIRSPEED_THR  = "Constant Airspeed & Throttle"
    CONST_THROTTLE_FPA  = "Constant Throttle & Flight Path Angle"
    CONST_AIRSPEED_FPA  = "Constant Airspeed & Flight Path Angle"
    FIXED_LD_CRUISE     = "Fixed L/D Cruise (Breguet)"


@dataclass
class MissionSegment:
    seg_type: SegmentType
    duration: float              # segment duration [s]

    # ── Fixed inputs (set those matching your segment type) ───────────────────
    delta_t: Optional[float] = None   # throttle fraction  [0–1]
    V:       Optional[float] = None   # airspeed           [m/s]
    zdot:    Optional[float] = None   # rate of climb      [m/s]  + = up
    gamma:   Optional[float] = None   # flight path angle  [rad]
    alpha:   Optional[float] = None   # angle of attack    [rad]

    dt: float = 2.0   # integration time step [s]


# ──────────────────────────────────────────────────────────────────────────────
# Trim Solver
# ──────────────────────────────────────────────────────────────────────────────
def _trim(ac: Aircraft, seg: MissionSegment,
          h: float, V_guess: float, gamma_guess: float
          ) -> Tuple[float, float, float, float]:
    """
    Solve for the trimmed (alpha, delta_t, gamma, V) at segment start.
    Steady-state: ẍ = z̈ = 0, so EoM reduce to two algebraic equations.
    """
    p = ac.p

    def eom_residuals(alpha, V, gamma, delta_t):
        L, D, _, _ = ac.aero(V, alpha, h)
        T          = ac.thrust(delta_t, h)
        g          = gamma
        res_x = T * np.cos(g + alpha) - D * np.cos(g) - L * np.sin(g)
        res_z = (L * np.cos(g) + T * np.sin(g + alpha)
                 - D * np.sin(g) - p.W)
        return res_x, res_z

    st = seg.seg_type

    # Physics-based velocity seed: V from L ≈ W at CL_seed
    rho_h     = isa_atmosphere(h)[0]
    CL_seed   = 0.6                              # typical cruise CL
    V_phys    = np.sqrt(2.0 * p.W / (rho_h * p.S * CL_seed))
    V_seed    = V_phys if V_guess < 1.0 else 0.5 * (V_phys + V_guess)

    # ── 1. Constant Throttle & Rate of Climb ──────────────────────────────────
    if st == SegmentType.CONST_THROTTLE_ROC:
        dt_fix   = seg.delta_t
        zdot_fix = seg.zdot

        def eqs(x):
            alpha, V = x
            V = max(V, 5.0)
            gamma = np.arcsin(np.clip(zdot_fix / V, -0.99, 0.99))
            rx, rz = eom_residuals(alpha, V, gamma, dt_fix)
            return [rx, rz]

        sol      = fsolve(eqs, [0.05, V_seed], full_output=True)
        alpha, V = sol[0]
        gamma    = np.arcsin(np.clip(zdot_fix / max(abs(V), 5.0), -0.99, 0.99))
        return alpha, dt_fix, gamma, abs(V)

    # ── 2. Constant Airspeed & Throttle ───────────────────────────────────────
    elif st == SegmentType.CONST_AIRSPEED_THR:
        V_fix  = seg.V
        dt_fix = seg.delta_t

        def eqs(x):
            alpha, gamma = x
            rx, rz = eom_residuals(alpha, V_fix, gamma, dt_fix)
            return [rx, rz]

        sol        = fsolve(eqs, [0.05, gamma_guess], full_output=True)
        alpha, gamma = sol[0]
        return alpha, dt_fix, gamma, V_fix

    # ── 3. Constant Throttle & Flight Path Angle ──────────────────────────────
    elif st == SegmentType.CONST_THROTTLE_FPA:
        dt_fix    = seg.delta_t
        gamma_fix = seg.gamma

        def eqs(x):
            alpha, V = x
            V = max(V, 5.0)
            rx, rz = eom_residuals(alpha, V, gamma_fix, dt_fix)
            return [rx, rz]

        sol      = fsolve(eqs, [0.05, V_seed], full_output=True)
        alpha, V = sol[0]
        return alpha, dt_fix, gamma_fix, abs(V)

    # ── 4. Constant Airspeed & Flight Path Angle ──────────────────────────────
    elif st == SegmentType.CONST_AIRSPEED_FPA:
        V_fix     = seg.V
        gamma_fix = seg.gamma

        def eqs(x):
            alpha, dt = x
            dt = np.clip(dt, 0.0, 1.0)
            rx, rz = eom_residuals(alpha, V_fix, gamma_fix, dt)
            return [rx, rz]

        sol       = fsolve(eqs, [0.05, 0.5], full_output=True)
        alpha, dt = sol[0]
        dt        = np.clip(dt, 0.0, 1.0)
        return alpha, dt, gamma_fix, V_fix

    # ── 5. Fixed L/D Cruise ───────────────────────────────────────────────────
    elif st == SegmentType.FIXED_LD_CRUISE:
        alpha_fix = seg.alpha
        V_fix     = seg.V
        gamma_fix = 0.0   # level cruise

        _, D, _, _ = ac.aero(V_fix, alpha_fix, h)
        T_needed   = D    # level: T = D (small-angle approx)
        dt         = np.clip(T_needed / ac.thrust(1.0, h), 0.0, 1.0)
        return alpha_fix, dt, gamma_fix, V_fix

    else:
        raise ValueError(f"Unknown segment type: {st}")


# ──────────────────────────────────────────────────────────────────────────────
# Mission Simulator
# ──────────────────────────────────────────────────────────────────────────────
class MissionSimulator:
    def __init__(self, aircraft: Aircraft):
        self.ac = aircraft

    # ── Single Segment ────────────────────────────────────────────────────────
    def run_segment(self, seg: MissionSegment,
                    state0: np.ndarray) -> dict:
        """
        Integrate one mission segment.
        state0 = [x0, z0, Vx0, Vz0]
        """
        x0, z0, Vx0, Vz0 = state0
        V0     = np.sqrt(Vx0**2 + Vz0**2) if (Vx0**2 + Vz0**2) > 0 else 60.0
        gam0   = np.arctan2(Vz0, Vx0)     if V0 > 1.0              else 0.0

        # Trim at the start-of-segment altitude
        alpha, delta_t, gamma, V = _trim(self.ac, seg, z0, V0, gam0)

        # Re-init state from trimmed values (direction is preserved by trim)
        Vx = V * np.cos(gamma)
        Vz = V * np.sin(gamma)
        s0 = np.array([x0, z0, Vx, Vz])

        t_eval = np.arange(0.0, seg.duration + seg.dt, seg.dt)
        t_eval = np.clip(t_eval, 0.0, seg.duration)

        sol = solve_ivp(
            self.ac.derivatives,
            (0.0, seg.duration),
            s0,
            args=(delta_t, alpha),
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6, atol=1e-9,
            dense_output=False,
        )

        t    = sol.t
        x_a  = sol.y[0]
        z_a  = sol.y[1]
        Vx_a = sol.y[2]
        Vz_a = sol.y[3]
        V_a  = np.sqrt(Vx_a**2 + Vz_a**2)
        gam_a = np.degrees(np.arctan2(Vz_a, Vx_a))

        # Derived quantities along trajectory
        n = len(t)
        L_a  = np.zeros(n); D_a = np.zeros(n)
        T_a  = np.zeros(n); CL_a = np.zeros(n); CD_a = np.zeros(n)
        for i in range(n):
            L_a[i], D_a[i], CL_a[i], CD_a[i] = self.ac.aero(V_a[i], alpha, z_a[i])
            T_a[i] = self.ac.thrust(delta_t, z_a[i])

        return {
            'name':        seg.seg_type.value,
            't':           t,
            'x':           x_a,
            'z':           z_a,
            'V':           V_a,
            'gamma_deg':   gam_a,
            'alpha_deg':   np.full(n, np.degrees(alpha)),
            'delta_t':     np.full(n, delta_t),
            'L':           L_a,
            'D':           D_a,
            'T':           T_a,
            'CL':          CL_a,
            'CD':          CD_a,
            'LD':          L_a / np.maximum(D_a, 1e-6),
            'state_final': sol.y[:, -1],
        }

    # ── Full Mission ──────────────────────────────────────────────────────────
    def run_mission(self, segments: List[MissionSegment],
                    state0: np.ndarray) -> List[dict]:
        """
        Run a complete multi-segment mission.
        Prints a one-line status per segment.
        """
        results      = []
        state        = state0.copy()
        t_offset     = 0.0

        print("=" * 60)
        print("  2DOF MISSION SIMULATION")
        print("=" * 60)

        for i, seg in enumerate(segments):
            print(f"  Seg {i+1:02d}  {seg.seg_type.value} …")
            res         = self.run_segment(seg, state)
            res['t']   += t_offset
            t_offset    = res['t'][-1]
            results.append(res)
            state       = res['state_final']

        print("=" * 60)
        print(f"  Total range : {results[-1]['x'][-1]/1000:.1f} km")
        print(f"  Total time  : {results[-1]['t'][-1]:.0f} s  "
              f"({results[-1]['t'][-1]/60:.1f} min)")
        print(f"  Final alt.  : {results[-1]['z'][-1]:.0f} m")
        print("=" * 60)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_mission(results: List[dict],
                 title: str = "2DOF Aircraft Mission Simulation",
                 save_path: str = None):
    n_seg  = len(results)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_seg))

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)

    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_traj  = fig.add_subplot(gs[0, :])    # full-width trajectory
    ax_V     = fig.add_subplot(gs[1, 0])
    ax_gamma = fig.add_subplot(gs[1, 1])
    ax_alpha = fig.add_subplot(gs[1, 2])
    ax_LD    = fig.add_subplot(gs[2, 0])
    ax_T     = fig.add_subplot(gs[2, 1])
    ax_dt    = fig.add_subplot(gs[2, 2])

    panel_cfg = [
        (ax_V,     'V [m/s]',      'Airspeed'),
        (ax_gamma, 'γ [deg]',      'Flight Path Angle'),
        (ax_alpha, 'α [deg]',      'Angle of Attack'),
        (ax_LD,    'L/D [-]',      'Lift-to-Drag'),
        (ax_T,     'Thrust [kN]',  'Thrust'),
        (ax_dt,    'δt [-]',       'Throttle'),
    ]
    keys = ['V', 'gamma_deg', 'alpha_deg', 'LD', 'T', 'delta_t']

    for i, res in enumerate(results):
        c   = colors[i]
        lbl = f"S{i+1}: {res['name'].split('&')[0].strip()}"
        ax_traj.plot(res['x']/1000, res['z'], color=c, lw=2, label=lbl)

        for (ax, ylabel, _), key in zip(panel_cfg, keys):
            val = res[key] / 1000 if key == 'T' else res[key]
            ax.plot(res['t'], val, color=c, lw=2)

    ax_traj.set_xlabel("Range [km]", fontsize=9)
    ax_traj.set_ylabel("Altitude [m]", fontsize=9)
    ax_traj.set_title("Flight Trajectory", fontsize=9)
    ax_traj.legend(loc='upper left', fontsize=7, ncol=n_seg)
    ax_traj.grid(True, alpha=0.25)

    for (ax, ylabel, ttl), _ in zip(panel_cfg, keys):
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ttl, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved → {save_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Example Mission
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. Define aircraft ───────────────────────────────────────────────────
    #
    #   Representative light transport / trainer:
    #     m = 5000 kg  →  W ≈ 49 kN
    #     T_max_sl = 13 000 N  →  T/W ≈ 0.26  (turboprop-like)
    #
    ac_params = AircraftParams(
        S=20.0, AR=8.0, e=0.85,
        CL0=0.20, CLa=5.5, CD0=0.020,
        m=5000.0, T_max_sl=13000.0, T_alt_exp=0.75,
    )
    aircraft = Aircraft(ac_params)
    sim      = MissionSimulator(aircraft)

    # ── 2. Initial state: [x, z, Vx, Vz] ────────────────────────────────────
    # Start near sea level at ~70 m/s (roughly 1.35× V_stall)
    state0 = np.array([0.0, 200.0, 70.0, 0.0])

    # ── 3. Define mission segments ───────────────────────────────────────────
    #
    #   Typical light-transport profile:
    #     Takeoff alt: 200 m → climb to ~2600 m → cruise → descend → approach
    #
    segments = [
        # Segment 1: Climb – constant throttle + fixed rate of climb
        #   Use ~50% throttle so the trim converges near cruise speed
        #   (high δt forces the solver onto the sub-stall drag branch)
        MissionSegment(
            seg_type = SegmentType.CONST_THROTTLE_ROC,
            duration = 480.0,       # 8 min climb
            delta_t  = 0.50,        # ~50% throttle → trim ≈ 75 m/s
            zdot     = 5.0,         # 5 m/s ≈ 1 000 ft/min
            dt       = 2.0,
        ),
        # Segment 2: Level cruise – constant airspeed, γ = 0
        #   (CONST_AIRSPEED_FPA with gamma=0 ensures level flight regardless of throttle)
        MissionSegment(
            seg_type = SegmentType.CONST_AIRSPEED_FPA,
            duration = 2400.0,      # 40 min cruise
            V        = 88.0,
            gamma    = 0.0,
            dt       = 10.0,
        ),
        # Segment 3: Descent – constant airspeed, −3° flight path
        MissionSegment(
            seg_type = SegmentType.CONST_AIRSPEED_FPA,
            duration = 360.0,       # 6 min descent
            V        = 75.0,
            gamma    = np.radians(-3.0),
            dt       = 2.0,
        ),
        # Segment 4: Short approach – fixed L/D level segment
        MissionSegment(
            seg_type = SegmentType.FIXED_LD_CRUISE,
            duration = 180.0,
            alpha    = np.radians(5.0),
            V        = 65.0,
            dt       = 5.0,
        ),
    ]

    # ── 4. Run ───────────────────────────────────────────────────────────────
    results = sim.run_mission(segments, state0)

    # ── 5. Segment summary table ─────────────────────────────────────────────
    print(f"\n{'Seg':<4} {'Type':<38} {'Dur[s]':<8} "
          f"{'Δx[km]':<8} {'Δz[m]':<8} {'V̄[m/s]':<8} {'L/D̄':<6}")
    print("-" * 82)
    for i, res in enumerate(results):
        dur  = res['t'][-1] - res['t'][0]
        dx   = (res['x'][-1] - res['x'][0]) / 1000
        dz   = res['z'][-1] - res['z'][0]
        Vmn  = res['V'].mean()
        LDmn = res['LD'].mean()
        name = res['name'][:36]
        print(f"{i+1:<4} {name:<38} {dur:<8.0f} "
              f"{dx:<8.2f} {dz:<8.1f} {Vmn:<8.1f} {LDmn:<6.2f}")

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    fig = plot_mission(results,
                       title="2DOF Aircraft Mission Simulation",
                       save_path="C:\\Users\\duran\\Downloads\\mission_sim_2dof.png")
    plt.show()