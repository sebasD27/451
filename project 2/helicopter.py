import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import brentq

@dataclass
class Component:
    name: str
    weight: float   # lbf
    x_cg: float     # ft from nose (longitudinal)
    z_cg: float     # ft from keel (vertical)

N_CREW        = 2
N_PAX         = 8
W_PERSON      = 200       # lbf per person (including luggage)
W_CREW        = N_CREW * W_PERSON
W_PAX         = N_PAX  * W_PERSON
W_ELECTRONICS = 1000      # lbf, electronics bay
W_WEAPONS     = 1000      # lbf, defensive armament
W_PAYLOAD     = W_CREW + W_PAX + W_ELECTRONICS + W_WEAPONS  # total fixed payload

RANGE_NMI     = 250       # nmi
CRUISE_KTS    = 150       # kts
MAX_KTS       = 170       # kts
CRUISE_ALT_FT = 3000      # ft
MAX_ALT_FT    = 14000     # ft
HELIPAD_FT    = 40        # ft, max rotor diameter constraint

#main rotor
def weight_main_rotor_blades(b, R, Omega):
    """
    W_bM = 0.026 * b^0.66 * R^1.3 * (Omega*R)^0.67
    b     : number of blades
    R     : rotor radius (ft)
    Omega : rotor angular velocity (rad/s)
    """
    return 0.026 * (b**0.66) * (R**1.3) * ((Omega * R)**0.67)

#rotor hub
def weight_main_rotor_hub(b, R, Omega, GW):

    g = 32.2  # ft/s^2
    W_bM = weight_main_rotor_blades(b, R, Omega)

    # Blade flapping moment of inertia (uniform blade approximation)
    J = (W_bM / g) * (R**2) / 3.0

    W_H = (0.0037 * (b**0.28) * (R**1.5) 
           * ((Omega * R)**0.43) 
           * (0.67 * W_bM + g * J / R**2)**0.55)

    return W_H
#tail rotor

def weight_tail_rotor(R_T, P_total_hp, Omega_M):
    """
    W_T = 1.4 * R_T^0.90 * (Transmission hp rating / Omega_M)^0.90
    R_T       : tail rotor radius (ft)
    P_total_hp: transmission hp rating
    Omega_M   : main rotor angular velocity (rad/s)
    """
    return 1.4 * (R_T**0.90) * (P_total_hp / Omega_M)**0.90

#fuselage
def weight_fuselage(GW, L_fus, W_fus, H_fus):
    """
    W_F = 6.9 * (GW/1000)^0.49 * L_F^0.61 * (wetted_area)^0.25
    Approximate wetted area from fuselage box dimensions
    wetted ~ 2*(L*W + L*H + W*H)
    """
    S_wet = 2 * (L_fus * W_fus + L_fus * H_fus + W_fus * H_fus)
    return 6.9 * ((GW / 1000)**0.49) * (L_fus**0.61) * (S_wet**0.25)

#landing gear
def weight_landing_gear(GW, n_legs=3, retractable=False):
    """
    W_LG = 40 * (GW/1000)^0.87 * n_legs^0.34
    +10% if retractable
    """
    W = 40 * ((GW / 1000)**0.87) * (n_legs**0.34)
    if retractable:
        W *= 1.10
    return W

#motor 

def weight_motors(n_motors, P_motor_hp):
    """
    Electric motors — approximate: ~0.5 lb/hp for modern high-power density motors
    """
    lbs_per_hp = (0.7457 * 2.2046) / 3.0

    return n_motors * P_motor_hp * lbs_per_hp

#drive system (gearboxes, shafts, etc.)
def weight_drive_system(P_total_hp, P_tail_hp, Omega_M, Omega_T, rpm_eng, n_gearboxes=2):
   
    W_DS = (13.6 
            * (P_total_hp**0.82)
            * (rpm_eng / 1000)**0.037
            * ((P_tail_hp / P_total_hp) * (Omega_M / Omega_T))**0.068
            * (n_gearboxes**0.066)
            / (Omega_M**0.64))
    return W_DS

#cockpit controls
def weight_cockpit_controls(GW):
    """W_CC = 11.5 * (GW/1000)^0.40"""
    return 11.5 * ((GW / 1000)**0.40)

#flight control system (actuators, linkages, etc.)
def weight_system_controls(b, Omega_M, R, c):
    """W_SC = 36 * b^2.2 * (Omega*R/1000)^3.2"""
    return 36 * (b**2.2) * (c**2.2) * ((Omega_M * R / 1000)**3.2)

def weight_instruments(GW):
    """W_T = 3.5 * (GW/1000)^1.5"""
    return 3.5 * ((GW / 1000)**1.3)

def weight_hydraulics(b, Omega_M, R, c):
    """W_hyd = 37 * b^0.13 * (Omega*R/1000)^2.1"""
    return 37 * (b**0.13) * (c**1.3) * ((Omega_M * R / 1000)**2.1)

def weight_electrical(P_trans_hp, GW, W_hyd):
    """W_EL = 9.6*(P_trans)^0.45 / (GW/1000)^0.40 - W_hyd"""
    return 9.6 * (P_trans_hp**0.65) / ((GW / 1000)**0.40) - W_hyd

def weight_avionics(level='avg'):
    """W_av = 50 (low), 150 (avg), 400 (high)"""
    return {'low': 50, 'avg': 150, 'high': 400}[level]

def weight_furnishings(GW, level='avg'):
    """W_FE = 6/13/23 * (GW/1000)^1.3 for low/avg/high"""
    k = {'low': 6, 'avg': 13, 'high': 23}[level]
    return k * ((GW / 1000)**1.3)

def weight_ac_antiice(GW):
    """W_ACAI = 8 * (GW/1000)"""
    return 8 * (GW / 1000)

def weight_manufacturing_variation(GW):
    """W_MV = 4 * (GW/1000)"""
    return 4 * (GW / 1000)


def weight_total(GW, b, R, Omega_M, c, P_total_hp, P_tail_hp, rpm_eng, n_motors, P_motor_hp):


    W_bM   = weight_main_rotor_blades(b, R, Omega_M)
    W_H    = weight_main_rotor_hub(b, R, Omega_M, GW)
    W_T    = weight_tail_rotor(R_T=R/4, P_total_hp=P_total_hp, Omega_M=Omega_M)
    W_F    = weight_fuselage(GW, L_fus=R*2.5, W_fus=R/2, H_fus=R/2)
    W_m    = weight_motors(n_motors, P_motor_hp)
    W_LG   = weight_landing_gear(GW)
    W_DS   = weight_drive_system(P_total_hp, P_tail_hp, Omega_M, Omega_T=Omega_M/5, rpm_eng=rpm_eng)
    W_CC   = weight_cockpit_controls(GW)
    W_SC   = weight_system_controls(b, Omega_M, R, c)
    W_inst = weight_instruments(GW)
    W_hyd  = weight_hydraulics(b, Omega_M, R, c)
    W_EL   = weight_electrical(P_total_hp, GW, W_hyd)
    W_av   = weight_avionics(level='avg')
    W_FE   = weight_furnishings(GW, level='avg')
    W_ACAI = weight_ac_antiice(GW)
    W_MV   = weight_manufacturing_variation(GW)

    ETOW = (W_bM + W_H + W_T + W_F + W_LG + W_DS + W_CC + 
            W_SC + W_hyd + W_EL + W_av + W_FE + W_ACAI + 
            W_MV  + W_m + W_inst)

    L  = 50
    H  = 10 

    components = [
        # Propulsion
        Component("Main Rotor Blades", W_bM,            0.5*L,  H*0.9),
        Component("Main Rotor Hub",    W_H,             0.5*L,  H*0.8),
        Component("Tail Rotor",        W_T,             0.9*L,  H*0.9),
        Component("Fuselage",          W_F,             0.5*L,  H*0.5),
        Component("Motors",           W_m,              0.5*L,  H*0.7),
        Component("Landing Gear",      W_LG,            0.5*L,   H*0.2),     
        Component("Drive System",     W_DS,             0.45*L,  H*0.8),

        # Systems
        Component("Cockpit Controls", W_CC,             0.10*L,  0.6*H),
        Component("System Controls",  W_SC,             0.45*L,  0.7*H),
        Component("Instruments",      W_inst,           0.08*L,  0.7*H),
        Component("Hydraulics",       W_hyd,            0.45*L,  0.4*H),
        Component("Electrical",       W_EL,             0.45*L,  0.3*H),
        Component("Avionics",        W_av,              0.45*L,  0.2*H),
        Component("Furnishings",     W_FE,              0.5*L,   H*0.5),
        Component("AC/Anti-ice",     W_ACAI,            0.5*L,   H*0.4),
        Component("Manufacturing Var.", W_MV,           0.5*L,   H*0.3),
 
        # Payload
        Component("Crew",                W_CREW  ,      0.20*L,  0.5*H),
        Component("Passengers",          W_PAX,         0.45*L,  0.5*H),
        Component("Electronics Bay",     W_ELECTRONICS, 0.35*L,  0.3*H),
        Component("Weapons Bay",         W_WEAPONS,     0.55*L,  0.3*H),
    ]

    W_total = sum(c.weight for c in components)

    # Weighted CG
    x_cg = sum(c.weight * c.x_cg for c in components) / W_total
    z_cg = sum(c.weight * c.z_cg for c in components) / W_total

    return ETOW

w = weight_total(GW=20000, b=2, R=20, Omega_M=30, c=2, P_total_hp=5400, P_tail_hp=800, rpm_eng=5000, n_motors=2, P_motor_hp=2700)    

print("heres w:", w)



def forward_flight_power(V_fwd, V_c, GW, rho, R, Omega, 
                          sigma, Cd0, kappa, f, K=4.7):
    """
    Leishman forward flight power
    V_fwd : forward airspeed (ft/s)
    V_c   : climb velocity (ft/s), 0 in cruise
    """
    A  = np.pi * R**2
    mu = V_fwd / (Omega * R)        # advance ratio
    CT = GW / (rho * A * (Omega * R)**2)  # thrust coefficient
    
    CP = (kappa * CT**2 / (2 * mu + 1e-6)   # avoid divide by zero at hover
        + sigma * Cd0 / 8 * (1 + K * mu**2)
        + 0.5 * (f/A) * mu**3
        + V_c * GW / (rho * A * (Omega * R)**3))
    
    P = rho * A * (Omega * R)**3 * CP   # power in ft*lbf/s
    P_hp = P / 550                       # convert to horsepower
    return P_hp

def total_power(V_fwd, V_c, GW, rho, R, Omega, sigma, Cd0, kappa, f):
    if V_fwd < 1.0:  # hover / vertical flight
        # use hover equation
        A  = np.pi * R**2
        vh = np.sqrt(GW / (2 * rho * A))
        Ph = kappa * GW * vh  # hover power (ft*lbf/s)
        x  = V_c / vh
        if x >= 0:
            ratio = x/2 + np.sqrt((x/2)**2 + 1)
        elif x >= -2:
            ratio = x + (kappa - 1.125*x - 1.372*x**2 
                        - 1.718*x**3 - 0.655*x**4)
        else:
            ratio = x/2 - np.sqrt((x/2)**2 - 1)
        return (Ph * ratio) / 550  # hp

    else:  # forward flight
        # use Leishman equation
        return forward_flight_power(V_fwd, V_c, GW, rho, 
                                     R, Omega, sigma, Cd0, kappa, f)

def mission_energy(GW, h_cruise, V_climb, V_cruise, V_descent, range_ft, 
                   rho_sl, rho_cruise, R, Omega, sigma, Cd0, kappa, f, eta):    
    # PHASE 1: Vertical Climb (0 → h_cruise)
    # -------------------------------------------------------------------------
    t_climb   = h_cruise / V_climb                          # seconds
    P_climb   = total_power(0, V_climb, GW, rho_sl, R, Omega,
                             sigma, Cd0, kappa, f)          # hp (use SL density, conservative)
    E_climb   = P_climb * 550 * t_climb / eta               # ft*lbf
    E_climb_Wh = E_climb / 2655.22                          # convert ft*lbf to Wh

    # -------------------------------------------------------------------------
    # PHASE 2: Cruise (at h_cruise, V_cruise, V_c=0)
    # -------------------------------------------------------------------------
    P_cruise  = total_power(V_cruise, 0, GW, rho_cruise, R, Omega,
                             sigma, Cd0, kappa, f)          # hp

    # Subtract horizontal distance covered during climb/descent
    # (climb/descent are vertical so no horizontal distance)
    range_cruise_ft = range_ft                              # all range in cruise
    t_cruise        = range_cruise_ft / V_cruise            # seconds
    E_cruise        = P_cruise * 550 * t_cruise / eta       # ft*lbf
    E_cruise_Wh     = E_cruise / 2655.22                    # Wh

    # -------------------------------------------------------------------------
    # PHASE 3: Vertical Descent (h_cruise → 0)
    # -------------------------------------------------------------------------
    t_descent  = h_cruise / V_descent                       # seconds
    P_descent  = total_power(0, -V_descent, GW, rho_cruise, R, Omega,
                              sigma, Cd0, kappa, f)         # hp
    E_descent  = P_descent * 550 * t_descent / eta          # ft*lbf
    E_descent_Wh = E_descent / 2655.22                      # Wh

    # -------------------------------------------------------------------------
    # TOTALS
    # -------------------------------------------------------------------------
    E_total_Wh = E_climb_Wh + E_cruise_Wh + E_descent_Wh
    return E_total_Wh

E = mission_energy(GW=20000, h_cruise=3000, V_climb=500, V_cruise=170, V_descent=500,
                   range_ft=RANGE_NMI*6076, rho_sl=0.002377, rho_cruise=0.002, R=20, Omega=30, sigma=0.08, Cd0=0.01, kappa=1.15, f=10, eta=0.7)
print(f"Estimated mission energy: {E:.1f} Wh")

batW = E / 1000
print(f"Estimated battery weight: {batW:.1f} lbf")

def residual(GW):
    W = weight_total(GW=GW, b=2, R=20, Omega_M=30, c=1, P_total_hp=5400, P_tail_hp=800, rpm_eng=5000, n_motors=2, P_motor_hp=2700)
    E = mission_energy(GW=GW, h_cruise=3000, V_climb=20, V_cruise=170, V_descent=10,
                   range_ft=RANGE_NMI*6076, rho_sl=0.002377, rho_cruise=0.002, R=20, Omega=30, sigma=0.08, Cd0=0.01, kappa=1.15, f=10, eta=0.9)
    W_bat = E / 3187 # battery weight based on energy requirement and specific energy
    return GW - (W + W_bat)

print(f"residual(10000)  = {residual(10000):.1f}")
print(f"residual(500000) = {residual(500000):.1f}")

GW_solution = brentq(residual, 10000, 500000)

print(f"Estimated Gross Weight: {GW_solution:.1f} lbf")

P_main = 5000
P_avail = P_main 


P_req = forward_flight_power(V_fwd=170, V_c=0, GW=GW_solution, rho=0.002, R=20, Omega=30, sigma=0.08, Cd0=0.01, kappa=1.15, f=10)




def max_speed(rho, GW, ):
    P_avail = 20000 # constant for electric
    
    def excess(V_fwd):
        P_req = forward_flight_power(V_fwd=V_fwd, V_c=0, GW=GW_solution, rho=rho, R=20, Omega=30, sigma=0.08, Cd0=0.01, kappa=1.15, f=10)
        return P_avail - P_req
    
    # Debug — check signs before brentq
    print(f"  excess(1.0)    = {excess(1.0):.1f}")
    print(f"  excess(1000.0) = {excess(1000.0):.1f}")

    # Find speed where excess power = 0
    V_max = brentq(excess, 1.0, 1000.0)  # ft/s
    return V_max / 1.68781  # convert to knots

V_max_cruise = max_speed(rho=0.002, GW=GW_solution)

print(f"Estimated max cruise speed: {V_max_cruise:.1f} knots")