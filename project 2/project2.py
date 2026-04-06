import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import brentq

# =============================================================================
# CONSTANTS & REQUIREMENTS
# =============================================================================
@dataclass
class Component:
    name: str; weight: float; x_cg: float; z_cg: float

N_CREW=2; N_PAX=8; W_PERSON=200
W_CREW=N_CREW*W_PERSON; W_PAX=N_PAX*W_PERSON
W_ELECTRONICS=1000; W_WEAPONS=1000
W_PAYLOAD = W_CREW+W_PAX+W_ELECTRONICS+W_WEAPONS  # 4000 lbf

RANGE_NMI=250; CRUISE_KTS=150; MAX_KTS=170
CRUISE_ALT_FT=3000; HELIPAD_FT=40
KTS_TO_FPS=1.68781; RHO_SL=0.002377

# NOTE: Specific energy of 113 Wh/lb is BELOW current Li-ion.
# This design requires ~3000+ Wh/lb — not physically achievable.
# Using 3187 Wh/lb here to demonstrate closure at the minimum feasible value.
# In a real design, switch to tiltrotor or hybrid-electric.
SPEC_ENERGY_WH_LB = 2000   # Wh/lb — minimum feasible for this mission

# =============================================================================
# ATMOSPHERE
# =============================================================================
def atmosphere(h_ft):
    T_sl=518.67; L=0.003566
    T=T_sl-L*h_ft
    return RHO_SL*(T/T_sl)**4.256

# =============================================================================
# WEIGHT FUNCTIONS
# =============================================================================
def weight_main_rotor_blades(b,R,Omega):
    return 0.026*(b**0.66)*(R**1.3)*((Omega*R)**0.67)

def weight_main_rotor_hub(b,R,Omega,GW):
    g=32.2; W_bM=weight_main_rotor_blades(b,R,Omega)
    J=(W_bM/g)*(R**2)/3.0
    return 0.0037*(b**0.28)*(R**1.5)*((Omega*R)**0.43)*(0.67*W_bM+g*J/R**2)**0.55

def weight_tail_rotor(R_T,P_total_hp,Omega_M):
    return 1.4*(R_T**0.90)*(P_total_hp/Omega_M)**0.90

def weight_fuselage(GW,L_fus,W_fus,H_fus):
    S_wet=2*(L_fus*W_fus+L_fus*H_fus+W_fus*H_fus)
    return 6.9*((GW/1000)**0.49)*(L_fus**0.61)*(S_wet**0.25)

def weight_landing_gear(GW,n_legs=3,retractable=True):
    W=40*((GW/1000)**0.87)*(n_legs**0.34)
    return W*1.10 if retractable else W

def weight_motors(n_motors,P_motor_hp):
    return n_motors*P_motor_hp*(0.7457*2.2046)/3.0

#def weight_motor_controllers(n_motors,P_motor_hp):
#    return n_motors*P_motor_hp*0.15

def weight_drive_system(P_total_hp,P_tail_hp,Omega_M,Omega_T,rpm_eng,n_gb=2):
    return (13.6*(P_total_hp**0.82)*(rpm_eng/1000)**0.037
            *((P_tail_hp/P_total_hp)*(Omega_M/Omega_T))**0.068
            *(n_gb**0.066)/(Omega_M**0.64))

def weight_cockpit_controls(GW):    return 11.5*((GW/1000)**0.40)
def weight_system_controls(b,Omega_M,R,c): return 36*(b**2.2)*(c**2.2)*((Omega_M*R/1000)**3.2)
def weight_instruments(GW):         return 3.5*((GW/1000)**1.3)
def weight_hydraulics(b,Omega_M,R,c): return 37*(b**0.13)*(c**1.3)*((Omega_M*R/1000)**2.1)
def weight_electrical(P_hp,GW,W_hyd): return max(9.6*(P_hp**0.65)/((GW/1000)**0.40)-W_hyd, 50)
def weight_avionics():              return 400   # high — presidential
def weight_furnishings(GW):         return 23*((GW/1000)**1.3)  # high — VIP
def weight_ac_antiice(GW):          return 8*(GW/1000)
def weight_manufacturing_variation(GW): return 4*(GW/1000)

# =============================================================================
# EMPTY WEIGHT (no payload, no battery)
# =============================================================================
def empty_weight(GW, b=2, R=20, Omega_M=30, c=1,
                 P_total_hp=2000, P_tail_hp=800,
                 rpm_eng=5000, n_motors=2, P_motor_hp=2700):
    W_bM  = weight_main_rotor_blades(b,R,Omega_M)
    W_H   = weight_main_rotor_hub(b,R,Omega_M,GW)
    W_T   = weight_tail_rotor(R/4,P_total_hp,Omega_M)
    W_F   = weight_fuselage(GW,R*2.5,R/2,R/2)
    W_m   = weight_motors(n_motors,P_motor_hp)
   # W_esc = weight_motor_controllers(n_motors,P_motor_hp)
    W_LG  = weight_landing_gear(GW)
    W_DS  = weight_drive_system(P_total_hp,P_tail_hp,Omega_M,Omega_M/5,rpm_eng)
    W_CC  = weight_cockpit_controls(GW)
    W_SC  = weight_system_controls(b,Omega_M,R,c)
    W_inst= weight_instruments(GW)
    W_hyd = weight_hydraulics(b,Omega_M,R,c)
    W_EL  = weight_electrical(P_total_hp,GW,W_hyd)
    W_av  = weight_avionics()
    W_FE  = weight_furnishings(GW)
    W_AC  = weight_ac_antiice(GW)
    W_MV  = weight_manufacturing_variation(GW)
    return (W_bM+W_H+W_T+W_F+W_m+W_LG+W_DS+
            W_CC+W_SC+W_inst+W_hyd+W_EL+W_av+W_FE+W_AC+W_MV)

# =============================================================================
# POWER MODEL
# =============================================================================
def hover_power(V_c, GW, rho, R, kappa=1.15):
    """Vertical flight power via momentum theory"""
    A=np.pi*R**2; vh=np.sqrt(GW/(2*rho*A)); x=V_c/vh
    if x>=0:    ratio=x/2+np.sqrt((x/2)**2+1)
    elif x>=-2: ratio=x+(kappa-1.125*x-1.372*x**2-1.718*x**3-0.655*x**4)
    else:       ratio=x/2-np.sqrt((x/2)**2-1)
    return max(kappa*GW*vh*ratio+GW*V_c, 0)/550

def forward_flight_power(V_fwd, V_c, GW, rho, R=20, Omega=30,
                          sigma=0.08, Cd0=0.01, kappa=1.15, f=10, K=4.7):
    """Leishman forward flight power — V_fwd and V_c in ft/s"""
    A=np.pi*R**2; mu=V_fwd/(Omega*R)
    CT=GW/(rho*A*(Omega*R)**2)
    CP=(kappa*CT**2/(2*mu+1e-9)
       +sigma*Cd0/8*(1+K*mu**2)
       +0.5*(f/A)*mu**3
       +V_c*GW/(rho*A*(Omega*R)**3))
    return rho*A*(Omega*R)**3*CP/550

def total_power(V_fwd, V_c, GW, rho, R=20, Omega=30,
                sigma=0.08, Cd0=0.01, kappa=1.15, f=10):
    if abs(V_fwd)<1.0:
        return hover_power(V_c,GW,rho,R,kappa)
    return forward_flight_power(V_fwd,V_c,GW,rho,R,Omega,sigma,Cd0,kappa,f)

# =============================================================================
# MISSION ENERGY  (all velocities in ft/s)
# =============================================================================
def mission_energy(GW, V_climb=20, V_cruise_fps=None, V_descent=10,
                   h_cruise=3000, range_nmi=250,
                   R=20, Omega=30, sigma=0.08, Cd0=0.01, kappa=1.15, f=10, eta=0.9):
    if V_cruise_fps is None:
        V_cruise_fps = CRUISE_KTS * KTS_TO_FPS   # 150 kts → ft/s

    rho_sl  = atmosphere(0)
    rho_cr  = atmosphere(h_cruise)
    conv    = 550/2655.22/eta   # hp → Wh conversion factor

    # Phase 1: vertical climb
    t_cl = h_cruise / V_climb
    E_cl = total_power(0, V_climb, GW, rho_sl, R, Omega, sigma, Cd0, kappa, f) * conv * t_cl

    # Phase 2: cruise
    t_cr = range_nmi*6076.12 / V_cruise_fps
    E_cr = total_power(V_cruise_fps, 0, GW, rho_cr, R, Omega, sigma, Cd0, kappa, f) * conv * t_cr

    # Phase 3: vertical descent
    t_de = h_cruise / V_descent
    E_de = total_power(0, -V_descent, GW, rho_cr, R, Omega, sigma, Cd0, kappa, f) * conv * t_de

    return E_cl+E_cr+E_de, E_cl, E_cr, E_de, t_cl, t_cr, t_de

# =============================================================================
# WEIGHT CLOSURE LOOP
# =============================================================================
def residual(GW):
    W_e   = empty_weight(GW)
    E,*_  = mission_energy(GW)
    W_bat = E / SPEC_ENERGY_WH_LB
    return GW - (W_e + W_bat + W_PAYLOAD)

print(f"residual(10000)  = {residual(10000):.1f}")
print(f"residual(500000) = {residual(500000):.1f}")

GW_solution = brentq(residual, 10000, 500000)
E_total,E_cl,E_cr,E_de,t_cl,t_cr,t_de = mission_energy(GW_solution)
W_bat_solution = E_total / SPEC_ENERGY_WH_LB
W_emp_solution = empty_weight(GW_solution)

print(f"\n{'='*50}")
print(f"  MTOW          : {GW_solution:.1f} lbf")
print(f"  Empty weight  : {W_emp_solution:.1f} lbf  ({W_emp_solution/GW_solution*100:.1f}%)")
print(f"  Battery weight: {W_bat_solution:.1f} lbf  ({W_bat_solution/GW_solution*100:.1f}%)")
print(f"  Payload       : {W_PAYLOAD:.1f} lbf  ({W_PAYLOAD/GW_solution*100:.1f}%)")
print(f"  Total energy  : {E_total/1000:.1f} kWh")
print(f"{'='*50}")


def defensive_mission(GW_initial, params):
    """
    Mission profile:
    1. Cruise from White House to Camp David (~60 nmi)
    2. Emergency vertical climb to 3000 ft at max climb speed
    3. Deploy weapons — GW drops by 1000 lbf
    4. Fly at max speed to Andrews AFB (~10 nmi)
    """

    # Phase 1 — cruise to Camp David (full weight)
    GW1 = GW_initial
    E1, *_ = mission_energy(GW1, range_nmi=60, 
                             V_cruise_fps=CRUISE_KTS*KTS_TO_FPS)

    # Phase 2 — emergency vertical climb (full weight)
    rho_sl = atmosphere(0)
    V_climb_max = 30  # ft/s — max vertical climb speed
    t_climb = 3000 / V_climb_max
    P_climb = hover_power(V_climb_max, GW1, rho_sl, R=20)
    E2 = P_climb * 550 * t_climb / eta / 2655.22  # Wh

    # ---- WEAPONS DEPLOY HERE ----
    GW2 = GW1 - W_WEAPONS - W_ELECTRONICS   # 1000 lbf lighter

    # Phase 3 — max speed back to Andrews (lighter weight)
    rho_cr = atmosphere(3000)
    V_max = max_speed_at_altitude(3000, GW2)  # recompute with lighter GW
    E3, *_ = mission_energy(GW2, range_nmi=10,
                             V_cruise_fps=V_max*KTS_TO_FPS)

    return E1+E2+E3, GW2


# =============================================================================
# WEIGHT STATEMENT — individual components at converged GW
# =============================================================================
GW = GW_solution
b=2; R=20; Omega_M=30; c=1; P_total_hp=2000; P_tail_hp=800; rpm_eng=5000
n_motors=2; P_motor_hp=2700

weights = {
    "Main Rotor Blades"    : weight_main_rotor_blades(b,R,Omega_M),
    "Main Rotor Hub"       : weight_main_rotor_hub(b,R,Omega_M,GW),
    "Tail Rotor"           : weight_tail_rotor(R/4,P_total_hp,Omega_M),
    "Fuselage"             : weight_fuselage(GW,R*2.5,R/2,R/2),
    "Landing Gear"         : weight_landing_gear(GW),
    "Motors"               : weight_motors(n_motors,P_motor_hp),
    #"Motor Controllers"    : weight_motor_controllers(n_motors,P_motor_hp),
    "Drive System"         : weight_drive_system(P_total_hp,P_tail_hp,Omega_M,Omega_M/5,rpm_eng),
    "Cockpit Controls"     : weight_cockpit_controls(GW),
    "System Controls"      : weight_system_controls(b,Omega_M,R,c),
    "Instruments"          : weight_instruments(GW),
    "Hydraulics"           : weight_hydraulics(b,Omega_M,R,c),
    "Electrical"           : weight_electrical(P_total_hp,GW,weight_hydraulics(b,Omega_M,R,c)),
    "Avionics"             : weight_avionics(),
    "Furnishings"          : weight_furnishings(GW),
    "A/C & Anti-Ice"       : weight_ac_antiice(GW),
    "Mfg Variation"        : weight_manufacturing_variation(GW),
}
print(f"\n{'Component':<25} {'Weight (lbf)':>12} {'% MTOW':>8}")
print("-"*47)
for name,w in weights.items():
    print(f"  {name:<23} {w:>12.1f} {w/GW_solution*100:>7.1f}%")
print("-"*47)
print(f"  {'Empty Weight':<23} {W_emp_solution:>12.1f} {W_emp_solution/GW_solution*100:>7.1f}%")
print(f"  {'Battery':<23} {W_bat_solution:>12.1f} {W_bat_solution/GW_solution*100:>7.1f}%")
print(f"  {'Payload':<23} {W_PAYLOAD:>12.1f} {W_PAYLOAD/GW_solution*100:>7.1f}%")
print(f"  {'MTOW':<23} {GW_solution:>12.1f} {'100.0%':>8}")

# =============================================================================
# ENERGY PROFILE PLOT
# =============================================================================
times    = [0, t_cl/60, (t_cl+t_cr)/60, (t_cl+t_cr+t_de)/60]
energies = [E_total, E_total-E_cl, E_total-E_cl-E_cr, 0]

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
ax1.plot(times,[e/1000 for e in energies],'b-o',linewidth=2,markersize=8)
ax1.fill_between(times,[e/1000 for e in energies],alpha=0.2)
ax1.axvline(t_cl/60,color='gray',linestyle='--',alpha=0.5)
ax1.axvline((t_cl+t_cr)/60,color='gray',linestyle='--',alpha=0.5)
ax1.set_xlabel("Time (minutes)"); ax1.set_ylabel("Energy Remaining (kWh)")
ax1.set_title(f"Energy Onboard — 250 nmi Mission  (MTOW = {GW_solution:.0f} lbf)")
ax1.text(t_cl/120,          E_total/1000*0.9,'Climb',   ha='center',color='green',fontsize=10)
ax1.text((t_cl+t_cl+t_cr)/120/2+t_cl/60/2, E_total/1000*0.9,'Cruise\n150 kts @ 3000 ft',ha='center',color='blue',fontsize=10)
ax1.text((t_cl+t_cr)/60+t_de/120, E_total/1000*0.9,'Descent',ha='center',color='red',fontsize=10)
ax1.grid(True,alpha=0.3)

phases=['Climb','Cruise','Descent']
P_cl=total_power(0,20,GW_solution,atmosphere(0))
P_cr=total_power(CRUISE_KTS*KTS_TO_FPS,0,GW_solution,atmosphere(3000))
P_de=total_power(0,-10,GW_solution,atmosphere(3000))
bars=ax2.bar(phases,[P_cl,P_cr,P_de],color=['green','blue','red'],alpha=0.7,edgecolor='black')
ax2.set_ylabel("Power Required (hp)"); ax2.set_title("Power by Phase")
ax2.grid(True,alpha=0.3,axis='y')
for bar,p in zip(bars,[P_cl,P_cr,P_de]):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+50,f'{p:.0f}hp',ha='center',fontsize=10)
plt.tight_layout()
plt.savefig("energy_profile.png",dpi=150,bbox_inches='tight')
plt.show()

# =============================================================================
# MAX SPEED VS ALTITUDE
# =============================================================================
P_MAIN_HP = 2000   # main rotor motor power (hp)

def max_speed_at_altitude(h_ft, GW):
    rho = atmosphere(h_ft)
    def excess(V_fps):
        return P_MAIN_HP - forward_flight_power(V_fps, 0, GW, rho)
    # Check feasibility
    if excess(50) < 0:
        return 0   # can't even fly at this altitude
    try:
        V_max_fps = brentq(excess, 50.0, 2000.0)
        return V_max_fps / KTS_TO_FPS
    except:
        return 0

altitudes = np.linspace(0, 14000, 60)
V_max_kts = [max_speed_at_altitude(h, GW_solution) for h in altitudes]

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(V_max_kts, altitudes/1000, 'b-', linewidth=2)
ax.axvline(170, color='r', linestyle='--', label='170 kt requirement')
ax.set_xlabel("Maximum Speed (kts)"); ax.set_ylabel("Altitude (1000 ft)")
ax.set_title("Maximum Speed vs Altitude")
ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("max_speed.png",dpi=150,bbox_inches='tight')
plt.show()

# =============================================================================
# MAX RATE OF CLIMB VS ALTITUDE
# =============================================================================
def max_roc_at_altitude(h_ft, GW):
    rho = atmosphere(h_ft)
    
    # Sweep speeds to find max excess power
    V_range = np.linspace(10, 400, 200)  # ft/s
    roc_values = []
    for V in V_range:
        P_req = forward_flight_power(V, 0, GW, rho)
        P_excess = P_MAIN_HP - P_req
        roc = max(P_excess * 550 / GW * 60, 0)  # ft/min
        roc_values.append(roc)
    
    return max(roc_values)

ROC = [max_roc_at_altitude(h, GW_solution) for h in altitudes]
print("\nMax Rate of Climb at Altitudes:", max_roc_at_altitude(3000, GW_solution)  )


fig,ax = plt.subplots(figsize=(8,6))
ax.plot(ROC, altitudes/1000, 'g-', linewidth=2)
ax.axvline(100, color='r', linestyle='--', label='100 ft/min service ceiling criterion')
ax.set_xlabel("Max Rate of Climb (ft/min)"); ax.set_ylabel("Altitude (1000 ft)")
ax.set_title("Maximum Rate of Climb vs Altitude")
ax.legend(); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("rate_of_climb.png",dpi=150,bbox_inches='tight')
plt.show()

print("\nAll plots saved.")