import pyCAPS
import numpy as np
import matplotlib.pyplot as plt


#geomtery(1-2): 3.5 hours

#weight model(3-4): 2.5 hours

#lift model(5): 3.5 hours

#D vs V (6): 2 hours

#Performance(7): 5-6 hours + 

#beam theory(8): 30 min 

#Sweep(9): 

#extra fun pain: 5 hours


filename = r"C:\Users\duran\Downloads\esp\ESP127\EngSketchPad\bin\451 plane.csm"
capsProblem = pyCAPS.Problem(problemName = "sin_Transport",
                             capsFile = filename,
                             outLevel = 0)

plane = capsProblem.geometry

foamDensity = 30

motorW = 0.079

wingloc = plane.despmtr["wingLoc"].value
chord = plane.outpmtr["chord"].value

hchord = plane.outpmtr["htail:chord"].value
htailloc = plane.outpmtr["htailLoc"].value

vchord = plane.outpmtr["vtail:rchord"].value
vtailloc = plane.outpmtr["vtailLoc"].value

wingV  = plane.outpmtr["wingV"].value
fuseV  = plane.outpmtr["fuseV"].value
htailV = plane.outpmtr["htailV"].value
vtailV = plane.outpmtr["htailV"].value
motorV = plane.outpmtr["motorV"].value

propcg = plane.outpmtr["propellercg"].value
batterycg = .018
servocg   = 0.3
ALreceivercg =  wingloc+(0.7*chord)
ARreceivercg = wingloc+(0.7*chord)
Ereceivercg  =  htailloc+(0.7*hchord)
Rreceivercg  =   vtailloc+(0.7*vchord) 
wingcg  = plane.outpmtr["wingcg"].value
fusecg  = plane.outpmtr["fusecg"].value
htailcg = plane.outpmtr["htailcg"].value
vtailcg = plane.outpmtr["htailcg"].value
motorcg = plane.outpmtr["motorcg"].value

battery = 0.348
servo  = 0.0166
ALreceiver = 0.0145  
ARreceiver = 0.0145
Ereceiver = 0.0145
Rreceiver = 0.0145
propW  = 0.019
motorW = 0.079
wingW  = foamDensity *wingV 
fuseW  = foamDensity *fuseV 
htailW = foamDensity *htailV
vtailW = foamDensity *vtailV

parts = {
    "wing"    : (wingW,  wingcg),
    "fuselage": (fuseW,  fusecg),
    "htail"   : (htailW, htailcg),
    "vtail"   : (vtailW, vtailcg),
    "battery" : (battery, batterycg),
    "servo"   : (servo,   servocg),
    "ALreceiver": (ALreceiver, ALreceiver),
    "ARreceiver": (ARreceiver, ARreceivercg),
    "Ereceiver" : (Ereceiver, Ereceivercg),
    "Rreceiver" : (Rreceiver, Rreceivercg),
    "motor"   : (motorW,  motorcg),
    "prop"    : (propW,   propcg),  
}
#print("heres parts", parts)

def cg_1d(parts):
    
    total_w = 0.0
    total_moment = 0.0

    for name, (w, x) in parts.items():
        if w is None or x is None:
            raise ValueError(f"Missing data for {name}: w={w}, x={x}")
        total_w += w
        total_moment += w * x

    if total_w <= 0:
        raise ValueError("Total weight/mass must be > 0")

    return total_moment / total_w, total_w

x_cg_total, totalW = cg_1d(parts)



print(f"Total mass/weight = {totalW:.6f}")
print(f"Aircraft X_CG from nose = {x_cg_total:.6f}" )

#lift and drag 

lift = totalW*9.81
print("lift",lift)
rho = 1.225
rhoSL = 1.225
s_w = plane.outpmtr["S_wet"].value
s_ref = plane.despmtr["S"].value
velo = np.linspace(2,25, 100)
q = 0.5*rho*velo**2
mew = 1.789e-5
CL = lift/(q*s_ref)
print("CL", CL)

##### FUSELAGE #####

length = plane.despmtr["fuselength"].value
diameter = plane.despmtr["fuseDiameter"].value
fineness = length/diameter

print("fineness", fineness)

CDfuse = ((length*diameter)/s_ref)*(0.25 * (fineness ** -0.9) + 0.05 * (fineness ** 0.57))

print("CDfuse", CDfuse)

##### Wing coefficients ####

CL_w = 0.2
CD_w = 0.010

b = plane.outpmtr["wingspan"].value

AR = b**2/s_ref
e  = 0.8

CDi = (CL**2)/(np.pi*e*AR)
print("CDi", CDi)

##### tail coefficients ####
htailS = plane.outpmtr["htail:S"].value

vtailS = plane.outpmtr["vtail:S"].value

CDht = 0.015 * (htailS/ s_ref)
print("CDht", CDht)
CDvt = 0.015 * (vtailS/ s_ref)
print("CDvt", CDvt)
## total drag 
CD0 = CD_w + CDfuse + CDht + CDvt
CD =  CD0 + CDi

print("CD", CD)


D = q*s_ref*CD
print("D", D)
Ts =1 #throttle setting 


T = 15 * (rho/rhoSL) * (1 - (velo/50))* Ts
t1 = 15 * (rho/rhoSL) * (1 - (velo/50))* 0.5
t2 = 15 * (rho/rhoSL) * (1 - (velo/50))* 0.2

plt.figure()
plt.plot(velo, D, label="Total Drag D")
plt.plot(velo, T, label=f"Thrust Available (throttle={Ts:.2f})")
plt.plot(velo, t1, label=f"Thrust Available (throttle={0.5:.2f})")
plt.plot(velo, t2, label=f"Thrust Available (throttle={0.2:.2f})")
plt.xlabel("Velocity V (m/s)")
plt.ylabel("Drag D (N)")
plt.title("Drag vs Velocity")
plt.grid(True)
plt.legend()
plt.show()

#best Range 
Wh_batt = 48.84           
E_batt = Wh_batt * 3600

n_eff = 0.85
C = 50 
Bestrange = n_eff*E_batt *(CL/CD)*(battery*9.81/totalW)
print(Bestrange)

#Endurance 
P = D * velo # min power req is endurance 
i = np.argmin(P)
V_best_endurance = velo[i]
P_min = P[i]
Endurance_s = n_eff * E_batt / P_min
print("V_best", V_best_endurance)

#ceiling

#bisection method 

def max_excess_power(sigma, velo):
    rho = sigma * rhoSL
    q = 0.5 * rho * velo**2

    T = 15.0 * sigma * (1.0 - velo/50.0)
    T = np.maximum(T, 0.0)
    
    # drag 
    CL = lift/(q*s_ref)
    CDi = (CL**2)/(np.pi*e*AR)
    CD =  CD0 + CDi
    D = q*s_ref*CD

    Pex = (T - D) * velo
    return np.max(Pex)

lo, hi = 0.05, 1.0   # sigma bounds (high altitude to sea level)

for _ in range(60):
    mid = 0.5*(lo+hi)
    if max_excess_power(mid, velo) > 0:
        hi = mid   # still can climb -> go higher (lower sigma)
    else:
        lo = mid   # can't climb -> go lower (higher sigma)

sigma_ceiling = 0.5*(lo+hi)
rho_ceiling = sigma_ceiling * rhoSL

print("sigma_ceiling =", sigma_ceiling)
print("rho_ceiling =", rho_ceiling)


#vstall 



CL_max = 0.9

phi_deg = np.linspace(0, 65, 60)
phi = np.deg2rad(phi_deg)
n=1/(np.cos(phi))

Vs = np.sqrt((2 * n * totalW) / (rhoSL * s_ref * CL_max))
plt.figure()
plt.plot(phi_deg, Vs, label="velocity")
plt.xlabel("Bank Angle (deg)")
plt.ylabel("Stall Speed (m/s)")
plt.title("Stall Speed vs Bank Angle (Sea Level)")
plt.grid(True)
plt.legend()
plt.show()


#Dash speed - solve when T=D then back solve the Velo 

phi_deg = np.linspace(0, 65, 100)
phi = np.deg2rad(phi_deg)
n=1/(np.cos(phi))
V_dash = np.full_like(phi_deg, np.nan, dtype=float)

for j in range(len(phi_deg)):

    q = 0.5 * rhoSL * velo**2
    CL = (n[j]*lift)/(q*s_ref)
    CDi = (CL**2)/(np.pi*e*AR)
    CD =  CD0 + CDi
    D = q*s_ref*CD
    f = T - D  # positive means excess thrust
    idx = np.where(np.diff(np.sign(f)) != 0)[0]  # sign-change indices
    if len(idx) == 0:
        # no level flight possible at this bank angle
        V_dash[j] = np.nan
        continue

    i = idx[-1]  # right-most intersection
    # linear interpolation between velo[i] and velo[i+1]
    V_dash[j] = velo[i] - f[i] * (velo[i+1]-velo[i]) / (f[i+1]-f[i])



print("Dash speed =", V_dash, "m/s")

plt.figure()
plt.plot(phi_deg, V_dash, label="velocity")
plt.xlabel("Bank Angle (deg)")
plt.ylabel("Dash Speed (m/s)")
plt.title("Dash Speed vs Bank Angle (Sea Level)")
plt.grid(True)
plt.legend()
plt.show()


#turning radius 

n = 2.5
radius = (velo**2)/(9.81*np.sqrt((n**2)-1))
i = np.argmin(radius)
print("min radius", np.rad2deg(radius[i])) 


#beam theory
chord = plane.outpmtr["chord"].value 
 
inertia = 6.84e-5 * chord**4 
y = 4*b/6*np.pi
c = 0.06*chord
moment = lift* y 

stress = moment*c/inertia 

s_allow = 100000
n_max = (2*s_allow*inertia) / (totalW*y*c)
phi_max = np.degrees(np.arccos(1/n_max))
print("n",n_max)          
print("phi", phi_max)         
