import pyCAPS
import numpy as np
import matplotlib.pyplot as plt

filename = r"C:\Users\duran\Downloads\esp\ESP127\EngSketchPad\bin\451 plane.csm"
capsProblem = pyCAPS.Problem(problemName = "sin_Transport",
                             capsFile = filename,
                             outLevel = 0)

plane = capsProblem.geometry

foamDensity = 30

motorW = 0.079


wingV  = plane.outpmtr["wingV"].value
fuseV  = plane.outpmtr["fuseV"].value
htailV = plane.outpmtr["htailV"].value
vtailV = plane.outpmtr["htailV"].value
motorV = plane.outpmtr["motorV"].value

propcg = 0.009
batterycg = .018
servocg   = 0.3
receivercg  = 0.15 
wingcg  = plane.outpmtr["wingcg"].value
fusecg  = plane.outpmtr["fusecg"].value
htailcg = plane.outpmtr["htailcg"].value
vtailcg = plane.outpmtr["htailcg"].value
motorcg = plane.outpmtr["motorcg"].value

battery = 0.348
servo  = 0.0166
receiver = 0.0145  
propW  = 0.019
motorW = 0.079
wingW  = foamDensity *wingV 
fuseW  = foamDensity *fuseV 
htailW = foamDensity *htailV
vtailW = foamDensity *vtailV
print (wingW)

parts = {
    "wing"    : (wingW,  wingcg),
    "fuselage": (fuseW,  fusecg),
    "htail"   : (htailW, htailcg),
    "vtail"   : (vtailW, vtailcg),
    "battery" : (battery, batterycg),
    "servo"   : (servo,   servocg),
    "receiver": (receiver,receivercg),
    "motor"   : (motorW,  motorcg),
    "prop"    : (propW,   propcg),  # prop CG usually near motor; adjust if different
}
#print("heres parts", parts)

def cg_1d(parts):
    """
    parts: dict of {name: (weight_or_mass, x_cg)}
    returns: (x_cg_total, total_weight_or_mass)
    """
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

velo = np.linspace(2,40, 100)
q = 0.5*rho*velo**2
mew = 1.789e-5
CL = lift/(q*s_ref)
print("C", CL)

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
CD = CD_w + CDfuse + CDht + CDvt + CDi

print("CD", CD)


D = 0.5*q*s_ref*CD
print("D", D)
Ts =0.5 #throttle setting 


T = 15 * (rho/rhoSL) * (1 - (velo/50))* Ts

plt.figure()
plt.plot(velo, D, label="Total Drag D")
plt.plot(velo, T, label=f"Thrust Available (throttle={Ts:.2f})")
plt.xlabel("Velocity V (m/s)")
plt.ylabel("Drag D (N)")
plt.title("Drag vs Velocity")
plt.grid(True)
plt.legend()
plt.show()

#best Range 
n_eff = 0.85
C = 50 
range = n_eff*(C/9.81)*(CL/CD)*(battery*9.81/totalW)
print(range)

#Endurance 
P = D * velo # min power req is endurance 
i = np.argmin(P)
V_best_endurance = velo[i]
P_min = P[i]
print("V_best", V_best_endurance)

#ceiling 


#vstall 
phi = 10
n=1/(np.cos(phi))

Vs = np.sqrt((2*n*(totalW/s_ref))/(rhoSL*CL))

#Dash speed - solve when T=D then back solve the Velo 

#turning radius 

radius = (velo**2)/(9.81*np.sqrt((n**2)-1))

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