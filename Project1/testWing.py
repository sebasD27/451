# Import pyCAPS module
import pyCAPS
import numpy as np
from kulfan import Kulfan


#np.set_printoptions(precision=16, floatmode='maxprec_equal')

#afl = Kulfan()

#airfoil = r"C:\Users\duran\Downloads\ONERA_D_selig.dat"


#afl.readfile(airfoil)
#upper = np.asarray(afl.upperCoefficients.magnitude, dtype=float)
#lower = np.asarray(afl.lowerCoefficients.magnitude, dtype=float)
#
#print("Upper Coefficients:", upper)
#print("Lower Coefficients:", lower)

# Import os
import os

#------------------------------------------------------------------------------#

# Load CSM file
filename = r"C:\Users\duran\Downloads\esp\ESP127\EngSketchPad\bin\451 plane.csm"
capsProblem = pyCAPS.Problem(problemName = "sin_Transport",
                             capsFile = filename,
                             outLevel = 0)
aflr4 = capsProblem.geometry

#aflr4.despmtr["aupper"].value = upper
#aflr4.despmtr["alower"].value = lower

#naca = 2412   #float(input("Enter NACA 4-digit airfoil number (e.g., 2412): "))

sweep = 26.7    #float(input("Enter wing sweep angle in degrees (e.g., 25): "))

dihedral = 0  #  float(input("Enter wing dihedral angle in degrees (e.g., 5): "))

taper = 0.56   #float(input("Enter wing taper ratio (e.g., 0.5): "))

#aflr4.despmtr["wing:naca"].value = naca
#aflr4.despmtr["wing:sweep"].value = sweep
#aflr4.despmtr["wing:dihedral"].value = dihedral
#aflr4.despmtr["wing:taper"].value = taper

# Create aflr4 aim
aflr4 = capsProblem.analysis.create(aim = "aflr4AIM",
                                    name = "aflr4")

aflr4.input.Proj_Name = "TestWing"
aflr4.input.Mesh_Format = "SU2"

aflr4.input.curv_factor = 0.15


aflr4.input.ff_cdfr  = 1.10

# Scaling factor to compute AFLR4 ’ref_len’ parameter via
# ref_len = capsMeshLength * Mesh_Length_Factor
aflr4.input.Mesh_Length_Factor = 2.5

# Relative scale of maximum spacing bound relative to ref_len
# max_spacing = max_scale * ref_len
aflr4.input.max_scale =3

# Relative scale of minimum spacing bound relative to ref_len
# min_spacing = min_scale * ref_len
aflr4.input.min_scale = 0.01

# Absolute scale of minimum spacing bound for proximity
# abs_min_spacing = abs_min_scale * ref_len
aflr4.input.abs_min_scale = 0.001

# Mark capsMesh == Farfield with a Farfield bcType  TRANSP_SRC_UG3_GBC  TRANSP_BL_INT_UG3_GBC 
#differnet option include: farfield, scalefactor, edgeWeight
aflr4.input.Mesh_Sizing = {"BC_3": { "edgeWeight":2, "scaleFactor":0.25 },
                           "BC_2": {"bcType":"symmetry"},
                           "BC_1": {"bcType":"Farfield"}}

# Run AIM                         
aflr4.runAnalysis()
# View the surface tessellation
#aflr4.geometry.view()
# Create AFLR3 AIM to generate the volume mesh
aflr3 = capsProblem.analysis.create(aim  = "aflr3AIM",
                                    name = "aflr3")

## Link the aflr4 Surface_Mesh as input to aflr3
aflr3.input["Surface_Mesh"].link(aflr4.output["Surface_Mesh"])
# Dump VTK files for visualization
aflr3.input.Proj_Name   = "VISTestWing"
aflr3.input.Mesh_Format = "SU2"

aflr3.input.BL_Max_Layers      =  50
aflr3.input.BL_Initial_Spacing = 3.0e-5
aflr3.input.BL_Thickness       = 0.01
#
#Specify prism boundary layer elements
aflr3.input.Mesh_Gen_Input_String = "-blc -mblfinal 1 -qall"


aflr3.runAnalysis()


