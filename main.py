from sys import path
import os
if os.environ['COMPUTERNAME'] == 'GBW-L-W2003':
    path.append(r"C:/Users/u0101727/Documents/Software/CasADi/casadi-windows-py37-v3.5.1-64bit")
    pathOS = "C:/Users/u0101727/Documents/MyRepositories/opensim-fork/install/sdk/Python"
elif os.environ['COMPUTERNAME'] == 'GBW-D-W0529':
    path.append(r"D:/u0101727/MySoftware/casadi-windows-py37-v3.5.1-64bit")
    pathOS = "C:/OpenSim_4.1/sdk/Python"
elif os.environ['COMPUTERNAME'] == 'GBW-D-W2711':
    path.append(r"C:/Users/Public/Documents/Software/casadi-windows-py37-v3.5.1-64bit")
    pathOS = "C:/OpenSim_4.1/sdk/Python"
import casadi as ca
import numpy as np

ndof = 14
NVec3 = 3

# Paths
pathMain = os.getcwd()
pathExternalFunctions = os.path.join(pathMain, 'ExternalFunctions')
os.chdir(pathExternalFunctions)
F = ca.external('F','ShoulderModel_consB.dll')   
os.chdir(pathMain)
vec1 = -np.ones((ndof, 1))
vec2 = -np.ones((ndof, 1))
vec12 = np.zeros((ndof*2, 1))
vec12[::2, :] = vec1
vec12[1::2, :] = vec2  
vec3 = -np.ones((ndof, 1))
vecall = np.concatenate((vec12,vec3))
res1 = F(vecall).full()

# CasADi expressions
q_in = ca.MX.sym('q_in', ndof)
qdot_in = ca.MX.sym('qdot_in', ndof)
qdotdotin = ca.MX.sym('qdotdotin', ndof)


qqdot_in = ca.MX(ndof*2, 1)
qqdot_in[::2, :] = q_in
qqdot_in[1::2, :] = qdot_in  
F_out = F(ca.vertcat(qqdot_in, qdotdotin))
idxMarker = {}
idxMarker["AC_clavicle"] = list(range(ndof, ndof+NVec3))
idxMarker["AC_scapula"] = list(range(1 + idxMarker["AC_clavicle"][-1], 
                                     1 + idxMarker["AC_clavicle"][-1] + NVec3))
phi = F_out[idxMarker["AC_scapula"]] - F_out[idxMarker["AC_clavicle"]]

jac_phi_q_in = ca.jacobian(phi, q_in)
f_pathConstraintsJacobian = ca.Function("f_pathConstraintsJacobian",
                                        [q_in, qdot_in, qdotdotin],
                                        [jac_phi_q_in])      
    
# Numerical values
G = f_pathConstraintsJacobian(vec1, vec2, vec3).full()
multipliers = -np.ones((phi.shape[0], 1)) # assumed lambda = 1
constraintForces = np.matmul(G.T, multipliers)

# print(res1[0:idxMarker["AC_clavicle"][0]] + constraintForces)
# print(res1[0:idxMarker["AC_clavicle"][0]] - constraintForces)




