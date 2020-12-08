# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:24:43 2020

@author: u0101727
"""
from sys import path
import os
path.append(r"C:/Users/u0101727/Documents/Software/CasADi/casadi-windows-py37-v3.5.1-64bit")
import casadi as ca
import numpy as np

# Nelt = 9
# # Paths
# F = ca.external('F', 'RError_FD.dll', dict(
#     enable_fd=True, enable_forward=False, enable_reverse=False,
#     enable_jacobian=False, fd_method='forward'))


# angle1 = 20*np.pi/180
# vec1 = np.array([[np.cos(angle1),1,np.sin(angle1),
#                   0,1,0,
#                   -np.sin(angle1),0,np.cos(angle1)]])

# angle2 = -20*np.pi/180
# vec2 = np.array([[np.cos(angle2),1,np.sin(angle2),
#                   0,1,0,
#                   -np.sin(angle2),0,np.cos(angle2)]])

# res1 = F(vec1,vec2).full()
# print(res1)

Nelt = 3
# Paths
F = ca.external('F', 'RError_Euler_FD.dll', dict(
    enable_fd=True, enable_forward=False, enable_reverse=False,
    enable_jacobian=False, fd_method='forward'))


angle1 = 20*np.pi/180
vec1 = np.array([[0, angle1, 0]])

angle2 = 250*np.pi/180
vec2 = np.array([[0, angle2, 0]])

res1 = F(vec1,vec2).full()
print(res1)