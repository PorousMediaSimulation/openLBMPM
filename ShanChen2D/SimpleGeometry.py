"""
Define the simple geometry domain to test differernt phenomena and boundary 
conditions
"""

import os, sys

import numpy as np
import scipy as sp

def defineGeometry(xDomain, yDomain):
    tmpVoid = np.ones([yDomain, xDomain], dtype = np.bool)
    tmpSolid = np.zeros([yDomain, xDomain], dtype = np.bool)
#    tmpVoid[0, :] = 0; tmpVoid[-1, :] = 0
#    tmpSolid[0, :] = 1; tmpSolid[-1, :] = 1
#    tmpVoid[:, 0] = 0; tmpVoid[:, -1] = 0
#    tmpSolid[:, 0] = 1; tmpSolid[:, -1] = 1
    tmpVoid[10:-10, 0] = 0; tmpVoid[10:-10, -1] = 0
    tmpSolid[10:-10, 0] = 1; tmpSolid[10:-10, -1] = 1
#    tmpVoid[10:-10, :16] = 0; tmpVoid[10:-10, 80:] = 0
#    tmpSolid[10:-10, :16] = 1; tmpSolid[10:-10, 80:] = 1
#    tmpVoid[108:148, 1:16] = 1; tmpVoid[108:148, 80:-1] = 1
#    tmpSolid[108:148, 1:16] = 0; tmpSolid[108:148, 80:-1] = 0
    
#    tmpVoid[-1, :] = 0; tmpSolid[-1, :] = 1
#    tmpVoid[0, :] = 0; tmpSolid[0, :] = 1
    return tmpVoid, tmpSolid