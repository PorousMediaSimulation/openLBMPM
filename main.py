"""
Main file for implementing LBM modeling. It will check whether there is existing 
dirctory for saving results, then create one. It will also check whether .ini
files exists or not, because the file includes fluids properties, initial and 
boundary conditions. Then the simulation will run.
===============================================================================
author: Pei Li
Email: pei.li@weizmann.ac.il
"""

import os, sys, getpass
import configparser

from ShanChenD2Q9 import ShanChenD2Q9
from Transport2D import Transport2D

from ShanChenD3Q19 import ShanChenD3Q19

from RKD2Q9 import RKColorGradientLBM
from Transport2DRK import Transport2DRK

from RKColorGradientD3Q19 import RKColorGradient3D

#search and create LBMResults directory
username = getpass.getuser()
pathForResultDirectoryDensity = os.path.expanduser('~/LBMResults')
pathForResultDirectoryVelocity = os.path.expanduser('~/LBMVelocity')
pathForResultDirectionDensity3D = os.path.expanduser('~/LBMResults3D')

if (os.path.exists(pathForResultDirectoryDensity) == False):
    os.mkdir('/home/' + username + '/LBMResults' )
if (os.path.exists(pathForResultDirectoryVelocity) == False):
    os.mkdir('/home/' + username + '/LBMVelocity')
    
#search the directory for .ini file
iniFilePath = 'IniFiles'
if (os.path.isdir(iniFilePath) == False):
    sys.exit()

modelDimension = input("Please choose 2D/3D model (enter 2D or 3D):")
modelType = input("Please choose the type of the simulation (enter flow or transport): ")

LBMType = input("Please choose ShanChen(SC) or Color Gradient(CG) methods for flow: ")

if modelDimension == "2D":
    print("Run 2D simulation.")
    if (modelType == "flow" and LBMType == "SC"):
        flow2D = ShanChenD2Q9(iniFilePath)
        flow2D.runTypeSCmodel()
    elif (modelType == "flow" and LBMType == "CG"):
        flow2D = RKColorGradientLBM(iniFilePath)
#        flow2D.runRKColorGradient2D()
        flow2D.runModifiedRKColorGradient2D()
    elif (modelType == "transport" and LBMType == "SC"):
        flowType = input("Please choose the force scheme (OR: original or EF: explicit forcing): ")
        transport2D = Transport2D(iniFilePath)
        if (flowType == "OR"):
    #transport2D.runTransport2DStaticInterface()
            transport2D.runTransport2DMPMC()
        elif (flowType == "EF"):
            transport2D.runTransportMPMCEFS()
    #transport2D.runTransportCPU()
    elif (modelType == "transport" and LBMType == "CG"):
        transport2D = Transport2DRK(iniFilePath)
        transport2D.runTransport2DMPMCRK()
        
    elif (modelType == "transport" and LBMType == "No"):
        transport2D = Transport2D(iniFilePath)
        transport2D.testTransportMRTScheme()
        print("The chosen type does not exist in current version. Stop here.")
        sys.exit()
elif modelDimension == "3D":
    if (modelType == "flow" and LBMType == "SC"):
        print("Run 3D simulation with SC.")
        MCMPLBM3D = ShanChenD3Q19(iniFilePath)
#        MCMPLBM3D.runOriginalSC3DGPU()
        MCMPLBM3D.runEFS4LBM3DGPU()
    elif (modelType == "flow" and LBMType == "CG"):
        print("Run 3D simulation with CG.")
        RKCGLBM3D = RKColorGradient3D(iniFilePath)
        RKCGLBM3D.runRKColorGradient3D()