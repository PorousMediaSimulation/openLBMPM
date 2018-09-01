"""
class: ShanChenD2Q9
usage: Implement Shan-Chen model for multiphase flow with SRT and MRT scheme in 
2D condition. Currently,it includes original Shan-Chen model and high isotropy-
explicit forcing scheme. GPUs accleration is also included.
===============================================================================
author: Pei Li
E-mail: li.pei1228@outlook.com
"""
import sys, os, getpass, math
import configparser
from copy import deepcopy
from timeit import default_timer as timer

import numpy as np
import scipy as sp
import scipy.linalg as slin
import scipy.ndimage as sciimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import tables as tb

from numba import jit, autojit
#from numba import cuda, vectorize, guvectorize

from numba import cuda
#from accelerate import cuda as acuda
#from accelerate import numba as anumba

from SimpleD2Q9 import BasicD2Q9
from AccelerateGPU2D import *
from OptimizedD2Q9GPU import *
from ExplicitD2Q9GPU import *
from SimpleGeometry import defineGeometry

class ShanChenD2Q9(BasicD2Q9):
    def __init__(self, pathIniFile):
        BasicD2Q9.__init__(self, pathIniFile)
        self.path = pathIniFile
        config = configparser.ConfigParser()
        config.read(self.path + '/' + 'twophasesetup.ini')
        #set up the dimensions and whether ther is an image for usage
        try:
            self.PictureExistance = config['PictureSetup']['Exist']
        except KeyError:
            print("Cannot find the parameter's name for image existance.")
        if (self.PictureExistance == "'no'"):
            try:
                self.borderX = int(config['SeparationBorder']['xGrid'])
                self.borderY = int(config['SeparationBorder']['yGrid'])
                self.nx = self.borderX; self.ny = self.borderY
            except KeyError:
                print('Could not read the right domain size, please check .ini file')
#            print(self.borderX, self.borderY)
            print("The dimension of simple geometry is %g and %g." % (self.borderX, self.borderY))
        #set up how many fluids interact each other in the simulation
        try:
            self.typesFluids = int(config['FluidsTypes']['NumberOfFluids'])
        except KeyError:
            print("Counld not find the parameter's name for number of fluids in .ini file.")
        except ValueError:
            print("There is an error for the value of kinds of fluids.")
        except:
            print("Unknown error happens when the number of fluids is read.")
        print("There are %g types of fluids in the domain." % self.typesFluids)
        #set up which kind of interaction calculation is used in the simulation
        try:
            self.interactionType = config['InterType']['InteractionType']
        except KeyError:
            print("Could not find the parameter's name for fluids' interaction types.")
        except:
            print("Unknown error happens when the type of fluids' interaction is read.")
        print("The interaction between fluids is:")
        print(self.interactionType)
        #set up the parameters for GPU parallel computation
        self.Parallel = config['Parallelism']['Parallel']
        if (self.Parallel == "'yes'"):
            self.xDimension = int(config['Parallelism']['xDimension'])
            self.threadNum = int(config['Parallelism']['ThreadsNum'])
            print("GPU will be used and the parameters for partiion domain are: ")
            print("The x-dimension of grid is %g." % self.xDimension)
            print("The number of thread in each grid is %g." % self.threadNum)
        #set up SRT or MRT or TRT is implemented in the simulation
        try:
            self.relaxationType = config['RelaxationType']['Type']
        except KeyError:
            print('Could not read the choice for the relaxation type, please check .ini file.')
            sys.exit('Error happened when reading the type of relaxation rate.')
        except:
            print('Unknown error for choosing the type of relaxation rate.')
            raise
        if (self.relaxationType == "'MRT'"):
            print("The relaxation scheme is MRT.")
            self.inverseTransformation = slin.inv(self.transformationMatrix)
            self.relaxationMatrix = np.zeros([self.typesFluids, 9, 9], \
                                              dtype = np.float64)
            self.diagonalValues = np.ones([self.typesFluids, 9], dtype = np.float64)
            self.diagonalValues[0, 1] = 0.6; self.diagonalValues[0, 2] = 1.5
            self.diagonalValues[0, 0] = 1.
            self.diagonalValues[0, 4] = 1.2; self.diagonalValues[0, 6] = 1.2
            
            self.diagonalValues[1, 1] = 0.6; self.diagonalValues[1, 2] = 1.5
            self.diagonalValues[1, 0] = 1.
            self.diagonalValues[1, 4] = 1.2; self.diagonalValues[1, 6] = 1.2
            
            self.collisionMatrix = np.empty([self.typesFluids, 9, 9], \
                                            dtype = np.float64)
#            self.collisionMatrix1 = np.empty([9, 9], dtype = np.float64)
#            self.s0 = float(config['RelaxationRate']['s0'])
#            self.s1 = float(config['RelaxationRate']['s1'])
#            self.s2 = float(config['RelaxationRate']['s2'])
#            self.s3 = float(config['RelaxationRate']['s3'])
#            self.s4 = float(config['RelaxationRate']['s4'])
#            self.s5 = float(config['RelaxationRate']['s5'])
#            self.s6 = float(config['RelaxationRate']['s6'])
#            self.s7 = float(config['RelaxationRate']['s7'])
#            self.s8 = float(config['RelaxationRate']['s8'])
#            for i in sp.arange(9):
#                for j in sp.arange(9):
#                    for k in sp.arange(self.typesFluids):
#                        if (i == j):
#                            self.relaxationMatrix[k, i, j] = self.diagonalValues[k, i]
##                            self.relaxationMatrix1[i, j] = self.diagonalValues1[i]
#                            if (i == 7 or i == 8):
#                                self.relaxationMatrix[k, i, j] = 1./self.tau[k]
##                                self.relaxationMatrix1[i, j] = 1./self.tau1
#            for i in sp.arange(self.typesFluids):
#                self.collisionMatrix[i] = np.dot(np.dot(self.inverseTransformation, \
#                    self.relaxationMatrix[i]), self.transformationMatrix)
#                print("The collision matrix for the fluid: ", i)
#                print(self.collisionMatrix[i])
        elif self.relaxationType == "'SRT'":
            print("The SRT-BGM relaxation method is used.")
        elif self.relaxationType == "'TRT'":
            print("The TRT relaxation scheme is used.")
        #set up whether the image domain is duplicated or not
        try:
            self.duplicateDomain = config['DuplicateDomain']['Option']
        except KeyError:
            print('Could not read the choice for duplicating the domain, please check .ini file.')
            sys.exit('Error happened when reading the set-up for duplicating the image.')
        except:
            print('Unknown error happened when reading the set-up of duplicating image.')
            raise
        #choose to having drainage-imbibition cycles or not
        try:
            self.isCycles = config['DICycles']['Option']
        except KeyError:
            print("Cannot read the choice for drainaeg-imbibition cycles, please check .ini file.")
            sys.exit("Error happened when reading the option for running drainage-imbibition cycles.")
        except:
            print("Unknown error happened when reading the set-up for old-new water cycling.")
            raise
        if (self.isCycles == "'yes'"):
            self.lastStep = int(config['DICycles']['LastStep'])
        self.unitEX = np.array([0., 1., 0., -1., 0., 1., -1., -1., 1.])
        self.unitEY = np.array([0., 0., 1., 0., -1., 1., 1., -1., -1.])
        #read parameters for different calculations for interactions between fluids
        input("If all the information is correct, please hit Enter.")
        if self.interactionType == "'ShanChen'":
            self.__readIniFileShanChen()
        elif self.interactionType == "'EFS'":
            self.__readIniFileEFS()
        #Generate .h5 file to save results after a certain time steps
        self.__createHDF5File()
            
    """
    Read .ini file for original Shan-Chenm model
    """
    def __readIniFileShanChen(self,):
        print("Read .ini file for original ShanChen model.")
        config = configparser.ConfigParser()
        config.read(self.path + '/' + "shanchen2D.ini")
        #set up fluids properties in the domain
        print("Read parameters for fluids' properties.")
        try:
            tmpDensities = config['FluidProperties']['InitialDensities']
            tmpBackground = config['FluidProperties']['BackgroundDensities']
        except KeyError:
            print("Could not find the parameter's name for fluid densities in .ini file.")
        self.initialDensities = np.asarray([], dtype = np.float64)
        self.backgroundDensities = np.asarray([], dtype = np.float64)
        tmpDensitiesList = tmpDensities.split(',')
        tmpBackgroundList = tmpBackground.split(',')
        if (len(tmpDensitiesList) == self.typesFluids):
            for i in tmpDensitiesList:
                self.initialDensities = np.append(self.initialDensities, float(i))
            print("The initial densities for fluids in the domain are:")
            print(self.initialDensities)
        else:
            print("The number of fluids does not match the number of densities.")
            sys.exit()
        if (len(tmpBackgroundList) == self.typesFluids):
            for i in tmpBackgroundList:
                self.backgroundDensities = np.append(self.backgroundDensities, float(i))
            print("The background densities for fluids in the domain are:")
            print(self.backgroundDensities)
        else:
            print("The number of fluids does not match the number of background densities.")
        #set up the values of taus for each fluids   
        try:
            tmpTau = config['FluidProperties']['FluidsTau']
        except KeyError:
            print("Could not find the parameter's name for tau of fluids in .ini file.")
        except:
            print("Unknown error happens when the tau are read.")
        self.tau = np.asarray([], dtype = np.float64)
        tmpTauList = tmpTau.split(',')
        if (len(tmpTauList) == self.typesFluids):
            for i in tmpTauList:
               self.tau = np.append(self.tau, float(i))
            print("The values of taus for each fluid are:")
            print(self.tau)
        else:
            print("The number of fluids does not match the number of viscosities.")
            sys.exit()
        #set up interaction coefficients between fluids
        try:
            tmpInterF = config['ShanChenParameters']['InteractionFluid']
        except KeyError:
            sys.exit()
        tmpInterFList = tmpInterF.split(',')
        tmpCount = 0
        self.interCoeff = np.zeros([self.typesFluids, self.typesFluids], dtype = np.float64)
        for i in np.arange(self.typesFluids - 1):
            for j in np.arange(i+1, self.typesFluids):
                self.interCoeff[i, j] = float(tmpInterFList[tmpCount])
                self.interCoeff[j, i] = self.interCoeff[i, j]
                tmpCount += 1
            print("Interaction coefficients are:")
            print(self.interCoeff)
        #set up interaction coefficients between fluids and solid
        try:
            tmpInterS = config['ShanChenParameters']['InteractionSolid']
        except KeyError:
            print("Cannot find the parameter's name for interaction with solid.")
        self.interactionSolid = np.asarray([], dtype = np.float64)
        tmpInterSList = tmpInterS.split(',')
        if (len(tmpInterSList) == self.typesFluids):
            for i in tmpInterSList:
                self.interactionSolid = np.append(self.interactionSolid, float(i))
            print("The interaction strength with solid phase is: ")
            print(self.interactionSolid)
        else:
#            print("The number of fluids does not match the number of interaction coeff with solid.")
            sys.exit()
        #set up boundary
        self.boundaryTypeInlet = config['BoundaryDefinition']['BoundaryTypeInlet']
        self.boundaryMethod = config['BoundaryDefinition']['BoundaryMethod']
        self.boundaryTypeOutlet = config['BoundaryDefinition']['BoundaryTypeOutlet']
        if (self.boundaryTypeInlet == "'Dirichlet'"):
            print("The boundary type is " + "Dirichlet.")
            try:
                tmpPressureValuesHigher = config['PressureBoundary']['PressureInlet']
            except KeyError:
                sys.exit()
            tmpPressureHigher = tmpPressureValuesHigher.split(',')
            self.inletPressures = np.asarray([], dtype = np.float64)
            if (len(tmpPressureHigher) == self.typesFluids):
                for i in tmpPressureHigher:
                    self.inletPressures = np.append(self.inletPressures, float(i))
        elif (self.boundaryTypeInlet == "'Neumann'"):
            print("The boundary type is " + "Neumann.")
            try:
                tmpVelocityX = config['VelocityBoundary']['velocityX']
                tmpVelocityY = config['VelocityBoundary']['velocityY']
            except KeyError:
                sys.exit()
            tmpVXList = tmpVelocityX.split(',')
            tmpVYList = tmpVelocityY.split(',')
            print(tmpVYList)
            self.velocityXInlet = np.asarray([], dtype = np.float64)
            self.velocityYInlet = np.asarray([], dtype = np.float64)
            if (len(tmpVXList) == self.typesFluids and len(tmpVYList) == \
                self.typesFluids):
                for i in tmpVXList:
                    self.velocityXInlet = np.append(self.velocityXInlet, float(i))
                for j in tmpVYList:
                    self.velocityYInlet = np.append(self.velocityYInlet, float(j))
            print("The velocities of inlet on the x- and y- directions are:")
            print(self.velocityXInlet)
            print(self.velocityYInlet)
            
        if (self.boundaryTypeOutlet == "'Dirichlet'"):
            print("The outlet boundary condition is the constant pressure type.")
            try:
                tmpPressureValuesLower = config['PressureBoundary']['PressureOutlet']
            except KeyError:
                sys.exit()
            tmpPressureLower = tmpPressureValuesLower.split(',')
            self.outletPressures = np.asarray([], dtype = np.float64)
            if (len(tmpPressureLower) == self.typesFluids):
                for i in tmpPressureLower:
                    self.outletPressures = np.append(self.outletPressures, float(i))
        #set up whether the body force shoudl be included or not
        print("Read parameters on body force.")
        try:
            self.bodyForceOption = config['BodyForce']['Option']
        except KeyError:
            print("Can't find the parameter on having body force or not.")
        except:
            print("Unknow error happens when the option for body force is read.")
            sys.exit()
        if self.bodyForceOption == "'yes'":
            try:
                self.bodyForceXG = float(config['BodyForce']['forceXG'])
                self.bodyForceYG = float(config['bodyForce']['forceYG'])
            except KeyError:
                print("Can't find the parameters' name for body force.")
            except ValueError:
                print("The value for body force must be float.")
            except:
                print("Unknown error happens when the body force parameters are read.")
            print("The gravity or other acceleration in the domain is: ")
            print(self.bodyForceXG)
            print(self.bodyForceYG)
        #set up time steps for the simulation
        self.numTimeStep = int(config['Time']['numberTimeStep'])
        print("The number of time step is %g." % self.numTimeStep)
        input("If all the information on original ShanChen is correct, please hit Enter.")
        
    def __readIniFileEFS(self, ):
        #choose the type of model
        print("Read .ini file for explicit forcing scheme.")
        config = configparser.ConfigParser()
        config.read(self.path + '/' + 'efs2D.ini')
        try:
            tmpDensities = config['FluidProperties']['InitialDensities']
            tmpBackground = config['FluidProperties']['BackgroundDensities']
        except KeyError:
            print("Could not find the parameter's name for fluid densities in .ini file.")
        self.initialDensities = np.asarray([], dtype = np.float64)
        self.backgroundDensities = np.asarray([], dtype = np.float64)
        tmpDensitiesList = tmpDensities.split(',')
        tmpBackgroundList = tmpBackground.split(',')
        if (len(tmpDensitiesList) == self.typesFluids):
            for i in tmpDensitiesList:
                self.initialDensities = np.append(self.initialDensities, float(i))
            print("The initial densities for fluids in the domain are:")
            print(self.initialDensities)
        else:
            print("The number of fluids does not match the number of densities.")
            sys.exit()
        if (len(tmpBackgroundList) == self.typesFluids):
            for i in tmpBackgroundList:
                self.backgroundDensities = np.append(self.backgroundDensities, float(i))
            print("The background densities for fluids in the domain are:")
            print(self.backgroundDensities)
        else:
            print("The number of fluids does not match the number of background densities.")
        #set up the values of taus for each fluids   
        try:
            tmpTau = config['FluidProperties']['FluidsTau']
        except KeyError:
            print("Could not find the parameter's name for tau of fluids in .ini file.")
        except:
            print("Unknown error happens when the tau are read.")
        self.tau = np.asarray([], dtype = np.float64)
        tmpTauList = tmpTau.split(',')
        if (len(tmpTauList) == self.typesFluids):
            for i in tmpTauList:
               self.tau = np.append(self.tau, float(i))
            print("The values of taus for each fluid are:")
            print(self.tau)
        else:
            print("The number of fluids does not match the number of viscosities.")
            sys.exit()
        #set up interaction coefficients between fluids
        try:
            tmpInterF = config['EFSParameters']['InteractionFluid']
        except KeyError:
            sys.exit()
        tmpInterFList = tmpInterF.split(',')
        tmpCount = 0
        self.interCoeff = np.zeros([self.typesFluids, self.typesFluids], dtype = np.float64)
        for i in np.arange(self.typesFluids - 1):
            for j in np.arange(i+1, self.typesFluids):
                self.interCoeff[i, j] = float(tmpInterFList[tmpCount])
                self.interCoeff[j, i] = self.interCoeff[i, j]
                tmpCount += 1
            print("Interaction coefficients are:")
            print(self.interCoeff)
        #set up interaction coefficients between fluids and solid
        try:
            tmpInterS = config['EFSParameters']['InteractionSolid']
        except KeyError:
            print("Cannot find the parameter's name for interaction with solid.")
        self.interactionSolid = np.asarray([], dtype = np.float64)
        tmpInterSList = tmpInterS.split(',')
        if (len(tmpInterSList) == self.typesFluids):
            for i in tmpInterSList:
                self.interactionSolid = np.append(self.interactionSolid, float(i))
            print("The interaction strength with solid phase is: ")
            print(self.interactionSolid)
        else:
#            print("The number of fluids does not match the number of interaction coeff with solid.")
            sys.exit()
        #set up forcing scheme
        try:
            self.explicitScheme = int(config['ForceScheme']['ExplicitScheme'])
        except KeyError:
            print("Can't find the parameter for forcing scheme.")
        except ValueError:
            print("The number for the scheme must be an integer.")
        except:
            print("Unknown error happened when the number of scheme is read.")
            sys.exit()
        #set up boundary
        self.boundaryTypeInlet = config['BoundaryDefinition']['BoundaryTypeInlet']
        self.boundaryMethod = config['BoundaryDefinition']['BoundaryMethod']
        self.boundaryTypeOutlet = config['BoundaryDefinition']['BoundaryTypeOutlet']
        if (self.boundaryTypeInlet == "'Dirichlet'"):
            print("The inlet boundary type is " + "Dirichlet.")
            try:
                tmpPressureValuesHigher = config['PressureBoundary']['PressureInlet']
            except KeyError:
                sys.exit()
            tmpPressureHigher = tmpPressureValuesHigher.split(',')
            self.inletPressures = np.asarray([], dtype = np.float64)
            if (len(tmpPressureHigher) == self.typesFluids):
                for i in tmpPressureHigher:
                    self.inletPressures = np.append(self.inletPressures, float(i))
        elif (self.boundaryTypeInlet == "'Neumann'"):
            print("The boundary type is " + "Neumann.")
            try:
                tmpVelocityX = config['VelocityBoundary']['velocityX']
                tmpVelocityY = config['VelocityBoundary']['velocityY']
            except KeyError:
                sys.exit()
            tmpVXList = tmpVelocityX.split(',')
            tmpVYList = tmpVelocityY.split(',')
            print(tmpVYList)
            self.velocityXInlet = np.asarray([], dtype = np.float64)
            self.velocityYInlet = np.asarray([], dtype = np.float64)
            if (len(tmpVXList) == self.typesFluids and len(tmpVYList) == \
                self.typesFluids):
                for i in tmpVXList:
                    self.velocityXInlet = np.append(self.velocityXInlet, float(i))
                for j in tmpVYList:
                    self.velocityYInlet = np.append(self.velocityYInlet, float(j))
            print("The velocities of inlet on the x- and y- directions are:")
            print(self.velocityXInlet)
            print(self.velocityYInlet)
        #set up whether the body force shoudl be included or not
        if (self.boundaryTypeOutlet == "'Dirichlet'"):
            print("The outlet boundary condition is the constant pressure type.")
            try:
                tmpPressureValuesLower = config['PressureBoundary']['PressureOutlet']
            except KeyError:
                sys.exit()
            tmpPressureLower = tmpPressureValuesLower.split(',')
            self.outletPressures = np.asarray([], dtype = np.float64)
            if (len(tmpPressureLower) == self.typesFluids):
                for i in tmpPressureLower:
                    self.outletPressures = np.append(self.outletPressures, float(i))
                    
        print("Read parameters on body force.")
        try:
            self.bodyForceOption = config['BodyForce']['Option']
        except KeyError:
            print("Can't find the parameter on having body force or not.")
        except:
            print("Unknow error happens when the option for body force is read.")
            sys.exit()
        if self.bodyForceOption == "'yes'":
            try:
                self.bodyForceXG = float(config['BodyForce']['forceXG'])
                self.bodyForceYG = float(config['bodyForce']['forceYG'])
            except KeyError:
                print("Can't find the parameters' name for body force.")
            except ValueError:
                print("The value for body force must be float.")
            except:
                print("Unknown error happens when the body force parameters are read.")
                sys.exit()
            print("The gravity or other acceleration in the domain is: ")
            print(self.bodyForceXG)
            print(self.bodyForceYG)
        #set up time steps for the simulation
        self.numTimeStep = int(config['Time']['numberTimeStep'])
        print("The number of time step is %g." % self.numTimeStep)
        if (self.relaxationType == "'MRT'"):
            for i in sp.arange(9):
                for j in sp.arange(9):
                    for k in sp.arange(self.typesFluids):
                        if (i == j):
                            self.relaxationMatrix[k, i, j] = self.diagonalValues[k, i]
#                            self.relaxationMatrix1[i, j] = self.diagonalValues1[i]
                            if (i == 7 or i == 8):
                                self.relaxationMatrix[k, i, j] = 1./self.tau[k]
#                                self.relaxationMatrix1[i, j] = 1./self.tau1
            for i in sp.arange(self.typesFluids):
                self.collisionMatrix[i] = np.dot(np.dot(self.inverseTransformation, \
                    self.relaxationMatrix[i]), self.transformationMatrix)
                print("The collision matrix for the fluid: ", i)
                print(self.collisionMatrix[i])
        input("If all the information on original ShanChen is correct, please hit Enter.")
                
    """
    Create .h5 file for saving the results on fluids
    """
    def __createHDF5File(self, ):
        print("Create .h5 (HDF file) to save the results of fluids.")
        username = getpass.getuser()
        pathfile = '/home/'+ username + '/LBMResults/'
        file = tb.open_file(pathfile + 'SimulationResults.h5', 'w')
        file.create_group(file.root, 'FluidMacro', 'MacroData')
        file.create_group(file.root, 'FluidVelocity', 'MacroVelocity')
        file.close()
        print("The file of .h5 has been created.")
        
    def __expandImageDomain(self, arrayDomain, xDirectionNum, yDirectionNum):
        """
        Generate larger domain from an image by duplicating it periodically
        """
        reverseX = np.fliplr(arrayDomain)
        reverseY = np.flipud(arrayDomain)
        reverseYX = np.fliplr(reverseY)
        newDomain = arrayDomain
        for i in sp.arange(yDirectionNum):
            if (i % 2 == 0):
                tmpDomainRow = np.vstack((arrayDomain))
                for j in sp.arange(1, xDirectionNum):
                    if (j % 2 != 0):
                        tmpDomainRow = np.hstack((tmpDomainRow, reverseX))
                    if (j % 2 == 0):
                        tmpDomainRow = np.hstack((tmpDomainRow, arrayDomain))
            elif (i % 2 != 0):
                tmpDomainRow = np.vstack((reverseY))
                for j in sp.arange(1, xDirectionNum):
                    if (j % 2 != 0):
                        tmpDomainRow = np.hstack((tmpDomainRow, reverseYX))
                    if (j % 2 == 0):
                        tmpDomainRow = np.hstack((tmpDomainRow, reverseY))
            if (i == 0):
                newDomain = tmpDomainRow
            elif (i > 0):
                newDomain = np.vstack((newDomain, tmpDomainRow))
        return newDomain
        
        
    def __processImage(self):
        """
        Load the image file and redefine the domain size
        """
        userName = getpass.getuser()
        pathImage = os.path.expanduser('~/StructureImage')
        imageFile = pathImage + '/structure.png'
        try:
            print('read the image')
            binaryImage = sciimage.imread(imageFile, True)
        except FileNotFoundError:
            print('The image file or the directory does not exist.')
        except:
            print('Other errors happen.')
        ySize, xSize = binaryImage.shape
        xPosition = []; yPosition = []
        for i in sp.arange(ySize):
            for j in sp.arange(xSize):
                if (binaryImage[i, j] == 0.0):
                    yPosition.append(i)
                    xPosition.append(j)
        xPosition = np.array(xPosition); yPosition = np.array(yPosition)
        xMin = xPosition.min(); xMax = xPosition.max()
        yMin = yPosition.min(); yMax = yPosition.max()
        #redefine the domain
        if (self.duplicateDomain == "'no'"):
            self.effectiveDomain = binaryImage[yMin:(yMax + 1), xMin:(xMax + 1)]
        elif (self.duplicateDomain == "'yes'"):
            tmpDomain = binaryImage[yMin:(yMax + 1), xMin:(xMax + 1)]
            xDirectionNum = int(input("Number of duplication in x direction: "))
            yDirectionNum = int(input("Number of duplication in y direction: "))
            self.effectiveDomain = self.__expandImageDomain(tmpDomain, xDirectionNum, \
                                                          yDirectionNum)
        yDimension, xDimension = self.effectiveDomain.shape
        self.effectiveDomain[:, 0] = 0.; self.effectiveDomain[:, -1] = 0.
        tmpBufferLayer = np.zeros(xDimension, dtype = np.float64)
        tmpBufferLayer[:] = 255.
        for i in sp.arange(40):
            if (i < 20):
                self.effectiveDomain = np.vstack((tmpBufferLayer, self.effectiveDomain))
            else:
                self.effectiveDomain = np.vstack((self.effectiveDomain, tmpBufferLayer))
                
    def optimizeFluidArray(self):
        """
        Convert 2D array of porous media matrix to 1D array for fluid nodes
        """
        print("Run the function for optimization.")
        self.fluidNodes = np.empty(self.voidSpace, dtype = np.int64)
        ySize = self.ny; xSize = self.nx
        print("Start to fill effective fluid nodes.")
        tmpIndicesDomain = -np.ones(self.isDomain.shape, dtype = np.int64)
        tmpIndicesFN = 0
        for i in sp.arange(ySize):
            for j in sp.arange(xSize):
                if (self.isDomain[i, j] == 1):
#                if (self.effectiveDomain[i, j] == 255.):
                    tmpIndices = i * xSize + j
                    self.fluidNodes[tmpIndicesFN] = tmpIndices
                    tmpIndicesDomain[i, j] = tmpIndicesFN
                    tmpIndicesFN += 1
        self.neighboringNodes = np.zeros(self.fluidNodes.size * 8, dtype = np.int64)
        if self.interactionType == "'EFS'":
            if self.explicitScheme == 8:
                self.neighboringNodesISO8 = np.zeros(self.fluidNodes.size * 24, \
                                                     dtype = np.int64)
            elif self.explicitScheme == 10:
                self.neighboringNodesISO10 = np.zeros(self.fluidNodes.size * 36, \
                                                     dtype = np.int64)
        totalNodes = self.fluidNodes.size
        #use cuda to generate the array for neighboring nodes
        print("Start to fill neighboring nodes")
        deviceFluidNodes = cuda.to_device(self.fluidNodes)
        devicetmpIndicesDomain = cuda.to_device(tmpIndicesDomain)
#        deviceIsDomain = cuda.to_device(self.isDomain)
        deviceNeighboringNodes = cuda.to_device(self.neighboringNodes)
        blockNumX = int(self.xDimension / self.threadNum) 
        blockNumY = math.ceil(self.fluidNodes.size / self.xDimension)
        threadPerBlock1D = (self.threadNum, 1)
        grid = (blockNumX, blockNumY)

        fillNeighboringNodes[grid, threadPerBlock1D](totalNodes, self.nx, self.ny, \
                            self.xDimension, deviceFluidNodes, devicetmpIndicesDomain, \
                            deviceNeighboringNodes)
        self.neighboringNodes = deviceNeighboringNodes.copy_to_host()
        if self.interactionType == "'EFS'":
            if self.explicitScheme == 8:
                deviceNeighboringNodesISO8 = cuda.to_device(self.neighboringNodesISO8)
                fillNeighboringNodesISO8[grid, threadPerBlock1D](totalNodes, self.nx, self.ny, \
                                    self.xDimension, deviceFluidNodes, devicetmpIndicesDomain, \
                                    deviceNeighboringNodesISO8)
                self.neighboringNodesISO8 = deviceNeighboringNodesISO8.copy_to_host()
            elif self.explicitScheme == 10:
                deviceNeighboringNodesISO10 = cuda.to_device(self.neighboringNodesISO10)
                fillNeighboringNodesISO10[grid, threadPerBlock1D](totalNodes, self.nx, self.ny, \
                                    self.xDimension, deviceFluidNodes, devicetmpIndicesDomain, \
                                    deviceNeighboringNodesISO10)
                self.neighboringNodesISO10 = deviceNeighboringNodesISO10.copy_to_host()
        
        print("Redefine the fluid nodes.")
#        cuda.current_context().trashing.clear()
        self.optFluidPDF = np.empty([self.typesFluids, self.fluidNodes.size, 9])
        self.optFluidRho = np.empty([self.typesFluids, self.fluidNodes.size])
        self.optMacroVelocity = np.zeros(self.fluidNodes.size)
        self.optMacroVelocityX = np.zeros(self.fluidNodes.size, dtype = np.float64)
        self.optMacroVelocityY = np.zeros(self.fluidNodes.size, dtype = np.float64)
        self.optForceX = np.zeros([self.typesFluids, self.fluidNodes.size], \
                                  dtype = np.float64)
        self.optForceY = np.zeros([self.typesFluids, self.fluidNodes.size], \
                                  dtype = np.float64)
        tmpDomain = np.array([i == 1 for i in self.isDomain.reshape(ySize * xSize)])
        for i in sp.arange(self.typesFluids):
            self.optFluidRho[i] = self.fluidsDensity.reshape(self.typesFluids, \
                            ySize * xSize)[i, tmpDomain]
            self.optFluidPDF[i] = self.fluidPDF.reshape(self.typesFluids, ySize * \
                            xSize, 9)[i, tmpDomain]
#        tmpPVX = np.zeros(self.fluidNodes.size); tmpPVY = np.zeros(self.fluidNodes.size)
#        devicePVX = cuda.to_device(tmpPVX); devicePVY = cuda.to_device(tmpPVY)
#        deviceFX = cuda.device_array_like(self.optFluidRho)
#        deviceFY = cuda.device_array_like(self.optFluidRho)
#        deviceEVX = cuda.device_array_like(self.optFluidRho)
#        deviceEVY = cuda.device_array_like(self.optFluidRho)
#        devicePDF = cuda.device_array_like(self.optFluidPDF)
#        weightCoeff = np.array([4./ 9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., \
#                        1./36.])
#        deviceWeightCoeff = cuda.to_device(weightCoeff)
#
#        weightInter = np.array([1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., \
#                                1./36.])
##        weightInter = np.array([1./3., 1./3., 1./3., 1./3., 1./12., 1./12., 1./12., \
##                                1./12.])
#        deviceWeightInter = cuda.to_device(weightInter)
#        
#        deviceTau = cuda.to_device(self.tau)
#        deviceInteractionCoeff = cuda.to_device(self.interCoeff)
#        deviceInterSolid = cuda.to_device(self.interactionSolid)
#        deviceRho = cuda.to_device(self.optFluidRho)
#        devicePotential = cuda.device_array_like(self.optFluidRho)
#        
#        calFluidPotentialGPUEql[grid, threadPerBlock1D](totalNodes, self.typesFluids, \
#                               self.xDimension, deviceRho, devicePotential)
#        calInteractionForce[grid, threadPerBlock1D](totalNodes, self.typesFluids, \
#                           self.nx, self.ny, self.xDimension, deviceFluidNodes, \
#                           deviceNeighboringNodes, deviceWeightInter, deviceInteractionCoeff, \
#                           deviceInterSolid, devicePotential, deviceFX, deviceFY)
#        calEquilibriumVGPU[grid, threadPerBlock1D](totalNodes, self.typesFluids, \
#                          self.xDimension, deviceTau, deviceRho, deviceFX, deviceFY, \
#                          devicePVX, devicePVY, deviceEVX, deviceEVY)
#        calEquilibriumFuncGPU[grid, threadPerBlock1D](totalNodes, self.typesFluids, \
#                             self.xDimension, deviceWeightCoeff, deviceRho, \
#                             deviceEVX, deviceEVY, devicePDF)
#        calFluidRhoGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, self.xDimension, \
#                              deviceRho, devicePDF)
#        self.optFluidPDF = devicePDF.copy_to_host()
#        self.optFluidRho = deviceRho.copy_to_host()
        
    @jit
    def initializeDomainBorder(self):
        """
        define the wall position in 2D domain
        """
        #Read image of the structure
        if (self.PictureExistance == "'yes'"):
            self.__processImage()
            #re-define the domain size with the layers of boundaries and ghost points
            self.ny, self.nx = self.effectiveDomain.shape
            print('Now the size of domain is %g and %g' %(self.ny, self.nx))
        else:
            self.isDomain = sp.empty([self.ny, self.nx], dtype = np.bool)
            self.isSolid = sp.empty([self.ny, self.nx], dtype = np.bool)
            self.isDomain, self.isSolid = defineGeometry(self.nx, self.ny)
        if (self.PictureExistance == "'yes'"):
            self.originalXdim = self.nx
            self.isDomain = sp.empty([self.ny, self.nx], dtype = np.bool)
            self.isSolid = sp.empty([self.ny, self.nx], dtype = np.bool)
            #define the boundary position
            if (self.isCycles == "'yes'"):
                self.isBoundaryFluid2 = sp.empty([self.ny, self.nx], \
                                        dtype = np.bool)
#            self.isFluidBoundary = sp.empty([self.ny, self.nx], dtype = np.bool)
            self.isDomain[:, :] = 1; self.isSolid[:, :] = 0
            for i in sp.arange(self.ny):
                for j in sp.arange(self.nx):
                        if (self.effectiveDomain[i, j] == 0.0):
                            self.isDomain[i, j] = 0
                            self.isSolid[i, j] = 1
        self.voidSpace = np.count_nonzero(self.isDomain)
        print('The number of vexls in void space is %g.' % self.voidSpace)
        print('The porosity of the layout is %f.' % (self.voidSpace / (self.isDomain.size)))
    
    def initializeDomainCondition(self):
        """
        Initialize components distribution and velocity in the domain
        """
        print('Initialize the condition.')

        self.fluidPDF = np.zeros([self.typesFluids, self.ny, self.nx, 9])
        self.fluidsDensity = np.zeros([self.typesFluids, self.ny, self.nx])
        self.physicalVX = np.zeros([self.ny, self.nx])
        self.physicalVY = np.zeros([self.ny, self.nx])
        self.forceX = np.zeros([self.typesFluids, self.ny, self.nx])
        self.forceY = np.zeros([self.typesFluids, self.ny, self.nx])
        if (self.PictureExistance == "'no'"):
            for i in sp.arange(self.ny):
                for j in sp.arange(self.nx):
#                    for k in sp.arange(self.typesFluids):
                    tmpCenterX = int(self.nx / 2); tmpCenterY = int(self.ny / 2)
                    if (self.isDomain[i, j] == True):
#                        if (sp.sqrt((i - tmpCenterY) * (i - tmpCenterY) + (j - \
#                                tmpCenterX) * (j - tmpCenterX)) <= 15.):
#                        if (i < 15 and np.abs(j - tmpCenterX) < 15):
#                        if ((i >0 and i < 28) and (j >=102 and j < 154)):
                        if (i < self.ny - 10):
#                        if (i < 128 and i > 70):
                            self.fluidsDensity[0, i, j] = self.initialDensities[0]
                            self.fluidPDF[0, i, j, :] = self.weightsCoeff * self.initialDensities[0]
                            self.fluidsDensity[1, i, j] = self.backgroundDensities[1]
                            self.fluidPDF[1, i, j, :] = self.weightsCoeff * self.backgroundDensities[1]
                        else:
                            self.fluidsDensity[1, i, j] = self.initialDensities[1]
                            self.fluidPDF[1, i, j, :] = self.weightsCoeff * self.initialDensities[1]
                            self.fluidsDensity[0, i, j] = self.backgroundDensities[0]
                            self.fluidPDF[0, i, j, :] = self.weightsCoeff * self.backgroundDensities[0]    
                            
        if (self.isCycles == "'no'" and self.PictureExistance == "'yes'"):
            for i in sp.arange(self.ny):
                for j in sp.arange(self.nx):
                    if (i < self.ny - 20):
    #                if ( np.abs(i - 60) < 20):
                        for k in sp.arange(self.typesFluids):
                            if (k == 0 and self.isDomain[i, j] == 1):
                                self.fluidPDF[k, i, j, :] = self.initialDensities[k] * self.weightsCoeff
                                self.fluidsDensity[k, i, j] = self.initialDensities[k]
                            if (k == 1 and self.isDomain[i, j] == 1):
                                self.fluidPDF[k, i, j, :] = self.backgroundDensities[k] * self.weightsCoeff
                                self.fluidsDensity[k, i, j] = self.backgroundDensities[k]
                    else:
                        for k in sp.arange(self.typesFluids):
                            if (k == 0 and self.isDomain[i, j] == 1):
                                self.fluidPDF[k, i, j, :] = self.backgroundDensities[k] * self.weightsCoeff
                                self.fluidsDensity[k, i, j] = self.backgroundDensities[k]
                            if (k == 1 and self.isDomain[i, j] == 1):
                                self.fluidPDF[k, i, j, :] = self.initialDensities[k] * self.weightsCoeff
                                self.fluidsDensity[k, i, j] = self.initialDensities[k]
        elif (self.isCycles == "'yes'" and self.PictureExistance == "'yes'"):
            username = getpass.getuser()
            pathIniFile = '/home/' + username + '/LBMInitial/'
            if (os.path.exists(pathIniFile) == True):   
                #for the old fluid distribution
                #the domain of the network
                iniFile = tb.open_file(pathIniFile + 'SimulationResults.h5', 'r')
                for i in sp.arange(self.typesFluids-1):
                    self.fluidsDensity[i, :-30, :] = eval('iniFile.root.FluidMacro.FluidDensityType%gin%d[:-30, :]' % (i, self.lastStep))
                    self.fluidsDensity[i, -30:, :] = self.backgroundDensities[i]
                    for j in sp.arange(self.ny):
                        for k in sp.arange(self.nx):
                            self.fluidPDF[i, j, k, :] = self.weightsCoeff * \
                                self.fluidsDensity[i, j, k]
                iniFile.close()
#            for the new fluid in the domain
                for i in sp.arange(self.ny):
                    for j in sp.arange(self.nx):
                        if (i < self.ny - 30 and self.isDomain[i, j] == 1):
                            self.fluidsDensity[-1, i, j] = self.backgroundDensities[-1]
                            self.fluidPDF[-1, i, j, :] = self.backgroundDensities[-1] * \
                                self.weightsCoeff
#                            continue
                        elif (i >= self.ny - 30 and self.isDomain[i, j] == 1):
                            self.fluidsDensity[-1, i, j] = self.initialDensities[-1]
                            self.fluidPDF[-1, i, j, :] = self.initialDensities[-1] * \
                                self.weightsCoeff
            else:
                print("There is no file for initializing the domain.")
                sys.exit()

                                   
    
    def calStreamingProcess(self):
        print('Streaming part.')
        for i in sp.arange(9):
            self.distrFluid0[i, :, :] = np.roll(np.roll(self.distrFluid0[i,:, :], \
                                        int(self.microVelocity[i, 1]), axis = 0), \
                                        int(self.microVelocity[i, 0]), axis = 1)
            self.distrFluid1[i, :, :] = np.roll(np.roll(self.distrFluid1[i,:, :], \
                                        int(self.microVelocity[i, 1]), axis = 0), \
                                        int(self.microVelocity[i, 0]), axis = 1)
                                        
    def __calSteadyStateCritiria(self, timeStep):
        if (timeStep == 0):
            self.velocityRecord0X[:, :] = self.velocityX0[:, :]
            self.velocityRecrod0Y[:, :] = self.velocityY0[:, :]
            self.velocityRecord1X[:, :] = self.velocityX1[:, :]
            self.velocityRecord1Y[:, :] = self.velocityY1[:, :]
        else:
            relativeError = 0.0
            tmpVelocity = sp.sqrt(sp.power(self.velocityX, 2) + sp.power(self.velocityY, \
                2))
            tmpVelocityRecord = sp.sqrt(sp.power(self.velocityRecord0X, 2) + \
                sp.power(self.velocityRecord0Y, 2))
            tmpDiff = tmpVelocity - tmpVelocityRecord
            tmpDiffSquare = sp.power(tmpDiff, 2)
            tmpDiffSS = sp.sqrt(sp.sum(tmpDiffSquare))
            tmpVelocitySquareSum = sp.sum(sp.power(self.velocityX, 2) + \
                sp.power(self.velocityY, 2))
            tmpVelocitySS = sp.sqrt(tmpVelocitySquareSum)
            relativeError = tmpDiffSS / tmpVelocitySS
        return relativeError
                                        
    def __calContactAngle(self):
        tmpContactAngle = np.arccos((self.interactionSurface1 - self.interactionSurface0) \
                        /(self.interactionStrength * (self.fluid0Density - \
                        self.fluid1Density) / 2.))
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMResults/'
        fileName = pathResults + 'ContactAngle.dat'
        saveContactAngle = open(fileName, 'ab')
        np.savetxt(fileName, tmpContactAngle)
        saveContactAngle.close()
#        return tmpContactAngle

    def calMeasuredContactAngle(self):
        """
        calculate the contact angle from the simulation
        """
        #account the base
        bottomLength = 0
        arrayHeight = np.empty([0, ], dtype = 'int64')
        for i in sp.arange(self.nx):
            if (self.densityFluid1[1, i] >= 0.485):
                bottomLength += 1
        #account the height
        for m in sp.arange(self.nx):
            tmpHeight = 0
            for n in sp.arange(1, self.ny - 1):
                if (self.densityFluid1[n, m] >= 0.485):
                    tmpHeight += 1
            arrayHeight = np.append(arrayHeight, tmpHeight)
        heightH = np.amax(arrayHeight)
        #radius of droplet
        radiusD = (4. * np.power(heightH, 2.) + np.power(bottomLength, 2.)) / \
                (8. * heightH)
        contactAngle = np.arctan((bottomLength) / (2. * (radiusD - heightH))) 
        return contactAngle
        
    def plotDensityDistributionOPT(self, iStep, typeFluid):
        """
        Plot fluid 0 density distribution in the whole domain
        """
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMResults/'
        plotFluid = plt.figure()
        cmap = colors.ListedColormap(['green', 'blue', 'red'])
        bounds = [0.0, 0.01, 1.0, 2.0]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig = plt.imshow(self.fluidsDensity[typeFluid, 2:-2, :], origin = 'lower')
#        plt.colorbar(fig, cmap = cmap, norm = norm, boundaries = boundds, \
#                     ticks = [0.0, 0.06, 2.0])
#        plt.axis('off')
        plt.colorbar()
#        fig.axes.get_xaxis().set_visible(False)
#        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(pathResults + 'Fluid%gdistribution%05d.png' % (typeFluid, iStep))
#        , bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        
#        plt.plot(np.arange(self.ny), self.fluidsDensity[1, :, 25], '-')
#        plt.savefig(pathResults + 'ProfileDensity%05d.png' % iStep)
#        plt.close()
#        input()
        
    def plotPhysicalVelocity(self, iStep):
        xGrid = sp.arange(self.nx); yGrid = sp.arange(self.ny)
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMVelocity/'
        plotVelocity1 = plt.figure()
        velocityPlot = plt.quiver(xGrid[1:-1:2], yGrid[1:-1:2], \
                    self.physicalVX[1:-1:2, 1:-1:2], \
                    self.physicalVY[1:-1:2, 1:-1:2], \
                    pivot = 'mid', color = 'r', units='inches')
#        plt.axis([-15, self.nx + 15, -15, self.ny + 15])
        plt.savefig(pathResults + 'Velocity%05d.png' % (iStep))
        plt.close()
        
    def plotVelocityNorms(self, iStep):
        print("Caclulate the norm of velocity in the domain.")
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMResults/'
        tmpVelocityNorm = np.sqrt(np.power(self.physicalVX, 2.) + np.power(self.physicalVY, 2.))
        plt.imshow(tmpVelocityNorm, origin = 'lower')
        plt.colorbar()
        plt.savefig(pathResults + 'VelocityNorm%05d.png' % iStep)
        plt.close()
#        plt.plot(tmpVelocityNorm[4, :])
#        plt.savefig(pathResults + 'VelocityNormProfile%05d.png' % iStep)
#        plt.close()
        
    def resultInHDF5(self, iStep):
        """
        Save the data from the simulation in HDF5 fromat
        """
        filePath = os.path.expanduser('~/LBMResults')
        resultFile = filePath + '/SimulationResults.h5'
        dataFile = tb.open_file(resultFile, 'a')
        #output the densities of fluids
        for i in sp.arange(self.typesFluids):
            dataFile.create_array('/FluidMacro', 'FluidDensityType%gin%g' % (i, iStep), \
                                  self.fluidsDensity[i])
        dataFile.create_array('/FluidVelocity', 'FluidVelocityXAt%g' % iStep, \
                              self.physicalVX)
        dataFile.create_array('/FluidVelocity', 'FluidVelocityYAt%g' % iStep, \
                              self.physicalVY)
        dataFile.close()

    @autojit
    def convertOptTo2D(self):
        tmpIndex = 0
        for tmpPos in self.fluidNodes:
            tmpX = tmpPos % self.nx; tmpY = int(tmpPos / self.nx)
            for i in sp.arange(self.typesFluids):
                self.fluidsDensity[i, tmpY, tmpX] = self.optFluidRho[i, tmpIndex]
                self.fluidPDF[i, tmpY, tmpX] = self.optFluidPDF[i, tmpIndex]
            self.physicalVX[tmpY, tmpX] = self.optMacroVelocityX[tmpIndex]
            self.physicalVY[tmpY, tmpX] = self.optMacroVelocityY[tmpIndex]
            tmpIndex += 1        
                    
    def runMultiComponentEFGPU(self):
        #for the periodic boundary condition
        periodicLeft0 = np.zeros([3, self.ny], dtype = np.float64)
        periodicRight0 = np.zeros([3, self.ny], dtype = np.float64)
        periodicBottom0 = np.zeros([3, self.nx], dtype = np.float64)
        periodicTop0 = np.zeros([3, self.nx], dtype = np.float64)
        
        self.initializeDomainBorder()
        self.initializeDomainCondition()
        self.calMacroParametersGPU()
        self.convertArraysForGPU()
        #copy from host to device
        externalForce = np.zeros(self.densityF01D.size, dtype = 'float64')
        deviceDistrF0One = cuda.to_device(self.distrFHost0)
        deviceDistrF1One = cuda.to_device(self.distrFHost1)
        deviceDistrF0New = cuda.to_device(self.distrFHost0)
        deviceDistrF1New = cuda.to_device(self.distrFHost1)
        
        deviceDistrF0M = cuda.to_device(self.distrFHost0)
        deviceDistrF1M = cuda.to_device(self.distrFHost1)
        
        deviceFluid0Density1D = cuda.to_device(self.densityF01D)
        deviceFluid1Density1D = cuda.to_device(self.densityF11D)
        deviceVelocityX01D = cuda.to_device(self.velocityX01D)
        deviceVelocityY01D = cuda.to_device(self.velocityY01D)
        deviceVelocityX11D = cuda.to_device(self.velocityX01D)
        deviceVelocityY11D = cuda.to_device(self.velocityY01D)
        deviceDomain1D = cuda.to_device(self.isDomainHost)
        deviceSolid1D = cuda.to_device(self.isSolidHost)
        deviceBoundaryF0 = cuda.to_device(self.isBoundaryF0Host)
        deviceBoundaryF1 = cuda.to_device(self.isBoundaryF1Host)
        
        deviceEffectiveVX = cuda.to_device(self.velocityX01D)
        deviceEffectiveVY = cuda.to_device(self.velocityY01D)
        
        deviceEquilibriumFunc0 = cuda.to_device(self.distrFHost0)
        deviceEquilibriumFunc1 = cuda.to_device(self.distrFHost1)
        
        deviceExternalF0X = cuda.to_device(externalForce)
        deviceExternalF0Y = cuda.to_device(externalForce)
        deviceExternalF1X = cuda.to_device(externalForce)
        deviceExternalF1Y = cuda.to_device(externalForce)
        
        deviceForcingTerm0 = cuda.to_device(self.distrFHost0)
        deviceForcingTerm1 = cuda.to_device(self.distrFHost1)
        
        devicePeriodicLeft0 = cuda.to_device(periodicLeft0)
        devicePeriodicRight0 = cuda.to_device(periodicRight0)    
        devicePeriodicLeft1 = cuda.to_device(periodicLeft0)
        devicePeriodicRight1 = cuda.to_device(periodicRight0)
        
        devicePeriodicTop0 = cuda.to_device(periodicTop0)
        devicePeriodicBottom0 = cuda.to_device(periodicBottom0)
        devicePeriodicTop1 = cuda.to_device(periodicTop0)
        devicePeriodicBottom1 = cuda.to_device(periodicBottom0)
        
        threadNumber = 32
        blockNumX2D = int(np.ceil(float(self.nx) / threadNumber))
        blockNumY2D = int(np.ceil(float(self.ny) / threadNumber))
        grid2D = (blockNumX2D, blockNumY2D)
        threadPerBlock1D = (threadNumber, 1)
        blockNumX1D = math.ceil(float(self.nx) / threadNumber)
        blockNumY1D = math.ceil(float(self.ny) / threadNumber)
        grid1DX = (blockNumX1D, self.ny)
        grid1DY = (1, blockNumY1D)
        
        print("Calculate the effective velocity.")
        calEffectiveVGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, self.tau0, \
            self.tau1, deviceFluid0Density1D, deviceFluid1Density1D, \
            deviceVelocityX01D, deviceVelocityY01D, deviceVelocityX11D, \
            deviceVelocityY11D, deviceEffectiveVX, deviceEffectiveVY, \
            deviceDomain1D)
        print("End the calculation of effective velocity.")
        print("Start to calculate the equilibrium function of each fluid.")
        calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid0Density1D, deviceEffectiveVX, deviceEffectiveVY, \
            deviceEquilibriumFunc0, deviceDomain1D)
        calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid1Density1D, deviceEffectiveVX, deviceEffectiveVY, \
            deviceEquilibriumFunc1, deviceDomain1D)
        print("End the calculation of the equilibrium function of each fluid.")
        print("Start to calculate the force between fluids.")
        calInteractionForceEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, 6.0, \
            self.interactionStrength, deviceFluid0Density1D, deviceFluid1Density1D, \
            deviceExternalF0X, deviceExternalF0Y, deviceExternalF1X, deviceExternalF1Y, \
            deviceDomain1D, deviceSolid1D)
        print("End the calculation for the force between fluids.")
        print("Start to calculate the force between fluid and solid.")
        calExternalForceSolidEF[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            self.interactionSurface0, self.interactionSurface1, deviceFluid0Density1D, \
            deviceFluid1Density1D, deviceExternalF0X, deviceExternalF0Y, \
            deviceExternalF1X, deviceExternalF1Y, deviceDomain1D, deviceSolid1D)
        print("End the calculation for the force between solid and fluid.")
        print("Start to calculate forcing term for distribution function.")
        calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid0Density1D, deviceExternalF0X, deviceExternalF0Y, \
            deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc0, \
            deviceForcingTerm0, deviceDomain1D)
        calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid1Density1D, deviceExternalF1X, deviceExternalF1Y, \
            deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc1, \
            deviceForcingTerm1, deviceDomain1D)
        print("End the calculation of forcing term for distrubition function")
        print("Start to transform the distribution function with forcing term.")
        calTransformedDistrFuncGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0One, deviceForcingTerm0, deviceDomain1D)
        calTransformedDistrFuncGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1One, deviceForcingTerm1, deviceDomain1D)
        print("End transforming distrbution.")
        for i in sp.arange(self.numTimeStep + 1):
            print('The time step of %d' % i)
            starter = timer()
            #collision part
            print("Start the collision part.")
            calCollisionEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, self.tau0, \
                deviceDistrF0One,deviceEquilibriumFunc0, deviceForcingTerm0, deviceDomain1D)
            calCollisionEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, self.tau1, \
                deviceDistrF1One, deviceEquilibriumFunc1, deviceForcingTerm1, deviceDomain1D)
            print("Finish the collision part.")
            calHalfWallBounceBack[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0One, deviceDomain1D, deviceSolid1D)
            calHalfWallBounceBack[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1One, deviceDomain1D, deviceSolid1D)
            
            #streaming part
            print("Start the streaming process.")
            calStreamingStep1[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0One, deviceDistrF0M)
            calStreamingStep2[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0New, deviceDistrF0M)
            calStreamingStep1[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1One, deviceDistrF1M)
            calStreamingStep2[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1New, deviceDistrF1M)
            print("End the streaming process.")
            
            
            #calculate macroscopic parameters - density and velocity
            print("Recalculate the density of each lattice.")
            calMacroDensityGPU1D[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid0Density1D, \
            deviceDistrF0New, deviceDistrF0One, deviceDomain1D)
            calMacroDensityGPU1D[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid1Density1D, \
            deviceDistrF1New, deviceDistrF1One, deviceDomain1D)
            print("End the calculation of density")
#            input()
            print("Recalculate the velocity of each fluid.")
            print("Start to calculate the force between fluids.")
            calInteractionForceEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, 6.0, \
                self.interactionStrength, deviceFluid0Density1D, deviceFluid1Density1D, \
                deviceExternalF0X, deviceExternalF0Y, deviceExternalF1X, deviceExternalF1Y, \
                deviceDomain1D, deviceSolid1D)
            print("End the calculation for the force between fluids.")
            print("Start to calculate the force between fluid and solid.")
            calExternalForceSolidEF[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                self.interactionSurface0, self.interactionSurface1, deviceFluid0Density1D, \
                deviceFluid1Density1D, deviceExternalF0X, deviceExternalF0Y, \
                deviceExternalF1X, deviceExternalF1Y, deviceDomain1D, deviceSolid1D)
            calMacroVelocityEFGPU(self.nx, self.ny, deviceFluid0Density1D, \
                deviceExternalF0X, deviceExternalF0Y, deviceDistrF0One, \
                deviceVelocityX01D, deviceVelocityY01D, deviceDomain1D)
            calMacroVelocityEFGPU(self.nx, self.ny, deviceFluid1Density1D, \
                deviceExternalF1X, deviceExternalF1Y, deviceDistrF1One, \
                deviceVelocityX11D, deviceVelocityY11D, deviceDomain1D)
            if (i % 1 == 0):
                self.distrFHost0 = deviceDistrF0New.copy_to_host()
                self.distrFHost1 = deviceDistrF1New.copy_to_host()
                self.densityF01D = deviceFluid0Density1D.copy_to_host()
                self.densityF11D = deviceFluid1Density1D.copy_to_host()
                self.velocityX01D = deviceVelocityX01D.copy_to_host()
                self.velocityY01D = deviceVelocityY01D.copy_to_host()
                self.velocityX11D = deviceVelocityX11D.copy_to_host()
                self.velocityY11D = deviceVelocityY11D.copy_to_host()
                
                self.convert1DdistTo2D()
#                print('Plot the results.')
#                print(self.densityFluid1[1, :])
#                input()
                self.plotDensityDistribution0(i)
                self.plotDensityDistribution1(i)
#                self.plotVelocity(i)
#                self.plotVelocityFluid0(i)
#                self.plotVelocityFluid1(i)
        
            #for the next time step
            calEffectiveVGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, self.tau0, \
                self.tau1, deviceFluid0Density1D, deviceFluid1Density1D, \
                deviceVelocityX01D, deviceVelocityY01D, deviceVelocityX11D, \
                deviceVelocityY11D, deviceEffectiveVX, deviceEffectiveVY, \
                deviceDomain1D)
            print("End the calculation of effective velocity.")
            print("Start to calculate the equilibrium function of each fluid.")
            calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid0Density1D, deviceEffectiveVX, deviceEffectiveVY, \
                deviceEquilibriumFunc0, deviceDomain1D)
            calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid1Density1D, deviceEffectiveVX, deviceEffectiveVY, \
                deviceEquilibriumFunc1, deviceDomain1D)
            print("End the calculation of the equilibrium function of each fluid.")
#            print("Start to calculate the force between fluids.")
#            calInteractionForceEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, 6.0, \
#                self.interactionStrength, deviceFluid0Density1D, deviceFluid1Density1D, \
#                deviceExternalF0X, deviceExternalF0Y, deviceExternalF1X, deviceExternalF1Y, \
#                deviceDomain1D, deviceSolid1D)
#            print("End the calculation for the force between fluids.")
#            print("Start to calculate the force between fluid and solid.")
#            calExternalForceSolidEF[grid1DX, threadPerBlock1D](self.nx, self.ny, \
#                self.interactionSurface0, self.interactionSurface1, deviceFluid0Density1D, \
#                deviceFluid0Density1D, deviceExternalF0X, deviceExternalF0Y, \
#                deviceExternalF1X, deviceExternalF1Y, deviceDomain1D, deviceSolid1D)
#            print("End the calculation for the force between solid and fluid.")
            print("Start to calculate forcing term for distribution function.")
            calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid0Density1D, deviceExternalF0X, deviceExternalF0Y, \
                deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc0, \
                deviceForcingTerm0, deviceDomain1D)
            calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid1Density1D, deviceExternalF1X, deviceExternalF1Y, \
                deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc1, \
                deviceForcingTerm1, deviceDomain1D)
                
    def runMultiComponentEFMRTGPU(self):
        #for the periodic boundary condition
        periodicLeft0 = np.zeros([3, self.ny], dtype = np.float64)
        periodicRight0 = np.zeros([3, self.ny], dtype = np.float64)
        periodicBottom0 = np.zeros([3, self.nx], dtype = np.float64)
        periodicTop0 = np.zeros([3, self.nx], dtype = np.float64)
        
        self.initializeDomainBorder()
        self.initializeDomainCondition()
        self.calMacroParametersGPU()
        self.convertArraysForGPU()
        #copy from host to device
        externalForce = np.zeros(self.densityF01D.size, dtype = 'float64')
        deviceDistrF0One = cuda.to_device(self.distrFHost0)
        deviceDistrF1One = cuda.to_device(self.distrFHost1)
        deviceDistrF0New = cuda.to_device(self.distrFHost0)
        deviceDistrF1New = cuda.to_device(self.distrFHost1)
        
        deviceDistrF0M = cuda.to_device(self.distrFHost0)
        deviceDistrF1M = cuda.to_device(self.distrFHost1)
        
        deviceFluid0Density1D = cuda.to_device(self.densityF01D)
        deviceFluid1Density1D = cuda.to_device(self.densityF11D)
        deviceVelocityX01D = cuda.to_device(self.velocityX01D)
        deviceVelocityY01D = cuda.to_device(self.velocityY01D)
        deviceVelocityX11D = cuda.to_device(self.velocityX01D)
        deviceVelocityY11D = cuda.to_device(self.velocityY01D)
        deviceDomain1D = cuda.to_device(self.isDomainHost)
        deviceSolid1D = cuda.to_device(self.isSolidHost)
        deviceBoundaryF0 = cuda.to_device(self.isBoundaryF0Host)
        deviceBoundaryF1 = cuda.to_device(self.isBoundaryF1Host)
        
        deviceEffectiveVX = cuda.to_device(self.velocityX01D)
        deviceEffectiveVY = cuda.to_device(self.velocityY01D)
        
        deviceEquilibriumFunc0 = cuda.to_device(self.distrFHost0)
        deviceEquilibriumFunc1 = cuda.to_device(self.distrFHost1)
        
        deviceExternalF0X = cuda.to_device(externalForce)
        deviceExternalF0Y = cuda.to_device(externalForce)
        deviceExternalF1X = cuda.to_device(externalForce)
        deviceExternalF1Y = cuda.to_device(externalForce)
        
        deviceForcingTerm0 = cuda.to_device(self.distrFHost0)
        deviceForcingTerm1 = cuda.to_device(self.distrFHost1)
        
        devicePeriodicLeft0 = cuda.to_device(periodicLeft0)
        devicePeriodicRight0 = cuda.to_device(periodicRight0)    
        devicePeriodicLeft1 = cuda.to_device(periodicLeft0)
        devicePeriodicRight1 = cuda.to_device(periodicRight0)
        
        devicePeriodicTop0 = cuda.to_device(periodicTop0)
        devicePeriodicBottom0 = cuda.to_device(periodicBottom0)
        devicePeriodicTop1 = cuda.to_device(periodicTop0)
        devicePeriodicBottom1 = cuda.to_device(periodicBottom0)
        
        #Matrix in MRT (multi-relaxation scheme)
        deviceCollision0 = cuda.to_device(self.collisionMatrix0)
        deviceCollision1 = cuda.to_device(self.collisionMatrix1)
        conserveS0 = self.relaxationMatrix0[0, 0]
        conserveS1 = self.relaxationMatrix0[0, 0]
        deviceForceTermM0 = cuda.to_device(self.distrFHost0)
        deviceForceTermM1 = cuda.to_device(self.distrFHost1)
        deviceFluid0DistrM = cuda.to_device(self.distrFHost0)
        deviceFluid1DistrM = cuda.to_device(self.distrFHost1)
        
        
        threadNumber = 32
        threadPerBlock1D = (threadNumber, 1)
        blockNumX1D = math.ceil(float(self.nx) / threadNumber)
        blockNumY1D = math.ceil(float(self.ny) / threadNumber)
        grid1DX = (blockNumX1D, self.ny)
        grid1DY = (1, blockNumY1D)
        print("Start to calculate the effective velocity in MRT.")
        calEffectiveVGPUMRT[grid1DX, threadPerBlock1D](self.nx, self.ny, conserveS0, \
            conserveS1, deviceFluid0Density1D, deviceFluid1Density1D, deviceVelocityX01D, \
            deviceVelocityY01D, deviceVelocityX11D, deviceVelocityY11D, \
            deviceEffectiveVX, deviceEffectiveVY, deviceDomain1D)

        print("End the caclulation for the effective velocity in MRT")
        print("Start to calculate the equilibrium function of each fluid.")
        calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid0Density1D, deviceEffectiveVX, deviceEffectiveVY, \
            deviceEquilibriumFunc0, deviceDomain1D)
        calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid1Density1D, deviceEffectiveVX, deviceEffectiveVY, \
            deviceEquilibriumFunc1, deviceDomain1D)
        print("Start to calculate the force between fluids.")
        calInteractionForceEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, 6.0, \
            self.interactionStrength, deviceFluid0Density1D, deviceFluid1Density1D, \
            deviceExternalF0X, deviceExternalF0Y, deviceExternalF1X, deviceExternalF1Y, \
            deviceDomain1D, deviceSolid1D)
        print("End the calculation for the force between fluids.")
        print("Start to calculate the force between fluid and solid.")
        calExternalForceSolidEF[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            self.interactionSurface0, self.interactionSurface1, deviceFluid0Density1D, \
            deviceFluid1Density1D, deviceExternalF0X, deviceExternalF0Y, \
            deviceExternalF1X, deviceExternalF1Y, deviceDomain1D, deviceSolid1D)
        print("End the calculation for the force between solid and fluid.")
        print("Start to calculate forcing term for distribution function.")
        calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid0Density1D, deviceExternalF0X, deviceExternalF0Y, \
            deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc0, \
            deviceForcingTerm0, deviceDomain1D)
        calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid1Density1D, deviceExternalF1X, deviceExternalF1Y, \
            deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc1, \
            deviceForcingTerm1, deviceDomain1D)
        print("End the calculation of forcing term for distrubition function")
        print("Start to transform the distribution function with forcing term.")
        calTransformedDistrFuncGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0One, deviceForcingTerm0, deviceDomain1D)
        calTransformedDistrFuncGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1One, deviceForcingTerm1, deviceDomain1D)
        print("End transforming distrbution.")
        for i in sp.arange(self.numTimeStep + 1):
            print('Time step is %g' % i)
            print('Start the MRT transformation for the force term.')
            calTransformForceTerm[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceForcingTerm0, deviceCollision0, deviceForceTermM0, deviceDomain1D)
            calTransformForceTerm[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceForcingTerm1, deviceCollision1, deviceForceTermM1, deviceDomain1D)
            print('End the MRT transformation for the force term.')
            print('Start the MRT transformation for distrubition and equilibrium function term.')
            calTransformFandFeq[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceDistrF0One, deviceEquilibriumFunc0, deviceCollision0, \
                deviceFluid0DistrM, deviceDomain1D)
            calTransformFandFeq[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceDistrF1One, deviceEquilibriumFunc1, deviceCollision1, \
                deviceFluid1DistrM, deviceDomain1D)
            print('End the MRT transformation for distribution and equilibrium function term.')
#
            print('Start the collision of MRT.')
            calFinalTransformEFMRT[grid1DX, threadPerBlock1D](self.nx, self.ny, deviceDistrF0One, \
                deviceForcingTerm0, deviceEquilibriumFunc0, deviceFluid0DistrM, \
                deviceForceTermM0, deviceDomain1D)
            calFinalTransformEFMRT[grid1DX, threadPerBlock1D](self.nx, self.ny, deviceDistrF1One, \
                deviceForcingTerm1,deviceEquilibriumFunc1, deviceFluid1DistrM, \
                deviceForceTermM1, deviceDomain1D)
            print('End the collision of MRT.')
            
            calHalfWallBounceBack[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0One, deviceDomain1D, deviceSolid1D)
            calHalfWallBounceBack[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1One, deviceDomain1D, deviceSolid1D)
            
            #streaming part
            print("Start the streaming process.")
            calStreamingStep1[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0One, deviceDistrF0M)
            calStreamingStep2[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF0New, deviceDistrF0M)
            calStreamingStep1[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1One, deviceDistrF1M)
            calStreamingStep2[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceDistrF1New, deviceDistrF1M)
            print("End the streaming process.")
            #calculate macroscopic parameters - density and velocity
            print("Recalculate the density of each lattice.")
            calMacroDensityGPU1D[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid0Density1D, \
            deviceDistrF0New, deviceDistrF0One, deviceDomain1D)
            calMacroDensityGPU1D[grid1DX, threadPerBlock1D](self.nx, self.ny, \
            deviceFluid1Density1D, \
            deviceDistrF1New, deviceDistrF1One, deviceDomain1D)
            print("End the calculation of density")
#            input()
            print("Recalculate the velocity of each fluid.")
            print("Start to calculate the force between fluids.")
            calInteractionForceEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, 6.0, \
                self.interactionStrength, deviceFluid0Density1D, deviceFluid1Density1D, \
                deviceExternalF0X, deviceExternalF0Y, deviceExternalF1X, deviceExternalF1Y, \
                deviceDomain1D, deviceSolid1D)
            print("End the calculation for the force between fluids.")
            print("Start to calculate the force between fluid and solid.")
            calExternalForceSolidEF[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                self.interactionSurface0, self.interactionSurface1, deviceFluid0Density1D, \
                deviceFluid1Density1D, deviceExternalF0X, deviceExternalF0Y, \
                deviceExternalF1X, deviceExternalF1Y, deviceDomain1D, deviceSolid1D)
            calMacroVelocityEFGPU(self.nx, self.ny, deviceFluid0Density1D, \
                deviceExternalF0X, deviceExternalF0Y, deviceDistrF0One, \
                deviceVelocityX01D, deviceVelocityY01D, deviceDomain1D)
            calMacroVelocityEFGPU(self.nx, self.ny, deviceFluid1Density1D, \
                deviceExternalF1X, deviceExternalF1Y, deviceDistrF1One, \
                deviceVelocityX11D, deviceVelocityY11D, deviceDomain1D)
            if (i % 10 == 0):
                self.distrFHost0 = deviceDistrF0New.copy_to_host()
                self.distrFHost1 = deviceDistrF1New.copy_to_host()
                self.densityF01D = deviceFluid0Density1D.copy_to_host()
                self.densityF11D = deviceFluid1Density1D.copy_to_host()
                self.velocityX01D = deviceVelocityX01D.copy_to_host()
                self.velocityY01D = deviceVelocityY01D.copy_to_host()
                self.velocityX11D = deviceVelocityX11D.copy_to_host()
                self.velocityY11D = deviceVelocityY11D.copy_to_host()
                
                self.convert1DdistTo2D()
#                print('Plot the results.')
#                print(self.densityFluid1[1, :])
#                input()
                self.plotDensityDistribution0(i)
                self.plotDensityDistribution1(i)
#                self.plotVelocity(i)
#                self.plotVelocityFluid0(i)
#                self.plotVelocityFluid1(i)
            calEffectiveVGPUMRT[grid1DX, threadPerBlock1D](self.nx, self.ny, conserveS0, \
                conserveS1, deviceFluid0Density1D, deviceFluid1Density1D, deviceVelocityX01D, \
                deviceVelocityY01D, deviceVelocityX11D, deviceVelocityY11D, \
                deviceEffectiveVX, deviceEffectiveVY, deviceDomain1D)
            print("End the calculation of effective velocity.")
            print("Start to calculate the equilibrium function of each fluid.")
            calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid0Density1D, deviceEffectiveVX, deviceEffectiveVY, \
                deviceEquilibriumFunc0, deviceDomain1D)
            calEquilibriumFuncEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid1Density1D, deviceEffectiveVX, deviceEffectiveVY, \
                deviceEquilibriumFunc1, deviceDomain1D)
            print("End the calculation of the equilibrium function of each fluid.")
#            print("Start to calculate the force between fluids.")
#            calInteractionForceEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, 6.0, \
#                self.interactionStrength, deviceFluid0Density1D, deviceFluid1Density1D, \
#                deviceExternalF0X, deviceExternalF0Y, deviceExternalF1X, deviceExternalF1Y, \
#                deviceDomain1D, deviceSolid1D)
#            print("End the calculation for the force between fluids.")
#            print("Start to calculate the force between fluid and solid.")
#            calExternalForceSolidEF[grid1DX, threadPerBlock1D](self.nx, self.ny, \
#                self.interactionSurface0, self.interactionSurface1, deviceFluid0Density1D, \
#                deviceFluid0Density1D, deviceExternalF0X, deviceExternalF0Y, \
#                deviceExternalF1X, deviceExternalF1Y, deviceDomain1D, deviceSolid1D)
#            print("End the calculation for the force between solid and fluid.")
            print("Start to calculate forcing term for distribution function.")
            calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid0Density1D, deviceExternalF0X, deviceExternalF0Y, \
                deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc0, \
                deviceForcingTerm0, deviceDomain1D)
            calForcingTermEFGPU[grid1DX, threadPerBlock1D](self.nx, self.ny, \
                deviceFluid1Density1D, deviceExternalF1X, deviceExternalF1Y, \
                deviceEffectiveVX, deviceEffectiveVY, deviceEquilibriumFunc1, \
                deviceForcingTerm1, deviceDomain1D)
            
    """
    Function for optimized data structure of LBM with multiphase flow (not just two phase)
    """
    def runOptimizedLBM(self):
        print("Start to initialize the original 2D domain.")
        self.initializeDomainBorder()
        self.initializeDomainCondition()
#        self.calMacroParametersGPU()
        print("Finish the initialization of the original 2D domain..")
        print("Convert the data in the original 2D domain to the optimized way.")
        self. optimizeFluidArray()
        
#        input()
        print("Finish the conversion of the data.")
        print("Start to set up arrays in GPU.")
        deviceFluidPDF = cuda.to_device(self.optFluidPDF)
        deviceFluidPDFold = cuda.to_device(self.optFluidPDF)
        deviceFluidPDFNew = cuda.to_device(self.optFluidPDF)
        deviceFluidRho = cuda.to_device(self.optFluidRho)
        deviceFluidIndices = cuda.to_device(self.fluidNodes)
        deviceNeighboringNodes = cuda.to_device(self.neighboringNodes)
        
        initialFX = np.zeros(self.optFluidRho.shape, dtype = np.float64)
        initialFY = np.zeros(self.optFluidRho.shape, dtype = np.float64)
        
        deviceFluidPotential = cuda.device_array_like(self.optFluidRho)
        deviceEquilibriumVX = cuda.device_array_like(self.optFluidRho)
        deviceEquilibriumVY = cuda.device_array_like(self.optFluidRho)
        deviceForceX = cuda.to_device(initialFX)
        deviceForceY = cuda.to_device(initialFY)
        devicePrimeVX = cuda.to_device(self.optMacroVelocity)
        devicePrimeVY = cuda.to_device(self.optMacroVelocity)
        deviceEquiliriumFunc = cuda.device_array_like(self.optFluidPDF)
        devicePhysicalVX = cuda.to_device(self.optMacroVelocity)
        devicePhysicalVY = cuda.to_device(self.optMacroVelocity)
        deviceEX = cuda.to_device(self.unitEX)
        deviceEY = cuda.to_device(self.unitEY)
        
        deviceTau = cuda.to_device(self.tau)
        deviceInteractionCoeff = cuda.to_device(self.interCoeff)
        deviceInterSolid = cuda.to_device(self.interactionSolid)
        #Distribute the thread
        blockNumX = int(self.xDimension / self.threadNum)
        blockNumY = math.ceil(self.fluidNodes.size / self.xDimension)
        threadPerBlock1D = (self.threadNum, 1)
#        grid1D = (blockNumX, 1)
        grid1D = (blockNumX, blockNumY)
        #define array for interaction coefficients
        weightInter = np.array([1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., \
                                1./36.])
#        weightInter = np.array([1./3., 1./3., 1./3., 1./3., 1./12., 1./12., 1./12., \
#                                1./12.])
        deviceWeightInter = cuda.to_device(weightInter)
        weightCoeff = np.array([4./ 9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., \
                        1./36.])
        deviceWeightCoeff = cuda.to_device(weightCoeff)
        #Start to run the optimized simulation
        totalNodes = self.fluidNodes.size
        if self.boundaryTypeInlet == "'Neumann'":
            deviceVelocityY = cuda.to_device(self.velocityYInlet)
        saturationOld = 1.0; saturationNew = 1.0; tmpStep = 0; recordStep = 0
        relativeChangeS = 1.0
        while (tmpStep < self.numTimeStep + 1 and relativeChangeS > 1.0e-5):
            starter = timer()
            print("The step is %g." % tmpStep)
            tmpStep += 1
            if (self.boundaryTypeInlet == "'Dirichlet'"):
                rhoHigher = self.specificRho1Upper; rhoLower = self.specificRho0Lower
                if (self.boundaryMethod == "'ZouHe'"):
                    constantPressureZouHeBoundaryHigher[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny ,self.xDimension, rhoHigher, \
                                    deviceFluidIndices, deviceFluidRho, deviceFluidPDF)
                    constantPressureZouHeBoundaryLower[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.xDimension, rhoLower, \
                                    deviceFluidIndices, deviceFluidRho, deviceFluidPDF)
                elif (self.boundaryMethod == "'Chang'"):
                    calPressureBoundaryHigherChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    rhoHigher, deviceFluidIndices, deviceFluidRho, \
                                    deviceForceX, deviceForceY, deviceFluidPDFold, \
                                    deviceFluidPDF)
                    calPressureBoundaryLowerChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    rhoLower, deviceFluidIndices, deviceFluidRho, \
                                    deviceForceX, deviceForceY, deviceFluidPDFold, \
                                    deviceFluidPDF)
                ghostPointsConstantPressureInlet[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.ny, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
            elif (self.boundaryTypeInlet == "'Neumann'"):
                print("Run Neumann boundary condition.")
                if (self.boundaryMethod == "'ZouHe'"):
                    constantVelocityZouHeBoundaryHigher[grid1D, threadPerBlock1D](\
                                    totalNodes, self.typesFluids, self.nx, self.ny, \
                                    self.xDimension, deviceVelocityY, deviceFluidIndices, \
                                    deviceFluidRho, deviceFluidPDF)
#                    input()
#                    self.optFluidRho = deviceFluidRho.copy_to_host()
                elif (self.boundaryMethod == "'Chang'"):
                    calVelocityBoundaryHigherChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    deviceVelocityY, deviceFluidIndices, deviceFluidRho, \
                                    deviceForceX, deviceForceY, deviceFluidPDFold, \
                                    deviceFluidPDF)
                print("Run the ghost nodes.")
                ghostPointsConstantVelocityInlet[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.ny, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes,\
                                deviceFluidRho, deviceFluidPDF)
            savePDFLastStep[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                           self.xDimension,deviceFluidPDF, deviceFluidPDFold)
#            if ((tmpStep - 1) % 50000 == 0 and (tmpStep - 1) != 0):
#                self.optFluidRho = deviceFluidRho.copy_to_host()
#                tmpBeginning = 20 * self.nx
#                tmpEnd = self.optFluidRho[-1].size - self.nx * 20
#                tmpStatus = (self.optFluidRho[-1, tmpBeginning:tmpEnd] >= 1.0)
#                tmpNum = np.count_nonzero(tmpStatus) 
#                saturationNew = tmpNum / (tmpStatus.size)
#                relativeChangeS = np.fabs(saturationNew - saturationOld)
#                saturationOld = saturationNew
#                if (relativeChangeS <= 1.0e-5):
#                    self.convertOptTo2D()
#                    self.resultInHDF5(tmpStep)
#                    self.plotDensityDistribution0OPT(tmpStep - 1)
#                    self.plotDensityDistribution1OPT(tmpStep - 1)
#                    if (self.typesFluids > 2):
#                        self.plotDensityDistribution2OPT(tmpStep - 1)
#                    print("Simulation stops, because the saturation is stable.")

#                input()
            if ((tmpStep-1) % 80 == 0):
                self.optFluidRho = deviceFluidRho.copy_to_host()
                self.optMacroVelocityX = devicePhysicalVX.copy_to_host()
                self.optMacroVelocityY = devicePhysicalVY.copy_to_host()
                self.optFluidPDF = deviceFluidPDF.copy_to_host()
                self.convertOptTo2D()

                for i in sp.arange(self.typesFluids):
                    self.plotDensityDistributionOPT(tmpStep - 1, i)
                self.plotPhysicalVelocity(tmpStep - 1)
                self.resultInHDF5(recordStep)
                recordStep += 1
            print("Calculate the Macro-scale mixture velocity.")
            calMacroWholeVelocity[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, self.xDimension, \
                                 deviceTau, deviceFluidRho, deviceFluidPDF, devicePrimeVX, \
                                 devicePrimeVY)

            print("Calculate the potential of each fluids.")
            calFluidRhoGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, self.xDimension, \
                          deviceFluidRho, deviceFluidPDF)
            calFluidPotentialGPUEql[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, self.xDimension, \
                                   deviceFluidRho, deviceFluidPotential)
            print("Calculate the force on the fluids and collision process.")
            interactionCollisionProcess[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                       self.xDimension, deviceWeightInter, deviceTau, \
                                       deviceInteractionCoeff, deviceInterSolid, \
                                       deviceWeightCoeff, deviceFluidRho, deviceFluidPotential, \
                                       deviceFluidPDF, deviceFluidPDFNew, deviceFluidIndices, \
                                       deviceNeighboringNodes, deviceForceX, deviceForceY)

            print("Calculate the streaming part 1.")
            calStreaming1GPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                            self.xDimension, deviceFluidIndices, deviceNeighboringNodes, \
                            deviceFluidPDF, deviceFluidPDFNew)
#            print("Calculate the streaming part 1 with bounce-back line boundary.")
#            calStreaming1withLinkGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
#                                    self.xDimension, deviceFluidIndices, deviceNeighboringNodes, \
#                                    deviceFluidRho, deviceFluidPDF, deviceFluidPDFNew, \
#                                    devicePhysicalVX, devicePhysicalVY, deviceWeightCoeff)
            print("Calculate the streaming part 2.")
            calStreaming2GPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, self.xDimension, \
                            deviceFluidPDFNew, deviceFluidPDF)

            if (self.boundaryTypeOutlet == "'Convective'"):
                print("Run the outlet boundary condition.")
                convectiveOutletGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                   self.nx, self.xDimension, \
                                   deviceFluidIndices, deviceNeighboringNodes, deviceFluidPDF, \
                                   deviceFluidRho)
                print("Run the ghost 2nd layer.")
                convectiveOutletGhost2GPU[grid1D, threadPerBlock1D](totalNodes, \
                                         self.typesFluids, self.nx, self.xDimension, \
                                         deviceFluidIndices, deviceNeighboringNodes, \
                                         deviceFluidPDF, \
                                         deviceFluidRho)

                print("Run the ghost 1st layer.")
                convectiveOutletGhost3GPU[grid1D, threadPerBlock1D](totalNodes, \
                                         self.typesFluids, self.nx, self.xDimension, \
                                         deviceFluidIndices, deviceNeighboringNodes,\
                                         deviceFluidPDF, \
                                         deviceFluidRho)
            print("Calculate the macro-density.")
            calFluidRhoGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, self.xDimension, \
                          deviceFluidRho, deviceFluidPDF)
            calPhysicalVelocity[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                               self.xDimension, deviceFluidPDF, deviceFluidRho,\
                               deviceForceX, deviceForceY, devicePhysicalVX, \
                               devicePhysicalVY)

    def runOptimizedEFLBM(self):
        print("Start to initialize the original 2D domain.")
        self.initializeDomainBorder()
        self.initializeDomainCondition()
#        self.calMacroParametersGPU()
        print("Finish the initialization of the original 2D domain..")
        print("Convert the data in the original 2D domain to the optimized way.")
        self. optimizeFluidArray()
        print("Finish the conversion of the data.")
        print("Start to set up arrays in GPU.")
        deviceFluidPDF = cuda.to_device(self.optFluidPDF)
        deviceFluidPDFold = cuda.to_device(self.optFluidPDF)
        deviceFluidPDFNew = cuda.to_device(self.optFluidPDF)
        deviceFluidRho = cuda.to_device(self.optFluidRho)
        deviceFluidIndices = cuda.to_device(self.fluidNodes)
        deviceNeighboringNodes = cuda.to_device(self.neighboringNodes)
        if self.explicitScheme == 8:
            deviceNeighboringNodesISO8 = cuda.to_device(self.neighboringNodesISO8)
        elif self.explicitScheme == 10:
            deviceNeighboringNodesISO10 = cuda.to_device(self.neighboringNodesISO10)
        deviceFluidPotential = cuda.device_array_like(self.optFluidRho)
        deviceEquilibriumVX = cuda.to_device(self.optMacroVelocityX)
        deviceEquilibriumVY = cuda.to_device(self.optMacroVelocityY)
        deviceForceX = cuda.device_array_like(self.optFluidRho)
        deviceForceY = cuda.device_array_like(self.optFluidRho)
        deviceEquiliriumFunc = cuda.to_device(self.optFluidPDF)
        deviceForcePDF = cuda.device_array_like(self.optFluidPDF)
        devicePhysicalVX = cuda.to_device(self.optMacroVelocity)
        devicePhysicalVY = cuda.to_device(self.optMacroVelocity)
        
        deviceTau = cuda.to_device(self.tau)
        deviceInteractionCoeff = cuda.to_device(self.interCoeff)
        deviceInterSolid = cuda.to_device(self.interactionSolid)
        #Distribute the thread\
        totalNodes = self.fluidNodes.size
        blockNumX = int(self.xDimension / self.threadNum)
        blockNumY = math.ceil(self.fluidNodes.size / self.xDimension)
        threadPerBlock1D = (self.threadNum, 1)
        grid1D = (blockNumX, blockNumY)
        
        eX = np.array([0., 1., 0., -1., 0., 1., -1., -1., 1.])
        eY = np.array([0., 0., 1., 0., -1., 1., 1., -1., -1.])
        deviceEX = cuda.to_device(eX)
        deviceEY = cuda.to_device(eY)
        weightInter4 = np.array([1./3., 1./3., 1./3., 1./3., 1./12., 1./12., 1./12., \
                                1./12.])
        weightInter8 = np.array([4./21., 4./21., 4./21., 4./21., 4./45., 4./45., \
                                 4./45., 4./45., 1./60., 1./60., 1./60., 1./60., \
                                 1./5040., 1./5040., 1./5040., 1./5040., \
                                 2./315., 2./315., 2./315., 2./315., 2./315., \
                                 2./315., 2./315., 2./315.])
        weightInter10 = np.array([262./1785., 262./1785., 262./1785., 262./1785., \
                                  93./1190., 93./1190., 93./1190., 93./1190., \
                                  7./340., 7./340., 7./340., 7./340., 9./9520., \
                                  9./9520., 9./9520., 9./9520., 6./595., 6./595., \
                                  6./595., 6./595., 6./595., 6./595., 6./595., 6./595., \
                                  2./5355., 2./5355., 2./5355., 2./5355., 1./7140., \
                                  1./7140., 1./7140., 1./7140., 1./7140., 1./7140., \
                                  1./7140., 1./7140.])
        if (self.explicitScheme == 4):
            deviceWeightInter = cuda.to_device(weightInter4)
        elif (self.explicitScheme == 8):
            deviceWeightInter = cuda.to_device(weightInter8)
        elif (self.explicitScheme == 10):
            deviceWeightInter = cuda.to_device(weightInter10)
        weightCoeff = np.array([4./ 9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., \
                        1./36.])
        #Using MRT scheme
        if (self.relaxationType == "'MRT'"):
            deviceCollisionM = cuda.to_device(self.collisionMatrix)
            self.conserveS = np.ones(self.typesFluids, dtype = np.float64)
            for i in np.arange(self.typesFluids):
                self.conserveS[i] = self.diagonalValues[i, 0]
            deviceConserveS = cuda.to_device(self.conserveS)
            deviceForcePDFM = cuda.device_array_like(self.optFluidPDF)
            deviceFluidPDFM = cuda.to_device(self.optFluidPDF)
        deviceWeightCoeff = cuda.to_device(weightCoeff)
        if (self.boundaryTypeInlet == "'Neumann'"):
            print('The velocity on the boundary is: ')
            print(self.velocityYInlet)
            deviceVelocityY = cuda.to_device(self.velocityYInlet)
        print('Start the computation.')
        #Initialize the transformation because the force term in PDF
        calFluidPotentialGPUEql[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                               self.xDimension, deviceFluidRho, deviceFluidPotential) 
        print('calculate the force term')
        if (self.explicitScheme == 4):
            calExplicit4thOrderScheme[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                     self.xDimension, deviceFluidIndices, deviceNeighboringNodes,\
                                     deviceWeightInter, deviceInteractionCoeff, \
                                     deviceInterSolid, deviceFluidPotential, \
                                     deviceForceX, deviceForceY)
        elif (self.explicitScheme == 8):
            calExplicit8thOrderScheme[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                     self.xDimension, deviceFluidIndices, deviceNeighboringNodesISO8,\
                                     deviceWeightInter, deviceInteractionCoeff, \
                                     deviceInterSolid, deviceFluidPotential, \
                                     deviceForceX, deviceForceY)
        elif (self.explicitScheme == 10):
            calExplicit10thOrderScheme[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                     self.xDimension, deviceFluidIndices, deviceNeighboringNodesISO10,\
                                     deviceWeightInter, deviceInteractionCoeff, \
                                     deviceInterSolid, deviceFluidPotential, \
                                     deviceForceX, deviceForceY)
        print('calculate the equilibrium function.')
        if self.relaxationType == "'SRT'":
            print('calculate the equilibrium velocity.')
            calEquilibriumVEFGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                    self.xDimension, deviceTau, deviceEX, deviceEY, \
                                    deviceFluidRho, deviceForceX, deviceForceY, \
                                    deviceFluidPDF, deviceEquilibriumVX, \
                                    deviceEquilibriumVY)
            
        elif self.relaxationType == "'MRT'":
            transformEquilibriumVelocity[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.xDimension, deviceEX, 
                                    deviceEY, deviceFluidRho, deviceForceX, \
                                    deviceForceY, deviceFluidPDF, deviceConserveS, \
                                    deviceEquilibriumVX, deviceEquilibriumVY)
        calEquilibriumFuncEFGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                               self.xDimension, deviceWeightCoeff, deviceEX, deviceEY, \
                               deviceFluidRho, deviceEquilibriumVX, deviceEquilibriumVY, \
                               deviceEquiliriumFunc)
        print('Calculate the force PDF.')
        calForceDistrGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                        self.xDimension, deviceEX, deviceEY, deviceEquilibriumVX, \
                        deviceEquilibriumVY, deviceFluidRho, deviceForceX, \
                        deviceForceY, deviceEquiliriumFunc, deviceForcePDF)
        print('Start to transform the PDF.')
        transformPDFGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                       self.xDimension, deviceFluidPDF, deviceForcePDF)
        if (self.boundaryTypeInlet == "'Dirichlet'"):
            print("It is Dirichlet boundary condition (constant pressure).")
            rhoHigher = self.specificRho1Upper; rhoLower = self.specificRho0Lower
            if (self.boundaryMethod == "'ZouHe'"):
                if (self.explicitScheme == 4):
                    constantPressureZouHeBoundaryHigher[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny ,self.xDimension, rhoHigher, \
                                    deviceFluidIndices, deviceFluidRho, deviceFluidPDF)
                    constantPressureZouHeBoundaryLower[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.xDimension, rhoLower, \
                                    deviceFluidIndices, deviceFluidRho, deviceFluidPDF)
            elif (self.boundaryMethod == "'Chang'"):
                if (self.explicitScheme == 4):
                    calPressureBoundaryHigherChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    rhoHigher, deviceFluidIndices, deviceFluidRho, \
                                    deviceForceX, deviceForceY, deviceFluidPDFold, \
                                    deviceFluidPDF)
                    calPressureBoundaryLowerChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    rhoLower, deviceFluidIndices, deviceFluidRho, \
                                    deviceForceX, deviceForceY, deviceFluidPDFold, \
                                    deviceFluidPDF)
            if (self.explicitScheme == 4):
                ghostPointsConstantPressure[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.ny, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
        elif (self.boundaryTypeInlet == "'Neumann'"):
            print("It is Von Neumann boundary condition (constant velocity).")
            if (self.boundaryMethod == "'ZouHe'"):
                if (self.explicitScheme == 4):
                    constantVelocityZouHeBoundaryHigher[grid1D, threadPerBlock1D](\
                                    totalNodes, self.typesFluids, self.nx, self.ny, \
                                    self.xDimension, deviceVelocityY, deviceFluidIndices, \
                                    deviceFluidRho, deviceFluidPDF)
                if (self.explicitScheme == 8):
                    constantVelocityZouHeBoundaryHigher8[grid1D, threadPerBlock1D](\
                                    totalNodes, self.typesFluids, self.nx, self.ny, \
                                    self.xDimension, deviceVelocityY, deviceFluidIndices, \
                                    deviceFluidRho, deviceFluidPDF)
            elif (self.boundaryMethod == "'Chang'"):
                if (self.explicitScheme == 4):
                    calVelocityBoundaryHigherChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    deviceVelocityY, deviceFluidIndices, deviceFluidRho, \
                                    deviceForceX, deviceForceY, deviceFluidPDFold, \
                                    deviceFluidPDF)
            print("Run the ghost nodes.")
            if (self.explicitScheme == 4):
                ghostPointsConstantVelocityInlet[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.ny, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
            if (self.explicitScheme == 8):
                ghostPointsConstantVelocity8[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.ny, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
                ghostPointsConstantVelocity82[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.ny, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
                
        if (self.boundaryTypeOutlet == "'Dirichlet'"):
            lowerDensity = 1.002
            if (self.explicitScheme == 4):
                constantPressureZouHeBoundaryLower[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.xDimension, \
                                        lowerDensity, deviceFluidIndices, deviceFluidRho, \
                                        deviceFluidPDF)
                ghostPointsConstantPressureOutlet[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
            elif (self.explicitScheme == 8):
                constantPressureZouHeBoundaryLower8[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.xDimension, \
                                        lowerDensity, deviceFluidIndices, deviceFluidRho, \
                                        deviceFluidPDF)
                ghostPointsConstantPressureOutlet8[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
                ghostPointsConstantPressureOutlet82[grid1D, threadPerBlock1D](totalNodes, \
                                self.typesFluids, self.nx, self.xDimension, \
                                deviceFluidIndices, deviceNeighboringNodes, \
                                deviceFluidRho, deviceFluidPDF)
        print('Star the loop.')
        tmpStep = 0
        for i in sp.arange(self.numTimeStep + 1):
            savePDFLastStep[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                           self.xDimension,deviceFluidPDF, deviceFluidPDFold)
            if self.relaxationType == "'MRT'":
                print("Check MRT algorithm's force part.")
                transfromForceTerm[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                  self.xDimension, deviceForcePDF, deviceCollisionM, \
                                  deviceForcePDFM)

                transformPDFandEquil[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                    self.xDimension, deviceFluidPDF, deviceEquiliriumFunc, \
                                    deviceCollisionM, deviceFluidPDFM)
                
            if (self.boundaryTypeOutlet == "'Freeflow'"):
                print("Run the outlet boundary condition.")
                convectiveOutletGPUEFS[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                   self.nx, self.xDimension, \
                                   deviceFluidIndices, deviceNeighboringNodes, deviceFluidPDF, \
                                   deviceFluidRho, deviceForcePDF, deviceEquiliriumFunc)
#                self.optFluidRho = deviceFluidRho.copy_to_host()
#                input()
#                print("Run the ghost 2nd layer.")
                convectiveOutletGhost2GPUEFS[grid1D, threadPerBlock1D](totalNodes, \
                                         self.typesFluids, self.nx, self.xDimension, \
                                         deviceFluidIndices, deviceNeighboringNodes, \
                                         deviceFluidPDF, \
                                         deviceFluidRho, deviceForcePDF, deviceEquiliriumFunc)
                print("Run the ghost 1st layer.")
                convectiveOutletGhost3GPUEFS[grid1D, threadPerBlock1D](totalNodes, \
                                         self.typesFluids, self.nx, self.xDimension, \
                                         deviceFluidIndices, deviceNeighboringNodes, \
                                         deviceFluidPDF, \
                                         deviceFluidRho, deviceForcePDF, deviceEquiliriumFunc)
                
            print('The step is %g.' % i)
            print('Calculate the collision of explicit scheme.')
            if (self.relaxationType == "'SRT'"):
                calCollisionEXGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                 self.xDimension, deviceTau, deviceFluidPDF, \
                                 deviceEquiliriumFunc, deviceForcePDF)
            elif (self.relaxationType == "'MRT'"):
                calAfterCollisionMRT[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                    self.xDimension, deviceFluidPDF, deviceForcePDF, \
                                    deviceEquiliriumFunc, deviceFluidPDFM, deviceForcePDFM)
            print("Calculate the streaming part 1.")
            calStreaming1GPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                            self.xDimension, deviceFluidIndices, deviceNeighboringNodes, \
                            deviceFluidPDF, deviceFluidPDFNew)
            print("Calculate the streaming part 2.")
            calStreaming2GPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, self.xDimension, \
                            deviceFluidPDFNew, deviceFluidPDF)

            print("Calculate the macro-density.")
            calFluidRhoGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                          self.xDimension, deviceFluidRho, deviceFluidPDF)
            calPhysicalVelocity[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                               self.xDimension, deviceFluidPDF, deviceFluidRho,\
                               deviceForceX, deviceForceY, devicePhysicalVX, \
                               devicePhysicalVY)
            
#            self.optFluidRho = deviceFluidRho.copy_to_host()
            if (self.boundaryTypeOutlet == "'Convective'"):
                print("The convective boundary condition is on the outlet.")
                #for isotropic value is 4
                convectiveOutletEachGPU[grid1D, threadPerBlock1D](totalNodes, \
                                       self.typesFluids, self.nx, self.xDimension, deviceFluidIndices, \
                                       deviceNeighboringNodes, deviceFluidPDF, \
                                       deviceFluidPDFold, deviceFluidRho, \
                                       devicePhysicalVY)
                convectiveOutletEach2GPU[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.xDimension, deviceFluidIndices, \
                                        deviceNeighboringNodes, deviceFluidPDF, \
                                        deviceFluidPDFold, deviceFluidRho, \
                                        devicePhysicalVY)
                convectiveOutletEach3GPU[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.xDimension, deviceFluidIndices, \
                                        deviceNeighboringNodes, deviceFluidPDF, \
                                        deviceFluidPDFold, deviceFluidRho, \
                                        devicePhysicalVY)
            if (self.boundaryTypeOutlet == "'Dirichlet'"):
                lowerDensity = 1.002
                if (self.explicitScheme == 4):
                    constantPressureZouHeBoundaryLower[grid1D, threadPerBlock1D](totalNodes, \
                                            self.typesFluids, self.nx, self.xDimension, \
                                            lowerDensity, deviceFluidIndices, deviceFluidRho, \
                                            deviceFluidPDF)
                    ghostPointsConstantPressureOutlet[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.xDimension, \
                                    deviceFluidIndices, deviceNeighboringNodes, \
                                    deviceFluidRho, deviceFluidPDF)
                elif (self.explicitScheme == 8):
                    constantPressureZouHeBoundaryLower8[grid1D, threadPerBlock1D](totalNodes, \
                                            self.typesFluids, self.nx, self.xDimension, \
                                            lowerDensity, deviceFluidIndices, deviceFluidRho, \
                                            deviceFluidPDF)
                    ghostPointsConstantPressureOutlet8[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.xDimension, \
                                    deviceFluidIndices, deviceNeighboringNodes, \
                                    deviceFluidRho, deviceFluidPDF)
                    ghostPointsConstantPressureOutlet82[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.xDimension, \
                                    deviceFluidIndices, deviceNeighboringNodes, \
                                    deviceFluidRho, deviceFluidPDF)

#                input()
            #Implement boundary condition
            if (self.boundaryTypeInlet == "'Dirichlet'"):
                print("It is Dirichlet boundary condition (constant pressure).")
                rhoHigher = self.specificRho1Upper; rhoLower = self.specificRho0Lower
                if (self.boundaryMethod == "'ZouHe'"):
                    if (self.explicitScheme == 4):
                        constantPressureZouHeBoundaryHigher[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.ny ,self.xDimension, rhoHigher, \
                                        deviceFluidIndices, deviceFluidRho, deviceFluidPDF)
                        constantPressureZouHeBoundaryLower[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.xDimension, rhoLower, \
                                        deviceFluidIndices, deviceFluidRho, deviceFluidPDF)
                elif (self.boundaryMethod == "'Chang'"):
                    if (self.explicitScheme == 4):
                        calPressureBoundaryHigherChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.ny, self.xDimension, \
                                        rhoHigher, deviceFluidIndices, deviceFluidRho, \
                                        deviceForceX, deviceForceY, deviceFluidPDFold, \
                                        deviceFluidPDF)
                        calPressureBoundaryLowerChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.ny, self.xDimension, \
                                        rhoLower, deviceFluidIndices, deviceFluidRho, \
                                        deviceForceX, deviceForceY, deviceFluidPDFold, \
                                        deviceFluidPDF)
                if (self.explicitScheme == 4):
                    ghostPointsConstantPressure[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    deviceFluidIndices, deviceNeighboringNodes, \
                                    deviceFluidRho, deviceFluidPDF)
            elif (self.boundaryTypeInlet == "'Neumann'"):
                print("It is Von Neumann boundary condition (constant velocity).")
                if (self.boundaryMethod == "'ZouHe'"):
                    if (self.explicitScheme == 4):
                        constantVelocityZouHeBoundaryHigher[grid1D, threadPerBlock1D](\
                                        totalNodes, self.typesFluids, self.nx, self.ny, \
                                        self.xDimension, deviceVelocityY, deviceFluidIndices, \
                                        deviceFluidRho, deviceFluidPDF)
                    if (self.explicitScheme == 8):
                        constantVelocityZouHeBoundaryHigher8[grid1D, threadPerBlock1D](\
                                        totalNodes, self.typesFluids, self.nx, self.ny, \
                                        self.xDimension, deviceVelocityY, deviceFluidIndices, \
                                        deviceFluidRho, deviceFluidPDF)
                elif (self.boundaryMethod == "'Chang'"):
                    if (self.explicitScheme == 4):
                        calVelocityBoundaryHigherChangGPU[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.nx, self.ny, self.xDimension, \
                                        deviceVelocityY, deviceFluidIndices, deviceFluidRho, \
                                        deviceForceX, deviceForceY, deviceFluidPDFold, \
                                        deviceFluidPDF)
                print("Run the ghost nodes.")
                if (self.explicitScheme == 4):
                    ghostPointsConstantVelocityInlet[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    deviceFluidIndices, deviceNeighboringNodes, \
                                    deviceFluidRho, deviceFluidPDF)
                if (self.explicitScheme == 8):
                    ghostPointsConstantVelocity8[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    deviceFluidIndices, deviceNeighboringNodes, \
                                    deviceFluidRho, deviceFluidPDF)
                    ghostPointsConstantVelocity82[grid1D, threadPerBlock1D](totalNodes, \
                                    self.typesFluids, self.nx, self.ny, self.xDimension, \
                                    deviceFluidIndices, deviceNeighboringNodes, \
                                    deviceFluidRho, deviceFluidPDF)
            print("Calculate the macro-density.")
            calFluidRhoGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                          self.xDimension, deviceFluidRho, deviceFluidPDF)
            calPhysicalVelocity[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                               self.xDimension, deviceFluidPDF, deviceFluidRho,\
                               deviceForceX, deviceForceY, devicePhysicalVX, \
                               devicePhysicalVY)

            if (i % 1000 == 0):
                self.optFluidRho = deviceFluidRho.copy_to_host()
                self.optMacroVelocityX = devicePhysicalVX.copy_to_host()
                self.optMacroVelocityY = devicePhysicalVY.copy_to_host()
                self.convertOptTo2D()
                for j in sp.arange(self.typesFluids):
                    self.plotDensityDistributionOPT(tmpStep, j)
                self.resultInHDF5(tmpStep)
                tmpStep += 1
            print('Calculate the potential.')
            calFluidPotentialGPUEql[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                   self.xDimension, deviceFluidRho, deviceFluidPotential)

            print('Calculate the force.')
            if (self.explicitScheme == 4):
                calExplicit4thOrderScheme[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                         self.xDimension, deviceFluidIndices, deviceNeighboringNodes,\
                                         deviceWeightInter, deviceInteractionCoeff, \
                                         deviceInterSolid, deviceFluidPotential, \
                                         deviceForceX, deviceForceY)
            elif (self.explicitScheme == 8):
                calExplicit8thOrderScheme[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                         self.xDimension, deviceFluidIndices, deviceNeighboringNodesISO8,\
                                         deviceWeightInter, deviceInteractionCoeff, \
                                         deviceInterSolid, deviceFluidPotential, \
                                         deviceForceX, deviceForceY)
            elif (self.explicitScheme == 10):
                calExplicit10thOrderScheme[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                         self.xDimension, deviceFluidIndices, deviceNeighboringNodesISO10,\
                                         deviceWeightInter, deviceInteractionCoeff, \
                                         deviceInterSolid, deviceFluidPotential, \
                                         deviceForceX, deviceForceY)
            if self.relaxationType == "'SRT'":
                print("Calculate the equilibrium velocity.")
                calEquilibriumVEFGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                                        self.xDimension, deviceTau, deviceEX, deviceEY, \
                                        deviceFluidRho, deviceForceX, deviceForceY, \
                                        deviceFluidPDF, deviceEquilibriumVX, \
                                        deviceEquilibriumVY)
    
            elif self.relaxationType == "'MRT'":
                transformEquilibriumVelocity[grid1D, threadPerBlock1D](totalNodes, \
                                        self.typesFluids, self.xDimension, deviceEX, 
                                        deviceEY, deviceFluidRho, deviceForceX, \
                                        deviceForceY, deviceFluidPDF, deviceConserveS, \
                                        deviceEquilibriumVX, deviceEquilibriumVY)
            print('Calculate the equilibrium function.')
            calEquilibriumFuncEFGPU[grid1D, threadPerBlock1D](totalNodes, \
                                       self.typesFluids, self.xDimension, \
                                       deviceWeightCoeff, deviceEX, deviceEY, \
                                       deviceFluidRho, deviceEquilibriumVX, \
                                       deviceEquilibriumVY, deviceEquiliriumFunc)
            

            print('Calculate the force term.')
            calForceDistrGPU[grid1D, threadPerBlock1D](totalNodes, self.typesFluids, \
                        self.xDimension, deviceEX, deviceEY, deviceEquilibriumVX, \
                        deviceEquilibriumVY, deviceFluidRho, deviceForceX, \
                        deviceForceY, deviceEquiliriumFunc, deviceForcePDF)
            
    def runTypeSCmodel(self):
        if (self.interactionType == "'ShanChen'" and self.Parallel == "'yes'"):
            self.runOptimizedLBM()
        elif (self.interactionType == "'EFS'" and self.Parallel == "'yes'"):
            self.runOptimizedEFLBM()

