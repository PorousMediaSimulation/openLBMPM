"""
Color-Gradient LBM main file. All the parameters are set up here and the simulation 
is run on GPU. All the functions for GPU bases on Anaconda package.

"""

import os, sys, getpass
import configparser
import math

import numpy as np
import scipy as sp
import numpy.linalg as nplinear
import scipy.ndimage as sciimage
import matplotlib.pyplot as plt

import tables as tb
from numba import autojit, jit, cuda

import AcceleratedRKGPU2D as RKGPU2D
import RKGPU2DBoundary as RKCG2DBC
from SimpleGeometryRK import defineGeometry

class RKColorGradientLBM():
    def __init__(self, pathIniFile):
        self.pathIni = pathIniFile
        config = configparser.ConfigParser()
        config.read(self.pathIni + '/RKtwophasesetup2D.ini')
        try:
            self.imageExist = config['ImageSetup']['Existance']
        except KeyError:
            print("Cannot find the parameters name for the existence of image.")
            sys.exit()
        except ValueError:
            print("The value should be string.")
            sys.exit()
        if (self.imageExist == "'no'"):
            print("Load the domain size of the simulation.")
            try:
                self.xDomain = int(config['DomainSize']['xDomain'])
                self.yDomain = int(config['DomainSize']['yDomain'])
            except KeyError:
                print("Cannot find the parameter's name for the domain size.")
                sys.exit()
        try:
            self.numBufferingLayers = int(config['DomainSize']['numBufferingLayers'])
        except KeyError:
            print("Cannot find the parameters for the number of buffering layer on the boundary.")
            sys.exit()
        except ValueError:
            print("The value of the number of the buffering layers on the boundary should be \
                  an integer.")
            sys.exit()
        try:
            self.ratioTopToBottom = float(config['DomainSize']['ratioTopToBottom'])
        except KeyError:
            print("Cannot find the parameter for the number of the buffering layer ration.")
            sys.exit()
        except ValueError:
            print("The number for the buffering layer ratio should be a float.")
            sys.exit()
        print("Read the parameters on the density ratio.")
        try:
            self.surfaceTensionType = config['SurfaceTension']['SurfaceTensionType']
        except KeyError:
            print("Cannot find the parameter for the type of the surface tension.")
            sys.exit()
        except ValueError:
            print("The definition of the surface tension type should be string.")
            sys.exit()
        if self.surfaceTensionType == "'CSF'":
            try:
                self.surfaceTension = float(config['SurfaceTension']['SurfaceTensionValue'])
            except KeyError:
                print("Cannot find the paramete for the value of surface tension.")
                sys.exit()
            except ValueError:
                print("The value for the surface tension should be a float.")
                sys.exit()
            try:
                self.contactAngle = float(config['SurfaceTension']['ContactAngle'])
            except KeyError:
                print("Cannot find the parameter for the value of contact angle.")
                sys.exit()
            except ValueError:
                print("The contact angle shoudl be a float number.")
                sys.exit()
            self.cosTheta = np.cos(self.contactAngle / 180. * np.pi)
            self.sinTheta = np.sin(self.contactAngle / 180. * np.pi)
            try:
                self.wettingType = int(config['SurfaceTension']['WettingType'])
            except KeyError:
                print("Cannot find the parameter for the value of wetting boundary type.")
                sys.exit()
            except ValueError:
                print("The contact angle shoudl be an integer.")
                sys.exit()
        try:
            self.betaThickness = float(config['RKParameters']['BetaThickness'])
        except KeyError:
            print( "Cannot find the parameter's name for the thickness of the interface.")
            sys.exit()
        except ValueError:
            print("The value should be float.")
            sys.exit()
        if self.surfaceTensionType == "'Perturbation'":
            try:
                self.alphaR = float(config['RKParameters']['AlphaR'])
                self.alphaB = float(config['RKParameters']['AlphaB'])
            except KeyError:
                print("Cannot find the parameters name for the density ratio.")
                sys.exit()
            except ValueError:
                print("The vlaue should be float.")
            print("Read the parameter on the thickness of the interface.")
    
            print("Read the parameters on surface strength.")
            try:
                self.AkR = float(config['RKParameters']['AkR'])
                self.AkB = float(config['RKParameters']['AkB'])
            except KeyError:
                print("Cannot find the parameter's name for the surface strength.")
                sys.exit()
            except ValueError:
                print("The value for the surface strength shoudl be floats.")
                sys.exit()
            print("Read the value for determining tau.")

            self.constantCR = np.zeros(9, dtype = np.float64)
            self.constantCB = np.zeros(9, dtype = np.float64)
            self.constantB = np.zeros(9, dtype = np.float64)
            self.constantCR[0] = self.alphaR
            self.constantCR[1:5] = (1. - self.alphaR) / 5.
            self.constantCR[5:] = (1. - self.alphaR) / 20.
            self.constantCB[0] = self.alphaB
            self.constantCB[1:5] = (1. - self.alphaB) / 5.
            self.constantCB[5:] = (1. - self.alphaB) / 20.
            self.constantB[0] = -4./27.
            self.constantB[1: 5] = 2./27.; self.constantB[5:] = 5./108.
            #new values for constant B from Liu et.al 2014
            self.constantBNew = np.ones(9, dtype = np.float64)
            self.constantBNew[0] = -2./9.
            self.constantBNew[1:5] = 1./9.; self.constantBNew[5:] = 1./36.
        try:
            self.deltaValue = float(config['RKParameters']['DeltaValue'])
        except KeyError:
            print("Cannot find the parameter's name for determine the tau values.")
            sys.exit()
        except ValueError:
            print("The value for determine tau should be a float.")
        print("Read the values for taus.")
        try:
            self.tauR = float(config['FluidParameters']['TauR'])
            self.tauB = float(config['FluidParameters']['TauB'])
        except KeyError:
            print("Cannot find the parameter's name for  tau values")
            sys.exit()
        except ValueError:
            print("The values for tau should be floats.")
            sys.exit()
        try:
            self.initialRhoR = float(config['FluidParameters']['InitialRhoR'])
            self.initialRhoB = float(config['FluidParameters']['InitialRhoB'])
        except KeyError:
            print("Cannot find the parameter's name for densities values")
            sys.exit()
        except ValueError:
            print("The values for tau should be floats.")
            sys.exit()
        try:
            self.tauCalculation = int(config['FluidParameters']['TauType'])
        except KeyError:
            print("Cannot find the parameter's name for calculating tau value.")
            sys.exit()
        except ValueError:
            print("The value for the type of calculating tau should be an integer.")
            sys.exit()
        try:
            self.solidPhi = float(config['SolidBoundarySetup']['SolidColorDiff'])
        except KeyError:
            print("Cannot find the parameter's name for densities values on solid.")
            sys.exit()
        except ValueError:
            print("The values for tau should be floats.")
            sys.exit()
        try:
            self.gradientType = config['GradientType']['Type']
        except KeyError:
            print("Cannot find the parameter's name for gradient type.")
            sys.exit()
        
        #Option for body force
        self.bodyFX = float(config['BodyForce']['bodyForceX'])
        self.bodyFY = float(config['BodyForce']['bodyForceY'])
        #Read the simulation steps and time interval for exporting the results
        try:
            self.timeSteps = int(config['TimeSetup']['TimeSteps'])
        except KeyError:
            print("Cannot find the parameter's name for time steps.")
            sys.exit()
        try:
            self.timeInterval = int(config['TimeSetup']['TimeInterval'])
        except KeyError:
            print("Cannot find the parameter's name for time interval.")
            sys.exit()
        except ValueError:
            print("The value of the time interval should be an integer.")
            sys.exit()
            
        self.Parallel = config['Parallelism']['Parallel']
        if (self.Parallel == "'yes'"):
            self.xDimension = int(config['Parallelism']['xDimension'])
            self.threadNum = int(config['Parallelism']['ThreadsNum'])
            print("GPU will be used and the parameters for partiion domain are: ")
            print("The x-dimension of grid is %g." % self.xDimension)
            print("The number of thread in each grid is %g." % self.threadNum)
        try:
            self.relaxationType = config['RelaxationType']['Type']
        except KeyError:
            print("Cannot find the parameter's name for RK relaxation type.")
            sys.exit()
        except ValueError:
            print('The value should be a string.')
            sys.exit()
        print("Define other constants.")
        try:
            self.boundaryTypeInlet = config['BoundaryCondition']['BoundaryTypeInlet']
        except KeyError:
            print("Cannot find the parameter's name for the inlet boundary type.")
            sys.exit()
        except ValueError:
            print("The value for the inlet boundary type should be string.")
            sys.exit()
        
        try:
            self.boundaryTypeOutlet = config['BoundaryCondition']['BoundaryTypeOutlet']
        except KeyError:
            print("Cannot find the parameter's name for the outlet boundary type")
            sys.exit()
        except ValueError:
            print("The value for the outlet boundary type should be string.")
            sys.exit()
        #Inlet boundary condition
        if self.boundaryTypeInlet == "'Neumann'":
            try:
                self.neumannType = config['BoundaryCondition']['NeumannType']
            except KeyError:
                print("Cannot find the parameter's name for neumann type.")
                sys.exit()
            try:
                self.velocityYR = float(config['BoundaryCondition']['VelocityYR'])
                self.velocityYB = float(config['BoundaryCondition']['VelocityYB'])
            except ValueError:
                print("The value should be floats for the velocities.")
        elif self.boundaryTypeInlet == "'Dirichlet'":
            try:
                self.densityRhoBH = float(config['BoundaryCondition']['densityBH'])
                self.densityRhoRH = float(config['BoundaryCondition']['densityRH'])
            except ValueError:
                print("The value should be floats for pressure/density for inlet.")
                sys.exit()
        #Outlet boundary condition
        if self.boundaryTypeOutlet == "'Dirichlet'":
            try:
                self.densityRhoBL = float(config['BoundaryCondition']['densityBL'])
                self.densityRhoRL = float(config['BoundaryCondition']['densityRL'])
            except ValueError:
                print("The value should be floats for pressure/density for outlet.")
                sys.exit()
        try:
            self.isCycles = config['CyclesSetup']['IsCycle']
        except KeyError:
            print("Cannot find the parameter's name for D-I cycles.")
            sys.exit()
        except ValueError:
            print("The value for D-I cycles shoudl be string.")
        if (self.isCycles == "'yes'"):
            try:
                self.lastStep = int(config['CyclesSetup']['LastStep'])
            except KeyError:
                print("Cannot find the parameter's name for read the last time step file.")
                sys.exit()
            except ValueError:
                print("The value for the last step must be a number.")
                sys.exit()

        self.weightsCoeff = np.zeros(9, dtype = np.float64)
        self.unitEX = np.array([0., 1., 0., -1., 0., 1., -1., -1., 1.])
        self.unitEY = np.array([0., 0., 1., 0., -1., 1., 1., -1., -1.])
        self.weightsCoeff[0] = 4./9.; self.weightsCoeff[1:5] = 1./9.
        self.weightsCoeff[5:] = 1./36.
        self.solidPhi = 0.7
        self.gradientScheme = np.ones(9, dtype = np.float64)
        if (self.gradientType == "'Anisotropic'"):
            self.gradientScheme[1:5] = 1./3.; self.gradientScheme[5:] = 1./12.
        if self.relaxationType == "'MRT'":
            self.transformationM = np.zeros([9, 9], dtype = np.float64)
            self.transformationM[0, :] = 1.

            self.transformationM[1, :] = -1.; self.transformationM[1, 0] = -4.
            self.transformationM[1, 5:] = 2.

            self.transformationM[2, :] = 1.; self.transformationM[2, 0] = 4.
            self.transformationM[2, 1:5] = -2.

            self.transformationM[3, :] = 0.; self.transformationM[3, 1] = 1.
            self.transformationM[3, 3] = -1.; self.transformationM[3, 5] = 1.
            self.transformationM[3, 6:-1] = -1.; self.transformationM[3, -1] = 1.
            
            self.transformationM[4, 1] = -2.; self.transformationM[4, 2] = 0.
            self.transformationM[4, 3] = 2.; self.transformationM[4, 5] = 1.
            self.transformationM[4, 6:-1] = -1.; self.transformationM[4, -1] = 1.
            
            self.transformationM[5, 2] = 1.; self.transformationM[5, 4] = -1.
            self.transformationM[5, 5:7] = 1.; self.transformationM[5, 7:] = -1.;
            
            self.transformationM[6, 2] = -2.; self.transformationM[6, 4] = 2.
            self.transformationM[6, 5:7] = 1.; self.transformationM[6, 7:] = -1.

            self.transformationM[7, 1] = 1.; self.transformationM[7, 2] = -1.
            self.transformationM[7, 3] = 1.; self.transformationM[7, 4] = -1.;
            
            self.transformationM[8, 5] = 1.; self.transformationM[8, 6] = -1.
            self.transformationM[8, 7] = 1.; self.transformationM[8, 8] = -1.
            self.invTransformationM = nplinear.inv(self.transformationM)
            self.collisionS = np.zeros(9, dtype = np.float64)
            self.collisionS[1] = 1.64; self.collisionS[2] = 1.54
            self.collisionS[4] = 1.9; self.collisionS[6] = 1.9

        self.__createHDF5File()
        self.__checkGPUAvailability()
    
    """
    Create .h5 file for saving the results on fluids
    """
    def __createHDF5File(self, ):
        print("Create .h5 (HDF file) to save the results of fluids.")
        username = getpass.getuser()
        pathfile = '/home/'+ username + '/LBMResults/'
        file = tb.open_file(pathfile + 'SimulationResultsRK.h5', 'w')
        file.create_group(file.root, 'FluidMacro', 'MacroData')
        file.create_group(file.root, 'FluidPDF', 'MicroData')
        file.create_group(file.root, 'FluidVelocity', 'MacroVelocity')
        file.close()
        print("The file of .h5 has been created.") 
        
    """
    Check GPU device availabibity for simulation
    """
    def __checkGPUAvailability(self, ):
        print("Check the GPU device availability for simulation.")
        if cuda.is_available():
            print("GPU device is available.")
            print("The device is: ")
            print(cuda.gpus)
            print(cuda.detect())
        else:
            print("There is no GPU device for CUDA available in the system, so the simulation will stop.")
            sys.exit("No GPU device.")
            
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
#        if (self.duplicateDomain == "'no'"):
        self.effectiveDomain = binaryImage[yMin:(yMax + 1), xMin:(xMax + 1)]
#        elif (self.duplicateDomain == "'yes'"):
#            tmpDomain = binaryImage[yMin:(yMax + 1), xMin:(xMax + 1)]
#            xDirectionNum = int(input("Number of duplication in x direction: "))
#            yDirectionNum = int(input("Number of duplication in y direction: "))
#            self.effectiveDomain = self.__expandImageDomain(tmpDomain, xDirectionNum, \
#                                                          yDirectionNum)
        yDimension, xDimension = self.effectiveDomain.shape
        self.effectiveDomain[:, 0] = 0.; self.effectiveDomain[:, -1] = 0.
        tmpBufferLayer = np.zeros(xDimension, dtype = np.float64)
        tmpBufferLayer[:] = 255.
        for i in sp.arange(self.numBufferingLayers):
            if (i < int(self.numBufferingLayers * self.ratioTopToBottom) ):
                self.effectiveDomain = np.vstack((tmpBufferLayer, self.effectiveDomain))
            else:
                self.effectiveDomain = np.vstack((self.effectiveDomain, tmpBufferLayer))
        
    @jit
    def initializeDomainBorder(self):
        """
        define the wall position in 2D domain
        """
        #Read image of the structure
        if (self.imageExist == "'yes'"):
            self.__processImage()
            #re-define the domain size with the layers of boundaries and ghost points
            self.yDomain, self.xDomain = self.effectiveDomain.shape
            print('Now the size of domain is %g and %g' %(self.yDomain, self.xDomain))
        else:
            self.isDomain = sp.empty([self.yDomain, self.xDomain], dtype = np.bool)
            self.isSolid = sp.empty([self.yDomain, self.xDomain], dtype = np.bool)
            self.isDomain, self.isSolid = defineGeometry(self.xDomain, self.yDomain)
        if (self.imageExist == "'yes'"):
            self.originalXdim = self.xDomain
            self.isDomain = sp.empty([self.yDomain, self.xDomain], dtype = np.bool)
            self.isSolid = sp.empty([self.yDomain, self.xDomain], dtype = np.bool)
            self.isDomain[:, :] = 1; self.isSolid[:, :] = 0
            for i in sp.arange(self.yDomain):
                for j in sp.arange(self.xDomain):
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

        self.fluidPDFR = np.zeros([self.yDomain, self.xDomain, 9], dtype = np.float64)
        self.fluidPDFB = np.zeros([self.yDomain, self.xDomain, 9], dtype = np.float64)
        self.fluidsRhoR = np.zeros([self.yDomain, self.xDomain], dtype = np.float64)
        self.fluidsRhoB = np.zeros([self.yDomain, self.xDomain], dtype = np.float64)
        self.physicalVX = np.zeros([self.yDomain, self.xDomain], dtype = np.float64)
        self.physicalVY = np.zeros([self.yDomain, self.xDomain], dtype = np.float64)

        if (self.imageExist == "'no'"):
            if self.isCycles == "'no'":
                for i in sp.arange(self.yDomain):
                    for j in sp.arange(self.xDomain):
    #                    for k in sp.arange(self.typesFluids):
                        tmpCenterX = int(self.xDomain / 2); tmpCenterY = int(self.yDomain / 2)
                        if (self.isDomain[i, j] == True):
#                            if (sp.sqrt((i - tmpCenterY) * (i - tmpCenterY) + (j - \
#                                    tmpCenterX) * (j - tmpCenterX)) <= 20.):
#                            if (np.abs(j - tmpCenterX) <= 75):
                            if i >= self.yDomain - self.numBufferingLayers:
                                self.fluidsRhoR[i, j] = self.initialRhoR
                                self.fluidsRhoB[i, j] = 5.0e-8
    #                            self.fluidsRhoB[i, j] = self.initialRhoB
                                self.fluidPDFR[i, j], self.fluidPDFB[i, j] = \
                                    self.__initializeFluidPDF(self.fluidsRhoR[i, j], \
                                    self.fluidsRhoB[i, j], self.physicalVX[i, j], \
                                    self.physicalVY[i, j])
                            else:
    #                            self.fluidsRhoR[i, j] = self.initialRhoR
                                self.fluidsRhoB[i, j] = self.initialRhoB
                                self.fluidsRhoR[i, j] = 5.0e-8
                                self.fluidPDFR[i, j], self.fluidPDFB[i, j] = \
                                    self.__initializeFluidPDF(self.fluidsRhoR[i, j], \
                                    self.fluidsRhoB[i, j], self.physicalVX[i, j], \
                                    self.physicalVY[i, j])
            elif self.isCycles == "'yes'":
                pathFile = os.path.expanduser('~/LBMInitial/')
                if os.path.exists(pathFile):
                    dataFile = tb.open_file(pathFile + 'SimulationResultsRK.h5', 'r')
                    self.fluidsRhoR[:, :] = eval('dataFile.root.FluidMacro.FluidDensityRin%d[:, :]' % self.lastStep)
                    self.fluidsRhoB[:, :] = eval('dataFile.root.FluidMacro.FluidDensityBin%d[:, :]' % self.lastStep)
                    self.physicalVX[:, :] = eval('dataFile.root.FluidVelocity.FluidVelocityXAt%d[:, :]' % self.lastStep)
                    self.physicalVY[:, :] = eval('dataFile.root.FluidVelocity.FluidVelocityYAt%d[:, :]' % self.lastStep)
                    self.fluidsRhoR[-20:, :] = 0.; self.fluidsRhoB[-20:, :] = self.initialRhoB
#                    self.physicalVX[-10:, :] = 0.; self.physicalVY[-10:, :] = 0.
                    for i in sp.arange(self.yDomain):
                        for j in sp.arange(self.xDomain):
                            if self.isDomain[i, j] == True:
                                self.fluidPDFR[i, j], self.fluidPDFB[i, j] = \
                                    self.__initializeFluidPDF(self.fluidsRhoR[i, j], \
                                    self.fluidsRhoB[i, j], self.physicalVX[i, j], \
                                    self.physicalVY[i, j])
        
                
                                    
        elif (self.imageExist == "'yes'"):
            if self.isCycles == "'no'":
                for i in sp.arange(self.yDomain):
                    for j in sp.arange(self.xDomain):
                        if (self.isDomain[i, j] == True):
                            if i >= self.yDomain - 20:
                                self.fluidsRhoR[i, j] = self.initialRhoR
#                                self.fluidsRhoB[i, j] = 1.0e-8
    #                            self.fluidsRhoB[i, j] = self.initialRhoB
                                self.fluidPDFR[i, j], self.fluidPDFB[i, j] = \
                                    self.__initializeFluidPDF(self.fluidsRhoR[i, j], \
                                    self.fluidsRhoB[i, j], self.physicalVX[i, j], \
                                    self.physicalVY[i, j])
                            else:
    #                            self.fluidsRhoR[i, j] = self.initialRhoR
                                self.fluidsRhoB[i, j] = self.initialRhoB
#                                self.fluidsRhoR[i, j] = 1.0e-8
                                self.fluidPDFR[i, j], self.fluidPDFB[i, j] = \
                                    self.__initializeFluidPDF(self.fluidsRhoR[i, j], \
                                    self.fluidsRhoB[i, j], self.physicalVX[i, j], \
                                    self.physicalVY[i, j])
            elif self.isCycles == "'yes'":
                print("Initialize the domain for the D-I cycles simulation.")
                pathFile = os.path.expanduser('~/LBMInitial/')
                if os.path.exists(pathFile):
                    dataFile = tb.open_file(pathFile + 'SimulationResultsRK.h5', 'r')
                    self.fluidsRhoR[:, :] = eval('dataFile.root.FluidMacro.FluidDensityRin%d[:, :]' % self.lastStep)
                    self.fluidsRhoB[:, :] = eval('dataFile.root.FluidMacro.FluidDensityBin%d[:, :]' % self.lastStep)
                    self.physicalVX[:, :] = eval('dataFile.root.FluidVelocity.FluidVelocityXAt%d[:, :]' % self.lastStep)
                    self.physicalVY[:, :] = eval('dataFile.root.FluidVelocity.FluidVelocityYAt%d[:, :]' % self.lastStep)
                    self.fluidsRhoR[-20:, :] = 0.; self.fluidsRhoB[-20:, :] = self.initialRhoB
#                    self.physicalVX[-30:, :] = 0.; self.physicalVY[-30:, :] = 0.
                    for i in sp.arange(self.yDomain):
                        for j in sp.arange(self.xDomain):
                            if self.isDomain[i, j] == True:
                                self.fluidPDFR[i, j], self.fluidPDFB[i, j] = \
                                    self.__initializeFluidPDF(self.fluidsRhoR[i, j], \
                                    self.fluidsRhoB[i, j], self.physicalVX[i, j], \
                                    self.physicalVY[i, j])
        if self.boundaryTypeOutlet == "'Dirichlet'":
            self.fluidsRhoB[1, :] = self.densityRhoBL
            self.fluidsRhoR[1, :] = self.densityRhoRL
            for i in sp.arange(self.xDomain):
                self.fluidPDFR[1, i], self.fluidPDFB[1, i] = self.__initializeFluidPDF(self.densityRhoRL, \
                              self.densityRhoBL, self.physicalVX[1, i], self.physicalVY[1, i])
    @autojit
    def __initializeFluidPDF(self, RhoR, RhoB, velocityX, velocityY):
        tmpPDFR = np.empty(9, dtype = np.float64)
        tmpPDFB = np.empty(9, dtype = np.float64)
        for i in np.arange(9):
#            tmpPDFR[i] = RhoR * (self.constantCR[i] + self.weightsCoeff[i] * (\
#                   3. * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) + \
#                   4.5 * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) * \
#                   (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) - 1.5 * \
#                   (velocityX * velocityX + velocityY * velocityY)))
#            tmpPDFB[i] = RhoB * (self.constantCB[i] + self.weightsCoeff[i] * (\
#                   3. * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) + \
#                   4.5 * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) * \
#                   (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) - 1.5 * \
#                   (velocityX * velocityX + velocityY * velocityY)))
            tmpPDFR[i] = RhoR * self.weightsCoeff[i] * (1 + (\
                   3. * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) + \
                   4.5 * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) * \
                   (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) - 1.5 * \
                   (velocityX * velocityX + velocityY * velocityY)))
            tmpPDFB[i] = RhoB * self.weightsCoeff[i] * (1 + (\
                   3. * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) + \
                   4.5 * (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) * \
                   (self.unitEX[i] * velocityX + self.unitEY[i] * velocityY) - 1.5 * \
                   (velocityX * velocityX + velocityY * velocityY)))
        return tmpPDFR, tmpPDFB
    
    def optimizeFluidArray(self):
        """
        Convert 2D array of porous media matrix to 1D array for fluid nodes
        """
        print("Run the function for optimization.")
        self.fluidNodes = np.empty(self.voidSpace, dtype = np.int64)
        print("Start to fill effective fluid nodes.")
        tmpIndicesDomain = -np.ones(self.isDomain.shape, dtype = np.int64)
        tmpIndicesFN = 0
        for i in sp.arange(self.yDomain):
            for j in sp.arange(self.xDomain):
                if (self.isDomain[i, j] == 1):
#                if (self.effectiveDomain[i, j] == 255.):
                    tmpIndices = i * self.xDomain + j
                    self.fluidNodes[tmpIndicesFN] = tmpIndices
                    tmpIndicesDomain[i, j] = tmpIndicesFN
                    tmpIndicesFN += 1
        self.neighboringNodes = np.zeros(self.fluidNodes.size * 8, dtype = np.int64)
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

        RKGPU2D.fillNeighboringNodes[grid, threadPerBlock1D](totalNodes, self.xDomain, self.yDomain, \
                            self.xDimension, deviceFluidNodes, devicetmpIndicesDomain, \
                            deviceNeighboringNodes)
        self.neighboringNodes = deviceNeighboringNodes.copy_to_host()       
        print("Redefine the fluid nodes.")
#        cuda.current_context().trashing.clear()
        self.optFluidPDFR = np.empty([self.fluidNodes.size, 9], dtype = np.float64)
        self.optFluidPDFB = np.empty([self.fluidNodes.size, 9], dtype = np.float64)
        self.optFluidRhoR = np.empty([self.fluidNodes.size], dtype = np.float64)
        self.optFluidRhoB = np.empty([self.fluidNodes.size], dtype = np.float64)

        self.optMacroVelocityX = np.zeros(self.fluidNodes.size, dtype = np.float64)
        self.optMacroVelocityY = np.zeros(self.fluidNodes.size, dtype = np.float64)

        tmpDomain = np.array([i == 1 for i in self.isDomain.reshape(self.yDomain * \
                            self.xDomain)])
        self.optFluidRhoR = self.fluidsRhoR.reshape(self.yDomain * self.xDomain)[tmpDomain]
        self.optFluidRhoB = self.fluidsRhoB.reshape(self.yDomain * self.xDomain)[tmpDomain]
        
        self.optFluidPDFR = self.fluidPDFR.reshape(self.yDomain * self.xDomain, \
                                                   9)[tmpDomain]
        self.optFluidPDFB = self.fluidPDFB.reshape(self.yDomain * self.xDomain, \
                                                   9)[tmpDomain]
        
    def optimizeFluidandSolidArray(self):
        """
        Convert 2D array of porous media matrix to 1D array for fluid nodes for 
        color gradient method in Xu et.al 2017
        """
        print("Run the function for optimization.")
        self.fluidNodes = np.empty(self.voidSpace, dtype = np.int64)
        self.wettingSolidNodes = np.array([], dtype = np.int64)
        newIndicesWetting = -2
        print("Start to fill effective fluid nodes.")
        self.newIndicesDomain = -np.ones(self.isDomain.shape, dtype = np.int64)
        tmpIndicesFN = 0
        for i in sp.arange(self.yDomain):
            for j in sp.arange(self.xDomain):
                if (self.isDomain[i, j] == 1):
#                if (self.effectiveDomain[i, j] == 255.):
                    tmpIndices = i * self.xDomain + j
                    self.fluidNodes[tmpIndicesFN] = tmpIndices
                    self.newIndicesDomain[i, j] = tmpIndicesFN
                    tmpIndicesFN += 1
                else:
                    tmpCount = 0
                    for m in np.arange(-1, 2):
                        for n in np.arange(-1, 2):
                            tmpY = i + m if i + m < self.yDomain else 0
                            tmpX = j + n if j + n < self.xDomain else 0
                            if (self.isDomain[tmpY, tmpX] == 1):
                                tmpCount += 1
                    if tmpCount > 0:
                        tmpSolidLoc = i * self.xDomain + j
                        self.wettingSolidNodes = np.append(self.wettingSolidNodes, tmpSolidLoc)
                        self.newIndicesDomain[i, j] = newIndicesWetting
                        newIndicesWetting -= 1
        print('Indices for solid near to the fluid points.')
        print(self.wettingSolidNodes)
        self.neighboringNodes = np.zeros(self.fluidNodes.size * 8, dtype = np.int64)
        self.neighboringWettingSolidNodes = np.zeros(self.wettingSolidNodes.size * 8, dtype = np.int64)
        totalNodes = self.fluidNodes.size
        totalWettingNodes = self.wettingSolidNodes.size
        #use cuda to generate the array for neighboring nodes
        print("Start to fill neighboring nodes")
        deviceFluidNodes = cuda.to_device(self.fluidNodes)
        deviceWettingSolidNodes = cuda.to_device(self.wettingSolidNodes)
        devicetmpIndicesDomain = cuda.to_device(self.newIndicesDomain)
#        deviceIsDomain = cuda.to_device(self.isDomain)
        deviceNeighboringNodes = cuda.to_device(self.neighboringNodes)
        deviceNeighboringWettingSolidNodes = cuda.to_device(self.neighboringWettingSolidNodes)
        blockNumX = int(self.xDimension / self.threadNum) 
        blockNumY = math.ceil(self.fluidNodes.size / self.xDimension)
        threadPerBlock1D = (self.threadNum, 1)
        grid = (blockNumX, blockNumY)
        #neighboring nodes for all the fluid nodes
        RKGPU2D.fillNeighboringNodes[grid, threadPerBlock1D](totalNodes, self.xDomain, self.yDomain, \
                            self.xDimension, deviceFluidNodes, devicetmpIndicesDomain, \
                            deviceNeighboringNodes)
        self.neighboringNodes = deviceNeighboringNodes.copy_to_host()
        #neighboring nodes for all the wetting nodes in solid
        RKGPU2D.fillNeighboringWettingNodes[grid, threadPerBlock1D](totalWettingNodes, \
                            self.xDomain, self.yDomain, self.xDimension, deviceWettingSolidNodes, \
                            devicetmpIndicesDomain, deviceNeighboringWettingSolidNodes)
        self.neighboringWettingSolidNodes = deviceNeighboringWettingSolidNodes.copy_to_host()
        print("Redefine the fluid nodes.")
#        cuda.current_context().trashing.clear()
        self.optFluidPDFR = np.empty([self.fluidNodes.size, 9], dtype = np.float64)
        self.optFluidPDFB = np.empty([self.fluidNodes.size, 9], dtype = np.float64)
        self.optFluidRhoR = np.empty([self.fluidNodes.size], dtype = np.float64)
        self.optFluidRhoB = np.empty([self.fluidNodes.size], dtype = np.float64)

        self.optMacroVelocityX = np.zeros(self.fluidNodes.size, dtype = np.float64)
        self.optMacroVelocityY = np.zeros(self.fluidNodes.size, dtype = np.float64)

        tmpDomain = np.array([i == 1 for i in self.isDomain.reshape(self.yDomain * \
                            self.xDomain)])
        self.optFluidRhoR = self.fluidsRhoR.reshape(self.yDomain * self.xDomain)[tmpDomain]
        self.optFluidRhoB = self.fluidsRhoB.reshape(self.yDomain * self.xDomain)[tmpDomain]
        
        self.optFluidPDFR = self.fluidPDFR.reshape(self.yDomain * self.xDomain, \
                                                   9)[tmpDomain]
        self.optFluidPDFB = self.fluidPDFB.reshape(self.yDomain * self.xDomain, \
                                                   9)[tmpDomain]

    """
    Generate the list of fluid nodes having solid node neighbors
    """        
    def sortOutFluidNodesToSolid(self,):
        print('List the fluid nodes near to the solid phase.')
        self.fluidNodesWithSolidGPU = np.array([], dtype = np.int64)
        self.fluidNodesWithSolidOriginal = np.array([], dtype = np.int64)
        for i in np.arange(self.yDomain):
            for j in np.arange(self.xDomain):
                if self.isDomain[i, j] == 1:
                    tmpCount = 0
                    for m in np.arange(-1, 2):
                        for n in np.arange(-1, 2):
                            tmpY = i + m if i + m < self.yDomain else 0
                            tmpX = j + n if j + n < self.xDomain else 0
                            if (self.isDomain[tmpY, tmpX] == 0):
                                tmpCount += 1
                    if tmpCount > 0:
                        tmpIndices = i * self.xDomain + j
                        self.fluidNodesWithSolidGPU = np.append(self.fluidNodesWithSolidGPU, \
                                                self.newIndicesDomain[i, j])
                        self.fluidNodesWithSolidOriginal = np.append(self.fluidNodesWithSolidOriginal, \
                                                tmpIndices)
#        print('The wetting nodes in the new indices array are:')
#        print(self.fluidNodesWithSolidGPU)
#        print(self.fluidNodesWithSolidOriginal)
        
    """
    Calculate the vector values normal to the solid surface. Isotropy here is 8.
    """
    def calVectorNormaltoSolid(self,):
        print('Calculate the normal vector for the solid surface.')
        print(self.isDomain[-1])
        self.nsX = np.empty(self.fluidNodesWithSolidOriginal.size, dtype = np.float64)
        self.nsY = np.empty(self.fluidNodesWithSolidOriginal.size, dtype = np.float64)
        tmpIndices = 0
        tmpXVSum = 0.; tmpYVSum = 0.
        for i in self.fluidNodesWithSolidOriginal:
            yPosition = int(i / self.xDomain); xPosition = i % self.xDomain
            tmpE1 = xPosition + 1 if xPosition < self.xDomain - 1 else 0
            tmpW1 = xPosition - 1 if xPosition > 0 else self.xDomain - 1
            tmpN1 = yPosition + 1 if yPosition < self.yDomain - 1 else 0
            tmpS1 = yPosition - 1 if yPosition > 0 else self.yDomain - 1
            
#            tmpE2 = xPosition + 2 if xPosition < self.xDomain - 2 else 0
#            tmpW2 = xPosition - 2 if xPosition > 1 else self.xDomain - 1
#            tmpN2 = yPosition + 2 if yPosition < self.yDomain - 2 else 0
#            tmpS2 = yPosition - 2 if yPosition > 1 else self.yDomain - 1
            if xPosition < self.xDomain - 2:
                tmpE2 = xPosition + 2
            elif xPosition == self.xDomain - 2:
                tmpE2 = 0
            elif xPosition == self.xDomain - 1:
                tmpE2 = 1
            if xPosition > 1:
                tmpW2 = xPosition - 2
            elif xPosition == 1:
                tmpW2 = self.xDomain - 1
            elif xPosition == 0:
                tmpW2 = self.xDomain - 2
            if yPosition < self.yDomain - 2:
                tmpN2 = yPosition + 2
            elif yPosition == self.yDomain - 2:
                tmpN2 = 0
            elif yPosition == self.yDomain - 1:
                tmpN2 = 1
            if yPosition > 1:
                tmpS2 = yPosition - 2
            elif yPosition == 1:
                tmpS2 = self.yDomain - 1
            elif yPosition == 0:
                tmpS2 = self.yDomain - 2
            #Eastern D = 1 point c = (1, 0)
            if self.isDomain[yPosition, tmpE1] == 0:
                tmpXVSum += 4./21. * 1. * (1.); tmpYVSum += 4./21. * 1. * (0.)
            #Northern D = 1 point c = (0, 1)
            if self.isDomain[tmpN1, xPosition] == 0:
                tmpXVSum += 4./21. * 1. * (0.); tmpYVSum += 4./21. * 1. * (1.)
            #Western D = 1 point C = (-1., 0)
            if self.isDomain[yPosition, tmpW1] == 0:
                tmpXVSum += 4./21. * 1. * (-1.); tmpYVSum += 4./21. * 1. * (0.)
            #Southern D = 1 point C = (0, -1)
            if self.isDomain[tmpS1, xPosition] == 0:
                tmpXVSum += 4./21. * 1. * (0.); tmpYVSum += 4./21. * 1. * (-1.)
                
            #Northeastern D = sqrt(2) c = (1, 1)
            if self.isDomain[tmpN1, tmpE1] == 0:
                tmpXVSum += 4./45. * 1. * (1.); tmpYVSum += 4./45. * 1. * (1.)
            #Northwestern D = sqrt(2) c = (-1, 1)
            if self.isDomain[tmpN1, tmpW1] == 0:
                tmpXVSum += 4./45. * 1. * (-1.); tmpYVSum += 4./45. * 1. * (1.)
            #Southwestern D = sqrt(2) c = (-1, -1)
            if self.isDomain[tmpS1, tmpW1] == 0:
                tmpXVSum += 4./45. * 1. * (-1.); tmpYVSum += 4./45. * 1. * (-1.)
            #Southeastern D = sqrt(2) c = (1, -1)
            if self.isDomain[tmpS1, tmpE1] == 0:
                tmpXVSum += 4./45. * 1. * (1.); tmpYVSum += 4./45. * 1. * (-1.)
                
            #Eastern D = 2 c = (2, 0)
            if self.isDomain[yPosition, tmpE2] == 0:
                tmpXVSum += 1./60. * 1. * (2.); tmpYVSum += 1./60. * 1. * (0.)
            #Northern D = 2 c = (0, 2)
            if self.isDomain[tmpN2, xPosition] == 0:
                tmpXVSum += 1./60. * 1. * (0.); tmpYVSum += 1./60. * 1. * (2.)
            #Western D = 2 c = (-2, 0)
            if self.isDomain[yPosition, tmpW2] == 0:
                tmpXVSum += 1./60. * 1. * (-2.); tmpYVSum += 1./60. * 1. * (0.)
            #Southern D = 2 c = (0, -2)
            if self.isDomain[tmpS2, xPosition] == 0:
                tmpXVSum += 1./60. * 1. * (0.); tmpYVSum += 1./60. * 1. * (-2.)
                
            #Northeastern D = sqrt(5) c = (2, 1)
            if self.isDomain[tmpN1, tmpE2] == 0:
                tmpXVSum += 2./315. * 1. * (2.); tmpYVSum += 2./315. * 1. * (1.)
            #Northeastern D = sqrt(5) c = (1, 2)
            if self.isDomain[tmpN2, tmpE1] == 0:
                tmpXVSum += 2./315. * 1. * (1.); tmpYVSum += 2./315. * 1. * (2.)
            #Northwestern D = sqrt(5) c = (-1, 2)
            if self.isDomain[tmpN2, tmpW1] == 0:
                tmpXVSum += 2./315. * 1. * (-1.); tmpYVSum += 2./315. * 1. * (2.)
            #Northwestern D = sqrt(5) c = (-2, 1)
            if self.isDomain[tmpN1, tmpW2] == 0:
                tmpXVSum += 2./315. * 1. * (-2.); tmpYVSum += 2./315. * 1. * (1.)
            #Southwestern D = sqrt(5) c = (-2, -1)
            if self.isDomain[tmpS1, tmpW2] == 0:
                tmpXVSum += 2./315. * 1. * (-2.); tmpYVSum += 2./315. * 1. * (-1.)
            #Southwestern D = sqrt(5), c = (-1, -2)
            if self.isDomain[tmpS2, tmpW1] == 0:
                tmpXVSum += 2./315. * 1. * (-1.); tmpYVSum += 2./315. * 1. * (-2.)
            #Southeastern D = sqrt(5), c = (1, -2)
            if self.isDomain[tmpS2, tmpE1] == 0:
                tmpXVSum += 2./315. * 1. * (1.); tmpYVSum += 2./315. * 1. * (-2.)
            #Southeastern D = sqrt(5) c = (2, -1)
            if self.isDomain[tmpS1, tmpE2] == 0:
                tmpXVSum += 2./315. * 1. * (2.); tmpYVSum += 2./315. * 1. * (-1.)
                
            #Northeastern D = sqrt(8) c = (2, 2)
            if self.isDomain[tmpN2, tmpE2] == 0:
                tmpXVSum += 1./5040. * 1. * (2.); tmpYVSum += 1./5040. * 1. * (2.)
            #Northwestern D = sqrt(8) c = (-2, 2)
            if self.isDomain[tmpN2, tmpW2] == 0:
                tmpXVSum += 1./5040. * 1. * (-2.); tmpYVSum += 1./5040. * 1. * (2.)
            #Southwestern D = sqrt(8) c = (-2, -2)
            if self.isDomain[tmpS2, tmpW2] == 0:
                tmpXVSum += 1./5040. * 1. * (-2.); tmpYVSum += 1./5040. * 1. * (-2.)
            #Southeastern D = sqrt(8) c = (2, -2)
            if self.isDomain[tmpS2, tmpE2] == 0:
                tmpXVSum += 1./5040. * 1. * (2.); tmpYVSum += 1./5040. * 1. * (-2.)
                
            #calculate the unit vector
            tmpVectorNorm = math.sqrt(tmpXVSum * tmpXVSum + tmpYVSum * tmpYVSum)
            self.nsX[tmpIndices] = tmpXVSum / tmpVectorNorm
            self.nsY[tmpIndices] = tmpYVSum / tmpVectorNorm
            tmpIndices += 1
            tmpXVSum = 0.; tmpYVSum = 0.
#        print(self.nsX)
#        print(self.nsY)
#        print(self.nsX.size)
#        print(self.nsY.size)
                
    @autojit
    def convertOptTo2D(self):
        tmpIndex = 0
        for tmpPos in self.fluidNodes:
            tmpX = tmpPos % self.xDomain; tmpY = int(tmpPos / self.xDomain)
            self.fluidsRhoR[tmpY, tmpX] = self.optFluidRhoR[tmpIndex]
            self.fluidsRhoB[tmpY, tmpX] = self.optFluidRhoB[tmpIndex]
            self.fluidPDFB[tmpY, tmpX, :] = self.optFluidPDFB[tmpIndex, :]
            self.fluidPDFR[tmpY, tmpX, :] = self.optFluidPDFR[tmpIndex, :]
            self.physicalVX[tmpY, tmpX] = self.optMacroVelocityX[tmpIndex]
            self.physicalVY[tmpY, tmpX] = self.optMacroVelocityY[tmpIndex]
            tmpIndex += 1
        
    def resultInHDF5(self, iStep):
        """
        Save the data from the simulation in HDF5 fromat
        """
        filePath = os.path.expanduser('~/LBMResults')
        resultFile = filePath + '/SimulationResultsRK.h5'
        dataFile = tb.open_file(resultFile, 'a')
        #output the densities of fluids
        dataFile.create_array('/FluidMacro', 'FluidDensityRin%g' % (iStep), \
                                  self.fluidsRhoR)
        dataFile.create_array('/FluidMacro', 'FluidDensityBin%g' % (iStep), \
                                  self.fluidsRhoB)
        dataFile.create_array('/FluidPDF', 'FluidPDFBat%g' % (iStep), self.fluidPDFB)
        dataFile.create_array('/FluidPDF', 'FluidPDFRat%g' % (iStep), self.fluidPDFR)
        dataFile.create_array('/FluidVelocity', 'FluidVelocityXAt%g' % iStep, \
                              self.physicalVX)
        dataFile.create_array('/FluidVelocity', 'FluidVelocityYAt%g' % iStep, \
                              self.physicalVY)
        dataFile.close()
        
    def plotDensityDistributionOPT(self, iStep):
        """
        Plot fluid 0 density distribution in the whole domain
        """
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMResults/'
#        plt.subplot(121)
        plt.imshow(self.fluidsRhoR, origin = 'low')
        plt.colorbar()
        plt.savefig(pathResults + 'FluidsRDistributionAt%05d.png' % (iStep))
        plt.close()
#        plt.colorbar()
#        plt.subplot(122)
        plt.imshow(self.fluidsRhoB, origin = 'low')
        plt.colorbar()
        plt.savefig(pathResults + 'FluidsBDistributionAt%05d.png' % (iStep))
        plt.close()

    def runRKColorGradient2DCSF(self, ):
        print("Start to run R-K color gradient lattice Boltzmann method.")
        self.initializeDomainBorder()
        self.initializeDomainCondition()
        print("Finish initialize the original simulated domain.")
        self.optimizeFluidandSolidArray()
        self.numColorSolid = 0;
        self.numWettingFluid = 0

        if self.wettingSolidNodes.size > 0:
            self.sortOutFluidNodesToSolid()
            print(self.fluidNodesWithSolidOriginal.size)
            print(self.fluidNodesWithSolidGPU.size)
            self.numWettingFluid = self.fluidNodesWithSolidGPU.size
            self.calVectorNormaltoSolid()
            self.numColorSolid = self.wettingSolidNodes.size
        self.solidColorValue = np.zeros(self.numColorSolid, dtype=np.float64)

        print("Start to set up arrays for the device.")
        deviceFluidRhoR = cuda.to_device(self.optFluidRhoR)
        deviceFluidRhoB = cuda.to_device(self.optFluidRhoB)
        deviceFluidPDFR = cuda.to_device(self.optFluidPDFR)
        deviceFluidPDFB = cuda.to_device(self.optFluidPDFB)
        deviceFluidPDFRNew = cuda.device_array_like(self.optFluidPDFR)
        deviceFluidPDFBNew = cuda.device_array_like(self.optFluidPDFB)
        devicePhysicalVX = cuda.to_device(self.optMacroVelocityX)
        devicePhysicalVY = cuda.to_device(self.optMacroVelocityY)

        deviceConstCR = cuda.to_device(self.constantCR)
        deviceConstCB = cuda.to_device(self.constantCB)

        deviceColorValue = cuda.device_array_like(self.optFluidRhoB)
        colorValue = np.array(self.optFluidRhoB.size, dtype=np.float64)
        colorValueOld = np.array(self.optFluidRhoB, dtype=np.float64)

        totalFluidPDFTotal = self.optFluidPDFB + self.optFluidPDFR
        deviceFluidPDFTotal = cuda.to_device(totalFluidPDFTotal)
        deviceForceX = cuda.device_array_like(self.optFluidRhoB)
        deviceForceY = cuda.device_array_like(self.optFluidRhoB)
        deviceGradientX = cuda.device_array_like(self.optFluidRhoB)
        deviceGradientY = cuda.device_array_like(self.optFluidRhoR)
        deviceSolidColor = cuda.to_device(self.solidColorValue)
        deviceKValue = cuda.to_device(self.optFluidRhoB)
        if self.wettingSolidNodes.size > 0:
            deviceNeighboringWettingSolid = cuda.to_device(self.neighboringWettingSolidNodes)
            deviceFluidNodesWithSolid = cuda.to_device(self.fluidNodesWithSolidGPU)
            deviceUnitNsx = cuda.to_device(self.nsX)
            deviceUnitNsy = cuda.to_device(self.nsY)

        deviceFluidNodes = cuda.to_device(self.fluidNodes)
        deviceNeighboringNodes = cuda.to_device(self.neighboringNodes)

        deviceWeightsCoeff = cuda.to_device(self.weightsCoeff)
        deviceUnitEX = cuda.to_device(self.unitEX)
        deviceUnitEY = cuda.to_device(self.unitEY)

        if self.relaxationType == "'MRT'":
            deviceTransformationM = cuda.to_device(self.transformationM)
            deviceTransformationIM = cuda.to_device(self.invTransformationM)
            deviceCollisionM = cuda.to_device(self.collisionS)

        if self.boundaryTypeOutlet == "'AverageConvective'":
            deviceFluidPDFROld = cuda.to_device(self.optFluidPDFR)
            deviceFluidPDFBOld = cuda.to_device(self.optFluidPDFB)
        totalNodes = self.fluidNodes.size
        blockNumX = int(self.xDimension / self.threadNum)
        blockNumY = math.ceil(self.fluidNodes.size / self.xDimension)
        threadPerBlock1D = (self.threadNum, 1)
        grid1D = (blockNumX, blockNumY)
        iStep = 0;
        recordStep = 0
        stopStandard = 1.0
        while (iStep < self.timeSteps):
            #        while (stopStandard > 1.0e-10):
            print("At the time step %d." % iStep)
            iStep += 1
            print("Start the first step of streaming for both fluids.")
            RKGPU2D.calStreaming1GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidNodes, deviceNeighboringNodes, \
                                                               deviceFluidPDFR, deviceFluidPDFRNew)
            RKGPU2D.calStreaming1GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidNodes, deviceNeighboringNodes, \
                                                               deviceFluidPDFB, deviceFluidPDFBNew)
            print("Start the second step of streaming for both fluids.")
            RKGPU2D.calStreaming2GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidPDFRNew, deviceFluidPDFR)
            RKGPU2D.calStreaming2GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidPDFBNew, deviceFluidPDFB)

            #            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
            #                                          self.xDimension, deviceFluidPDFR, \
            #                                          deviceFluidPDFB, deviceFluidRhoR, \
            #                                          deviceFluidRhoB)
            RKGPU2D.calTotalFluidPDF[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidPDFR, deviceFluidPDFB,
                                                               deviceFluidPDFTotal)
            if self.boundaryTypeOutlet == "'Convective'":
                print("Free boundary at the outlet.")
                RKGPU2D.convectiveOutletGPU[grid1D, threadPerBlock1D](totalNodes, self.xDomain, \
                                                                      self.xDimension, deviceFluidNodes,
                                                                      deviceNeighboringNodes, deviceFluidPDFR, \
                                                                      deviceFluidPDFB, deviceFluidRhoR,
                                                                      deviceFluidRhoB)
                RKGPU2D.convectiveOutletGhost2GPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                            self.xDomain, self.xDimension,
                                                                            deviceFluidNodes,
                                                                            deviceNeighboringNodes, \
                                                                            deviceFluidPDFR, deviceFluidPDFB,
                                                                            deviceFluidRhoR, \
                                                                            deviceFluidRhoB)
                RKGPU2D.convectiveOutletGhost3GPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                            self.xDomain, self.xDimension,
                                                                            deviceFluidNodes,
                                                                            deviceNeighboringNodes, \
                                                                            deviceFluidPDFR, deviceFluidPDFB,
                                                                            deviceFluidRhoR, \
                                                                            deviceFluidRhoB)
            elif self.boundaryTypeOutlet == "'AverageConvective'":
                print("Free boundary with average method.")
                RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
                                                                         self.xDimension, deviceFluidPDFR, \
                                                                         deviceFluidPDFB, deviceFluidRhoR, \
                                                                         deviceFluidRhoB)
                RKGPU2D.calPhysicalVelocityRKGPU2DM[grid1D, threadPerBlock1D](totalNodes, \
                                                                              self.xDimension, deviceFluidPDFTotal,
                                                                              deviceFluidRhoR, \
                                                                              deviceFluidRhoB, devicePhysicalVX,
                                                                              devicePhysicalVY, \
                                                                              deviceForceX, deviceForceY)
                RKGPU2D.convectiveAverageBoundaryGPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                               self.xDomain, self.xDimension,
                                                                               deviceFluidNodes, \
                                                                               deviceNeighboringNodes,
                                                                               devicePhysicalVY, deviceFluidPDFR, \
                                                                               deviceFluidPDFB, deviceFluidPDFROld,
                                                                               deviceFluidPDFBOld)
                RKGPU2D.convectiveAverageBoundaryGPU2[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDomain, self.xDimension,
                                                                                deviceFluidNodes, \
                                                                                deviceNeighboringNodes,
                                                                                devicePhysicalVY, deviceFluidPDFR, \
                                                                                deviceFluidPDFB, deviceFluidPDFROld,
                                                                                deviceFluidPDFBOld)
                RKGPU2D.convectiveAverageBoundaryGPU3[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDomain, self.xDimension,
                                                                                deviceFluidNodes, \
                                                                                deviceNeighboringNodes,
                                                                                devicePhysicalVY, deviceFluidPDFR, \
                                                                                deviceFluidPDFB, deviceFluidPDFROld,
                                                                                deviceFluidPDFBOld)
                RKGPU2D.copyFluidPDFLastStep[grid1D, threadPerBlock1D](totalNodes, \
                                                                       self.xDomain, self.xDimension,
                                                                       deviceFluidNodes, \
                                                                       deviceFluidPDFR, deviceFluidPDFB, \
                                                                       deviceFluidPDFROld, deviceFluidPDFBOld)
            elif self.boundaryTypeOutlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                #                RKGPU2D.calConstPressureLowerGPU[grid1D, threadPerBlock1D](totalNodes, \
                #                                self.xDomain, self.xDimension, self.densityRhoBL, \
                #                                self.densityRhoRL, deviceFluidNodes, deviceFluidRhoB, \
                #                                deviceFluidRhoR, deviceFluidPDFB, \
                #                                deviceFluidPDFR)
                totalPressure = self.densityRhoBL + self.densityRhoRL
                RKGPU2D.calConstPressureLowerGPUTotal[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDomain, self.xDimension,
                                                                                totalPressure, deviceFluidNodes, \
                                                                                deviceFluidPDFTotal,
                                                                                devicePhysicalVY, deviceFluidRhoR, \
                                                                                deviceFluidRhoB, deviceFluidPDFR,
                                                                                deviceFluidPDFB)
                RKGPU2D.ghostPointsConstPressureLowerRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                  self.xDomain, self.xDimension,
                                                                                  deviceFluidNodes, \
                                                                                  deviceNeighboringNodes,
                                                                                  deviceFluidRhoR, deviceFluidRhoB, \
                                                                                  deviceFluidPDFR, deviceFluidPDFB)

                #            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
            #                                          self.xDimension, deviceFluidPDFR, \
            #                                          deviceFluidPDFB, deviceFluidRhoR, \
            #                                          deviceFluidRhoB)

            if self.boundaryTypeInlet == "'Neumann'":
                #                RKGPU2D.constantVelocityZHBoundaryHigherRK[grid1D, threadPerBlock1D](totalNodes, \
                #                                        self.xDomain, self.yDomain, self.xDimension, \
                #                                        self.velocityYR, self.velocityYB, deviceFluidNodes, \
                #                                        deviceFluidRhoR, deviceFluidRhoB, deviceFluidPDFR, \
                #                                        deviceFluidPDFB)
                specificVY = self.velocityYB + self.velocityYR
                RKGPU2D.constantTotalVelocityInlet[grid1D, threadPerBlock1D](totalNodes, \
                                                                             self.xDomain, self.yDomain,
                                                                             self.xDimension, \
                                                                             specificVY, deviceFluidNodes,
                                                                             deviceNeighboringNodes, \
                                                                             deviceFluidRhoR, deviceFluidRhoB,
                                                                             deviceFluidPDFR, \
                                                                             deviceFluidPDFB, deviceFluidPDFTotal,
                                                                             devicePhysicalVY)
                #                RKGPU2D.constantVelocityZHBoundaryHigherNewRK[grid1D, threadPerBlock1D](totalNodes, \
                #                                        self.xDomain, self.yDomain, self.xDimension, \
                #                                        self.velocityYR, self.velocityYB, deviceFluidNodes, \
                #                                        deviceNeighboringNodes, deviceFluidRhoR, \
                #                                        deviceFluidRhoB, deviceFluidPDFR, \
                #                                        deviceFluidPDFB)
                RKGPU2D.ghostPointsConstantVelocityRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDomain, self.yDomain,
                                                                                self.xDimension, deviceFluidNodes, \
                                                                                deviceNeighboringNodes,
                                                                                deviceFluidRhoR, \
                                                                                deviceFluidRhoB, deviceFluidPDFR,
                                                                                deviceFluidPDFB, \
                                                                                deviceForceX, deviceForceY)
            if self.boundaryTypeInlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                RKGPU2D.calConstPressureInletGPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                           self.xDomain, self.yDomain,
                                                                           self.xDimension, self.densityRhoBH, \
                                                                           self.densityRhoRH, deviceFluidNodes,
                                                                           deviceFluidRhoB, \
                                                                           deviceFluidRhoR, deviceFluidPDFB, \
                                                                           deviceFluidPDFR)
                RKGPU2D.ghostPointsConstPressureInletRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                  self.xDomain, self.yDomain,
                                                                                  self.xDimension, deviceFluidNodes, \
                                                                                  deviceNeighboringNodes,
                                                                                  deviceFluidRhoR, deviceFluidRhoB, \
                                                                                  deviceFluidPDFR, deviceFluidPDFB)

            print("Calculate the macro-density of the fluids")
            RKGPU2D.calTotalFluidPDF[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidPDFR, deviceFluidPDFB,
                                                               deviceFluidPDFTotal)
            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
                                                                     self.xDomain, self.yDomain, self.xDimension,
                                                                     deviceFluidNodes, deviceFluidPDFR, \
                                                                     deviceFluidPDFB, deviceFluidRhoR, \
                                                                     deviceFluidRhoB)
            print("Calcuate the macroscopic velocity.")
            RKGPU2D.calPhysicalVelocityRKGPU2DM[grid1D, threadPerBlock1D](totalNodes, \
                                                                          self.xDomain, self.yDomain,
                                                                          self.xDimension, deviceFluidNodes, \
                                                                          deviceFluidPDFTotal, deviceFluidRhoR, \
                                                                          deviceFluidRhoB, devicePhysicalVX,
                                                                          devicePhysicalVY, \
                                                                          deviceForceX, deviceForceY)
            print("Calculate the color values on each fluid nodes.")
            RKGPU2D.calPhaseFieldPhi[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidRhoR, deviceFluidRhoB, deviceColorValue)
            if (self.boundaryTypeOutlet == "'Dirilcht'"):
                RKGPU2D.calNeumannPhiOutlet[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                                      self.xDomain, deviceFluidNodes,
                                                                      deviceNeighboringNodes, deviceColorValue)
            #            if (iStep - 1 == 0):
            #                colorValue = deviceColorValue.copy_to_host()
            #                colorValueOld = colorValue
            #            if ((iStep - 1) % 100 == 0 and iStep != 1):
            #                colorValue = deviceColorValue.copy_to_host()
            #                stopStandard = np.sum(np.power(colorValue - colorValueOld, 2)) / \
            #                        np.sum(np.power(colorValue, 2))
            #                print("The value of stoping loop is: ")
            #                print(stopStandard)
            #                colorValueOld = colorValue
            if ((iStep - 1) % self.timeInterval == 0 or stopStandard < 1.0e-10):
                print("Copy data to host for saving and plotting.")
                self.optFluidRhoR = deviceFluidRhoR.copy_to_host()
                self.optFluidRhoB = deviceFluidRhoB.copy_to_host()
                self.optMacroVelocityX = devicePhysicalVX.copy_to_host()
                self.optMacroVelocityY = devicePhysicalVY.copy_to_host()
                self.optFluidPDFB = deviceFluidPDFB.copy_to_host()
                self.optFluidPDFR = deviceFluidPDFR.copy_to_host()
                self.optFluidEql = deviceEquilibriumR.copy_to_host()
                self.convertOptTo2D()
                self.resultInHDF5(recordStep)
                self.plotDensityDistributionOPT(recordStep)
                recordStep += 1
            #                input()
            if self.wettingSolidNodes.size > 0:
                print("Calculate the color value on the solid nodes neighboring to fluid.")
                RKGPU2D.calColorValueOnSolid[grid1D, threadPerBlock1D](self.numColorSolid, \
                                                                       self.xDimension,
                                                                       deviceNeighboringWettingSolid, \
                                                                       deviceWeightsCoeff, deviceColorValue, \
                                                                       deviceSolidColor)
            print("Calculate the initial color gradient on fluid nodes.")
            RKGPU2D.calRKInitialGradient[grid1D, threadPerBlock1D](totalNodes, \
                                                                   self.xDimension, self.numColorSolid, \
                                                                   deviceFluidNodes, deviceNeighboringNodes, \
                                                                   deviceWeightsCoeff, deviceUnitEX, deviceUnitEY, \
                                                                   deviceColorValue, deviceSolidColor, \
                                                                   deviceGradientX, deviceGradientY)
            if self.wettingSolidNodes.size > 0:
                print("Update the color gradient values on the fluid nodes near to solid.")
                if self.wettingType == 1:
                    RKGPU2D.updateColorGradientOnWetting[grid1D, threadPerBlock1D](self.numWettingFluid, \
                                                                                   self.xDimension, self.cosTheta,
                                                                                   self.sinTheta,
                                                                                   deviceFluidNodesWithSolid, \
                                                                                   deviceUnitNsx, deviceUnitNsy,
                                                                                   deviceGradientX, deviceGradientY)
                elif self.wettingType == 2:
                    RKGPU2D.updateColorGradientOnWettingNew[grid1D, threadPerBlock1D](self.numWettingFluid, \
                                                                                      self.xDimension,
                                                                                      self.cosTheta, self.sinTheta,
                                                                                      deviceFluidNodesWithSolid, \
                                                                                      deviceUnitNsx, deviceUnitNsy,
                                                                                      deviceGradientX,
                                                                                      deviceGradientY)
            print("Calculate the force values in the domain")
            if self.wettingType == 1:
                RKGPU2D.calForceTermInColorGradient2D[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDimension,
                                                                                self.surfaceTension,
                                                                                deviceNeighboringNodes, \
                                                                                deviceWeightsCoeff, deviceUnitEX,
                                                                                deviceUnitEY, \
                                                                                deviceGradientX, deviceGradientY,
                                                                                deviceForceX, \
                                                                                deviceForceY, deviceKValue)
            elif self.wettingType == 2:
                RKGPU2D.calForceTermInColorGradientNew2D[grid1D, threadPerBlock1D](totalNodes, \
                                                                                   self.xDimension,
                                                                                   self.surfaceTension,
                                                                                   deviceNeighboringNodes, \
                                                                                   deviceWeightsCoeff, deviceUnitEX,
                                                                                   deviceUnitEY, \
                                                                                   deviceGradientX, deviceGradientY,
                                                                                   deviceForceX, \
                                                                                   deviceForceY, deviceKValue)
            print("Calculate the single phase collision for total distribution function.")
            if self.relaxationType == "'SRT'":
                RKGPU2D.calRKCollision1TotalGPU2DSRTM[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDimension,
                                                                                self.tauCalculation, self.tauR,
                                                                                self.tauB, \
                                                                                self.deltaValue, deviceUnitEX,
                                                                                deviceUnitEY, \
                                                                                deviceWeightsCoeff,
                                                                                devicePhysicalVX, \
                                                                                devicePhysicalVY, deviceFluidRhoR,
                                                                                deviceFluidRhoB, \
                                                                                deviceColorValue,
                                                                                deviceFluidPDFTotal)
                print("Calculate the force perturbation for the total distribution function.")
                RKGPU2D.calPerturbationFromForce2D[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                                             self.tauCalculation, self.tauR,
                                                                             self.tauB, self.deltaValue, \
                                                                             deviceWeightsCoeff, deviceUnitEX, \
                                                                             deviceUnitEY, devicePhysicalVX,
                                                                             devicePhysicalVY, \
                                                                             deviceForceX, deviceForceY,
                                                                             deviceColorValue, \
                                                                             deviceFluidPDFTotal, deviceFluidRhoR,
                                                                             deviceFluidRhoB)
            if self.relaxationType == "'MRT'":
                RKGPU2D.calRKCollision1TotalGPU2DMRTM[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDimension,
                                                                                self.tauCalculation, self.tauR,
                                                                                self.tauB, \
                                                                                self.deltaValue, deviceUnitEX,
                                                                                deviceUnitEY, deviceWeightsCoeff, \
                                                                                devicePhysicalVX, devicePhysicalVY, \
                                                                                deviceFluidRhoR, deviceFluidRhoB, \
                                                                                deviceColorValue,
                                                                                deviceFluidPDFTotal, \
                                                                                deviceTransformationM,
                                                                                deviceTransformationIM, \
                                                                                deviceCollisionM)
                RKGPU2D.calPerturbationFromForce2DMRT[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                                                self.tauCalculation, self.tauR,
                                                                                self.tauB, self.deltaValue, \
                                                                                deviceWeightsCoeff, deviceUnitEX, \
                                                                                deviceUnitEY, devicePhysicalVX,
                                                                                devicePhysicalVY, \
                                                                                deviceForceX, deviceForceY,
                                                                                deviceColorValue, \
                                                                                deviceFluidPDFTotal,
                                                                                deviceTransformationM, \
                                                                                deviceTransformationIM,
                                                                                deviceCollisionM, \
                                                                                deviceFluidRhoR, deviceFluidRhoB)

            print("Recoloring both fluids in the system.")
            RKGPU2D.calRecoloringProcessM[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                                    self.betaThickness, deviceWeightsCoeff,
                                                                    deviceFluidRhoR, \
                                                                    deviceFluidRhoB, deviceUnitEX, deviceUnitEY, \
                                                                    deviceGradientX, deviceGradientY,
                                                                    deviceFluidPDFR, \
                                                                    deviceFluidPDFB, deviceFluidPDFTotal)

    #            input('Finsih one loop.')

    def runRKColorGradient2DPerturbation(self, ):
        print("Start to run R-K color gradient lattice Boltzmann method.")
        #        self.__checkGPUAvailability()
        self.initializeDomainBorder()
        self.initializeDomainCondition()
        print("Finish initialize the original simulated domain.")
        self.optimizeFluidArray()
        print("Finish converting the data.")
        print("Start to set up arrays for the device.")
        deviceFluidRhoR = cuda.to_device(self.optFluidRhoR)
        deviceFluidRhoB = cuda.to_device(self.optFluidRhoB)
        deviceFluidPDFR = cuda.to_device(self.optFluidPDFR)
        deviceFluidPDFB = cuda.to_device(self.optFluidPDFB)
        deviceFluidPDFRNew = cuda.device_array_like(self.optFluidPDFR)
        deviceFluidPDFBNew = cuda.device_array_like(self.optFluidPDFB)
        devicePhysicalVX = cuda.to_device(self.optMacroVelocityX)
        devicePhysicalVY = cuda.to_device(self.optMacroVelocityY)
        devicePhiValue = cuda.device_array_like(self.optFluidRhoB)
        deviceCollisionR1 = cuda.device_array_like(self.optFluidPDFR)
        deviceCollisionB1 = cuda.device_array_like(self.optFluidPDFB)
        deviceCollisionTotal1 = cuda.device_array_like(self.optFluidPDFR)
        deviceCollisionTotal2 = cuda.device_array_like(self.optFluidPDFR)
        deviceFluidPDFTotal = cuda.device_array_like(self.optFluidPDFR)
        deviceGradientX = cuda.device_array_like(self.optFluidRhoB)
        deviceGradientY = cuda.device_array_like(self.optFluidRhoR)

        self.colorGradientX = np.zeros([self.yDomain, self.xDomain], dtype=np.float64)
        self.colorGradientY = np.zeros([self.yDomain, self.xDomain], dtype=np.float64)
        optCGX = np.zeros(self.yDomain * self.xDomain, dtype=np.float64)
        optCGY = np.zeros(self.yDomain * self.xDomain, dtype=np.float64)
        deviceCGX = cuda.to_device(optCGX)
        deviceCGY = cuda.to_device(optCGY)

        deviceFluidNodes = cuda.to_device(self.fluidNodes)
        deviceNeighboringNodes = cuda.to_device(self.neighboringNodes)

        deviceWeightsCoeff = cuda.to_device(self.weightsCoeff)
        deviceConstCR = cuda.to_device(self.constantCR)
        deviceConstCB = cuda.to_device(self.constantCB)
        deviceConstB = cuda.to_device(self.constantB)
        deviceUnitEX = cuda.to_device(self.unitEX)
        deviceUnitEY = cuda.to_device(self.unitEY)
        deviceScheme = cuda.to_device(self.gradientScheme)
        deviceConstBNew = cuda.to_device(self.constantBNew)

        if self.relaxationType == "'MRT'":
            deviceTransformationM = cuda.to_device(self.transformationM)
            deviceTransformationIM = cuda.to_device(self.invTransformationM)
            deviceCollisionM = cuda.to_device(self.collisionS)

        totalNodes = self.fluidNodes.size
        blockNumX = int(self.xDimension / self.threadNum)
        blockNumY = math.ceil(self.fluidNodes.size / self.xDimension)
        threadPerBlock1D = (self.threadNum, 1)
        grid1D = (blockNumX, blockNumY)
        #        print(self.weightsCoeff)
        #        print(self.constantCR)
        #        print(self.constantCB)
        #        print(self.constantB)
        #        print(self.unitEX)
        #        print(self.unitEY)
        #        print(self.gradientScheme)
        #        print(self.AkB, self.AkR)
        #        input()
        iStep = 0;
        recordStep = 0
        while (iStep < self.timeSteps):
            print("At the time step %d." % iStep)
            iStep += 1
            print("Start the first step of streaming for both fluids.")
            RKGPU2D.calStreaming1GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidNodes, deviceNeighboringNodes, \
                                                               deviceFluidPDFR, deviceFluidPDFRNew)
            RKGPU2D.calStreaming1GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidNodes, deviceNeighboringNodes, \
                                                               deviceFluidPDFB, deviceFluidPDFBNew)
            print("Start the second step of streaming for both fluids.")
            RKGPU2D.calStreaming2GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidPDFRNew, deviceFluidPDFR)
            RKGPU2D.calStreaming2GPU[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidPDFBNew, deviceFluidPDFB)
            #            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
            #                                          self.xDimension, deviceFluidPDFR, \
            #                                          deviceFluidPDFB, deviceFluidRhoR, \
            #                                          deviceFluidRhoB)
            if self.boundaryTypeOutlet == "'Convective'":
                print("Free boundary at the outlet.")
                RKGPU2D.convectiveOutletGPU[grid1D, threadPerBlock1D](totalNodes, self.xDomain, \
                                                                      self.xDimension, deviceFluidNodes,
                                                                      deviceNeighboringNodes, deviceFluidPDFR, \
                                                                      deviceFluidPDFB, deviceFluidRhoR,
                                                                      deviceFluidRhoB)
                RKGPU2D.convectiveOutletGhost2GPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                            self.xDomain, self.xDimension,
                                                                            deviceFluidNodes,
                                                                            deviceNeighboringNodes, \
                                                                            deviceFluidPDFR, deviceFluidPDFB,
                                                                            deviceFluidRhoR, \
                                                                            deviceFluidRhoB)
                RKGPU2D.convectiveOutletGhost3GPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                            self.xDomain, self.xDimension,
                                                                            deviceFluidNodes,
                                                                            deviceNeighboringNodes, \
                                                                            deviceFluidPDFR, deviceFluidPDFB,
                                                                            deviceFluidRhoR, \
                                                                            deviceFluidRhoB)
            elif self.boundaryTypeOutlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                RKGPU2D.calConstPressureLowerGPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                           self.xDomain, self.xDimension,
                                                                           self.densityRhoBL, \
                                                                           self.densityRhoRL, deviceFluidNodes,
                                                                           deviceFluidRhoB, \
                                                                           deviceFluidRhoR, deviceFluidPDFB, \
                                                                           deviceFluidPDFR)
                RKGPU2D.ghostPointsConstPressureLowerRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                  self.xDomain, self.xDimension,
                                                                                  deviceFluidNodes, \
                                                                                  deviceNeighboringNodes,
                                                                                  deviceFluidRhoR, deviceFluidRhoB, \
                                                                                  deviceFluidPDFR, deviceFluidPDFB)

            if self.boundaryTypeInlet == "'Neumann'":
                RKGPU2D.constantVelocityZHBoundaryHigherRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                     self.xDomain, self.yDomain,
                                                                                     self.xDimension, \
                                                                                     self.velocityYR,
                                                                                     self.velocityYB,
                                                                                     deviceFluidNodes, \
                                                                                     deviceFluidRhoR,
                                                                                     deviceFluidRhoB,
                                                                                     deviceFluidPDFR, \
                                                                                     deviceFluidPDFB)
                RKGPU2D.ghostPointsConstantVelocityRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDomain, self.yDomain,
                                                                                self.xDimension, deviceFluidNodes, \
                                                                                deviceNeighboringNodes,
                                                                                deviceFluidRhoR, \
                                                                                deviceFluidRhoB, deviceFluidPDFR,
                                                                                deviceFluidPDFB)
            if self.boundaryTypeInlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                RKGPU2D.calConstPressureInletGPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                           self.xDomain, self.yDomain,
                                                                           self.xDimension, self.densityRhoBH, \
                                                                           self.densityRhoRH, deviceFluidNodes,
                                                                           deviceFluidRhoB, \
                                                                           deviceFluidRhoR, deviceFluidPDFB, \
                                                                           deviceFluidPDFR)
                RKGPU2D.ghostPointsConstPressureInletRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                  self.xDomain, self.yDomain,
                                                                                  self.xDimension, deviceFluidNodes, \
                                                                                  deviceNeighboringNodes,
                                                                                  deviceFluidRhoR, deviceFluidRhoB, \
                                                                                  deviceFluidPDFR, deviceFluidPDFB)
            print("Calculate the macro-density of the fluids")
            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
                                                                     self.xDimension, deviceFluidPDFR, \
                                                                     deviceFluidPDFB, deviceFluidRhoR, \
                                                                     deviceFluidRhoB)
            print("Calculate the macroscale velocity of the fluids.")
            RKGPU2D.calPhysicalVelocityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, self.xDomain, \
                                                                         self.xDimension, deviceFluidNodes,
                                                                         deviceFluidPDFR, deviceFluidPDFB, \
                                                                         deviceFluidRhoR, deviceFluidRhoB,
                                                                         devicePhysicalVX, \
                                                                         devicePhysicalVY)
            #            RKGPU2D.calTotalFluidPDF[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
            #                            deviceFluidPDFR, deviceFluidPDFB, deviceFluidPDFTotal)
            if ((iStep - 1) % self.timeInterval == 0):
                print("Copy data to host for saving and plotting.")
                #                tmpRhoR = deviceFluidRhoR.copy_to_host()
                #                print(tmpRhoR.shape)
                #                print(self.optFluidRhoR.shape)
                #                input()
                self.optFluidRhoR = deviceFluidRhoR.copy_to_host()
                self.optFluidRhoB = deviceFluidRhoB.copy_to_host()
                self.optMacroVelocityX = devicePhysicalVX.copy_to_host()
                self.optMacroVelocityY = devicePhysicalVY.copy_to_host()
                self.optFluidPDFB = deviceFluidPDFB.copy_to_host()
                self.optFluidPDFR = deviceFluidPDFR.copy_to_host()
                #                print("convert the array.")
                self.convertOptTo2D()
                self.resultInHDF5(recordStep)
                self.plotDensityDistributionOPT(recordStep)
            #                recordStep += 1
            #                print(self.fluidsRhoB[1, :])
            #                print(self.fluidPDFB[1, 24, :])
            #                print(self.fluidsRhoB[2, :])
            #                input()
            RKGPU2D.calPhaseFieldPhi[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                               deviceFluidRhoR, deviceFluidRhoB, devicePhiValue)
            if (self.boundaryTypeOutlet == "'Dirilcht'"):
                RKGPU2D.calNeumannPhiOutlet[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                                                      self.xDomain, deviceFluidNodes,
                                                                      deviceNeighboringNodes, devicePhiValue)
            print("Calculate the first collision part.")
            if self.relaxationType == "'SRT'":
                #                RKGPU2D.calRKCollision1GPU2DSRT[grid1D, threadPerBlock1D](totalNodes, \
                #                                            self.xDimension, self.deltaValue, self.tauR, \
                #                                            self.tauB, deviceUnitEX, deviceUnitEY, \
                #                                            deviceConstCR, deviceConstCB, deviceWeightsCoeff, \
                #                                            devicePhysicalVX, devicePhysicalVY, deviceFluidRhoR, \
                #                                            deviceFluidRhoB, deviceFluidPDFR, deviceFluidPDFB)
                RKGPU2D.calRKCollision1GPU2DSRTNew[grid1D, threadPerBlock1D](totalNodes, \
                                                                             self.xDimension, self.deltaValue,
                                                                             self.tauR, \
                                                                             self.tauB, deviceUnitEX, deviceUnitEY, \
                                                                             deviceConstCR, deviceConstCB,
                                                                             deviceWeightsCoeff, \
                                                                             devicePhysicalVX, devicePhysicalVY,
                                                                             deviceFluidRhoR, \
                                                                             deviceFluidRhoB, devicePhiValue, \
                                                                             deviceFluidPDFR, deviceFluidPDFB,
                                                                             deviceCollisionR1, \
                                                                             deviceCollisionB1)
            #                RKGPU2D.calRKCollision1TotalGPU2DSRT[grid1D, threadPerBlock1D](totalNodes, \
            #                                            self.xDimension, self.tauR, self.tauB, \
            #                                            deviceUnitEX, deviceUnitEY, deviceConstCR, \
            #                                            deviceConstCB, deviceWeightsCoeff, devicePhysicalVX, \
            #                                            devicePhysicalVY, deviceFluidRhoR, deviceFluidRhoB, \
            #                                            devicePhiValue, deviceFluidPDFTotal, \
            #                                            deviceCollisionTotal1)
            elif self.relaxationType == "'MRT'":
                #                RKGPU2D.calRKCollision1GPU2DMRT[grid1D, threadPerBlock1D](totalNodes, \
                #                                            self.xDimension, self.deltaValue, self.tauR, \
                #                                            self.tauB, deviceUnitEX, deviceUnitEY, \
                #                                            deviceConstCR, deviceConstCB, deviceWeightsCoeff, \
                #                                            devicePhysicalVX, devicePhysicalVY, deviceFluidRhoR, \
                #                                            deviceFluidRhoB, deviceFluidPDFR, deviceFluidPDFB, \
                #                                            deviceTransformationM, deviceTransformationIM, \
                #                                            deviceCollisionM)
                RKGPU2D.calRKCollision1GPU2DMRTNew[grid1D, threadPerBlock1D](totalNodes, \
                                                                             self.xDimension, self.deltaValue,
                                                                             self.tauR, \
                                                                             self.tauB, self.bodyFX, self.bodyFY,
                                                                             deviceUnitEX, deviceUnitEY, \
                                                                             deviceConstCR, deviceConstCB,
                                                                             deviceWeightsCoeff, \
                                                                             devicePhysicalVX, devicePhysicalVY,
                                                                             deviceFluidRhoR, \
                                                                             deviceFluidRhoB, devicePhiValue,
                                                                             deviceFluidPDFR, \
                                                                             deviceFluidPDFB, deviceTransformationM, \
                                                                             deviceTransformationIM,
                                                                             deviceCollisionM)
            print("Calculate the second collision and re-coloring parts.")
            #            RKGPU2D.calRKCollision23GPU[grid1D, threadPerBlock1D](totalNodes, \
            #                                        self.xDimension, self.betaThickness, self.AkR, \
            #                                        self.AkB, self.solidPhi, \
            #                                        deviceFluidNodes, deviceNeighboringNodes, \
            #                                        deviceConstB, deviceWeightsCoeff, deviceUnitEX, \
            #                                        deviceUnitEY, deviceScheme, deviceFluidRhoR, \
            #                                        deviceFluidRhoB, deviceConstCR, deviceConstCR, \
            #                                        deviceFluidPDFR, deviceFluidPDFB, deviceCGX, \
            #                                        deviceCGY)
            RKGPU2D.calRKCollision23GPUNew[grid1D, threadPerBlock1D](totalNodes, \
                                                                     self.xDimension, self.betaThickness, self.AkR, \
                                                                     self.AkB, self.solidPhi, \
                                                                     deviceFluidNodes, deviceNeighboringNodes, \
                                                                     deviceConstBNew, deviceWeightsCoeff,
                                                                     deviceUnitEX, \
                                                                     deviceUnitEY, deviceScheme, deviceFluidRhoR, \
                                                                     deviceFluidRhoB, devicePhiValue, deviceConstCR, \
                                                                     deviceConstCB, deviceFluidPDFR,
                                                                     deviceFluidPDFB, \
                                                                     deviceCGX, deviceCGY)
            #            surfaceTA = self.AkR * 2.
            #            RKGPU2D.calRKCollision2TotalGPUNew[grid1D, threadPerBlock1D](totalNodes, \
            #                                    self.xDimension, surfaceTA, self.solidPhi, \
            #                                    deviceFluidNodes, deviceNeighboringNodes, \
            #                                    deviceConstBNew, deviceWeightsCoeff, deviceUnitEX, \
            #                                    deviceUnitEY, devicePhiValue, deviceCollisionTotal2, \
            #                                    deviceGradientX, deviceGradientY)
            RKGPU2D.calRecoloringProcess[grid1D, threadPerBlock1D](totalNodes, \
                                                                   self.xDimension, self.betaThickness,
                                                                   deviceWeightsCoeff, \
                                                                   deviceFluidRhoR, deviceFluidRhoB, deviceUnitEX, \
                                                                   deviceUnitEY, deviceGradientX, deviceGradientY, \
                                                                   deviceCollisionTotal1, deviceCollisionTotal2, \
                                                                   deviceFluidPDFR, deviceFluidPDFB,
                                                                   deviceFluidPDFTotal)

    def runRKColorGradient2D(self,):
        if self.surfaceTensionType == "'CSF'":
            self.runRKColorGradient2DCSF()
        elif self.surfaceTensionType == "'Perturbation'":
            self.runRKColorGradient2DPerturbation()
