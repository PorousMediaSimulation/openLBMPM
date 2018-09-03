"""
Include the transport phenomena with RK-Color gradient method, which calculate
the flow field for the transports. There are SRT and MRT 
"""

import sys, os, getpass, math
import configparser
from timeit import default_timer as timer

import numpy as np
import scipy as sp
import scipy.linalg as slin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tables as tb

from numba import jit, autojit
from numba import cuda
from SimpleGeometryRK import defineGeometry

import AcceleratedRKGPU2D as RKGPU2D
import AccelerateTransport2DRK as Transport2D
from RKD2Q9 import RKColorGradientLBM
import RKGPU2DBoundary as RKCG2DBC

class Transport2DRK(RKColorGradientLBM):
    def __init__(self, pathInFile):
        self.pathInFile = pathInFile
        config = configparser.ConfigParser()
        config.read(self.pathInFile + '/' + 'transportsetup.ini')

        try:
            self.systemType = config['SystemType']['Option']
        except KeyError:
            print("Cannot find the definition for the system for the transport.")
            sys.exit()
        except:
            print("Error happened when the type of system is defined.")
            raise
        if self.systemType == "'MPMC'":
            RKColorGradientLBM.__init__(self, pathInFile)
        else:
            self.xDomainT = 100; self.yDomainT = 50
            self.isDomain = np.ones([self.yDomainT, self.xDomainT], dtype = np.bool)
            self.isDomain[0, :] = 0; self.isDomain[-1, :] = 0
            self.isDomain[:, 0] = 0; self.isDomain[:, -1] = 0
            self.transportDomain = np.zeros([self.yDomainT, self.xDomainT], dtype = np.bool)
            self.transportDomain[1:-1, 1:31] = 1
            self.transportDomain[1:-1, -31:-1] = 1
            self.velocityX = np.zeros([self.yDomainT, self.xDomainT], dtype = np.float64)
            self.velocityY = np.zeros([self.yDomainT, self.xDomainT], dtype = np.float64)
            
            self.numVoidNodes = np.count_nonzero(self.isDomain)
            self.velocityX1D = np.zeros(self.numVoidNodes, dtype = np.float64)
            self.velocityY1D = np.zeros(self.numVoidNodes, dtype = np.float64)
            self.xDimension = 128; self.threadNum = 32
            self.timeSteps = 2100        
        self.unitVX = np.array([0., 1., -1, 0., 0.], dtype = np.float64)
        self.unitVY = np.array([0., 0., 0., 1., -1.], dtype = np.float64)
        try:
            self.reaction = config['SystemType']['Reaction']
        except KeyError:
            print("Cannot find the option to decide whether the reaction exists or not.")
            sys.exit()
        except:
            print("Error happened when defining the reaction exists nor not.")
            raise
        if self.reaction == "'yes'":
            try:
                self.numberReactions = int(config['Reaction']['NumberReaction'])
            except KeyError:
                print("Cannot find the parameter's name for the number of reactions.")
                sys.exit()
            except ValueError:
                print("The number of reactions in the system must be integer.")
            try:
                tmpReactionRate = config['Reaction']['ReactionRate']
            except KeyError:
                print("Cannot find the parameter's name for the reaction rate.")
            self.reactionRate = np.asarray([], dtype = np.float64)
            tmpRateList = tmpReactionRate.split(',')
            print(tmpRateList, tmpReactionRate)
            input()
#            if len(tmpReactionRate) == self.numberReactions:
            for i in tmpRateList:
                self.reactionRate = np.append(self.reactionRate, float(i))
            print(self.reactionRate)
        try:
            self.precipitation = config['SystemType']['Precipitation']
        except KeyError:
            print("Cannot find the option to decide whether the precipitation exists or not.")
            sys.exit()
        try:
            self.numSchemes = int(config['SystemType']['NumberSchemes'])
        except KeyError:
            print("Cannot find the option for the scheme of distribution function.")
            sys.exit()
        except:
            print("Error happened when finding the scheme for the distribution function.")
            raise
        #define parameters in the simulation for the transport phenomena
        try:
            self.numTracers = int(config['TransportParameters']['NumberTracers'])
        except KeyError:
            print('Cannot find the number for how many types of tracers in the domain.')
            sys.exit()
        except ValueError:
            print('The value for the number of tracers is not correct.')
            sys.exit()
        except:
            print("Error happened when defining the number of tracers in the domain.")
        print("The number of tracers in the domain is %g" % (self.numTracers))
        try:
            tmpDiffJ = config['TransportParameters']['DiffusionJ']
        except KeyError:
            print("Cannot find the values of J for calculating diffusion")
            sys.exit()
        except:
            print("Error happened when reading the values of J for each tracer.")
            raise
        tmpDiffJList = tmpDiffJ.split(',')
        self.diffJ = np.asarray([], dtype = np.float64)
        if (len(tmpDiffJList) == self.numTracers):
            for i in tmpDiffJList:
                self.diffJ = np.append(self.diffJ, float(i))
        else:
            print("The number of Js does not match the number of tracers in the domain.")
            sys.exit()
        try:
            tmpTransportTau = config['TransportParameters']['Tau']
        except KeyError:
            print("Cannto find the values for taus of tracers.")
            sys.exit()
        except:
            print("Error happened when reading the values of tau.")
        tmpTauList = tmpTransportTau.split(',')
        self.transportTau = np.asarray([], dtype = np.float64)
        if (len(tmpTauList) == self.numTracers):
            for i in tmpTauList:
                self.transportTau = np.append(self.transportTau, float(i))
        #define the initial condition of transport
        
        #define the boundary condition type and related parameters
        try:
            self.typeTracerInletBoundary = config['BoundaryCondition']['InletType']
        except KeyError:
            print("Cannot find the definition of the inlet boundary type for the tracer.")
            sys.exit()
        except ValueError:
            print("The type of inlet boundary is not string.")
            sys.exit()
        if self.typeTracerInletBoundary == "'Neumann'":
            tmpConcGradientIn = config['BoundaryCondition']['ConcentrationGradientInlet']
            tmpConcGradientListIn = tmpConcGradientIn.split(',')
            self.concGradientIn = np.asarray([], dtype = np.float64)
            if (len(tmpConcGradientListIn) == self.numTracers):
                for i in tmpConcGradientListIn:
                    self.concGradientIn = np.append(self.concGradientUpper, float(i))
        elif self.typeTracerInletBoundary == "'Dirichlet'":
            tmpTracerConc = config['BoundaryCondition']['ConcentrationInlet']
            tmpTracerConcListIn = tmpTracerConc.split(',')
            self.concTracerIn = np.asarray([], dtype = np.float64)
            if (len(tmpTracerConcListIn) == self.numTracers):
                for i in tmpTracerConcListIn:
                    self.concTracerIn = np.append(self.concTracerIn, float(i))
        
        try:
            self.typeTracerOutletBoundary = config['BoundaryCondition']['OutletType']
        except KeyError:
            print("Cannot find the definition of the outlet boundary type for the tracer.")
            sys.exit()
        except ValueError:
            print("The type of outlet boundary is no string.")
            sys.exit()
        if self.typeTracerOutletBoundary == "'VonNeumann'":
            tmpConcGradientOut = config['BoundaryCondition']['ConcentrationGradientOutlet']
            tmpConcGradientListOut = tmpConcGradientOut.split(',')
            self.concGradientOut = np.array([], dtype = np.float64)
            if (len(tmpConcGradientListOut) == self.numTracers):
                for i in tmpConcGradientListOut:
                    self.concGradientOut = np.append(self.concGradientOut, float(i))
        elif self.typeTracerOutletBoundary == "'Dirichlet'":
            tmpTracerConc = config['BoundaryCondition']['ConcentrationOutlet']
            tmpTracerConcList = tmpTracerConc.split(',')
            self.concTracerOut = np.asarray([], dtype = np.float64)
            if (len(tmpTracerConcList) == self.numTracers):
                for i in tmpTracerConcList:
                    self.concTracerOut = np.append(self.concTracerOut, float(i))
        elif self.typeTracerOutletBoundary == "'FreeFlow'":
            pass
        #define the initial condition for the transport phenomena
        try:
            self.initialConditionType = config['InitialCondition']['Type']
        except KeyError:
            print("Cannot find the definition of the initial boundary type for the tracer.")
            sys.exit()
        except ValueError:
            print("The type of initial condition is not string.")
            sys.exit()
        if self.initialConditionType == "'Homogeneous'":
            tmpTracerInit = config['InitialCondition']['TracerConc']
            tmpTracerInitList = tmpTracerInit.split(',')
            self.concTracerInit = np.asarray([], dtype = np.float64)
            if (len(tmpTracerInitList) == self.numTracers):
                for i in tmpTracerInitList:
                    self.concTracerInit = np.append(self.concTracerInit, float(i))
        elif self.initialConditionType == "'Heterogeneous'":
            #It should load a file which includes the tracer concentration in different 
            #parts of the domain.
            pass
        #Define which fluid is for transport
        try:
            self.fluidForTransport = int(config['FluidForTransport']['FluidType'])
        except KeyError:
            print("Cannot find which fluid's area for tracer transport.")
            sys.exit()
        except ValueError:
            print("The value should be an integer.")
        except:
            print("Other type of error occurs, please check .ini file.")
        #created hdf5 file to record the results
        print("Create file to save the concentration.")
        
        try:
            self.relaxationTypeTR = config['RelaxationType']['Relaxation']
        except KeyError:
            print('Cannot find the option for the relaxation type.')
        except ValueError:
            print('The type of relaxation should be string.')
        except:
            print('Other type of error occurs for the relaxation type, please check .ini file.')
        if self.relaxationTypeTR == "'MRT'":
            #Read the diffusion cofficient for collision matrix
            try:
                tmpDiffX = config['TransportMRT']['DiffusionX']
            except KeyError:
                print("Cannot find the values of diffusion in X direction")
                sys.exit()
            except:
                print("Error happened when reading the values of diffusion X in MRT.")
                raise
            tmpDiffusionX = tmpDiffX.split(',')
            diffusionX = np.asarray([], dtype = np.float64)
            if (len(tmpDiffusionX) == self.numTracers):
                for i in tmpDiffusionX:
                    diffusionX = np.append(diffusionX, float(i))
            else:
                print("The number of diffusion in x does not match the number of \
                      tracers in the domain.")
                sys.exit()
            try:
                tmpDiffY = config['TransportMRT']['DiffusionY']
            except KeyError:
                print("Cannot find the values of diffusion in Y direction")
                sys.exit()
            except:
                print("Error happened when reading the values of diffusion Y in MRT.")
                raise
            tmpDiffusionY = tmpDiffY.split(',')
            diffusionY = np.asarray([], dtype = np.float64)
            if (len(tmpDiffusionY) == self.numTracers):
                for i in tmpDiffusionY:
                    diffusionY = np.append(diffusionY, float(i))
            else:
                print("The number of diffusion in x does not match the number of \
                      tracers in the domain.")
                sys.exit()
                
            try:
                tmpDiffXY = config['TransportMRT']['DiffusionXY']
            except KeyError:
                print("Cannot find the values of diffusion in XY direction")
                sys.exit()
            except:
                print("Error happened when reading the values of diffusion XY in MRT.")
                raise
            tmpDiffusionXY = tmpDiffXY.split(',')
            diffusionXY = np.asarray([], dtype = np.float64)
            if (len(tmpDiffusionXY) == self.numTracers):
                for i in tmpDiffusionXY:
                    diffusionXY = np.append(diffusionXY, float(i))
            else:
                print("The number of diffusion in x does not match the number of \
                      tracers in the domain.")
                sys.exit()
                
            try:
                tmpDiffYX = config['TransportMRT']['DiffusionYX']
            except KeyError:
                print("Cannot find the values of diffusion in YX direction")
                sys.exit()
            except:
                print("Error happened when reading the values of diffusion YX in MRT.")
                raise
            tmpDiffusionYX = tmpDiffYX.split(',')
            diffusionYX = np.asarray([], dtype = np.float64)
            if (len(tmpDiffusionYX) == self.numTracers):
                for i in tmpDiffusionYX:
                    diffusionYX = np.append(diffusionYX, float(i))
            else:
                print("The number of diffusion in x does not match the number of \
                      tracers in the domain.")
                sys.exit()
        try:
            self.betaTracer = float(config['TransportParameters']['BetaInterface'])
        except KeyError:
            print("Cannot find the parameter for the transport around interface.")
            sys.exit()
        self.betaTracerArray = np.asarray([self.betaTracer])
        #Choose weight coefficients for the calculation
        if self.numSchemes == 5:
            self.weightsCoeffTR = np.array([1./3., 1./6., 1./6., 1./6., 1./6.], \
                                   dtype = np.float64)
            self.transportM = np.ones([5, 5], dtype = np.float64)
            self.transportM[1, 0] = 0; self.transportM[1, 2] = -1.
            self.transportM[1, 3:] = 0.
            self.transportM[2, :3] = 0.; self.transportM[2, 4] = -1.
            self.transportM[3, 0] = 4.; self.transportM[3, 1:] = -1.
            self.transportM[4, 0] = 0.; self.transportM[4, 3:] = -1.
            print(self.transportM)
#            input()
            #inverse matrix of transforming matrix T
            self.inverseTransportM = slin.inv(self.transportM)
            print(self.inverseTransportM)
            print(np.dot(self.transportM, self.inverseTransportM))
#            input()
            #collision matrix S when J_0 = 1./3. (self.diffJ in the code)
            self.relaxationS = np.zeros([self.numTracers, 5, 5], dtype = np.float64)
            for i in np.arange(self.numTracers):
                self.relaxationS[i, 1, 1] = (0.5 + 3. * diffusionX[i])
                self.relaxationS[i, 2, 2] = (0.5 + 3. * diffusionY[i])
                self.relaxationS[i, 1, 2] = 3. * diffusionXY
                self.relaxationS[i, 2, 1] = 3. * diffusionYX
            self.relaxationS[:, 0, 0] = 1.0
            self.relaxationS[:, 3, 3] = 1.0; self.relaxationS[:, 4, 4] = 1.0
            print(self.relaxationS)
#            input()
            self.relaxationSInverse = np.zeros([self.numTracers, 5, 5], dtype = np.float64)
            for i in np.arange(self.numTracers):
                self.relaxationSInverse[i] = slin.inv(self.relaxationS[i])
            #calculate the result of M^{-1}s
            self.inverseRelaxationMS = np.zeros([self.numTracers, 5, 5], dtype = np.float64)
            for i in np.arange(self.numTracers):
                self.inverseRelaxationMS[i] = -np.dot(self.inverseTransportM, \
                                       self.relaxationSInverse[i])
                
            self.relaxationSM = np.zeros([self.numTracers, 5, 5], dtype = np.float64)
            for i in np.arange(self.numTracers):
                self.relaxationSM[i, 1, 1] = (0.5 + 3. * diffusionX[i])
                self.relaxationSM[i, 2, 2] = (0.5 + 3. * diffusionY[i])
                self.relaxationSM[i, 1, 2] = 3. * diffusionXY
                self.relaxationSM[i, 2, 1] = 3. * diffusionYX
            self.relaxationSM[:, 0, 0] = 1.0
            self.relaxationSM[:, 3, 3] = 1.0; self.relaxationSM[:, 4, 4] = 1.0
            print(self.relaxationSM)
#            input()
            self.relaxationSMInverse = np.zeros([self.numTracers, 5, 5], dtype = np.float64)
            for i in np.arange(self.numTracers):
                self.relaxationSMInverse[i] = slin.inv(self.relaxationSM[i])
            #calculate the result of M^{-1}s
            self.inverseRelaxationMSM = np.zeros([self.numTracers, 5, 5], dtype = np.float64)
            for i in np.arange(self.numTracers):
                self.inverseRelaxationMSM[i] = -np.dot(self.inverseTransportM, \
                                       self.relaxationSMInverse[i])
        #MRT for D2Q9 scheme
        elif self.numSchemes == 9:
            self.weightsCoeffTR = np.zeros(9, dtype = np.float64)
            self.weightsCoeffTR[0] = 4./9.; self.weightsCoeffTR[1:5] = 1./9.
            self.weightsCoeffTR[5:] = 1./36.
            if self.relaxationTypeTR == "'MRT'":
                self.relaxationS = np.zeros([self.numTracers, 9, 9], dtype = np.float64)
                for i in np.arange(self.numTracers):
                    self.relaxationS[i, 0, 0] = 1.; self.relaxationS[i, 1, 1] = 1.
                    self.relaxationS[i, 2, 2] = 1.; self.relaxationS[i, 7, 7] = 1.
                    self.relaxationS[i, 8, 8] = 1.
                    self.relaxationS[i, 3, 3] = (0.5 + 3. * diffusionX[i])
                    self.relaxationS[i, 5, 5] = (0.5 + 3. * diffusionY[i])
                    self.relaxationS[i, 4, 4] = (0.5 + 3. * diffusionX[i])
                    self.relaxationS[i, 6, 6] = (0.5 + 3. * diffusionY[i])
                    self.relaxationS[i, 3, 5] = 3. * diffusionXY[i]
                    self.relaxationS[i, 5, 3] = 3. * diffusionYX[i]
                self.relaxationSInverse = np.zeros([self.numTracers, 9, 9], dtype = np.float64)
                for i in np.arange(self.numTracers):
                    self.relaxationSInverse[i] = slin.inv(self.relaxationS[i])
                self.inverseTransformRelaxation = np.zeros([self.numTracers, 9, 9], \
                                                           dtype = np.float64)
                for i in np.arange(self.numTracers):
                    self.inverseTransformRelaxation[i] = -np.dot(self.invTransformationM, \
                                                       self.relaxationSInverse[i])
        username = getpass.getuser()
        pathfile = '/home/'+ username + '/LBMResults/'
        file = tb.open_file(pathfile + 'ConcentrationResults.h5', 'w')
        file.create_group(file.root, 'TransportMacro', 'MacroData')
        file.close()
        input("If all the information is correct, please hit the Enter.")
        
    def initializeTransportDomain(self,):
        print("Start to initialize the concentration and distribution function in the domain.")
        self.tracerConc = np.zeros([self.numTracers, self.yDomainT, self.xDomainT])
        self.tracerDistr = np.zeros([self.numTracers, self.yDomainT, self.xDomainT, \
                                     self.numSchemes])
        self.diffJED = np.zeros([self.numTracers, 5])
        print("Initialize the parameters on diffusion.")
        for i in np.arange(5):
            if i == 0:
                self.diffJED[:, i] = self.diffJ[:]
            else:
                self.diffJED[:, i] = (1. - self.diffJ[:]) / 4.
        print(self.diffJED[0, :])
        #Initialize the velocity field for transport
        print("Initialize the array for velocity field.")
        self.velocityX = np.zeros([self.yDomainT, self.xDomainT], dtype = np.float64)
        self.velocityY = np.zeros([self.yDomainT, self.xDomainT], dtype = np.float64)
        #Initialize the concentration in the domain
        print("Initialize the concentration.")
        if self.imageExist == "'no'":
            if self.isCycles == "'no'":
                for i in np.arange(self.yDomain):
                    for j in np.arange(self.xDomain):
                        if (self.isDomain[i, j] == 1 and i <= self.yDomain - self.numBufferingLayers):
    #                    if (self.isDomain[i, j] == 1):
#                if (i < self.yDomain - 60 and i > 80):
#                        if (np.sqrt((i - self.yDomain / 2) * (i - self.yDomain / 2) + \
#                                    (j - self.xDomain / 2) * (j - self.xDomain / 2)) < 19 and \
#                                    j < self.xDomain / 2):
#                        if i < self.yDomain - self.numBufferingLayers and i > 90:
                            self.tracerConc[0, i, j] = 1.0
#            self.tracerConc[1, 180:200, 1:-1] = 1.0
            elif self.isCycles == "'yes'":
                pathFile = os.path.expanduser('~/LBMInitial/')
                if os.path.exists(pathFile):
                    dataFile = tb.open_file(pathFile + 'TransportResults.h5' )
                    for i in np.arange(self.numTracers):
                        self.tracerConc[:, :] = eval('dataFile.root.TransportMacro.TracerConcType%din%d[:, :]' % (i, self.lastStep))
                        self.velocityX[:, :] = self.physicalVX[:, :]
                        self.velocityY[:, :] = self.physicalVY[:, :]
        elif self.imageExist == "'yes'":
            if self.isCycles == "'no'":
                for i in np.arange(self.yDomainT):
                    for j in np.arange(self.xDomainT):
                        if (self.isDomain[i, j] == 1 and (i >= self.yDomainT - 10)):
                            self.tracerConc[:, i, j] = 1.0
            if self.isCycles == "'yes'":
                pathFile = os.path.expanduser('~/LBMInitial/')
                if os.path.exists(pathFile):
                    dataFile = tb.open_file(pathFile + 'TransportResults.h5' )
                    for i in np.arange(self.numTracers):
                        self.tracerConc[:, :] = eval('dataFile.root.TransportMacro.TracerConcType%din%d[:, :]' % (i, self.lastStep))
                        self.velocityX[:, :] = self.physicalVX[:, :]
                        self.velocityY[:, :] = self.physicalVY[:, :]
                        
#        Initial the concentration in the whole domain
        print("Initialize the distribution function.")
        for i in np.arange(self.yDomainT):
            for j in np.arange(self.xDomainT): 
#                for n in np.arange():
#                    self.tracerDistr[0, i, j, n] = self.tracerConc[0, i, j] * self.diffJED[0, n] 
#                for m in np.arange(5):
#                    for n in np.arange(self.numTracers):
#                        self.tracerDistr[n, i, j, m] = self.tracerConc[n, i, j] * \
#                                        (self.diffJED[n, m] + 1./2. * \
#                                         (self.unitVX[m] * self.velocityX[i, j] + \
#                                          self.unitVY[m] * self.velocityY[i, j]))
                for m in np.arange(self.numSchemes):
                    for n in np.arange(self.numTracers):
                        self.tracerDistr[n, i, j, m] = self.tracerConc[n, i, j] * \
                                                    self.weightsCoeffTR[m]
                                        
    @autojit
    def initialAreaForTransport(self, ):
        #Initialize the transport domain with the tracer concentration
        self.transportDomain = np.array(self.fluidsRhoR > 0.5)
        
    """
    update the status of nodes in the fluid domain
    """
    def updateNodeStatus(self, oldFluidOccup, newFluidOccup):
        tmpCompare = np.array(oldFluidOccup != newFluidOccup)
        tmpListLoc = np.array(np.where(tmpCompare == 1))
        tmpOldNodesChange = np.asarray([], dtype = np.int64)
        tmpNewNodesChange = np.asarray([], dtype = np.int64)
        for i in tmpListLoc:
            for j in i:
                
                if (oldFluidOccup[j] == 1):
                    print("The location in the old array is %d, %d" % (self.fluidNodes[j], j))
                    tmpOldNodesChange = np.append(tmpOldNodesChange, j)
                else:
                    print("The location in the new array is %d, %d" % (self.fluidNodes[j], j))
                    tmpNewNodesChange = np.append(tmpNewNodesChange, j)
        return tmpOldNodesChange, tmpNewNodesChange
    
    """
    Calculate the sum of concentration in the whole domain
    """
    @cuda.reduce
    def sumConcentration(self, concA1, concA2):
        return concA1 + concA2
    
    """
    Calculate the sum on the new fluid nodes
    """
    @autojit
    def sumNodesConcentration(self, listNodes):
        tmpSumConc = np.zeros(self.numTracers, dtype = np.float64)
        for i in np.arange(self.numTracers):
             for j in listNodes:
                tmpSumConc[i] += self.tracerConc[i, j]
        return tmpSumConc
    
    """
    Convert the array from 2D to 1D for porous media's structure
    """
    @autojit
    def cal1DArrayForFluidDomain(self, blockNumX, blockNumY):
        tmpFluidNodes = np.empty(self.numVoidNodes, dtype = np.int64)
        
        print("Start to fill effective fluid nodes.")
        tmpIndicesDomain = -np.ones(self.isDomain.shape, dtype = np.int64)
        tmpIndicesFN = 0
        for i in np.arange(self.yDomainT):
            for j in np.arange(self.xDomainT):
                if (self.isDomain[i, j] == 1):
                    tmpIndices = i * self.xDomainT + j
                    tmpFluidNodes[tmpIndicesFN] = tmpIndices
                    tmpIndicesDomain[i, j] = tmpIndicesFN
                    tmpIndicesFN += 1
        tmpNeighboringNodes = np.zeros(self.numVoidNodes * 4, dtype = np.int64)
        print("Start to fill the neighboring nodes for transport.")
        #copy related arrays to GPU
        deviceFluidNodes = cuda.to_device(tmpFluidNodes)
        deviceIndicesDomain = cuda.to_device(tmpIndicesDomain)
        deviceNeighboringNodes = cuda.to_device(tmpNeighboringNodes)
        grid = (blockNumX, blockNumY)
        threadPerBlock = (self.threadNum, 1)
        
        Transport2D.fillNeighboringNodesTransport[grid, threadPerBlock](self.numVoidNodes, self.xDomainT, \
                                     self.yDomainT, self.xDimension, deviceFluidNodes, \
                                     deviceIndicesDomain, deviceNeighboringNodes)
        tmpNeighboringNodes = deviceNeighboringNodes.copy_to_host()
        return tmpFluidNodes, tmpNeighboringNodes

    """
    Convert tracer's array to 1D array
    """
    def cal1DArrayTransport(self, ):
        print("Convert the 2D array of tracer to 1D.")
        self.tracerConc1D = np.zeros([self.numTracers, self.numVoidNodes], \
                                     dtype = np.float64)
        self.tracerPDF1D = np.zeros([self.numTracers, self.numVoidNodes, self.numSchemes], \
                                    dtype = np.float64)
        self.velocityTX1D = np.zeros(self.numVoidNodes, dtype = np.float64)
        self.velocityTY1D = np.zeros(self.numVoidNodes, dtype = np.float64)
        print(self.yDomainT, self.xDomainT)
        tmpDomain = np.array([i == 1 for i in self.isDomain.reshape(self.yDomainT * \
                            self.xDomainT)])
        for i in np.arange(self.numTracers):
            self.tracerConc1D[i] = self.tracerConc.reshape(self.numTracers, \
                             self.yDomainT * self.xDomainT)[i, tmpDomain]
            self.tracerPDF1D[i] = self.tracerDistr.reshape(self.numTracers, \
                            self.yDomainT * self.xDomainT, self.numSchemes)[i, tmpDomain]
        
    
    """
    Initialize the fluid domain for transport phenomena
    """
    @autojit
    def initializeDomainforTransport(self, xPosition, yPosition):
        tmpTransportDomain = np.zeros(self.numVoidNodes, dtype = np.bool)
        tmpCountNodes = 0
        for i in np.arange(self.yDomainT):
            for j in np.arange(self.xDomainT):
                if (self.isDomain[i, j] == 1):
                    if (i < yPosition and j < xPosition):
                        tmpTransportDomain[tmpCountNodes] = 1
                        tmpCountNodes += 1
        return tmpTransportDomain
    
    """
    Convert transport domain to 1D array
    """
    @autojit
    def convertTransportDomain1D(self, ):
        tmpTransportDomain1D = np.zeros(self.numVoidNodes, dtype = np.bool)
        tmpIndex = 0
        for i in np.arange(self.yDomainT):
            for j in np.arange(self.xDomainT):
                if (self.isDomain[i, j] == 1):
                    if (self.transportDomain[i, j] == 1):
                        tmpTransportDomain1D[tmpIndex] = 1
                    tmpIndex += 1
        return tmpTransportDomain1D
    
    """
    Convert 1D array of transport concentration to 1D
    """
    @autojit
    def convert1DConcTo2D(self, fluidNodes):
        tmpLoc = 0
        for i in fluidNodes:
            tmpY = int(i / self.xDomainT); tmpX = i % self.xDomainT
            self.tracerConc[:, tmpY, tmpX] = self.tracerConc1D[:, tmpLoc]
#            self.tracerDistr[:, tmpY, tmpX, :] = self.tracerPDF1D[:, tmpLoc, :]
            self.transportDomain[tmpY, tmpX] = self.transportDomain1D[tmpLoc]
            tmpLoc += 1
            
            
    """
    plot the tracer concentration
    """
    def plotConc2D(self, iStep, indexTracer):
        print("Plot the concentration in the 2D domain.")
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMResults/'
        fig = plt.imshow(self.tracerConc[indexTracer, :-2, :], origin = 'lower')
        plt.colorbar()
        plt.savefig(pathResults + "TransportConcentration%din%05d.png" % (indexTracer,\
                                                                          iStep))
        plt.close()
        print("Concentration in 2D has been saved.")
#        print(np.sum(self.tracerConc[0]))
    
    """
    plot a curve for the concentration
    """
    def plotCurve(self, iStep):
        print("Plot the concentration profile in y-direction.")
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMResults/'
        yshape, xshape = self.isDomain.shape
        plt.plot(np.arange(yshape), self.tracerConc[0, :, int(self.xDomain/2)])
        plt.xlabel('Domin size', fontsize = 13)
        plt.ylabel('Relative Concentration', fontsize = 13)
        plt.xticks(fontsize = 11)
        plt.xticks(fontsize = 11)
        plt.savefig(pathResults + "ConcentrationCurve%05d.png" % (iStep))
        plt.close()
        print("Concentration profile has been saved.")
        
    def plotTransportDomain(self, iStep):
        username = getpass.getuser()
        pathResults = '/home/' + username + '/LBMResults/'
#        print(self.transportDomain[:, 24])
        plt.imshow(self.transportDomain, origin = 'lower')
        plt.savefig(pathResults + "TransportDomain%05d.png" % iStep)
        
        plt.close()
        
    def saveConcentrationHDF5(self, index):
        """
        Save the concentration values in HDF5 file
        """
        filePath = os.path.expanduser('~/LBMResults')
        resultFile = filePath + '/ConcentrationResults.h5'
        dataFile = tb.open_file(resultFile, 'a')
        for i in sp.arange(self.numTracers):
            dataFile.create_array('/TransportMacro', 'TracerConcType%gin%g' % (i, index), \
                                  self.tracerConc[i])
        dataFile.close()
        
    def redistributeConc(self, transportDomain, neightboringNodesTR):
        nodesAtInterface = np.array([], dtype = np.int64)
        numNearInterface = 0
#        print(transportDomain.size)
#        print(self.fluidNodes.size)
#        print(transportDomain[0])
#        input()
        for i in np.arange(self.fluidNodes.size):
            tmpStart = 4 * i
            tmpCountNoTR = 0
            tmpCountYesTR = 0
            if (i > self.xDomain):
                for j in np.arange(4):
                    tmpLoc = neightboringNodesTR[tmpStart + j]
                    if tmpLoc >= 0:
                        if (transportDomain[tmpLoc] == 0 and transportDomain[i] == 1):
                            tmpCountNoTR += 1
                if tmpCountNoTR > 0:
                    numNearInterface += 1
                    nodesAtInterface = np.append(nodesAtInterface, i)
                    print(transportDomain[i])
        return numNearInterface, nodesAtInterface

    def findNewPoints(self, neighboringNodesTROld, neighboringNodesTRNew):
        nodesNew = np.array([], dtype = np.int64)
        for i in neighboringNodesTRNew:
            tmpCount = 0
            for j in neighboringNodesTROld:
                if i == j:
                    tmpCount += 1
            if tmpCount == 0:
                nodesNew = np.append(nodesNew, i)
        return nodesNew
        
        
    def runTransport2DMPMCRK(self, ):
        print("Start to run R-K color gradient lattice Boltzmann method.")
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
        
        fluidPDFRold = np.zeros([3 * (self.xDomain - 2), 9], dtype = np.float64)
        fluidPDFBold = np.zeros([3 * (self.xDomain - 2), 9], dtype = np.float64)
        deviceFluidPDFROld = cuda.to_device(fluidPDFRold)
        deviceFluidPDFBOld = cuda.to_device(fluidPDFBold)
        
        self.colorGradientX = np.zeros([self.yDomain, self.xDomain], dtype = np.float64)
        self.colorGradientY = np.zeros([self.yDomain, self.xDomain], dtype = np.float64)
        optCGX = np.zeros(self.yDomain * self.xDomain, dtype = np.float64)
        optCGY = np.zeros(self.yDomain * self.xDomain, dtype = np.float64)
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
        #Transport part
        print("Initialize tracer concentration and PDF in the domain.")
        self.numVoidNodes = self.fluidNodes.size
        self.xDomainT = self.xDomain; self.yDomainT = self.yDomain
        fluidNodes, neighboringTransportNodes = self.cal1DArrayForFluidDomain(blockNumX, \
                                                                    blockNumY)
        self.initializeTransportDomain()
        print("Initialize the area where the transport phenomenon can occur.")
        self.initialAreaForTransport()

        print("Generate the 1D array.")
        self.cal1DArrayTransport()
        self.transportDomain1D = self.convertTransportDomain1D()
        
        oldTransportArea = self.transportDomain1D
        print('For old area')
        self.weightsCoeffTR = np.array([1./3., 1./6., 1./6., 1./6., 1./6.], \
                                       dtype = np.float64)
        
        deviceTracerConc = cuda.to_device(self.tracerConc1D)
        deviceTracerConcNew = cuda.to_device(self.tracerConc1D)
        deviceTracerPDF = cuda.to_device(self.tracerPDF1D)
        deviceTracerPDFNew = cuda.to_device(self.tracerPDF1D)
        deviceNeighboringNodesTR = cuda.to_device(neighboringTransportNodes)
#        deviceFluidNodes = cuda.to_device(fluidNodes)
        deviceTransportDomain = cuda.to_device(self.transportDomain1D)
        deviceDiffJED = cuda.to_device(self.diffJED)
        deviceTauTransport = cuda.to_device(self.transportTau)
        if self.reaction == "'yes'":
            deviceReactionRate = cuda.to_device(self.reactionRate)
        deviceUnitVX = cuda.to_device(self.unitVX)
        deviceUnitVY = cuda.to_device(self.unitVY)
        deviceTransportM = cuda.to_device(self.transportM)
        deviceInverseRelaxationMS = cuda.to_device(self.inverseRelaxationMS)
        deviceWeightsTR = cuda.to_device(self.weightsCoeffTR)
        deviceInverseRelaxationMSM = cuda.to_device(self.inverseRelaxationMSM)
        
        concBoundary = np.array([0.0], dtype = np.float64)
        deviceConcBoundary = cuda.to_device(concBoundary)
        
        print("Start the simulation for multiphase-multicomponent fluid with transport.")
        criteriaFluidRho = 0.5
        tmpSteps = 0
        recordStep = 0
        #record the total mass of tracer
        tmpCalibrationConc = np.array([1.2], dtype = np.float64)
        deviceCalibrationConc = cuda.to_device(tmpCalibrationConc)
        iStep = 0; recordStep = 0
        nodesNearIFOld = np.array([], dtype = np.int64)
        
        filePath = os.path.expanduser('~/LBMResults')
        fullPath = filePath + '/NumNodesOccupied.dat'
        pointsFile = open(fullPath, 'ab')
        pointConc = filePath + '/ConcOnPoint.dat'
        concFile = open(pointConc, 'ab')
        while (iStep < self.timSteps):
            print("At the time step %d." % iStep)
            self.tracerConc1D = deviceTracerConc.copy_to_host()
            iStep += 1
            if self.boundaryTypeInlet == "'Neumann'":
                RKGPU2D.constantVelocityZHBoundaryHigherRK[grid1D, threadPerBlock1D](totalNodes, \
                                        self.xDomain, self.yDomain, self.xDimension, \
                                        self.velocityYR, self.velocityYB, deviceFluidNodes, \
                                        deviceFluidRhoR, deviceFluidRhoB, deviceFluidPDFR, \
                                        deviceFluidPDFB)
                RKGPU2D.ghostPointsConstantVelocityRK[grid1D, threadPerBlock1D](totalNodes, \
                                        self.xDomain, self.yDomain, self.xDimension, deviceFluidNodes, \
                                        deviceNeighboringNodes, deviceFluidRhoR, \
                                        deviceFluidRhoB, deviceFluidPDFR, deviceFluidPDFB)
            if self.boundaryTypeInlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                RKGPU2D.calConstPressureInletGPU[grid1D, threadPerBlock1D](totalNodes, \
                                self.xDomain, self.yDomain, self.xDimension, self.densityRhoBH, \
                                self.densityRhoRH, deviceFluidNodes, deviceFluidRhoB, \
                                deviceFluidRhoR, deviceFluidPDFB, \
                                deviceFluidPDFR)
                RKGPU2D.ghostPointsConstPressureInletRK[grid1D, threadPerBlock1D](totalNodes, \
                                self.xDomain, self.yDomain, self.xDimension, deviceFluidNodes, \
                                deviceNeighboringNodes, deviceFluidRhoR, deviceFluidRhoB, \
                                deviceFluidPDFR, deviceFluidPDFB)
            print("Calculate the macro-density of the fluids")
            RKGPU2D.copyFluidPDFLastStep[grid1D, threadPerBlock1D](totalNodes, \
                                        self.xDomain, self.xDimension, deviceFluidNodes, \
                                        deviceFluidPDFR, deviceFluidPDFB, \
                                        deviceFluidPDFROld, deviceFluidPDFBOld)
            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
                                          self.xDimension, deviceFluidPDFR, \
                                          deviceFluidPDFB, deviceFluidRhoR, \
                                          deviceFluidRhoB)
            print("Calculate the macroscale velocity of the fluids.")
            RKGPU2D.calPhysicalVelocityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, self.xDomain, \
                                        self.xDimension, deviceFluidNodes, deviceFluidPDFR, deviceFluidPDFB, \
                                        deviceFluidRhoR, deviceFluidRhoB, devicePhysicalVX, \
                                        devicePhysicalVY)
            RKGPU2D.calPhaseFieldPhi[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                            deviceFluidRhoR, deviceFluidRhoB, devicePhiValue)
            RKGPU2D.calNeumannPhiOutlet[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                            self.xDomain, deviceFluidNodes, deviceNeighboringNodes, devicePhiValue) 
            if ((iStep - 1) % self.timeInterval == 0):
                Transport2D.calUpdateDistributionGPU[grid1D, threadPerBlock1D](totalNodes, \
                                        self.xDimension, criteriaFluidRho, deviceFluidRhoR, \
                                        deviceTransportDomain)
                self.tracerConc1D = deviceTracerConc.copy_to_host()

                self.tracerPDF1D = deviceTracerPDF.copy_to_host()
                self.transportDomain1D = deviceTransportDomain.copy_to_host()
                self.convert1DConcTo2D(fluidNodes)
                for k in np.arange(self.numTracers):
                    self.plotConc2D(recordStep, k)
                self.plotTransportDomain(recordStep)
#                self.plotCurve(recordStep)
                self.saveConcentrationHDF5(recordStep)

            #RK color gradient part
            if ((iStep - 1) % self.timeInterval == 0):
                print("Copy data to host for saving and plotting.")
                self.optFluidRhoR = deviceFluidRhoR.copy_to_host()
                self.optFluidRhoB = deviceFluidRhoB.copy_to_host()
                self.optMacroVelocityX = devicePhysicalVX.copy_to_host()
                self.optMacroVelocityY = devicePhysicalVY.copy_to_host()
                self.optFluidPDFR = deviceFluidPDFR.copy_to_host()
                self.optFluidPDFB = deviceFluidPDFB.copy_to_host()
                self.convertOptTo2D()
                self.resultInHDF5(recordStep)
                self.plotDensityDistributionOPT(recordStep)
                recordStep += 1
            if self.relaxationType == "'MRT'":
                print("MRT collision process.")
                Transport2D.calCollisionTransportLinearEqlMRTGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                    self.xDimension, self.numTracers, deviceUnitVX, \
                                    deviceUnitVY, devicePhysicalVX, devicePhysicalVY, \
                                    deviceTracerConc, deviceTracerPDF, deviceTransportM, \
                                    deviceInverseRelaxationMS, deviceWeightsTR)
#                Transport2D.calCollisionQ9[grid1D, threadPerBlock1D](self.numVoidNodes, \
#                                          self.xDimension, self.numTracers, deviceUnitEX, \
#                                          deviceUnitEY, devicePhysicalVX, devicePhysicalVY, deviceTauTransport,\
#                                          deviceTracerConc, deviceTracerPDF, deviceWeightsCoeff)
#
#            if (self.reaction == "'yes'"):
#                print("Reaction(s) between tracers exists.")
#                Transport2D.calReactionTracersGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
#                                        self.numTracers, self.xDimension, deviceReactionRate, \
#                                        deviceDiffJED, deviceTracerConc, deviceTracerPDF)
#            print("Run the free flux boundary on outlet.")
#            Transport2D.calFreeConcBoundary1[grid1D, threadPerBlock1D](self.numVoidNodes, \
#                                self.numTracers, self.xDomain, self.xDimension, \
#                                deviceFluidNodes, deviceNeighboringNodesTR, \
#                                deviceTracerConc, deviceTracerPDF)
#            Transport2D.calFreeConcBoundary2[grid1D, threadPerBlock1D](self.numVoidNodes, \
#                                self.numTracers, self.xDomain, self.xDimension, \
#                                deviceFluidNodes, deviceNeighboringNodesTR, \
#                                deviceTracerConc, deviceTracerPDF)
            Transport2D.calFreeConcBoundary3[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                self.numTracers, self.xDomain, self.xDimension, \
                                deviceFluidNodes, deviceNeighboringNodesTR, \
                                deviceTracerConc, deviceTracerPDF)
            print("Calculate streaming process.")
            Transport2D.calStreamingTransportGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                    self.xDimension, self.numTracers, deviceNeighboringNodesTR,\
                                    deviceTracerPDF, deviceTracerPDFNew)
            Transport2D.calStreamingTransport2GPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                     self.numTracers, self.xDimension, deviceTracerPDFNew, \
                                     deviceTracerPDF)
#            Transport2D.calStreaming1GPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
#                                        self.numTracers, self.xDimension, deviceFluidNodes,\
#                                        deviceNeighboringNodes, deviceTracerPDF, \
#                                        deviceTracerPDFNew)
#            Transport2D.calStreaming2GPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
#                                        self.numTracers, self.xDimension, deviceTracerPDFNew, \
#                                        deviceTracerPDF)
#
#            #For boundary conditions
            print("Implement the upper boundary.")
            Transport2D.calInamuroConstConcBoundary[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                       self.xDimension, self.numTracers, self.yDomain, \
                                       self.xDomain, deviceFluidNodes, \
                                       deviceNeighboringNodesTR, deviceConcBoundary, \
                                       deviceWeightsTR, deviceTracerPDF)
#            
            #Concentration calculation
            print("Calculate the concentration on each lattice.")
            Transport2D.calConcentrationGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                               self.numTracers, self.xDimension, deviceTracerConc, \
                               deviceTracerPDF)


            if self.relaxationType == "'SRT'":
#                RKGPU2D.calRKCollision1GPU2DSRT[grid1D, threadPerBlock1D](totalNodes, \
#                                            self.xDimension, self.deltaValue, self.tauR, \
#                                            self.tauB, deviceUnitEX, deviceUnitEY, \
#                                            deviceConstCR, deviceConstCB, deviceWeightsCoeff, \
#                                            devicePhysicalVX, devicePhysicalVY, deviceFluidRhoR, \
#                                            deviceFluidRhoB, deviceFluidPDFR, deviceFluidPDFB)
                RKGPU2D.calRKCollision1GPU2DSRTNew[grid1D, threadPerBlock1D](totalNodes, \
                                            self.xDimension, self.deltaValue, self.tauR, \
                                            self.tauB, deviceUnitEX, deviceUnitEY, \
                                            deviceConstCR, deviceConstCB, deviceWeightsCoeff, \
                                            devicePhysicalVX, devicePhysicalVY, deviceFluidRhoR, \
                                            deviceFluidRhoB, devicePhiValue, \
                                            deviceFluidPDFR, deviceFluidPDFB)
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
                                            self.xDimension, self.deltaValue, self.tauR, \
                                            self.tauB, self.bodyFX, self.bodyFY, deviceUnitEX, deviceUnitEY, \
                                            deviceConstCR, deviceConstCB, deviceWeightsCoeff, \
                                            devicePhysicalVX, devicePhysicalVY, deviceFluidRhoR, \
                                            deviceFluidRhoB, devicePhiValue, deviceFluidPDFR, \
                                            deviceFluidPDFB, deviceTransformationM, \
                                            deviceTransformationIM, deviceCollisionM)
            print("Calculate the second collision and re-coloring parts.")
#            RKGPU2D.calRKCollision23GPU[grid1D, threadPerBlock1D](totalNodes, \
#                                        self.xDimension, self.betaThickness, self.AkR, \
#                                        self.AkB, self.solidRhoR, self.solidRhoB, \
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
                                        deviceConstBNew, deviceWeightsCoeff, deviceUnitEX, \
                                        deviceUnitEY, deviceScheme, deviceFluidRhoR, \
                                        deviceFluidRhoB, devicePhiValue, deviceConstCR, \
                                        deviceConstCB, deviceFluidPDFR, deviceFluidPDFB, \
                                        deviceCGX, deviceCGY)

            print("Start the first step of streaming for both fluids.")
#            RKGPU2D.copyFluidPDFRecoverOutlet[grid1D, threadPerBlock1D](totalNodes, \
#                                        self.xDomain, self.xDimension, deviceFluidNodes, \
#                                        deviceFluidPDFR, deviceFluidPDFB, \
#                                        deviceFluidPDFROld, deviceFluidPDFBOld)
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
            if (self.boundaryTypeInlet == "'Periodic'" and self.boundaryTypeOutlet == "'Periodic'"):
                RKGPU2D.calModifiedPeriodicBoundary[grid1D, threadPerBlock1D](totalNodes, \
                                                   self.xDomain, self.yDomain, \
                                                   self.xDimension, deviceFluidNodes, \
                                                   deviceNeighboringNodes, deviceFluidPDFR, \
                                                   deviceFluidPDFB)
            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
                                          self.xDimension, deviceFluidPDFR, \
                                          deviceFluidPDFB, deviceFluidRhoR, \
                                          deviceFluidRhoB)
            RKGPU2D.calPhysicalVelocityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, self.xDomain, \
                                        self.xDimension, deviceFluidNodes, deviceFluidPDFR, deviceFluidPDFB, \
                                        deviceFluidRhoR, deviceFluidRhoB, devicePhysicalVX, \
                                        devicePhysicalVY)
            
            if self.boundaryTypeOutlet == "'Neumann'":
                print("Boundary at the outlet.")
                RKGPU2D.convectiveOutletGPU[grid1D, threadPerBlock1D](totalNodes, self.xDomain, \
                                   self.xDimension, deviceFluidNodes, deviceNeighboringNodes, deviceFluidPDFR, \
                                   deviceFluidPDFB, deviceFluidRhoR, deviceFluidRhoB)
                RKGPU2D.convectiveOutletGhost2GPU[grid1D, threadPerBlock1D](totalNodes, \
                                self.xDomain, self.xDimension, deviceFluidNodes, deviceNeighboringNodes, \
                                deviceFluidPDFR, deviceFluidPDFB, deviceFluidRhoR, \
                                deviceFluidRhoB)
                RKGPU2D.convectiveOutletGhost3GPU[grid1D, threadPerBlock1D](totalNodes, \
                                self.xDomain, self.xDimension, deviceFluidNodes, deviceNeighboringNodes, \
                                deviceFluidPDFR, deviceFluidPDFB, deviceFluidRhoR, \
                                deviceFluidRhoB)
#                RKGPU2D.convectiveAverageBoundaryGPU[grid1D, threadPerBlock1D](totalNodes, \
#                                self.xDomain, self.xDimension, deviceFluidNodes, deviceNeighboringNodes,\
#                                devicePhysicalVY, deviceFluidPDFR, deviceFluidPDFB, \
#                                deviceFluidPDFROld, deviceFluidPDFBOld)
#                RKGPU2D.convectiveAverageBoundaryGPU2[grid1D, threadPerBlock1D](totalNodes, \
#                                self.xDomain, self.xDimension, deviceFluidNodes,  deviceNeighboringNodes,\
#                                devicePhysicalVY, deviceFluidPDFR, deviceFluidPDFB, \
#                                deviceFluidPDFROld, deviceFluidPDFBOld)
#                RKGPU2D.convectiveAverageBoundaryGPU3[grid1D, threadPerBlock1D](totalNodes, \
#                                self.xDomain, self.xDimension, deviceFluidNodes,  deviceNeighboringNodes,\
#                                devicePhysicalVY, deviceFluidPDFR, deviceFluidPDFB, \
#                                deviceFluidPDFROld, deviceFluidPDFBOld)
            elif self.boundaryTypeOutlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                RKGPU2D.calConstPressureLowerGPU[grid1D, threadPerBlock1D](totalNodes, \
                                self.xDomain, self.xDimension, self.densityRhoBL, \
                                self.densityRhoRL, deviceFluidNodes, deviceFluidRhoB, \
                                deviceFluidRhoR, deviceFluidPDFB, \
                                deviceFluidPDFR)
                RKGPU2D.ghostPointsConstPressureLowerRK[grid1D, threadPerBlock1D](totalNodes, \
                                self.xDomain, self.xDimension, deviceFluidNodes, \
                                deviceNeighboringNodes, deviceFluidRhoR, deviceFluidRhoB, \
                                deviceFluidPDFR, deviceFluidPDFB)
#        tracerFile.close()
        pointsFile.close()
        concFile.close()
        
    def runTransport2DMPMCRKNew(self,):
        print("Start to run R-K color gradient lattice Boltzmann method.")
        self.initializeDomainBorder()
        self.initializeDomainCondition()
        print("Finish initialize the original simulated domain.")
        self.optimizeFluidandSolidArray()
        self.numColorSolid = 0; self.numWettingFluid = 0
        if self.wettingSolidNodes.size > 0:
            self.sortOutFluidNodesToSolid()
            print(self.fluidNodesWithSolidOriginal.size)
            print(self.fluidNodesWithSolidGPU.size)
            self.numWettingFluid = self.fluidNodesWithSolidGPU.size
            self.calVectorNormaltoSolid()
            self.numColorSolid = self.wettingSolidNodes.size
        self.solidColorValue = np.zeros(self.numColorSolid, dtype = np.float64)
        
        print("Start to set up arrays for the device.")
        deviceFluidRhoR = cuda.to_device(self.optFluidRhoR)
        deviceFluidRhoB = cuda.to_device(self.optFluidRhoB)
        deviceFluidPDFR = cuda.to_device(self.optFluidPDFR)
        deviceFluidPDFB = cuda.to_device(self.optFluidPDFB)
        deviceFluidPDFRNew = cuda.device_array_like(self.optFluidPDFR)
        deviceFluidPDFBNew = cuda.device_array_like(self.optFluidPDFB)
        devicePhysicalVX = cuda.to_device(self.optMacroVelocityX)
        devicePhysicalVY = cuda.to_device(self.optMacroVelocityY)
        
        deviceColorValue = cuda.device_array_like(self.optFluidRhoB)
        colorValue = np.array(self.optFluidRhoB.size, dtype = np.float64)
        colorValueOld = np.array(self.optFluidRhoB, dtype = np.float64)
        
        deviceFluidPDFTotal = cuda.device_array_like(self.optFluidPDFR)
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
            
        
        totalNodes = self.fluidNodes.size
        blockNumX = int(self.xDimension / self.threadNum)
        blockNumY = math.ceil(self.fluidNodes.size / self.xDimension)
        threadPerBlock1D = (self.threadNum, 1)
        grid1D = (blockNumX, blockNumY)
        
        print("Initialize tracer concentration and PDF in the domain.")
        self.numVoidNodes = self.fluidNodes.size
        self.xDomainT = self.xDomain; self.yDomainT = self.yDomain
        fluidNodes, neighboringTransportNodes = self.cal1DArrayForFluidDomain(blockNumX, \
                                                                    blockNumY)
        self.initializeTransportDomain()
        print("Initialize the area where the transport phenomenon can occur.")
        self.initialAreaForTransport()

        print("Generate the 1D array.")
        self.cal1DArrayTransport()
        self.transportDomain1D = self.convertTransportDomain1D()
        
        oldTransportArea = self.transportDomain1D
        print('For old area')
        
        deviceTracerConc = cuda.to_device(self.tracerConc1D)
        deviceTracerConcNew = cuda.to_device(self.tracerConc1D)
        deviceTracerPDF = cuda.to_device(self.tracerPDF1D)
        deviceTracerPDFNew = cuda.to_device(self.tracerPDF1D)
        deviceNeighboringNodesTR = cuda.to_device(neighboringTransportNodes)
#        deviceFluidNodes = cuda.to_device(fluidNodes)
#        deviceTransportDomain = cuda.to_device(self.transportDomain1D)
        deviceDiffJED = cuda.to_device(self.diffJED)
        deviceTauTransport = cuda.to_device(self.transportTau)
        deviceWeightsTR = cuda.to_device(self.weightsCoeffTR)
        if self.reaction == "'yes'":
            deviceReactionRate = cuda.to_device(self.reactionRate)
        deviceUnitVX = cuda.to_device(self.unitVX)
        deviceUnitVY = cuda.to_device(self.unitVY)
        if self.numSchemes == 5 and self.relaxationTypeTR == "'MRT'":
            deviceTransportM = cuda.to_device(self.transportM)
            deviceInverseRelaxationMS = cuda.to_device(self.inverseRelaxationMS)
            deviceInverseRelaxationMSM = cuda.to_device(self.inverseRelaxationMSM)
        elif self.numSchemes == 9 and self.relaxationTypeTR == "'MRT'":
            deviceInverseRelaxationMS = cuda.to_device(self.inverseTransformRelaxation)
        
        self.valueTransportDomain = np.ones(self.tracerConc1D.size, dtype = np.float64)
        deviceValueTransportDomain = cuda.to_device(self.valueTransportDomain)
        deviceBetaTracer = cuda.to_device(self.betaTracerArray)
        concBoundary = np.array([1.0], dtype = np.float64)
        deviceConcBoundary = cuda.to_device(concBoundary)
        
        print("Start the simulation for multiphase-multicomponent fluid with transport.")
        self.criteriaFluidRho = 0.5
        tmpSteps = 0
        recordStep = 0
        #record the total mass of tracer
        iStep = 0; recordStep = 0
        nodesNearIFOld = np.array([], dtype = np.int64)
        
        filePath = os.path.expanduser('~/LBMResults')
        fullPath = filePath + '/NumNodesOccupied.dat'
        pointsFile = open(fullPath, 'ab')
        pointConc = filePath + '/ConcOnPoint.dat'
        concFile = open(pointConc, 'ab')
        while (iStep < self.timeSteps):
#        while (stopStandard > 1.0e-10):
            print("At the time step %d." % iStep)
            self.tracerConc1D = deviceTracerConc.copy_to_host()
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
            RKGPU2D.calTotalFluidPDF[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                            deviceFluidPDFR, deviceFluidPDFB, deviceFluidPDFTotal)
            if self.boundaryTypeOutlet == "'Convective'":
                print("Free boundary at the outlet.")
                RKCG2DBC.convectiveOutletGPU[grid1D, threadPerBlock1D](totalNodes, self.xDomain, \
                                                                      self.xDimension, deviceFluidNodes,
                                                                      deviceNeighboringNodes, deviceFluidPDFR, \
                                                                      deviceFluidPDFB, deviceFluidRhoR, deviceFluidRhoB)
                RKCG2DBC.convectiveOutletGhost2GPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                            self.xDomain, self.xDimension, deviceFluidNodes,
                                                                            deviceNeighboringNodes, \
                                                                            deviceFluidPDFR, deviceFluidPDFB,
                                                                            deviceFluidRhoR, \
                                                                            deviceFluidRhoB)
                RKCG2DBC.convectiveOutletGhost3GPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                            self.xDomain, self.xDimension, deviceFluidNodes,
                                                                            deviceNeighboringNodes, \
                                                                            deviceFluidPDFR, deviceFluidPDFB,
                                                                            deviceFluidRhoR, \
                                                                            deviceFluidRhoB)
            elif self.boundaryTypeOutlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                #                RKGPU2D.calConstPressureLowerGPU[grid1D, threadPerBlock1D](totalNodes, \
                #                                self.xDomain, self.xDimension, self.densityRhoBL, \
                #                                self.densityRhoRL, deviceFluidNodes, deviceFluidRhoB, \
                #                                deviceFluidRhoR, deviceFluidPDFB, \
                #                                deviceFluidPDFR)
                totalPressure = self.densityRhoBL + self.densityRhoRL
                RKCG2DBC.calConstPressureLowerGPUTotal[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDomain, self.xDimension,
                                                                                totalPressure, deviceFluidNodes, \
                                                                                deviceFluidPDFTotal, devicePhysicalVY,
                                                                                deviceFluidRhoR, \
                                                                                deviceFluidRhoB, deviceFluidPDFR,
                                                                                deviceFluidPDFB)
                RKCG2DBC.ghostPointsConstPressureLowerRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                  self.xDomain, self.xDimension,
                                                                                  deviceFluidNodes, \
                                                                                  deviceNeighboringNodes, deviceFluidRhoR,
                                                                                  deviceFluidRhoB, \
                                                                                  deviceFluidPDFR, deviceFluidPDFB)

            # RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
            #                                                          self.xDimension, deviceFluidPDFR, \
            #                                                          deviceFluidPDFB, deviceFluidRhoR, \
            #                                                          deviceFluidRhoB)

            if self.boundaryTypeInlet == "'Neumann'":
                #                RKGPU2D.constantVelocityZHBoundaryHigherRK[grid1D, threadPerBlock1D](totalNodes, \
                #                                        self.xDomain, self.yDomain, self.xDimension, \
                #                                        self.velocityYR, self.velocityYB, deviceFluidNodes, \
                #                                        deviceFluidRhoR, deviceFluidRhoB, deviceFluidPDFR, \
                #                                        deviceFluidPDFB)
                # RKGPU2D.constantVelocityZHBoundaryHigherNewRK[grid1D, threadPerBlock1D](totalNodes, \
                #                                                                         self.xDomain, self.yDomain,
                #                                                                         self.xDimension, \
                #                                                                         self.velocityYR, self.velocityYB,
                #                                                                         deviceFluidNodes, \
                #                                                                         deviceNeighboringNodes,
                #                                                                         deviceFluidRhoR, \
                #                                                                         deviceFluidRhoB, deviceFluidPDFR, \
                #                                                                         deviceFluidPDFB)
                specificVY = self.velocityYB + self.velocityYR
                RKCG2DBC.constantTotalVelocityInlet[grid1D, threadPerBlock1D](totalNodes, \
                                        self.xDomain, self.yDomain, self.xDimension, \
                                        specificVY, deviceFluidNodes, deviceNeighboringNodes, \
                                        deviceFluidRhoR, deviceFluidRhoB, deviceFluidPDFR, \
                                        deviceFluidPDFB, deviceFluidPDFTotal, devicePhysicalVY)
                RKCG2DBC.ghostPointsConstantVelocityRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                self.xDomain, self.yDomain, self.xDimension,
                                                                                deviceFluidNodes, \
                                                                                deviceNeighboringNodes, deviceFluidRhoR, \
                                                                                deviceFluidRhoB, deviceFluidPDFR,
                                                                                deviceFluidPDFB, \
                                                                                deviceForceX, deviceForceY)
            if self.boundaryTypeInlet == "'Dirichlet'":
                print("Use constant pressure/density boundary.")
                RKCG2DBC.calConstPressureInletGPU[grid1D, threadPerBlock1D](totalNodes, \
                                                                           self.xDomain, self.yDomain, self.xDimension,
                                                                           self.densityRhoBH, \
                                                                           self.densityRhoRH, deviceFluidNodes,
                                                                           deviceFluidRhoB, \
                                                                           deviceFluidRhoR, deviceFluidPDFB, \
                                                                           deviceFluidPDFR)
                RKCG2DBC.ghostPointsConstPressureInletRK[grid1D, threadPerBlock1D](totalNodes, \
                                                                                  self.xDomain, self.yDomain,
                                                                                  self.xDimension, deviceFluidNodes, \
                                                                                  deviceNeighboringNodes, deviceFluidRhoR,
                                                                                  deviceFluidRhoB, \
                                                                                  deviceFluidPDFR, deviceFluidPDFB)
                
            print("Calculate the macro-density of the fluids")
            RKGPU2D.calTotalFluidPDF[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                            deviceFluidPDFR, deviceFluidPDFB, deviceFluidPDFTotal)
            RKGPU2D.calMacroDensityRKGPU2D[grid1D, threadPerBlock1D](totalNodes, \
                                          self.xDimension, deviceFluidPDFR, \
                                          deviceFluidPDFB, deviceFluidRhoR, \
                                          deviceFluidRhoB)
            print("Calcuate the macroscopic velocity.")
            RKGPU2D.calPhysicalVelocityRKGPU2DM[grid1D, threadPerBlock1D](totalNodes, \
                            self.xDimension, deviceFluidPDFTotal, deviceFluidRhoR, \
                            deviceFluidRhoB, devicePhysicalVX, devicePhysicalVY, \
                            deviceForceX, deviceForceY)
            print("Calculate the color values on each fluid nodes.")
            RKGPU2D.calPhaseFieldPhi[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                            deviceFluidRhoR, deviceFluidRhoB, deviceColorValue)

            #RK color gradient part
            if ((iStep - 1) % self.timeInterval == 0):
                print("Copy data to host for saving and plotting.")
                self.optFluidRhoR = deviceFluidRhoR.copy_to_host()
                self.optFluidRhoB = deviceFluidRhoB.copy_to_host()
                self.optMacroVelocityX = devicePhysicalVX.copy_to_host()
                self.optMacroVelocityY = devicePhysicalVY.copy_to_host()
                self.optFluidPDFR = deviceFluidPDFR.copy_to_host()
                self.optFluidPDFB = deviceFluidPDFB.copy_to_host()
                self.convertOptTo2D()
                self.resultInHDF5(recordStep)
                self.plotDensityDistributionOPT(recordStep)
                recordStep += 1
#                input()
#
            if self.wettingSolidNodes.size > 0:
                print("Calculate the color value on the solid nodes neighboring to fluid.")
                RKGPU2D.calColorValueOnSolid[grid1D, threadPerBlock1D](self.numColorSolid, \
                                            self.xDimension, deviceNeighboringWettingSolid, \
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
                                            self.xDimension, self.cosTheta, self.sinTheta, deviceFluidNodesWithSolid, \
                                            deviceUnitNsx, deviceUnitNsy, deviceGradientX, deviceGradientY)
                elif self.wettingType == 2:
                    RKGPU2D.updateColorGradientOnWettingNew[grid1D, threadPerBlock1D](self.numWettingFluid, \
                                            self.xDimension, self.cosTheta, self.sinTheta, deviceFluidNodesWithSolid, \
                                            deviceUnitNsx, deviceUnitNsy, deviceGradientX, deviceGradientY) 
                    
            print("Calculate collision process.")
            Transport2D.calValueTransportDomain[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                               self.xDimension, self.criteriaFluidRho, \
                                               deviceValueTransportDomain, deviceFluidRhoR)
            if self.numSchemes == 5:
                if self.relaxationTypeTR == "'MRT'":
                    print("MRT collision process.")
                    Transport2D.calCollisionTransportLinearEqlMRTGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                        self.xDimension, self.numTracers, deviceUnitVX, \
                                        deviceUnitVY, devicePhysicalVX, devicePhysicalVY, \
                                        deviceTracerConc, deviceTracerPDF, deviceTransportM, \
                                        deviceInverseRelaxationMS, deviceWeightsTR)

                Transport2D.calTransportWithInterfaceD2Q5[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                    self.xDimension, self.numTracers, deviceBetaTracer, deviceValueTransportDomain, \
                                    deviceUnitEX, deviceUnitEY, deviceGradientX, deviceGradientY, \
                                    deviceWeightsTR, deviceTracerConc, deviceTracerPDF)

               if (self.reaction == "'yes'"):
                   print("Reaction(s) between tracers exists.")
                   Transport2D.calReactionTracersGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                           self.numTracers, self.xDimension, deviceReactionRate, \
                                           deviceDiffJED, deviceTracerConc, deviceTracerPDF)
                if self.typeTracerOutletBoundary == "'Freeflow'":
                    print("Implement the free flow boundary condition on the outlet.")
                    Transport2D.calFreeConcBoundary3[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                        self.numTracers, self.xDomain, self.xDimension, \
                                        deviceFluidNodes, deviceNeighboringNodesTR, \
                                        deviceTracerConc, deviceTracerPDF)
                print("Calculate streaming process.")
                Transport2D.calStreamingTransportGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                        self.xDimension, self.numTracers, deviceNeighboringNodesTR,\
                                        deviceTracerPDF, deviceTracerPDFNew)
                Transport2D.calStreamingTransport2GPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                         self.numTracers, self.xDimension, deviceTracerPDFNew, \
                                         deviceTracerPDF)
    
                #For boundary conditions
                if self.typeTracerInletBoundary == "'Dirichlet'":
                    print("Implement the upper boundary.")
                    Transport2D.calInamuroConstConcBoundary[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                               self.xDimension, self.numTracers, self.yDomain, \
                                               self.xDomain, deviceFluidNodes, \
                                               deviceNeighboringNodesTR, deviceConcBoundary, \
                                               deviceWeightsTR, deviceTracerPDF)
            elif self.numSchemes == 9:
                if self.relaxationTypeTR == "'SRT'":
                    print("Start the SRT collision process.")
                    Transport2D.calCollisionQ9[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                              self.xDimension, self.numTracers, deviceUnitEX, \
                                              deviceUnitEY, devicePhysicalVX, devicePhysicalVY, 
                                              deviceTauTransport, deviceTracerConc, \
                                              deviceTracerPDF, deviceWeightsTR)
                if self.relaxationTypeTR == "'MRT'":
                    print("Start MRT collision process.")
                    Transport2D.calCollisionTransportLinearEqlMRTGPUD2Q9[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                        self.xDimension, self.numTracers, deviceUnitVX, \
                                        deviceUnitVY, devicePhysicalVX, devicePhysicalVY, \
                                        deviceTracerConc, deviceTracerPDF, deviceTransformationM, \
                                        deviceInverseRelaxationMS, deviceWeightsTR)
                    
                print("Calculation with the effect of interface.")
                Transport2D.calTransportWithInterfaceD2Q9[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                    self.xDimension, self.numTracers, deviceBetaTracer, \
                                    deviceValueTransportDomain, deviceUnitEX, deviceUnitEY, \
                                    deviceGradientX, deviceGradientY, deviceWeightsTR, \
                                    deviceTracerConc, deviceTracerPDF)
                print("Star the streaming process.")
                Transport2D.calStreaming1GPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                    self.numTracers, self.xDimension, deviceFluidNodes, \
                                    deviceNeighboringNodes, deviceTracerPDF, \
                                    deviceTracerPDFNew)
                Transport2D.calStreaming2GPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                                    self.numTracers, self.xDimension, deviceTracerPDFNew, \
                                    deviceTracerPDF)
                
            #Concentration calculation
            print("Calculate the concentration on each lattice.")
            Transport2D.calConcentrationGPU[grid1D, threadPerBlock1D](self.numVoidNodes, \
                               self.numTracers, self.xDimension, self.numSchemes, \
                               deviceTracerConc, deviceTracerPDF)
            
            if ((iStep - 1) % self.timeInterval== 0):
                self.tracerConc1D = deviceTracerConc.copy_to_host()

                self.tracerPDF1D = deviceTracerPDF.copy_to_host()
                self.transportDomain1D = deviceValueTransportDomain.copy_to_host()
                self.convert1DConcTo2D(fluidNodes)
                for k in np.arange(self.numTracers):
                    self.plotConc2D(recordStep, k)
                self.plotTransportDomain(recordStep)
#                self.plotCurve(recordStep)
                self.saveConcentrationHDF5(recordStep)
#                input()
            print("Calculate the force values in the domain")
            if self.wettingType == 1:
                RKGPU2D.calForceTermInColorGradient2D[grid1D, threadPerBlock1D](totalNodes, \
                                        self.xDimension, self.surfaceTension, deviceNeighboringNodes, \
                                        deviceWeightsCoeff, deviceUnitEX, deviceUnitEY, \
                                        deviceGradientX, deviceGradientY, deviceForceX, \
                                        deviceForceY, deviceKValue)
            elif self.wettingType == 2:
                RKGPU2D.calForceTermInColorGradientNew2D[grid1D, threadPerBlock1D](totalNodes, \
                                        self.xDimension, self.surfaceTension, deviceNeighboringNodes, \
                                        deviceWeightsCoeff, deviceUnitEX, deviceUnitEY, \
                                        deviceGradientX, deviceGradientY, deviceForceX, \
                                        deviceForceY, deviceKValue)
            print("Calculate the single phase collision for total distribution function.")
            if self.relaxationType == "'SRT'":
                RKGPU2D.calRKCollision1TotalGPU2DSRTM[grid1D, threadPerBlock1D](totalNodes, \
                                                self.xDimension, self.tauCalculation, self.tauR, self.tauB, \
                                                self.deltaValue, deviceUnitEX, deviceUnitEY, \
                                                deviceWeightsCoeff, devicePhysicalVX, \
                                                devicePhysicalVY, deviceFluidRhoR, deviceFluidRhoB, \
                                                deviceColorValue, deviceFluidPDFTotal)
                print("Calculate the force perturbation for the total distribution function.")
                RKGPU2D.calPerturbationFromForce2D[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                          self.tauCalculation, self.tauR, self.tauB, self.deltaValue, \
                                          deviceWeightsCoeff, deviceUnitEX, \
                                          deviceUnitEY, devicePhysicalVX, devicePhysicalVY, \
                                          deviceForceX, deviceForceY, deviceColorValue, \
                                          deviceFluidPDFTotal, deviceFluidRhoR, deviceFluidRhoB)
            if self.relaxationType == "'MRT'":
                RKGPU2D.calRKCollision1TotalGPU2DMRTM[grid1D, threadPerBlock1D](totalNodes, \
                                            self.xDimension, self.tauCalculation, self.tauR, self.tauB, \
                                            self.deltaValue, deviceUnitEX, deviceUnitEY, deviceWeightsCoeff, \
                                            devicePhysicalVX, devicePhysicalVY, \
                                            deviceFluidRhoR, deviceFluidRhoB, \
                                            deviceColorValue, deviceFluidPDFTotal, \
                                            deviceTransformationM, deviceTransformationIM, \
                                            deviceCollisionM)
                RKGPU2D.calPerturbationFromForce2DMRT[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                          self.tauCalculation, self.tauR, self.tauB, self.deltaValue, \
                                          deviceWeightsCoeff, deviceUnitEX, \
                                          deviceUnitEY, devicePhysicalVX, devicePhysicalVY, \
                                          deviceForceX, deviceForceY, deviceColorValue, \
                                          deviceFluidPDFTotal, deviceTransformationM, \
                                          deviceTransformationIM, deviceCollisionM, \
                                          deviceFluidRhoR, deviceFluidRhoB)

            print("Recoloring both fluids in the system.")
            RKGPU2D.calRecoloringProcessM[grid1D, threadPerBlock1D](totalNodes, self.xDimension, \
                                    self.betaThickness, deviceWeightsCoeff, deviceFluidRhoR, \
                                    deviceFluidRhoB, deviceUnitEX, deviceUnitEY, \
                                    deviceGradientX, deviceGradientY, deviceFluidPDFR, \
                                    deviceFluidPDFB, deviceFluidPDFTotal)
