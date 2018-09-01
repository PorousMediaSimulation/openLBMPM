"""
Class: BasicD2Q9
Usage: includes basic algorithm (SRT & MRT) of LBM for fluid flow 
interactions.It can define 2D model with different boundary conditions and simple
geometry. The computationale parts are accelerated using Anaconda.
===============================================================================
author: Pei Li
Email: pei.li@weizmann.ac.il
"""
import sys, os
import configparser

import numpy as np
import scipy as sp
import scipy.linalg as slin 
import matplotlib.pyplot as plt

from numba import jit
from numba import vectorize, cuda

##sys.path.append("..")
##from BoundaryCondition import BoundaryConditions as BC


class BasicD2Q9:
    def __init__(self, pathIniFile):
        """
        use configparser to read the .ini file
        """
        self.path = pathIniFile
        config = configparser.ConfigParser()
        config.read(self.path + '/'+ 'basicsetup.ini')
        #read the grid number
        try:
            self.typeScheme = config['Scheme']['Type']
        except KeyError:
            print('Could not have right scheme for LBM, please check .ini file')
            sys.exit('error')
        try:
            self.length = float(config['Geometry']['length'])
            self.width = float(config['Geometry']['width'])
            self.nx = int(config['Geometry']['nx'])
            self.ny = int(config['Geometry']['ny']) 
        except KeyError:
            print('Could not give the right grid numbers, please check .ini file')
            sys.exit('error happens')
            #raise
        #read time and time step for simulation
        try:
            self.timeForSimulation = float(config['Time']['TimeLength'])
            self.timeStep = float(config['Time']['TimeStep'])
        except KeyError:
            print('Could not give the right time set up, please check .ini file')
            sys.exit('error happens')
        try:
            self.initialVXLB = float(config['InitialCondition']['VelocityXLB'])
            self.initialVYLB = float(config['InitialCondition']['VelocityYLB'])
        except KeyError:
            print('Could not have the intial velocity, please check .ini file')
            sys.exit('error happens')
        try:
            self.bodyForceG = float(config['BodyForce']['gValue'])
        except KeyError:
            print('Could not have the body force value, please check .ini file')
            sys.exit('error happens')
        try:
            self.xDomain = list(map(int, config.get('FlowDomain', 'xDomain').split(',')))
            self.yDomain = list(map(int, config.get('FlowDomain', 'yDomain').split(',')))
        except KeyError:
            print('Could not have the boundary position value, please check .ini file')
            sys.exit('error happens')
        self.deltaX = self.length / (self.nx - 1)
        self.deltaY = self.width / (self.ny - 1)
        #initialize coefficients in equilibrium function
        self.weightsCoeff = sp.empty(9) #D2Q9 algorithm
        self.weightsCoeff[0] = 4./ 9.
        self.weightsCoeff[1:5] = 1./ 9.
        self.weightsCoeff[5:] = 1./ 36.
        self.f1Eq = 3.; self.f2Eq = 9./2.; self.f3Eq = -3./2.
        #initialize microscopic velocities in each direction
        #self.microVelocity = sp.array([(c1, c2) for c1 in [0, -1, 1] \
#                           for c2 in [0, -1, 1]])
        self.microVelocity = sp.empty([9, 2])
        self.microVelocity[0] = [0., 0.]    #center of the lattice
        self.microVelocity[1] = [1.0, 0.]; self.microVelocity[2] = [0., 1.0]
        self.microVelocity[3] = [-1., 0.]; self.microVelocity[4] = [0., -1.0]
        self.microVelocity[5] = [1., 1.,]; self.microVelocity[6] = [-1., 1.]
        self.microVelocity[7] = [-1., -1.]; self.microVelocity[8] = [1., -1.]
        self.velocityCoeff = 1.0 # (self.deltaX) / (self.deltaY)
        self.soundSpeed = 1.0 / sp.sqrt(3.)
        #initialize relaxation parameter
        #TODO check the calculation method for omega
        self.omega = 0.0
        self.tau = 1.0  #default value for numerical stablility
        #initialize macroscopic parameters in the domain (density, veclocity)
        self.rho = sp.ones([self.ny, self.nx])
        self.velocityX = sp.empty([self.ny, self.nx])
        self.velocityY = sp.empty([self.ny, self.nx])
        #initialize distribution function of each direction in each lattice
        self.particleDisFunc = sp.ones([9, self.ny, self.nx])
        #initialize the array for equilibrium function on each direction of 
        #each lattice
        self.equilibriumDisFunc = sp.empty([9, self.ny, self.nx])
        self.isWall = sp.empty([self.ny, self.nx], dtype = 'bool')
        self.isDomain = sp.empty([self.ny, self.nx], dtype = 'bool')
        #define the transformation matrix in MRT scheme
        self.transformationMatrix = sp.zeros([9, 9], dtype = 'float64')
        self.transformationMatrix[0, :] = 1.; self.transformationMatrix[1, 0] = -4.
        self.transformationMatrix[1, 1:5] = -1.; self.transformationMatrix[1, 5:] = 2.
        self.transformationMatrix[2, 0] = 4.; self.transformationMatrix[2, 1:5] = -2. 
        self.transformationMatrix[2, 5:]= 1.; self.transformationMatrix[3, 1] = 1.
        self.transformationMatrix[3, 3] = -1.; self.transformationMatrix[3, 5] = 1. 
        self.transformationMatrix[3, 6:8] = -1.; self.transformationMatrix[3, -1] = 1. 
        self.transformationMatrix[4, 1] = -2.; self.transformationMatrix[4, 3] = 2.0
        self.transformationMatrix[4, 5] = 1.; self.transformationMatrix[4, 6:8] = -1. 
        self.transformationMatrix[4, 8] = 1. 
        self.transformationMatrix[5, 2] = 1.; self.transformationMatrix[5, 4] = -1
        self.transformationMatrix[5, 5:7] = 1.; self.transformationMatrix[5, 7:] = -1.
        self.transformationMatrix[6, 2] = -2.; self.transformationMatrix[6, 4] = 2. 
        self.transformationMatrix[6, 5:7] = 1.; self.transformationMatrix[6, 7:] = -1. 
        self.transformationMatrix[7, 1] = 1.; self.transformationMatrix[7, 2] = -1. 
        self.transformationMatrix[7, 3] = 1.; self.transformationMatrix[7, 4] = -1. 
        self.transformationMatrix[8, 5] = 1.; self.transformationMatrix[8, -3] = -1. 
        self.transformationMatrix[8, -2] = 1.; self.transformationMatrix[8, -1] = -1.
##        c = slin.inv(self.transformationMatrix)
##        print(np.dot(self.transformationMatrix, slin.inv(self.transformationMatrix)))
##        input()
        #define the relaxation rate in each direction (There are two options)
        self.relaxationS = sp.zeros(9, dtype = 'float64')
        self.relaxationS[1] = 1. / self.tau; self.relaxationS[2]= 1./self.tau
        self.relaxationS[7:] = 1./self.tau; 
        self.relaxationS[4] = 8.* (2.-1./self.tau) / (8. - 1./self.tau)
        
        
    """
    use perturbation function to give distribution function f_n the initial 
    condition:
    """
    @jit(target="cpu")
    def __initializeDomainCondition(self):
        """
        Initialize the distribution function value in the domain
        """
        #self.velocityX[:, 0] = 0.04
        for i in sp.arange(9):
            self.particleDisFunc[i, :, :] *= self.weightsCoeff[i]
        #self.equilibriumDisFunc = self.particleDisFunc
        self.velocityX[:, :] = self.initialVXLB
        self.velocityY[:, :] = self.initialVYLB
    
    @jit(target ="cpu")
    def __initializeSolidPosition(self):
        """
        Define the solid phase position and domain size
        """
##        self.isWall[:, :] = False
##        self.isWall[:, 0] = True; self.isWall[:, self.nx - 1] = True
##        self.isDomain[:, :] = True
##        self.isDomain[:, 0] = False; self.isDomain[:, self.nx - 1] = False
        self.isDomain[:, :] = True
        self.isDomain[self.yDomain[0], :] = False
        self.isDomain[self.yDomain[-1], :] = False
        self.isWall[:, :] = False
        self.isWall[self.yDomain[0], :] = True; self.isWall[self.yDomain[-1], :] = True
        
    @jit(target="cpu")                
    def __calMacroscopicParameters(self, ):
        """
        Update the density of each lattice and velocity by:
        rho = \sum_{n=0}^{9}particleDisFunc
        velocityX = 
        """
        self.rho[i, j] = sp.sum(self.particleDisFunc[:, i, j])
        self.velocityX[i, j] = sp.sum(self.particleDisFunc[:, i, j] \
                        * self.microVelocity[:, 0]) / self.rho[i, j]
        self.velocityY[i, j] = sp.sum(self.particleDisFunc[:, i, j] \
                        * self.microVelocity[:, 1]) / self.rho[i, j]

    @jit(target="cpu" )
    def __calEquilibriumDisFunc(self):
        """
        Compute the equilibrium distribution function in each direction of each 
        lattice
        """
        coeff1 = 3.0; coeff2 = 9./2.; coeff3 = -3./2.
        for i in sp.arange(self.ny):
            for j in sp.arange(self.nx):
                velocityAtLattice = np.array([self.velocityX[i, j],\
                                    self.velocityY[i, j] + self.bodyForceG])
                for k in sp.arange(9):
                    self.equilibriumDisFunc[k, i, j] = self.weightsCoeff[k] * \
                            self.rho[i, j] * (1.0 + coeff1 * (np.dot(self.microVelocity[k], \
                            velocityAtLattice)) + coeff2 * (sp.power(np.dot(self.microVelocity[k], \
                            velocityAtLattice), 2.0)) + coeff3 * np.dot(velocityAtLattice, \
                            velocityAtLattice))
                            
    @jit(target = "cpu")
    def calEquilibriumDisFuncLoc(self, macroDensity, macroVelocity):
        coeff1 = 3.0; coeff2 = 9./2.; coeff3 = -3./2.0
        tmpEquilibrium = np.ones(9, dtype = 'float64')
        for i in sp.arange(9):
            tmpEquilibrium[i] = self.weightsCoeff[i] * macroDensity * (1. + \
                    coeff1 * (np.dot(self.microVelocity[i], macroVelocity)) + \
                    coeff2 * (sp.power(np.dot(self.microVelocity[i], macroVelocity), \
                    2.0)) + coeff3 * np.dot(macroVelocity, macroVelocity))
        return tmpEquilibrium
    
    @jit(target="cpu")
    def __calCollisionProcess(self):
        """
        Compute the collision step
        """
        tempCollision = sp.empty([9, self.ny, self.nx])
        for i in sp.arange(9):
##                if (self.isDomain[i, j]):
                tempCollision[i, :, :] = self.particleDisFunc[i, :, :] - \
                (self.particleDisFunc[i, :, :] - self.equilibriumDisFunc[i, :, :]) /\
                self.tau
        return tempCollision

    @jit(target = "cpu")
    def __bounceBackBC(self, tempCollision):
        """
        bounce-back boundary condition
        """
        oppDirection = oppDirection = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        for i in sp.arange(9):
            tempCollision[i, self.isWall] = self.particleDisFunc[oppDirection[i], \
                                            self.isWall]
        return temCollision

    @jit(target="cpu")
    def __calStreamingProcess(self, tempParticleDisFunc):
        """
        Streaming step for each lattice
        """
        for i in sp.arange(9):
            self.particleDisFunc[i, :, :] = np.roll(np.roll(tempParticleDisFunc[i, \
                                            :, :], self.microVelocity[i, 1], 
                                            axis = 0), self.microVelocity[i, 0],\
                                            axis = 1)
        
    @jit(target = "cpu")
    def __calMRTMoments(self):
        """
        Transforming the distribution functions f to moments m 
        """
        tempMoments = sp.empty([9, self.ny, self.nx], dtype = 'float64')
        for i in sp.arange(self.ny):
            for j in sp.arange(self.nx):
                tempMoments[:, i, j] = np.dot(self.transformationMatrix, \
                                    self.particleDisFunc[:, i, j])
        return tempMoments
    
    @jit(target="cpu")
    def __calMRTMomentEquilbira(self):
        """
        Transforming the equilibrium distribution to equilibria in the moment space
        """
        tempMomentsEquilibria = sp.empty([9, self.ny, self.nx])
        for i in sp.arange(self.ny):
            for j in sp.arange(self.nx):
                tempMomentsEquilibra[:, i, j] = np.dot(self.transformationMatrix, \
                                                self.equilibriumDisFunc[:, i, j])
        return tempMomentsEquilibra
    
    @jit(target="cpu")
    def __calMRTCollision(self, moments, equilibria):
        """
        Calculate the collision in moment space
        """
        tempMRTCollision = sp.empty([9, self.ny, self.nx])
        tempRelaxationMatrix = sp.zeros([9, 9], dtype = 'float64')
        np.fill_diagonal(tempRelaxationMatrix, self.relaxationS)
        for i in sp.arange(9):
            tempMRTCollision[i, :, :] = moments[i, :, :] - np.dot(tempRelaxationMatrix, \
                                        (moments[i, :, :] - equilibria[i, :, :]))
        return tempMRTCollision
    
    @jit(target="cpu")
    def __calPostParticleDisFunc(self, momentsCollision):
        """
        Transforming the post-collision moments m' back to the post collision 
        distribution function f'
        """
        tempParticleDisFunc = sp.empty([9, self.ny, self.nx])
        inverseTransformation = slin.inv(self.transformationMatrix)
        for i in sp.arange(self.ny):
            for j in sp.arange(self.nx):
                tempParticleDisFunc[:, i, j] = np.dot(inverseTransformation, \
                                            momentsCollision[:, i, j])
        return tempParticleDisFunc
            
    def plotVelocityProfile(self, yPosition):
        plt.figure()
        tempX = sp.arange(self.ny)
        print(tempX)
        plt.plot(self.velocityY[yPosition, :], 'bo', label = 'velocity profile')
        plt.legend()
        plt.show()
                
    def runModeling(self):
        #BounceBack = BC
        #test for single phase and periodic boundary condition
        nTimeSteps = int(self.timeForSimulation / self.timeStep)
        #tempCollision = sp.empty([self.ny, self.nx, 9])
        #Initialize distribution function
        self.__initializeSolidPosition()
        self.__initializeDomainCondition()
        #self.__calMacroscopicParameters()
        #self.__calMacroscopicParameters()
        if (self.typeScheme == 'SRT'):
            for i in sp.arange(nTimeSteps):
                print('This is step %g' % i)
                self.__calMacroscopicParameters()
            #print(self.equilibriumDisFunc)
                self.__calEquilibriumDisFunc()
                tempParticleDisFunc = self.__calCollisionProcess()
                tempParticleDisFunc = self.__bounceBackBC(tempParticleDisFunc)
                self.__calStreamingProcess(tempParticleDisFunc)
        elif (self.typeScheme == 'MRT'):
            for i in sp.arange(nTimeSteps):
                print('This is step %g' % i)
                self.__calMacroscopicParameters()
                self.__calEquilibriumDisFunc()
                tempMoments = self.__calMRTMoments()
                tempEquilibria = self.__calMRTMomentEquilibria()
                tempMRTCollision = self.__calMRTCollision(tempMoments, tempEquilibria)
                tempParticleDisFunc = self.__calPostParticleDisFunc(tempMRTCollision)
                tempParticleDisFunc = self.__bounceBackBC(tempParticleDisFunc)
                self.__calStreamingProcess(tempParticleDisFunc)