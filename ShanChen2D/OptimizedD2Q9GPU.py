
"""
Module includes all functions for running optimized D2Q9 LBM in GPU
"""
 
import numpy as np
import scipy as sp
from math import sqrt
 
from numba import cuda, int64, float64
from numba import cuda, jit
 
#"""
#Generate the array for recording the neighboring nodes
#"""
#@cuda.jit('void(int64, int64, int64, int64[:])')
 
"""
Fill the array for neighboring nodes
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:, :], int64[:])')
def fillNeighboringNodes(totalNodes, nx, ny, xDim, fluidNodes, domainNewIndex, \
                         neighboringNodes):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
#    ty = cuda.threadIdx.y; by = cuda.blockIdx.y; bDimY = cuda.blockDim.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        tmpStart = 8 * indices
        tmpLoc = fluidNodes[indices]
        i = int(tmpLoc / nx); j = tmpLoc % nx
        tmpF = j + 1 if j < nx - 1 else 0
        tmpB = j - 1 if j > 0 else (nx - 1)
        tmpU = i + 1 if i < ny - 1 else 0
        tmpL = i - 1 if i > 0 else (ny - 1)
        #Eastern node
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpF]
        #Northern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, j]
        #Western node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpB]
        #Southern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, j]
#        if tmpL == ny - 1:
#            neighboringNodes[tmpStart] = -2
        #Northeastern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpF]
        #Northwestern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpB]
        #Southwestern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpB]
#        if tmpL == ny - 1:
#            neighboringNodes[tmpStart] = -2
        #southeastern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpF]
#        if tmpL  == ny - 1:
#            neighboringNodes[tmpStart] = -2
         
"""
Save the PDF values in the last time step
"""
@cuda.jit("void(int64, int64, int64, float64[:, :, :], float64[:, :, :])")
def savePDFLastStep(totalNodes, numFluids, xDim, fluidPDF, fluidPDFOld):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
     
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        for i in range(numFluids):
            for j in range(9):
                fluidPDFOld[i, indices, j] = fluidPDF[i, indices, j]
 
"""
Calculate the macro-denity of each fluids
"""
@cuda.jit('void(int64, int64, int64, float64[:, :], float64[:, :, :])')
def calFluidRhoGPU(totalNodes, numFluids, xDim, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
#    ty = cuda.threadIdx.y; by = cuda.blockIdx.y; bDimY = cuda.blockDim.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        for i in range(numFluids):
            fluidRho[i, indices] = 0.
            for j in range(9):
                fluidRho[i, indices] += fluidPDF[i, indices, j]
 
"""
Calculate the potential on each fluid node in potential = \rho[indices]
"""
@cuda.jit('void(int64, int64, int64, float64[:, :], float64[:, :])')
def calFluidPotentialGPUEql(totalNodes, numFluids, xDim, fluidRho, fluidPotential):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        for i in range(numFluids):
            fluidPotential[i, indices] = fluidRho[i, indices]
             
"""
Calculate the potential on each fluid node with P-R EOS (Yuan & Schaefer 2002)
"""
@cuda.jit('void(int64, int64, int64, float64, float64, float64, float64, float64, \
                float64, float64, float64[:, :], float64[:, :])')
def calFluidPotentialGPUPR(totalNodes, numFluids, xDim, constR, temperatureT, \
                           coeffA, coeffB, coeffAlpha, constC0, constG, \
                           fluidRho, fluidPotential):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        for i in range(numFluids):
            tmpP = (fluidRho[i, indices] * constR *  temperatureT) / (1. - \
                    coeffB * fluidRho[i, indices]) - (coeffA * coeffAlpha * \
                    fluidRho[i, indices] * fluidRho[i, indices]) / (1. + 2. * \
                    coeffB * fluidRho[i, indices] - coeffB * coeffB * \
                    fluidRho[i, indices] * fluidRho[i, indices])
            fluidPotential[i, indices] = sqrt(2. / (constC0 * constG) * (tmpP - \
                          1./3. * fluidRho[i, indices]))
         
 
"""
Calculate the macroscale pressure: 
     
"""
@cuda.jit('void(int64, int64, int64, float64[:, :], float64[:, :], float64[:])')
def calMacroPressure(totalNodes, numFluids, xDim, interactionCoeff, fluidRho, fluidPressure):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpRho = 0.; tmpPressure = 0.
        for i in range(numFluids):
            tmpRho += fluidRho[i, indices]
        tmpPressure = 1./3. * tmpRho
        for i in range(numFluids - 1):
            for j in range(i+1, numFluids):
                tmpPressure += 3./2. * interactionCoeff[i, j] * fluidRho[i, indices] * \
                            fluidRho[j, indices]
        fluidPressure[indices] = tmpPressure
                       
"""
Calculate the physical velocity in original Shan-Chen method
"""
@cuda.jit('void(int64, int64, int64, float64[:, :, :], float64[:, :], float64[:, :], \
                float64[:, :], float64[:], float64[:])')
def calPhysicalVelocity(totalNodes, numFluids, xDim, fluidPDF, fluidRho, forceX, forceY, \
                        velocityPX, velocityPY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpVX = 0.; tmpVY = 0.; tmpRho = 0.
        for i in range(numFluids):
            tmpVX += (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3] + \
                    fluidPDF[i, indices, 5] - fluidPDF[i, indices, 6] - \
                    fluidPDF[i, indices, 7] + fluidPDF[i, indices, 8]  + \
                    1./2. * forceX[i, indices])
            tmpVY += (fluidPDF[i, indices, 2] - fluidPDF[i, indices, 4] + \
                    fluidPDF[i, indices, 5] + fluidPDF[i, indices, 6] - \
                    fluidPDF[i, indices, 7] - fluidPDF[i, indices, 8] + \
                    1./2. * forceY[i, indices])
            tmpRho += fluidRho[i, indices]
        velocityPX[indices] = tmpVX / tmpRho
        velocityPY[indices] = tmpVY / tmpRho
         
#Calculate the maro-pressure in original Shan-Chen way
#"""
#@cuda.jit()
 
"""
Calculate the force on each fluid node 
"""
@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], int64[:], float64[:], float64[:, :], float64[:], \
        float64[:, :], float64[:, :], float64[:, :])')
def calInteractionForce(totaNodes, numFluids, nx, ny, xDim, fluidNodes, neighboringNodes, \
                        weightInter, interactionCoeff, interactionSolid, fluidPotential, \
                        forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
#    tmpWeight = cuda.shared.array(shape = (8, ), dtype=float64)
#    for i in range(8):
#        tmpWeight[i] = weightInter[i]
    if (indices < totaNodes):
        for i in range(numFluids):
            forceX[i, indices] = 0.
            forceY[i, indices] = 0.
#        tmpNodeNum = fluidNodes[indices]
        tmpStart = indices * 8  #starting point in the neighboring nodes array
#        tmpYpos = int(tmpNodeNum / nx); tmpXpos = tmpNodeNum % nx
        #Eastern point
        if (neighboringNodes[tmpStart] != -1):
            tmpE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -weightInter[0] * interactionCoeff[i, j] * \
                              fluidPotential[i, indices] * fluidPotential[j, tmpE] * (1.)
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Northern point
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpN = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -weightInter[1] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN] * (1.)
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceY[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Western Point
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -weightInter[2] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW] * (-1.)
        elif(neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Southern point
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpS = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -weightInter[3] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS] * (-1.)
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceY[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Northeastern point
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpNE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -weightInter[4] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
                    forceY[i, indices] += -weightInter[4] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
        elif(neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Northwestern point
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpNW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -weightInter[5] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (-1.)
                    forceY[i, indices] += -weightInter[5] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (1.)
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Southwestern point
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpSW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -weightInter[6] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
                    forceY[i, indices] += -weightInter[6] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Southeastern point
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpSE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -weightInter[7] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (1.)
                    forceY[i, indices] += -weightInter[7] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (-1.)
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
    cuda.syncthreads()
 
"""
Add the body force to the force term
"""                      
@cuda.jit('void(int64, int64, int64, float64, float64, float64[:, :], float64[:, :], \
                float64[:, :])')
def addBodyForceGPU(totalNum, numFluids, xDim, bodyFX, bodyFY, forceX, forceY, \
                    fluidRho):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNum):
        for i in range(numFluids):
            if (i == 2):
                forceX[i, indices] += bodyFX * fluidRho[i, indices]  
                forceY[i, indices] += bodyFY * fluidRho[i, indices]
 
"""
Calculate the velocity of whole fluids
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:, :], float64[:, :, :], \
                float64[:], float64[:])')
def calMacroWholeVelocity(totalNodes, numFluids, xDim, tau, fluidRho, fluidPDF, primeVX, \
                          primeVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bDimX * bx + tx
    
    if (indices < totalNodes):
        tmpVX = 0.; tmpVY = 0.; tmpD = 0.
        for i in range(numFluids):
            tmpVX += (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3] + \
                      fluidPDF[i, indices, 5] - fluidPDF[i, indices, 6] - \
                              fluidPDF[i, indices, 7] + fluidPDF[i, indices, 8]) / tau[i]
            tmpVY += (fluidPDF[i, indices, 2] - fluidPDF[i, indices, 4] + \
                      fluidPDF[i, indices, 5] + fluidPDF[i, indices, 6] - \
                              fluidPDF[i, indices, 7] - fluidPDF[i, indices, 8]) / tau[i]
            tmpD += fluidRho[i, indices] / tau[i]
        primeVX[indices] = tmpVX / tmpD
        primeVY[indices] = tmpVY / tmpD
                
               
"""
Calculate the equilibrium Velocity on each fluid nodes
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:, :], float64[:, :], float64[:, :], \
                float64[:], float64[:], float64[:, :], float64[:, :])')
def calEquilibriumVGPU(totalNum, numFluids, xDim, tau, fluidRho, forceX, forceY, \
                       mixtureVX, mixtureVY, equilibriumVX, equilibriumVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNum):
        for i in range(numFluids):
            equilibriumVX[i, indices] = mixtureVX[indices] + tau[i] * \
                         forceX[i, indices] / fluidRho[i, indices]
            equilibriumVY[i, indices] = mixtureVY[indices] + tau[i] * \
                         forceY[i, indices] / fluidRho[i, indices]
    cuda.syncthreads()
                          
"""
Calculate the equilibrium function for collision process (most common f_eq)
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:, :], float64[:, :], \
        float64[:, :], float64[:, :, :])')
def calEquilibriumFuncGPU(totalNum, numFluids, xDim, weightCoeff, fluidRho, equilibriumVX, \
                          equilibriumVY, fEq):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    sharedWeightCoeff = cuda.shared.array(shape = (9, ), dtype = float64)
    for i in range(9):
        sharedWeightCoeff[i] = weightCoeff[i]
    if (indices < totalNum):
        for i in range(numFluids):
            squareV = equilibriumVX[i, indices] * equilibriumVX[i, indices] + \
                    equilibriumVY[i, indices] * equilibriumVY[i, indices]
            fEq[i, indices, 0] = sharedWeightCoeff[0] * fluidRho[i, indices] * \
                   (1. - squareV / (2. * 1./3.))
            fEq[i, indices, 1] = sharedWeightCoeff[1] * fluidRho[i, indices] * \
                   (1. + 3. * equilibriumVX[i, indices] + 9./2. *\
                    (equilibriumVX[i, indices] * equilibriumVX[i, indices]) - \
                     squareV / (2. * 1./3.))
            fEq[i, indices, 2] = sharedWeightCoeff[2] * fluidRho[i, indices] * \
                   (1. + 3. * equilibriumVY[i, indices] + 9./2. * \
                    (equilibriumVY[i, indices] * equilibriumVY[i, indices]) - \
                     squareV / (2. * 1./3.))
            fEq[i, indices, 3] = sharedWeightCoeff[3] * fluidRho[i, indices] * \
                   (1. + 3. * (-1. * equilibriumVX[i, indices]) + 9./2. * \
                    (equilibriumVX[i, indices] * equilibriumVX[i, indices]) - \
                     squareV / (2. * 1./3.))
            fEq[i, indices, 4] = sharedWeightCoeff[4] * fluidRho[i, indices] * \
                   (1. + 3. * (-1. * equilibriumVY[i, indices]) + 9./2. * \
                    (equilibriumVY[i, indices] * equilibriumVY[i, indices]) - \
                     squareV / (2. * 1./3.))
            fEq[i, indices, 5] = sharedWeightCoeff[5] * fluidRho[i, indices] * \
                   (1. + 3. * (equilibriumVX[i, indices] + equilibriumVY[i, indices]) + \
                    9./2. * (equilibriumVX[i, indices] + equilibriumVY[i, indices]) * \
                    (equilibriumVX[i, indices] + equilibriumVY[i, indices]) - \
                     squareV / (2. * 1./3.))
            fEq[i, indices, 6] = sharedWeightCoeff[6] * fluidRho[i, indices] * \
                   (1. + 3. * (-equilibriumVX[i, indices] + equilibriumVY[i, indices]) + \
                    9./2. * (-equilibriumVX[i, indices] + equilibriumVY[i, indices]) * \
                    (-equilibriumVX[i, indices] + equilibriumVY[i, indices]) - \
                     squareV / (2. * 1./3.))
            fEq[i, indices, 7] = sharedWeightCoeff[7] * fluidRho[i, indices] * \
                   (1. + 3. * (-equilibriumVX[i, indices] - equilibriumVY[i, indices]) + \
                    9./2. * (-equilibriumVX[i, indices] - equilibriumVY[i, indices]) * \
                    (-equilibriumVX[i, indices] - equilibriumVY[i, indices]) - \
                     squareV / (2. * 1./3.))
            fEq[i, indices, 8] = sharedWeightCoeff[8] * fluidRho[i, indices] * \
                   (1. + 3. * (equilibriumVX[i, indices] - equilibriumVY[i, indices]) + \
                    9./2. * (equilibriumVX[i, indices] - equilibriumVY[i, indices]) * \
                    (equilibriumVX[i, indices] - equilibriumVY[i, indices]) - \
                     squareV / (2. * 1./3.))
    cuda.syncthreads()
                    
"""
Calculate the collision process
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:, :, :], float64[:, :, :])')
def calCollisionSRTGPU(totalNum, numFluids, xDim, tau, fluidPDF, fEq):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNum):
        for i in range(numFluids):
            for j in range(9):
                fluidPDF[i, indices, j] = fluidPDF[i, indices, j] - 1./tau[i] * \
                        (fluidPDF[i, indices, j] - fEq[i, indices, j])
    cuda.syncthreads()
 
"""
Calculate the streaming process1
"""
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
            float64[:, :, :])')
def calStreaming1GPU(totalNum, numFluids, xDim, fluidNodes, neighboringNodes, \
                    fluidPDF, fluidPDFNew):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNum):
        tmpArray = cuda.local.array(shape = (8, ), dtype = float64)
        for i in range(8):
            tmpArray[i] = 0.
        #Eastern node
        tmpStart = 8 * indices
        if (neighboringNodes[tmpStart] != -1):
            tmpE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpE, 1] = fluidPDF[i, indices, 1]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 3] = fluidPDF[i, indices, 1]
        #Northern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpN = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpN, 2] = fluidPDF[i, indices, 2]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 4] = fluidPDF[i, indices, 2]
        #Western node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpW, 3] = fluidPDF[i, indices, 3]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 1] = fluidPDF[i, indices, 3]
        #Southern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpS = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpS, 4] = fluidPDF[i, indices, 4]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 2] = fluidPDF[i, indices, 4]
        #Northeastern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpNE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpNE, 5] = fluidPDF[i, indices, 5]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 7] = fluidPDF[i, indices, 5]
        #Northwestern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpNW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpNW, 6] = fluidPDF[i, indices, 6]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 8] = fluidPDF[i, indices, 6]
        #Southwestern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpSW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpSW, 7] = fluidPDF[i, indices, 7]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 5] = fluidPDF[i, indices, 7]
        #Sourtheastern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpSE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpSE, 8] = fluidPDF[i, indices, 8]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                fluidPDFNew[i, indices, 6] = fluidPDF[i, indices, 8]
    cuda.syncthreads()
                 
"""
Calculate the streaming process 2
"""
@cuda.jit('void(int64, int64, int64, float64[:, :, :], float64[:, :, :])')
def calStreaming2GPU(totalNum, numFluids, xDim, fluidPDFNew, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNum):
        for i in range(numFluids):
            for j in range(1, 9):
                fluidPDF[i, indices, j] = fluidPDFNew[i, indices, j]
    cuda.syncthreads()
                 
"""
Constant pressure/density boundary condition Zou-He method (Lower)
"""
@cuda.jit('void(int64, int64, int64, int64, float64, int64[:], float64[:, :], \
            float64[:, :, :])')
def constantPressureZouHeBoundaryLower(totalNodes, numFluids, nx, xDim, densityL, fluidNodes, \
                                       fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    tmpArray = cuda.local.array(shape = (2, ), dtype = float64)
    tmpArray[0] = 1.0; tmpArray[1] = 0.02
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        tmpTotalRho = 0
        if (tmpIndex >= nx and tmpIndex < 2 * nx):
            for i in range(numFluids):
                tmpTotalRho += fluidRho[i, indices]
            for i in range(numFluids):
                tmpDensity = tmpArray[i]#fluidRho[i, indices] / tmpTotalRho * densityL
                velocityY = 1. - (fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 3] + 2.* (fluidPDF[i, indices, 4] + \
                        fluidPDF[i, indices, 7] + fluidPDF[i, indices, 8])) / \
                        tmpDensity
                fluidPDF[i, indices, 2] = fluidPDF[i, indices, 4] + 2./3. * velocityY * \
                       tmpDensity
                fluidPDF[i, indices, 5] = fluidPDF[i, indices, 7] + \
                    1./2. * (fluidPDF[i, indices, 3] - fluidPDF[i, indices, 1]) + \
                    1./6. * tmpDensity * velocityY
                fluidPDF[i, indices, 6] = fluidPDF[i, indices, 8] - 1./2. * \
                    (fluidPDF[i, indices, 3] - fluidPDF[i, indices, 1]) + \
                    1./6. * tmpDensity * velocityY
                fluidRho[i, indices] = tmpDensity
    cuda.syncthreads()

"""
Constant pressure/density boundary condition Zou-He method (Lower)
"""
@cuda.jit('void(int64, int64, int64, int64, float64, int64[:], float64[:, :], \
            float64[:, :, :])')
def constantPressureZouHeBoundaryLower8(totalNodes, numFluids, nx, xDim, densityL, fluidNodes, \
                                       fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    tmpArray = cuda.local.array(shape = (2, ), dtype = float64)
    tmpArray[0] = 1.0; tmpArray[1] = 0.02
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        tmpTotalRho = 0
        if (tmpIndex >= 2 * nx and tmpIndex < 3 * nx):
            for i in range(numFluids):
                tmpTotalRho += fluidRho[i, indices]
            for i in range(numFluids):
                tmpDensity = tmpArray[i]#fluidRho[i, indices] / tmpTotalRho * densityL
                velocityY = 1. - (fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 3] + 2.* (fluidPDF[i, indices, 4] + \
                        fluidPDF[i, indices, 7] + fluidPDF[i, indices, 8])) / \
                        tmpDensity
                fluidPDF[i, indices, 2] = fluidPDF[i, indices, 4] + 2./3. * velocityY * \
                       tmpDensity
                fluidPDF[i, indices, 5] = fluidPDF[i, indices, 7] + \
                    1./2. * (fluidPDF[i, indices, 3] - fluidPDF[i, indices, 1]) + \
                    1./6. * tmpDensity * velocityY
                fluidPDF[i, indices, 6] = fluidPDF[i, indices, 8] - 1./2. * \
                    (fluidPDF[i, indices, 3] - fluidPDF[i, indices, 1]) + \
                    1./6. * tmpDensity * velocityY
                fluidRho[i, indices] = tmpDensity
    cuda.syncthreads()
    
"""
Constant pressure/density boundary condition Zou-He method (Higher)
"""
@cuda.jit('void(int64, int64, int64, int64, int64, float64, int64[:], float64[:, :], \
                float64[:, :, :])')
def constantPressureZouHeBoundaryHigher(totalNodes, numFluids, nx, ny, xDim, \
                                        densityH, fluidNodes, fluidRho, fluidPDF): 
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        tmpTotalRho = 0.
        if (tmpIndex < nx * (ny - 1) and tmpIndex >= nx * (ny - 2)):
            for i in range(numFluids):
                tmpTotalRho += fluidRho[i, indices]
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices] / tmpTotalRho * densityH
                velocityY = -1. + (fluidPDF[i, indices, 0] + \
                    fluidPDF[i, indices, 1] + fluidPDF[i, indices, 3] + \
                    2. * (fluidPDF[i, indices, 2] + fluidPDF[i, indices, 5] + \
                    fluidPDF[i, indices, 6])) / tmpDensity
                fluidPDF[i, indices, 4] = fluidPDF[i, indices, 2] - 2. / 3. * \
                        tmpDensity * velocityY
                fluidPDF[i, indices, 7] = fluidPDF[i, indices, 5] + 1./2. * \
                    (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3]) - \
                    1./6. * tmpDensity * velocityY
                fluidPDF[i, indices, 8] = fluidPDF[i, indices, 6] - 1./2. * \
                    (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3]) - \
                    1./6. * tmpDensity * velocityY
                fluidRho[i, indices] = tmpDensity
    cuda.syncthreads()
                         
"""
Update the ghost points with constant pressure/density boundary
"""
@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], int64[:], float64[:, :], \
            float64[:, :, :])')
def ghostPointsConstantPressureInlet(totalNodes, numFluids, nx, ny, xDim, fluidNodes, \
                                neighboringNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < nx):
            tmpStart = 8 * indices + 1
            tmpH = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpH, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpH, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpH, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpH, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpH, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpH, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpH, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpH, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpH, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
        if (tmpIndex < ny * nx and tmpIndex >= (ny - 1) * nx):
            tmpStart = 8 * indices + 3
            tmpL = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpL, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpL, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpL, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpL, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpL, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpL, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpL, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpL, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpL, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
    cuda.syncthreads()
     
"""
Update the ghost points with constant velocity boundary
"""
@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], int64[:], \
                float64[:, :], float64[:, :, :])')
def ghostPointsConstantVelocityInlet(totalNodes, numFluids, nx, ny, xDim, fluidNodes, \
                                neighboringNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < ny * nx and tmpIndex >= (ny - 1) * nx):
            tmpStart = 8 * indices + 3
            tmpL = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpL, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpL, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpL, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpL, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpL, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpL, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpL, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpL, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpL, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
    cuda.syncthreads()
    
"""
Update the ghost points on the outlet boundary
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :], \
            float64[:, :, :])')
def ghostPointsConstantPressureOutlet(totalNodes, numFluids, nx, xDim, fluidNodes, \
                                neighboringNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < nx):
            tmpStart = 8 * indices + 1
            tmpH = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpH, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpH, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpH, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpH, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpH, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpH, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpH, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpH, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpH, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
                        
"""
Update the ghost points on the outlet boundary for isotropy being 8
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :], \
            float64[:, :, :])')
def ghostPointsConstantPressureOutlet8(totalNodes, numFluids, nx, xDim, fluidNodes, \
                                neighboringNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 2 * nx and tmpIndex >= nx):
            tmpStart = 8 * indices + 1
            tmpH = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpH, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpH, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpH, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpH, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpH, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpH, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpH, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpH, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpH, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
                        
"""
Update the ghost points on the outlet boundary for isotropy being 8
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :], \
            float64[:, :, :])')
def ghostPointsConstantPressureOutlet82(totalNodes, numFluids, nx, xDim, fluidNodes, \
                                neighboringNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < nx and tmpIndex >= 0):
            tmpStart = 8 * indices + 1
            tmpH = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpH, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpH, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpH, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpH, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpH, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpH, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpH, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpH, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpH, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
    
"""
Calculate Von Neumann boundary condition with Zou-He method
"""
@cuda.jit('void(int64, int64, int64, int64, int64, float64[:], int64[:], \
                float64[:, :], float64[:, :, :])')
def constantVelocityZouHeBoundaryHigher(totalNodes, numFluids, nx, ny, xDim, \
                                        specificVY, fluidNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < (ny - 1) * nx and tmpIndex >= (ny - 2) * nx):
            for i in range(numFluids):
                fluidRho[i, indices] = (fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 3] + 2. * (fluidPDF[i, indices, 2] + \
                        fluidPDF[i, indices, 5] + fluidPDF[i, indices, 6])) / \
                        (1. + specificVY[i])
                fluidPDF[i, indices, 4] = fluidPDF[i, indices, 2] - 2./3. * \
                        fluidRho[i, indices] * specificVY[i]
                fluidPDF[i, indices, 7] = fluidPDF[i, indices, 5] + \
                        (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3]) / 2. - \
                        1./6. * fluidRho[i, indices] * specificVY[i]
                fluidPDF[i, indices, 8] = fluidPDF[i, indices, 6] - \
                        (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3]) / 2. - \
                        1./6. * fluidRho[i, indices] * specificVY[i]
    cuda.syncthreads()
    
"""
Calculate Von Neumann boundary condition with Zou-He method
"""
@cuda.jit('void(int64, int64, int64, int64, int64, float64[:], int64[:], \
                float64[:, :], float64[:, :, :])')
def constantVelocityZouHeBoundaryHigher8(totalNodes, numFluids, nx, ny, xDim, \
                                        specificVY, fluidNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < (ny - 2) * nx and tmpIndex >= (ny - 3) * nx):
            for i in range(numFluids):
                fluidRho[i, indices] = (fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 3] + 2. * (fluidPDF[i, indices, 2] + \
                        fluidPDF[i, indices, 5] + fluidPDF[i, indices, 6])) / \
                        (1. + specificVY[i])
                fluidPDF[i, indices, 4] = fluidPDF[i, indices, 2] - 2./3. * \
                        fluidRho[i, indices] * specificVY[i]
                fluidPDF[i, indices, 7] = fluidPDF[i, indices, 5] + \
                        (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3]) / 2. - \
                        1./6. * fluidRho[i, indices] * specificVY[i]
                fluidPDF[i, indices, 8] = fluidPDF[i, indices, 6] - \
                        (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3]) / 2. - \
                        1./6. * fluidRho[i, indices] * specificVY[i]
    cuda.syncthreads()
    
"""
Update the ghost points with constant velocity boundary
"""
@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], int64[:], \
                float64[:, :], float64[:, :, :])')
def ghostPointsConstantVelocity8(totalNodes, numFluids, nx, ny, xDim, fluidNodes, \
                                neighboringNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < (ny - 1) * nx and tmpIndex >= (ny - 2) * nx):
            tmpStart = 8 * indices + 3
            tmpL = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpL, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpL, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpL, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpL, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpL, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpL, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpL, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpL, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpL, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
    cuda.syncthreads()

@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], int64[:], \
                float64[:, :], float64[:, :, :])')
def ghostPointsConstantVelocity82(totalNodes, numFluids, nx, ny, xDim, fluidNodes, \
                                neighboringNodes, fluidRho, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < ny * nx and tmpIndex >= (ny - 1) * nx):
            tmpStart = 8 * indices + 3
            tmpL = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDF[i, indices, 0] = fluidPDF[i, tmpL, 0]
                fluidPDF[i, indices, 1] = fluidPDF[i, tmpL, 1]
                fluidPDF[i, indices, 2] = fluidPDF[i, tmpL, 2]
                fluidPDF[i, indices, 3] = fluidPDF[i, tmpL, 3]
                fluidPDF[i, indices, 4] = fluidPDF[i, tmpL, 4]
                fluidPDF[i, indices, 5] = fluidPDF[i, tmpL, 5]
                fluidPDF[i, indices, 6] = fluidPDF[i, tmpL, 6]
                fluidPDF[i, indices, 7] = fluidPDF[i, tmpL, 7]
                fluidPDF[i, indices, 8] = fluidPDF[i, tmpL, 8]
                fluidRho[i, indices] = fluidPDF[i, indices, 0] + fluidPDF[i, indices, 1] + \
                        fluidPDF[i, indices, 2] + fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 4] + fluidPDF[i, indices, 5] + \
                        fluidPDF[i, indices, 6] + fluidPDF[i, indices, 7] + \
                        fluidPDF[i, indices, 8]
    cuda.syncthreads()
 
"""
Calculate the outlet boundary with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :])')
def convectiveOutletGPU(totalNodes, numFluids, nx, xDim, fluidNodes, neighboringNodes, \
                        fluidPDFNew, fluidRho):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    #calculate the average velocity
    tmpSumV = 0.
    tmpStart = 3 * nx
 
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 3 * nx and tmpIndex >= 2 * nx):
            tmpIndices = neighboringNodes[8 * indices + 1]
            for i in range(numFluids):
                fluidRho[i, indices] = 0.
                for j in range(9):
                    fluidPDFNew[i, indices, j] = fluidPDFNew[i, tmpIndices, j]
#                               (fluidPDFOld[i, indices, j] + \
#                            averageV * fluidPDFNew[i, indices + nx, j]) / (1. + \
#                            averageV)
                    fluidRho[i, indices] += fluidPDFNew[i, tmpIndices, j]
#    cuda.syncthreads()
     
"""
Calculate the outlet boundary ghost nodes in second layer with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :])')
def convectiveOutletGhost2GPU(totalNodes, numFluids, nx, xDim, \
                             fluidNodes, neighboringNodes, fluidPDFNew, fluidRho):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    #calculate the average velocity
    tmpSumV = 0.
    tmpStart = 3 * nx
 
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 2 * nx and tmpIndex >= nx):
            tmpIndices = neighboringNodes[8 * indices + 1]
            for i in range(numFluids):
                fluidRho[i, indices] = 0.
                for j in range(9):
                    fluidPDFNew[i, indices, j] = fluidPDFNew[i, tmpIndices, j]
#                               (fluidPDFOld[i, indices, j] + \
#                            averageV * fluidPDFNew[i, indices + nx, j]) / (1. + \
#                            averageV)
                    fluidRho[i, indices] += fluidPDFNew[i, tmpIndices, j]
#    cuda.syncthreads()
     
"""
Calculate the outlet boundary ghost nodes in first layer with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :])')
def convectiveOutletGhost3GPU(totalNodes, numFluids, nx, xDim, \
                             fluidNodes, neighboringNodes, fluidPDFNew, fluidRho):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    #calculate the average velocity
    tmpSumV = 0.
    tmpStart = 3 * nx
 
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < nx and tmpIndex >= 0):
            tmpIndices = neighboringNodes[8 * indices + 1]
            for i in range(numFluids):
                fluidRho[i, indices] = 0.
                for j in range(9):
                    fluidPDFNew[i, indices, j] = fluidPDFNew[i, tmpIndices, j]
#                               (fluidPDFOld[i, indices, j] + \
#                            averageV * fluidPDFNew[i, indices + nx, j]) / (1. + \
#                            averageV)
                    fluidRho[i, indices] += fluidPDFNew[i, tmpIndices, j]
#    cuda.syncthreads()

"""
Calculate the outlet boundary with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :, :], float64[:, :], float64[:])')
def convectiveOutletEachGPU(totalNodes, numFluids, nx, xDim, fluidNodes, \
                            neighboringNodes, fluidPDFNew, fluidPDFOld, fluidRho, \
                            physicalVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    #calculate the average velocity 
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 3 * nx and tmpIndex >= 2 * nx):
            tmpNeighbor = neighboringNodes[indices * 8 + 1]
            tmpVelocity = abs(physicalVY[tmpNeighbor])
            for i in range(numFluids):
                fluidRho[i, indices] = 0.
                for j in range(9):
                    fluidPDFNew[i, indices, j] = (fluidPDFOld[i, indices, j] + \
                            tmpVelocity * fluidPDFNew[i, tmpNeighbor, j]) / (1. + \
                            tmpVelocity)
                    fluidRho[i, indices] += fluidPDFNew[i, indices, j]
    cuda.syncthreads()
     
"""
Calculate the outlet boundary ghost nodes in second layer with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :, :], float64[:, :], float64[:])')
def convectiveOutletEach2GPU(totalNodes, numFluids, nx, xDim, fluidNodes, \
                             neighboringNodes, fluidPDFNew, fluidPDFOld, fluidRho, \
                             physicalVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    #calculate the average velocity

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 2 * nx and tmpIndex >= nx):
            tmpNeighbor1 = neighboringNodes[indices * 8 + 1]
            tmpNeighbor = neighboringNodes[tmpNeighbor1 * 8 + 1]
            tmpVelocity = abs(physicalVY[tmpNeighbor])
            for i in range(numFluids):
                fluidRho[i, indices] = 0.
                for j in range(9):
                    fluidPDFNew[i, indices, j] = (fluidPDFOld[i, indices, j] + \
                            tmpVelocity * fluidPDFNew[i, tmpNeighbor1, j]) / (1. + \
                            tmpVelocity)
                    fluidRho[i, indices] += fluidPDFNew[i, indices, j]
    cuda.syncthreads()
     
"""
Calculate the outlet boundary ghost nodes in first layer with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :, :], float64[:, :], float64[:])')
def convectiveOutletEach3GPU(totalNodes, numFluids, nx, xDim, fluidNodes, \
                             neighboringNodes, fluidPDFNew, fluidPDFOld, fluidRho, \
                             physicalVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    #calculate the average velocity 
    
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < nx and tmpIndex >= 0):
            tmpNeighbor2 = neighboringNodes[indices * 8 + 1]
            tmpNeighbor1 = neighboringNodes[tmpNeighbor2 * 8 + 1]
            tmpNeighbor = neighboringNodes[tmpNeighbor1 * 8 + 1]
            tmpVelocity = abs(physicalVY[tmpNeighbor])
            for i in range(numFluids):
                fluidRho[i, indices] = 0.
                for j in range(9):
                    fluidPDFNew[i, indices, j] = (fluidPDFOld[i, indices, j] + \
                            tmpVelocity * fluidPDFNew[i, tmpNeighbor2, j]) / (1. + \
                            tmpVelocity)
                    fluidRho[i, indices] += fluidPDFNew[i, indices, j]
#    cuda.syncthreads()
"""
Calculate the velocity boundary condition with Chang, C et.al., 2009
"""
@cuda.jit('void(int64, int64, int64, int64, int64, float64[:], int64[:], \
                float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], \
                float64[:, :, :])')
def calVelocityBoundaryHigherChangGPU(totalNodes, numFluids, nx, ny, xDim, \
                            specificVY, fluidNodes, fluidRho, forceX, forceY, \
                            fluidPDFOld, fluidPDFNew):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < (ny - 1) * nx and tmpIndex >= (ny - 2) * nx):
            for i in range(numFluids):
                fluidRho[i, indices] = (fluidPDFNew[i, indices, 0] + fluidPDFNew[i, indices, 1] + \
                        fluidPDFNew[i, indices, 3] + 2. * (fluidPDFNew[i, indices, 2] + \
                        fluidPDFNew[i, indices, 5] + fluidPDFNew[i, indices, 6])) / \
                        (1. + specificVY[i]) 
#                        + 1./2. * forceY[i, indices] / (1. + \
#                        specificVY[i])
                fluidPDFNew[i, indices, 4] = fluidPDFOld[i, indices, 4] - 2./3. * \
                           (fluidRho[i, indices] * specificVY[i] + fluidPDFOld[i, indices, 4] + \
                            fluidPDFOld[i, indices, 7] + fluidPDFOld[i, indices, 8]) + \
                            2./3. * (fluidPDFNew[i, indices, 2] + fluidPDFNew[i, indices, 5] + \
                            fluidPDFNew[i, indices, 6])
#                            + 1./2. * forceY[i, indices])
                fluidPDFNew[i, indices, 7] = fluidPDFOld[i, indices, 7] + 1./2. * \
                           (fluidPDFNew[i, indices, 1] - fluidPDFNew[i, indices, 3]) +\
                            1./6. * (fluidPDFNew[i, indices, 2] - fluidPDFOld[i, indices, 4]) +\
                            2./3. * (fluidPDFNew[i, indices, 5] - fluidPDFOld[i, indices, 7]) - \
                            1./3. * (fluidPDFNew[i, indices, 6] - fluidPDFOld[i, indices, 8]) - \
                            1./6. * fluidRho[i, indices] * specificVY[i]  
#                                            \+
#                            1./4. * forceX[i, indices] - 1./12. * forceY[i, indices] 
                fluidPDFNew[i, indices, 8] = fluidPDFOld[i, indices, 8] - 1./6. * \
                            fluidRho[i, indices] * specificVY[i] - 1./2. * (fluidPDFNew[i, indices, 1] - \
                            fluidPDFNew[i, indices, 3]) + 1./6. * (fluidPDFNew[i, indices, 2] - \
                            fluidPDFOld[i, indices, 4]) - 1./3. * (fluidPDFNew[i, indices, 5] - \
                            fluidPDFOld[i, indices, 7]) + 2./3. * (fluidPDFNew[i, indices, 6] - \
                            fluidPDFOld[i, indices, 8]) 
#                            - 1./4. * forceX[i, indices] + \
#                            1./12. * forceY[i, indices]
 
"""
Calculate the density/pressure condition with Chang, C et.al., 2009
"""
@cuda.jit('void(int64, int64, int64, int64, int64, float64, int64[:], float64[:, :], \
                float64[:, :], float64[:, :], float64[:, :, :], float64[:, :, :])')
def calPressureBoundaryHigherChangGPU(totalNodes, numFluids, nx, ny, xDim, specificRhoH, \
                                      fluidNodes, fluidRho, forceX, forceY, \
                                      fluidPDFOld, fluidPDFNew):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        tmpTotalRho = 0.
        if (tmpIndex < (ny - 1) * nx and tmpIndex >= (ny - 2) * nx):
            for i in range(numFluids):
                tmpTotalRho += fluidRho[i, indices]
            for i in range(numFluids):
                forceX[i, indices] = 0.; forceY[i, indices] = 0.
                tmpDensity = fluidRho[i, indices] / tmpTotalRho * specificRhoH
                velocityY = -1. + (fluidPDFNew[i, indices, 0] + fluidPDFNew[i, indices, 1] + \
                            fluidPDFNew[i, indices, 3] + 2. * (fluidPDFNew[i, indices, 2] + \
                            fluidPDFNew[i, indices, 5] + fluidPDFNew[i, indices, 6])) / \
                            tmpDensity + 1./2. * forceY[i, indices] / tmpDensity
                fluidPDFNew[i, indices, 4] = fluidPDFOld[i,indices, 4] - 2./3. * \
                           (tmpDensity * velocityY + fluidPDFOld[i, indices, 4] + \
                            fluidPDFOld[i, indices, 7] + fluidPDFOld[i, indices, 8] - \
                            fluidPDFNew[i, indices, 2] - fluidPDFNew[i, indices, 5] - \
                            fluidPDFNew[i, indices, 6] - 1./2. * forceY[i, indices])
                fluidPDFNew[i, indices, 7] = fluidPDFOld[i, indices, 7] - 1./2. * \
                           (fluidPDFNew[i, indices, 3] + fluidPDFNew[i, indices, 6] + \
                            fluidPDFOld[i, indices, 7] - fluidPDFNew[i, indices, 1] - \
                            fluidPDFNew[i, indices, 5] - fluidPDFOld[i, indices, 8] - \
                            1./2. * forceX[i, indices]) - 1./6. * (tmpDensity * \
                            velocityY + fluidPDFOld[i, indices, 7] + fluidPDFOld[i, indices, 8] + \
                            fluidPDFOld[i, indices, 4] - fluidPDFNew[i, indices, 2] - \
                            fluidPDFNew[i, indices, 5] - fluidPDFNew[i, indices, 6] - \
                            1./2. * forceY[i, indices])
                fluidPDFNew[i, indices, 8] = fluidPDFOld[i, indices, 8] + 1./2. * \
                           (fluidPDFNew[i, indices, 3] + fluidPDFNew[i, indices, 6] + \
                            fluidPDFOld[i, indices, 7] - fluidPDFNew[i, indices, 1] - \
                            fluidPDFNew[i, indices, 5] - fluidPDFOld[i, indices, 8] - \
                            1./2. * forceX[i, indices]) - 1./6. * (tmpDensity * \
                            velocityY + fluidPDFOld[i, indices, 7] + fluidPDFOld[i, indices, 8] + \
                            fluidPDFOld[i, indices, 4] - fluidPDFNew[i, indices, 2] - \
                            fluidPDFOld[i, indices, 5] - fluidPDFNew[i, indices, 6] - \
                            1./2. * forceY[i, indices])
                fluidRho[i, indices] = tmpDensity
                 
"""
Calculate the density/pressure condition with Chang, C et.al., 2009
"""
@cuda.jit('void(int64, int64, int64, int64, int64, float64, int64[:], float64[:, :], \
                float64[:, :], float64[:, :], float64[:, :, :], float64[:, :, :])')
def calPressureBoundaryLowerChangGPU(totalNodes, numFluids, nx, ny, xDim, specificRhoL, \
                                      fluidNodes, fluidRho, forceX, forceY, \
                                      fluidPDFOld, fluidPDFNew):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        tmpTotalRho = 0.
        if (tmpIndex < 2 * nx and tmpIndex >= nx):
            for i in range(numFluids):
                tmpTotalRho += fluidRho[i, indices]
            for i in range(numFluids):
                forceX[i, indices] = 0.; forceY[i, indices] = 0.
                tmpDensity = fluidRho[i, indices] / tmpTotalRho * specificRhoL
                velocityY = 1. - (fluidPDFNew[i, indices, 0] + fluidPDFNew[i, indices, 1] + \
                                fluidPDFNew[i, indices, 3] + 2. * (fluidPDFNew[i, indices, 4] \
                                + fluidPDFNew[i, indices, 7] + fluidPDFNew[i, indices, 8])) / \
                                tmpDensity + 1./2. * forceY[i, indices] / tmpDensity
                fluidPDFNew[i, indices, 2] = fluidPDFOld[i, indices, 2] + 2./3. * \
                                (tmpDensity * velocityY - fluidPDFOld[i, indices, 2] + \
                                fluidPDFNew[i, indices, 4] - fluidPDFOld[i, indices, 5] - \
                                fluidPDFOld[i, indices, 6] + fluidPDFNew[i, indices, 7] + \
                                fluidPDFNew[i, indices, 8])
                fluidPDFNew[i, indices, 5] = fluidPDFOld[i, indices, 5] + 1./2. * \
                               (-fluidPDFNew[i, indices, 1] + fluidPDFNew[i, indices, 3] - \
                                fluidPDFOld[i, indices, 5] + fluidPDFOld[i, indices, 6] + \
                                fluidPDFNew[i, indices, 7] - fluidPDFNew[i, indices, 8] - \
                                1./2. * forceY[i, indices]) + 1./6. * (tmpDensity * \
                                velocityY - fluidPDFOld[i, indices, 2] + fluidPDFNew[i, indices, 4] - \
                                fluidPDFOld[i, indices, 5] - fluidPDFOld[i, indices, 6] + \
                                fluidPDFNew[i, indices, 7] + fluidPDFNew[i, indices, 8] - \
                                1./2. * forceY[i, indices])
                fluidPDFNew[i, indices, 6] = fluidPDFOld[i, indices, 6] - 1./2. * \
                               (-fluidPDFNew[i, indices, 1] + fluidPDFNew[i, indices, 3] - \
                                fluidPDFOld[i, indices, 5] + fluidPDFOld[i, indices, 6] + \
                                fluidPDFNew[i, indices, 7] - fluidPDFNew[i, indices, 8] - \
                                1./2. * forceY[i, indices]) + 1./6. * (tmpDensity * \
                                velocityY - fluidPDFOld[i, indices, 2] + fluidPDFNew[i, indices, 4] - \
                                fluidPDFOld[i, indices, 5] - fluidPDFOld[i, indices, 6] + \
                                fluidPDFNew[i, indices, 7] + fluidPDFNew[i, indices, 8] - \
                                1./2. * forceY[i, indices])
                fluidRho[i, indices] = tmpDensity
                         
"""
New function for collision-streaming function for saving the memories used by 
equilibrium function, forces on X-Y direction and primary velocities
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:, :], float64[:], \
                float64[:], float64[:, :], float64[:, :], float64[:, :, :], \
                float64[:, :, :], int64[:], int64[:], float64[:, :], float64[:, :])')
def interactionCollisionProcess(totalNodes, numFluids, xDim, weightInter, tau, \
                                interCoeff, interSolid, weightsCoeff, fluidRho, \
                                fluidPotential, fluidPDF, fluidPDFNew, fluidNodes, \
                                neighboringNodes, forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        #primary velocity in x and y direction
        tmpVXTau = 0.; tmpVYTau = 0.
        tmpRhoTau = 0.;
        for i in range(numFluids):
            tmpVXTau += (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 5] - fluidPDF[i, indices, 6] - \
                        fluidPDF[i, indices, 7] + fluidPDF[i, indices, 8]) / tau[i]
            tmpVYTau += (fluidPDF[i, indices, 2] - fluidPDF[i, indices, 4] + \
                        fluidPDF[i, indices, 5] + fluidPDF[i, indices, 6] - \
                        fluidPDF[i, indices, 7] - fluidPDF[i, indices, 8]) / tau[i]
            tmpRhoTau += fluidRho[i, indices] / tau[i]
        tmpPrimeVx = tmpVXTau / tmpRhoTau; tmpPrimeVy = tmpVYTau / tmpRhoTau
        #Force on each lattice
 
        for i in range(numFluids):
            tmpStart = 8 * indices
            tmpFX = 0.; tmpFY = 0.
            forceX[i, indices] = 0.; forceY[i, indices] = 0.    
            if (neighboringNodes[tmpStart] != -1):
                tmpE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[0] * interCoeff[i, j] * \
                              fluidPotential[i, indices] * fluidPotential[j, tmpE] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./9. * interSolid[i] * fluidPotential[i, indices] * (1.)
            #Northern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpN = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFY += -weightInter[1] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFY += -1./9. * interSolid[i] * fluidPotential[i, indices] * (1.)
            #Western Point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[2] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW] * (-1.)
            elif(neighboringNodes[tmpStart] == -1):
                tmpFX += -1./9. * interSolid[i] * fluidPotential[i, indices] * (-1.)
            #Southern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpS = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFY += -weightInter[3] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                    tmpFY += -1./9. * interSolid[i] * fluidPotential[i, indices] * (-1.)
            #Northeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[4] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
                    tmpFY += -weightInter[4] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
            elif(neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
            #Northwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[5] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (-1.)
                    tmpFY += -weightInter[5] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
            #Southwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[6] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
                    tmpFY += -weightInter[6] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
            #Southeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[7] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (1.)
                    tmpFY += -weightInter[7] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
            forceX[i, indices] = tmpFX; forceY[i, indices] = tmpFY
            tmpEquilibriumVX = 0.; tmpEquilibriumVY = 0.
         
            tmpEquilibriumVX = tmpPrimeVx + tau[i] * tmpFX / fluidRho[i, indices]
            tmpEquilibriumVY = tmpPrimeVy + tau[i] * tmpFY / fluidRho[i, indices]
        #calculate the collision with the equilibrium function
 
            tmpEVSquare = tmpEquilibriumVX * tmpEquilibriumVX + tmpEquilibriumVY * \
                          tmpEquilibriumVY
            fluidPDF[i, indices, 0] = (1 - 1./tau[i]) * fluidPDF[i, indices, 0] + weightsCoeff[0] * \
                        fluidRho[i, indices] / tau[i] * (1. - 1.5 * tmpEVSquare)
            #Eastern node
 
            fluidPDF[i, indices, 1] = (1 - 1./tau[i]) * fluidPDF[i, indices, 1] + weightsCoeff[1] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * tmpEquilibriumVX + \
                        4.5 * (tmpEquilibriumVX * tmpEquilibriumVX) - \
                        1.5 * tmpEVSquare)
 
            fluidPDF[i, indices, 2] = (1. - 1./tau[i]) * fluidPDF[i, indices, 2] + weightsCoeff[2] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * tmpEquilibriumVY + \
                        4.5 * (tmpEquilibriumVY * tmpEquilibriumVY) - \
                        1.5 * tmpEVSquare)
 
            fluidPDF[i, indices, 3] = (1. - 1./tau[i]) * fluidPDF[i, indices, 3] + weightsCoeff[3] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * (-tmpEquilibriumVX) + \
                        4.5 * ((-tmpEquilibriumVX) * (-tmpEquilibriumVX)) - \
                        1.5 * tmpEVSquare)
 
            fluidPDF[i, indices, 4] = (1. - 1./tau[i]) * fluidPDF[i, indices, 4] + weightsCoeff[4] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * (-tmpEquilibriumVY) + \
                        4.5 * ((-tmpEquilibriumVY) * (-tmpEquilibriumVY)) - \
                        1.5 * tmpEVSquare)
 
            tmpSquare = (tmpEquilibriumVX + tmpEquilibriumVY) * (tmpEquilibriumVX + \
                        tmpEquilibriumVY)
            fluidPDF[i, indices, 5] = (1. - 1./tau[i]) * fluidPDF[i, indices, 5] + weightsCoeff[5] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * (tmpEquilibriumVX + \
                        tmpEquilibriumVY) + 4.5 * tmpSquare - 1.5 * tmpEVSquare)
 
            tmpSquare = (-tmpEquilibriumVX + tmpEquilibriumVY) * (-tmpEquilibriumVX + \
                        tmpEquilibriumVY)
            fluidPDF[i, indices, 6] = (1. - 1./tau[i]) * fluidPDF[i, indices, 6] + weightsCoeff[6] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * (-tmpEquilibriumVX + \
                        tmpEquilibriumVY) + 4.5 * tmpSquare - 1.5 * tmpEVSquare)
 
            tmpSquare = (-tmpEquilibriumVX - tmpEquilibriumVY) * (-tmpEquilibriumVX - \
                        tmpEquilibriumVY)
            fluidPDF[i, indices, 7] = (1. - 1./tau[i]) * fluidPDF[i, indices, 7] + weightsCoeff[7] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * (-tmpEquilibriumVX - \
                        tmpEquilibriumVY) + 4.5 * tmpSquare - 1.5 * tmpEVSquare)
 
            tmpSquare = (tmpEquilibriumVX - tmpEquilibriumVY) * (tmpEquilibriumVX - \
                        tmpEquilibriumVY)
            fluidPDF[i, indices, 8] = (1. - 1./tau[i]) * fluidPDF[i, indices, 8] + weightsCoeff[8] * \
                        fluidRho[i, indices] / tau[i] * (1. + 3. * (tmpEquilibriumVX - \
                        tmpEquilibriumVY) + 4.5 * tmpSquare - 1.5 * tmpEVSquare)
                        
"""
Implement EOF scheme for 2D lattice botlzmann Model
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:, :], \
                float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :, :], \
                float64[:, :, :], int64[:], int64[:], float64[:, :], float64[:, :])')
def interactionCollisionEOFProcess(totalNodes, numFluids, xDim, weightInter, \
                                   tauReverse, interCoeff, interSolid, weigthCoeff, 
                                   fluidRho, fluidPotential, fluidPDF, fluidPDFNew, \
                                   fluidNodes, neighboringNodes, forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        #velocity used for calculation f_eq
        tmpVXTau = 0.; tmpVYTau = 0.
        tmpRhoTau = 0.;
        for i in range(numFluids):
            tmpVXTau += (fluidPDF[i, indices, 1] - fluidPDF[i, indices, 3] + \
                        fluidPDF[i, indices, 5] - fluidPDF[i, indices, 6] - \
                        fluidPDF[i, indices, 7] + fluidPDF[i, indices, 8]) * tauReverse[i]
            tmpVYTau += (fluidPDF[i, indices, 2] - fluidPDF[i, indices, 4] + \
                        fluidPDF[i, indices, 5] + fluidPDF[i, indices, 6] - \
                        fluidPDF[i, indices, 7] - fluidPDF[i, indices, 8]) * tauReverse[i]
            tmpRhoTau += fluidRho[i, indices] * tauReverse[i]
        tmpPrimeVx = tmpVXTau / tmpRhoTau; tmpPrimeVy = tmpVYTau / tmpRhoTau
        tmpPrimeVSquare = tmpPrimeVx * tmpPrimeVx + tmpPrimeVy * tmpPrimeVy
        #force on each lattice
        #Force on each lattice
 
        for i in range(numFluids):
            tmpStart = 8 * indices
            tmpFX = 0.; tmpFY = 0.
            forceX[i, indices] = 0.; forceY[i, indices] = 0.    
            if (neighboringNodes[tmpStart] != -1):
                tmpE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[0] * interCoeff[i, j] * \
                              fluidPotential[i, indices] * fluidPotential[j, tmpE] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./9. * interSolid[i] * fluidPotential[i, indices] * (1.)
            #Northern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpN = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFY += -weightInter[1] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFY += -1./9. * interSolid[i] * fluidPotential[i, indices] * (1.)
            #Western Point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[2] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW] * (-1.)
            elif(neighboringNodes[tmpStart] == -1):
                tmpFX += -1./9. * interSolid[i] * fluidPotential[i, indices] * (-1.)
            #Southern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpS = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFY += -weightInter[3] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                    tmpFY += -1./9. * interSolid[i] * fluidPotential[i, indices] * (-1.)
            #Northeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[4] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
                    tmpFY += -weightInter[4] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
            elif(neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
            #Northwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[5] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (-1.)
                    tmpFY += -weightInter[5] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
            #Southwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[6] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
                    tmpFY += -weightInter[6] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
            #Southeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[7] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (1.)
                    tmpFY += -weightInter[7] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
            forceX[i, indices] = tmpFX; forceY[i, indices] = tmpFY
            #calculate f_F (force term)
            tmpEquilForce = cuda.local.array(shape = (9, ), dtype = float64)
            tmpEquil = cuda.local.array(shape = (9, ), dtype = float64)
            tmpRhoCs = 3. / fluidRho[i, indices]
            
            #Lattice f_{F, 0}
            tmpEquil[0] = 4./9 * fluidRho[i, indices] * (1. - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[0] = (tmpFX * (-tmpPrimeVx) + tmpFY * (-tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[0]
            #Lattice f_{F, 1}
            tmpEquil[1] = 1./9. * fluidRho[i, indices] * (1. + 3. * tmpPrimeVx + \
                        + 4.5 * (tmpPrimeVx) * (tmpPrimeVx) - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[1] = (tmpFX * (1. - tmpPrimeVx) + tmpFY * (-tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[1]
            #Lattice f_{F,2}
            tmpEquil[2] = 1./9. * fluidRho[i, indices] * (1. + 3. * tmpPrimeVy + \
                        + 4.5 * tmpPrimeVy * tmpPrimeVy - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[2] = (tmpFX * (-tmpPrimeVx) + tmpFY * (1. - tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[2]
            #Lattice f_{F, 3}
            tmpEquil[3] = 1./9. * fluidRho[i, indices] * (1. + 3. * (-tmpPrimeVx) + \
                        + 4.5 * (-tmpPrimeVx) * (-tmpPrimeVx) - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[3] = (tmpFX * (-1. - tmpPrimeVx) + tmpFY * (-tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[3]
            #Lattice f_{F, 4}
            tmpEquil[4] = 1./9. * fluidRho[i, indices] * (1. + 3. * (-tmpPrimeVy) + \
                        + 4.5 * (-tmpPrimeVy) * (-tmpPrimeVy) - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[4] = (tmpFX * (-tmpPrimeVx) + tmpFY * (-1. - tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[4]
            #Lattice f_{F, 5}
            tmpEquil[5] = 1./36. * fluidRho[i, indices] * (1. + 3. * (tmpPrimeVx + \
                        tmpPrimeVy) + 4.5 * (tmpPrimeVx + tmpPrimeVy) * (tmpPrimeVx + \
                        tmpPrimeVy) - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[5] = (tmpFX * (1. - tmpPrimeVx) + tmpFY * (1. - tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[5]
            #Lattice f_{F, 6}
            tmpEquil[6] = 1./36. * fluidRho[i, indices] * (1. + 3. * (-tmpPrimeVx + \
                        tmpPrimeVy) + 4.5 * (-tmpPrimeVx + tmpPrimeVy) * (-tmpPrimeVx + \
                        tmpPrimeVy) - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[6] = (tmpFX * (-1. - tmpPrimeVx) + tmpFY * (1. - tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[6]
            #Lattice f_{F, 7}
            tmpEquil[7]= 1./36. * fluidRho[i, indices] * (1. + 3. * (-tmpPrimeVx - \
                        tmpPrimeVy) + 4.5 * (-tmpPrimeVx - tmpPrimeVy) * (-tmpPrimeVx - \
                        tmpPrimeVy) - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[7] = (tmpFX * (-1. - tmpPrimeVx) + tmpFY * (-1. - tmpPrimeVy)) * \
                                tmpRhoCs * tmpEquil[7]
            #Lattice f_{8, 7}
            tmpEquil[8] = 1./36. * fluidRho[i, indices] * (1. + 3. * (tmpPrimeVx - \
                        tmpPrimeVy) + 4.5 * (tmpPrimeVx - tmpPrimeVy) * (tmpPrimeVx - \
                        tmpPrimeVy) - 1.5 * tmpPrimeVSquare)
            tmpEquilForce[8] = (tmpFX * (1. - tmpPrimeVx) + tmpFY * (-1. - tmpPrimeVy)) * \
                        tmpRhoCs * tmpEquil[8]
            
            #collision process
            for j in range(9):
                fluidPDF[i, indices, j] = fluidPDF[i, indices, j] * (1. - \
                                           tauReverse[i]) + tmpEquil[j] + tmpEquilForce[j] * \
                                           (1. - 0.5 * tauReverse[i])
                                         
                        
"""
Implement TRT collision process for all the functions of population of each fluid
"""
@cuda.jit(device=True)
def collisionTRTProcess(numDirection, typeFluids, direcArray, symmDirecArray, \
                        coeffRatio, fluidRho, evenRelaxation, oddRelaxation, fluidPDF):
    #for c_o: center of the lattice 
    tmpEquilibrium = 0.
    tmpCounterEquilibrium = 0.
    tmpSymmetric = 0.5 * fluidPDF[0]; tmpAntiSymmetric = 0.
    tmpSymmetricEquilib = 0.5 * tmpEquilibrium 
    tmpAntiSymmetricEquilib = 0.
    fluidPDF[0] = fluidPDF[0] - evenRelaxation * (tmpSymmetric - tmpSymmetricEquilib) - \
                    oddRelaxation* (tmpAntiSymmetric - tmpAntiSymmetricEquilib)
    #for c_1 to c_8
    for i in range(1, 9):
        tmpEquilibrium = 0.
        tmpCounterEquilibrium = 0.
        tmpSymmetric = 0.5 * (fluidPDF[direcArray[i]] + fluidPDF[symmDirecArray[i]])
        tmpAntiSymmetric = 0.5 * (fluidPDF[direcArray[i]] - fluidPDF[symmDirecArray[i]])
        tmpSymmetricEquilib = 0.5 * (tmpEquilibrium + tmpCounterEquilibrium)
        tmpAntiSymmetricEquilib = 0.5 * (tmpEquilibrium - tmpCounterEquilibrium)
        
        fluidPDF[i] = fluidPDF[i] - evenRelaxation * (tmpSymmetric - tmpSymmetricEquilib) - \
                            oddRelaxation * (tmpAntiSymmetric - tmpAntiSymmetricEquilib)
        
"""
device function for link-bounce back rule
""" 
@cuda.jit(device = True)
def calLinkBounceBack(density, wcoefff, eX, eY, velocityX, velocityY):
    linkBB = 3. * (2. * density * wcoefff * (eX * velocityX + eY * velocityY))
    return linkBB

"""
Calculate the streaming process1
"""
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:,:], \
            float64[:, :, :], float64[:, :, :], float64[:], float64[:], float64[:])')
def calStreaming1withLinkGPU(totalNum, numFluids, xDim, fluidNodes, neighboringNodes, \
                    fluidRho, fluidPDF, fluidPDFNew, physicalVX, physicalVY, weightCoeff):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNum):
        tmpWcoeff= cuda.local.array(shape = (9, ), dtype = float64)
        for i in range(9):
            tmpWcoeff[i] = weightCoeff[i]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        #Eastern node
        tmpStart = 8 * indices
        if (neighboringNodes[tmpStart] != -1):
            tmpE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpE, 1] = fluidPDF[i, indices, 1]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = 1.0; tmpEY = 0.
                tmpCoeff = tmpWcoeff[1]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 3] = fluidPDF[i, indices, 1] - tmpLinkBB
        #Northern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpN = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpN, 2] = fluidPDF[i, indices, 2]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = 0.; tmpEY = 1.0
                tmpCoeff = tmpWcoeff[2]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 4] = fluidPDF[i, indices, 2] - tmpLinkBB
        #Western node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpW, 3] = fluidPDF[i, indices, 3]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = -1.0; tmpEY = 0.0
                tmpCoeff = tmpWcoeff[3]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 1] = fluidPDF[i, indices, 3] - tmpLinkBB
        #Southern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpS = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpS, 4] = fluidPDF[i, indices, 4]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = 0.; tmpEY = -1.0
                tmpCoeff = tmpWcoeff[4]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 2] = fluidPDF[i, indices, 4] - tmpLinkBB
        #Northeastern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpNE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpNE, 5] = fluidPDF[i, indices, 5]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = 1.0; tmpEY = 1.0
                tmpCoeff = tmpWcoeff[5]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 7] = fluidPDF[i, indices, 5] - tmpLinkBB
        #Northwestern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpNW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpNW, 6] = fluidPDF[i, indices, 6]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = -1.0; tmpEY = 1.0
                tmpCoeff = tmpWcoeff[6]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 8] = fluidPDF[i, indices, 6] - tmpLinkBB
        #Southwestern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpSW = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpSW, 7] = fluidPDF[i, indices, 7]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = -1.0; tmpEY = -1.0
                tmpCoeff = tmpWcoeff[7]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 5] = fluidPDF[i, indices, 7] - tmpLinkBB
        #Sourtheastern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] != -1):
            tmpSE = neighboringNodes[tmpStart]
            for i in range(numFluids):
                fluidPDFNew[i, tmpSE, 8] = fluidPDF[i, indices, 8]
        elif (neighboringNodes[tmpStart] == -1):
            for i in range(numFluids):
                tmpDensity = fluidRho[i, indices]
                tmpEX = 1.0; tmpEY = -1.0
                tmpCoeff = tmpWcoeff[8]
                tmpLinkBB = calLinkBounceBack(tmpDensity, tmpCoeff, tmpEX, tmpEY, \
                                              tmpVX, tmpVY)
                fluidPDFNew[i, indices, 6] = fluidPDF[i, indices, 8] - tmpLinkBB
    cuda.syncthreads()
    
"""
Guo's force term in collision
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:, :], float64[:], \
                float64[:], float64[:, :], int64[:], int64[:], float64[:, :], \
                float64[:, :])')
def interactionForceGuo(totalNodes, numFluids, xDim, weightInter, \
                                interCoeff, interSolid, weightsCoeff, \
                                fluidPotential, fluidNodes, \
                                neighboringNodes, forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        #Force on each lattice
 
        for i in range(numFluids):
            tmpStart = 8 * indices
            tmpFX = 0.; tmpFY = 0.
            forceX[i, indices] = 0.; forceY[i, indices] = 0.    
            if (neighboringNodes[tmpStart] != -1):
                tmpE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[0] * interCoeff[i, j] * \
                              fluidPotential[i, indices] * fluidPotential[j, tmpE] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./9. * interSolid[i] * fluidPotential[i, indices] * (1.)
            #Northern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpN = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFY += -weightInter[1] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFY += -1./9. * interSolid[i] * fluidPotential[i, indices] * (1.)
            #Western Point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[2] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW] * (-1.)
            elif(neighboringNodes[tmpStart] == -1):
                tmpFX += -1./9. * interSolid[i] * fluidPotential[i, indices] * (-1.)
            #Southern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpS = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFY += -weightInter[3] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                    tmpFY += -1./9. * interSolid[i] * fluidPotential[i, indices] * (-1.)
            #Northeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[4] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
                    tmpFY += -weightInter[4] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
            elif(neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
            #Northwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[5] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (-1.)
                    tmpFY += -weightInter[5] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
            #Southwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[6] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
                    tmpFY += -weightInter[6] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
            #Southeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpFX += -weightInter[7] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (1.)
                    tmpFY += -weightInter[7] * interCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (-1.)
            elif (neighboringNodes[tmpStart] == -1):
                tmpFX += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFY += -1./36. * interSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
            forceX[i, indices] = tmpFX; forceY[i, indices] = tmpFY

"""
Collision process with Guo's scheme
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:], float64[:], float64[:, :, :])')
def calCollisionGuo(totalNodes, numFluids, xDim, tau, weightsCoeff, unitEX, unitEY, \
                                fluidRho, forceX, forceY, physicalVX, \
                                physicalVY, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        #Force on each lattice
        for i in range(numFluids):      
            tmpEquilVX = physicalVX[indices]; tmpEquilVY = physicalVY[indices]
            tmpEquilV2 = tmpEquilVX * tmpEquilVX + tmpEquilVY * tmpEquilVY
            tmpFX = forceX[i, indices]; tmpFY = forceY[i, indices]
            for j in range(9):
                tmpForceTerm = weightsCoeff[j] * ((3. * (unitEX[j] - physicalVX[indices]) + \
                                9. * unitEX[j] * (unitEX[j] * physicalVX[indices] + \
                                unitEY[j] * physicalVY[indices])) * tmpFX + (3. * \
                                (unitEY[j] - physicalVY[indices]) + 9. * unitEY[j] * \
                                (unitEX[j] * physicalVX[indices] + unitEY[j] * \
                                 physicalVY[indices])) * tmpFY)
                tmpEquilibrum = weightsCoeff[j] * fluidRho[i, indices] * (1. + 3. * \
                                (unitEX[j] * tmpEquilVX + unitEY[j] * tmpEquilVY) + \
                                4.5 * (unitEX[j] * tmpEquilVX + unitEY[j] * tmpEquilVY) * \
                                (unitEX[j] * tmpEquilVX + unitEY[j] * tmpEquilVY) - 1.5 * \
                                tmpEquilV2)
                fluidPDF[i, indices, j] = (1. - 1./tau[i]) * fluidPDF[i, indices, j]  + \
                                        1./tau[i] * tmpEquilibrum + (1. - 1./(2. * \
                                        tau[i])) * tmpForceTerm