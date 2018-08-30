"""
Accelerated R-K color gradient LBM for two-phase flow
"""

import sys, os
import math
import numpy as np

from numba import cuda, jit, float64

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
        #Northeastern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpF]
        #Northwestern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpB]
        #Southwestern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpB]
        #southeastern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpF]

"""
Calculate the neighboring nodes for wetting nodes in solid
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:, :], int64[:])')
def fillNeighboringWettingNodes(totalWettingNodes, nx, ny, xDim, wettingNodes, \
                                domainNewIndex, neighboringWettingNodes):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
#    ty = cuda.threadIdx.y; by = cuda.blockIdx.y; bDimY = cuda.blockDim.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalWettingNodes):
        tmpStart = 8 * indices
        tmpLoc = wettingNodes[indices]
        i = int(tmpLoc / nx); j = tmpLoc % nx
        tmpF = j + 1 if j < nx - 1 else 0
        tmpB = j - 1 if j > 0 else (nx - 1)
        tmpU = i + 1 if i < ny - 1 else 0
        tmpL = i - 1 if i > 0 else (ny - 1)
        #Eastern node
        neighboringWettingNodes[tmpStart] = domainNewIndex[i, tmpF]
        #Northern node
        tmpStart += 1
        neighboringWettingNodes[tmpStart] = domainNewIndex[tmpU, j]
        #Western node
        tmpStart += 1
        neighboringWettingNodes[tmpStart] = domainNewIndex[i, tmpB]
        #Southern node
        tmpStart += 1
        neighboringWettingNodes[tmpStart] = domainNewIndex[tmpL, j]
        #Northeastern node
        tmpStart += 1
        neighboringWettingNodes[tmpStart] = domainNewIndex[tmpU, tmpF]
        #Northwestern node
        tmpStart += 1
        neighboringWettingNodes[tmpStart] = domainNewIndex[tmpU, tmpB]
        #Southwestern node
        tmpStart += 1
        neighboringWettingNodes[tmpStart] = domainNewIndex[tmpL, tmpB]
        #southeastern node
        tmpStart += 1
        neighboringWettingNodes[tmpStart] = domainNewIndex[tmpL, tmpF]

 
"""
Calculate the macro-density for 'red' and 'blue' fluids
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:], float64[:])')
def calMacroDensityRKGPU2D(totalNodes, xDim, fluidPDFR, fluidPDFB, fluidRhoR, \
                           fluidRhoB):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpRhoR = 0.; tmpRhoB = 0.
        for i in range(9):
            tmpRhoR += fluidPDFR[indices, i]
            tmpRhoB += fluidPDFB[indices, i]
#            deviceCollisionR[indices, i] = fluidPDFR[indices, i]
#            deviceCollisionB[indices, i] = fluidPDFB[indices, i]
        fluidRhoR[indices] = tmpRhoR
        fluidRhoB[indices] = tmpRhoB
        
"""
Calculate the macro-scale velocity
"""
@cuda.jit('void(int64, int64, int64, int64[:], float64[:, :], float64[:, :], float64[:], float64[:], \
                float64[:], float64[:])')
def calPhysicalVelocityRKGPU2D(totalNodes, nx, xDim, fluidNodes, fluidPDFR, fluidPDFB, fluidRhoR, \
                               fluidRhoB, physicalVX, physicalVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        tmpVX = 0.; tmpVY = 0.; tmpRho = 0.
        tmpIndex = fluidNodes[indices]
#        if (indices >= 3 * nx):
#        if (tmpIndex >= 3 * nx):
        tmpRho = fluidRhoB[indices] + fluidRhoR[indices]
        tmpVX = fluidPDFR[indices, 1] - fluidPDFR[indices, 3] + fluidPDFR[indices, 5] - \
                fluidPDFR[indices, 6] - fluidPDFR[indices, 7] + fluidPDFR[indices, 8] + \
                fluidPDFB[indices, 1] - fluidPDFB[indices, 3] + fluidPDFB[indices, 5] - \
                fluidPDFB[indices, 6] - fluidPDFB[indices, 7] + fluidPDFB[indices, 8]
        physicalVX[indices] = tmpVX / tmpRho
        tmpVY = fluidPDFR[indices, 2] - fluidPDFR[indices, 4] + fluidPDFR[indices, 5] + \
                fluidPDFR[indices, 6] - fluidPDFR[indices, 7] - fluidPDFR[indices, 8] + \
                fluidPDFB[indices, 2] - fluidPDFB[indices, 4] + fluidPDFB[indices, 5] + \
                fluidPDFB[indices, 6] - fluidPDFB[indices, 7] - fluidPDFB[indices, 8]
        physicalVY[indices] = tmpVY / tmpRho
    
"""
Calcualte the tau's value for a location
"""
@cuda.jit(device=True)
def calTau1AtLocation(Phi, delta, tauR, tauB):
    S1 = 2. * tauR * tauB / (tauR + tauB)
    S2 = 2. * (tauR - S1) / delta
    S3 = -S2 / (2. * delta)
    tmpTau1 = S1 + S2 * Phi + S3 * Phi * Phi
    return tmpTau1

@cuda.jit(device=True)
def calTau2AtLocation(Phi, delta, tauR, tauB):
    T1 = 2. * tauR * tauB / (tauR + tauB)
    T2 = 2. * (T1 - tauB) / delta
    T3 = T2 / (2. * delta)
    tmpTau2 = T1 + T2 * Phi + T3 * Phi * Phi
    return tmpTau2

"""
Calculate the equilibrium function value on each direction
"""
@cuda.jit(device=True)
def calEquilibriumRK2D(fluidRho, weightsCoeff, eX, eY, physicalVX, physicalVY):
    tmpEquilibrium = fluidRho *  weightsCoeff * (1 +(3. * (eX * physicalVX + \
                    eY * physicalVY) + 4.5 * (eX * physicalVX + eY * physicalVY) * \
                    (eX * physicalVX + eY * physicalVY) - 1.5 * (physicalVX * \
                    physicalVX + physicalVY * physicalVY)))
    return tmpEquilibrium

"""
Calculate the equilibrium function value on each direction
"""
@cuda.jit(device=True)
def calEquilibriumRK2DOriginal(fluidRho, constantC, weightsCoeff, eX, eY, physicalVX, physicalVY):
    tmpEquilibrium = fluidRho * (constantC + weightsCoeff * (3. * (eX * physicalVX + \
                    eY * physicalVY) + 4.5 * (eX * physicalVX + eY * physicalVY) * \
                    (eX * physicalVX + eY * physicalVY) - 1.5 * (physicalVX * \
                    physicalVX + physicalVY * physicalVY)))
    return tmpEquilibrium

"""
Calculate the first collision part
"""
@cuda.jit('void(int64, int64, float64, float64, float64, float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])')
def calRKCollision1GPU2DSRT(totalNodes, xDim, delta, tauR, tauB, unitEX, unitEY, \
                         constantCR, constantCB, weightsCoeff, physicalVX, physicalVY, \
                         fluidRhoR, fluidRhoB, fluidPDFR, fluidPDFB):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        sharedEX = cuda.shared.array(shape = (9,), dtype = float64)
        sharedEY = cuda.shared.array(shape = (9,), dtype = float64)
        sharedCCR = cuda.shared.array(shape = (9,), dtype = float64)
        sharedCCB = cuda.shared.array(shape = (9,), dtype = float64)
        sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
        for i in range(9):
            sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
            sharedCCR[i] = constantCR[i]; sharedCCB[i] = constantCB[i]
            sharedWeights[i] = weightsCoeff[i]
            
        Phi = (fluidRhoR[indices] - fluidRhoB[indices]) / (fluidRhoR[indices] + \
              fluidRhoB[indices])
        tmpTau = 1.0
        if (Phi > delta):
            tmpTau = tauR
        elif (Phi > 0 and Phi <= delta):
            tmpTau = calTau1AtLocation(Phi, delta, tauR, tauB)
        elif (Phi <= 0 and Phi >= -delta):
            tmpTau = calTau2AtLocation(Phi, delta, tauR, tauB)
        elif (Phi < -delta):
            tmpTau = tauB
        tmpRhoR = fluidRhoR[indices]; tmpRhoB = fluidRhoB[indices]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        for i in range(9):
            tmpEquilibriumR = calEquilibriumRK2DOriginal(tmpRhoR, sharedCCR[i], sharedWeights[i], \
                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpCollisionR1 = -1./tmpTau * (fluidPDFR[indices, i] - tmpEquilibriumR)
            tmpEquilibriumB = calEquilibriumRK2DOriginal(tmpRhoB, sharedCCB[i], sharedWeights[i], \
                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpCollisionB1 = -1./tmpTau * (fluidPDFB[indices, i] - tmpEquilibriumB)
            
            fluidPDFR[indices, i] = fluidPDFR[indices, i] + tmpCollisionR1
            fluidPDFB[indices, i] = fluidPDFB[indices, i] + tmpCollisionB1

"""
Calculate the second and third collisions
"""        
@cuda.jit('void(int64, int64, float64, float64, float64, float64, float64, \
                int64[:], int64[:], float64[:], float64[:], float64[:], float64[:],\
                float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :], \
                float64[:, :], float64[:], float64[:])')
def calRKCollision23GPU(totalNodes, xDim, betaCoeff, AkR, AkB, solidRhoR, solidRhoB, \
                        fluidNodes, neighboringNodes, constantB, weightsCoeff, \
                        unitEX, unitEY, schemeGradient, fluidRhoR, fluidRhoB, \
                        constantCR, constantCB, fluidPDFR, fluidPDFB, CGX, CGY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNodes):
        sharedEX = cuda.shared.array((9,), dtype = float64)
        sharedEY = cuda.shared.array((9,), dtype = float64)
        sharedCB = cuda.shared.array((9,), dtype = float64)
        sharedWeights = cuda.shared.array((9,), dtype = float64)
        sharedScheme = cuda.shared.array((9,), dtype = float64)
        sharedConsCR = cuda.shared.array((9,), dtype = float64)
        sharedConsCB = cuda.shared.array((9,), dtype = float64)
        for i in range(9):
            sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
            sharedCB[i] = constantB[i]
            sharedWeights[i] = weightsCoeff[i]
            sharedScheme[i] = schemeGradient[i]
            sharedConsCR[i] = constantCR[i]
            sharedConsCB[i] = constantCB[i]
        #calculate the color gradient around the location
        tmpGradientX = 0.; tmpGradientY = 0.
        tmpStart = 8 * indices; tmpIndex = 0
        for i in range(8):
            tmpIndex += 1
            tmpNeighboring = tmpStart + i
            if (neighboringNodes[tmpNeighboring] != -1):
                tmpLocNeighbor = neighboringNodes[tmpNeighboring]
                tmpGradientX += sharedScheme[tmpIndex] * sharedEX[tmpIndex] * \
                                (fluidRhoR[tmpLocNeighbor] - fluidRhoB[tmpLocNeighbor])
                tmpGradientY += sharedScheme[tmpIndex] * sharedEY[tmpIndex] * \
                                (fluidRhoR[tmpLocNeighbor] - fluidRhoB[tmpLocNeighbor])
            else:
                tmpDiff = solidRhoR - solidRhoB
                tmpGradientX += sharedScheme[tmpIndex] * sharedEX[tmpIndex] * \
                                tmpDiff
                tmpGradientY += sharedScheme[tmpIndex] * sharedEY[tmpIndex] * \
                                tmpDiff
        #Calculate the second collision part
        tmpSquareGradient = tmpGradientX * tmpGradientX + tmpGradientY * \
                            tmpGradientY
        tmpNormGradient = math.sqrt(tmpSquareGradient)
        CGX[indices] = tmpGradientX; CGY[indices] = tmpSquareGradient
        CGY[indices] = 0.
        for i in range(9):
            tmpSecondCollisionR = 0.; tmpSecondCollisionB = 0.
            if (tmpSquareGradient == 0.):
#            CGY[indices] = tmpSquareGradient
#            if (math.isnan(tmpPartCollision) == 1 or math.isinf(tmpPartCollision) == 1):
                tmpSecondCollisionR = 0.; tmpSecondCollisionB = 0.
            else:
                tmpPartCollision = sharedWeights[i] * math.pow((sharedEX[i] * tmpGradientX + \
                                sharedEY[i] * tmpGradientY), 2) / tmpSquareGradient
                tmpSecondCollisionR = AkR * 0.5 * tmpNormGradient * (tmpPartCollision - \
                                    sharedCB[i])
                tmpSecondCollisionB = AkB * 0.5 * tmpNormGradient * (tmpPartCollision - \
                                    sharedCB[i])
            fluidPDFR[indices, i] = fluidPDFR[indices, i] + tmpSecondCollisionR
            fluidPDFB[indices, i] = fluidPDFB[indices, i] + tmpSecondCollisionB
        #Calculate the re-coloring part
        tmpRhoSum = fluidRhoR[indices] + fluidRhoB[indices]
        tmpRhoMul = fluidRhoR[indices] * fluidRhoB[indices]
        tmpRhoSumSquare = tmpRhoSum * tmpRhoSum
        for i in range(9):
            tmpUnitSqrt = math.sqrt(sharedEX[i] * sharedEX[i] + sharedEY[i] * \
                        sharedEY[i])
            cosTheta = 0.
            if (tmpUnitSqrt == 0. or tmpNormGradient == 0.):
                cosTheta = 0.
            else:
                cosTheta = (sharedEX[i] * tmpGradientX + sharedEY[i] * tmpGradientY) / \
                        (math.sqrt(sharedEX[i] * sharedEX[i] + sharedEY[i] * \
                        sharedEY[i]) * tmpNormGradient)
#            if (math.isnan(cosTheta) == 1 or math.isinf(cosTheta) == 1):
#                cosTheta = 0.
            tmpFeqRho = fluidRhoR[indices] * sharedConsCR[i] + fluidRhoB[indices] * \
                        sharedConsCB[i]
            tmpSumPDF = fluidPDFR[indices, i] + fluidPDFB[indices, i]
            #re-coloring
            fluidPDFR[indices, i] = fluidRhoR[indices]/tmpRhoSum * tmpSumPDF + \
                            (betaCoeff * tmpRhoMul / tmpRhoSumSquare) * tmpFeqRho * \
                            cosTheta
            fluidPDFB[indices, i] = fluidRhoB[indices]/tmpRhoSum * tmpSumPDF - \
                            (betaCoeff * tmpRhoMul / tmpRhoSumSquare) * tmpFeqRho * \
                            cosTheta
    cuda.syncthreads()

"""
Calculate the streaming process1
"""
@cuda.jit('void(int64, int64, int64[:], int64[:], float64[:, :], \
            float64[:, :])')                        
def calStreaming1GPU(totalNum, xDim, fluidNodes, neighboringNodes, \
                    fluidPDF, fluidPDFNew):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNum):

        #Eastern node
        tmpStart = 8 * indices
        if (neighboringNodes[tmpStart] >= 0):
            tmpE = neighboringNodes[tmpStart]
            fluidPDFNew[tmpE, 1] = fluidPDF[indices, 1]
        elif (neighboringNodes[tmpStart] <0):
            fluidPDFNew[indices, 3] = fluidPDF[indices, 1]
        #Northern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] >= 0):
            tmpN = neighboringNodes[tmpStart]
            fluidPDFNew[tmpN, 2] = fluidPDF[indices, 2]
        elif (neighboringNodes[tmpStart] < 0):
            fluidPDFNew[indices, 4] = fluidPDF[indices, 2]
        #Western node
        tmpStart += 1
        if (neighboringNodes[tmpStart] >= 0):
            tmpW = neighboringNodes[tmpStart]
            fluidPDFNew[tmpW, 3] = fluidPDF[indices, 3]
        elif (neighboringNodes[tmpStart] < 0):
            fluidPDFNew[indices, 1] = fluidPDF[indices, 3]
        #Southern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] >= 0):
            tmpS = neighboringNodes[tmpStart]
            fluidPDFNew[tmpS, 4] = fluidPDF[indices, 4]
        elif (neighboringNodes[tmpStart] < 0):
            fluidPDFNew[indices, 2] = fluidPDF[indices, 4]
        #Northeastern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] >= 0):
            tmpNE = neighboringNodes[tmpStart]
            fluidPDFNew[tmpNE, 5] = fluidPDF[indices, 5]
        elif (neighboringNodes[tmpStart] < 0):
            fluidPDFNew[indices, 7] = fluidPDF[indices, 5]
        #Northwestern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] >= 0):
            tmpNW = neighboringNodes[tmpStart]
            fluidPDFNew[tmpNW, 6] = fluidPDF[indices, 6]
        elif (neighboringNodes[tmpStart] < 0):
            fluidPDFNew[indices, 8] = fluidPDF[indices, 6]
        #Southwestern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] >= 0):
            tmpSW = neighboringNodes[tmpStart]
            fluidPDFNew[tmpSW, 7] = fluidPDF[indices, 7]
        elif (neighboringNodes[tmpStart] < 0):
            fluidPDFNew[indices, 5] = fluidPDF[indices, 7]
        #Sourtheastern node
        tmpStart += 1
        if (neighboringNodes[tmpStart] >= 0):
            tmpSE = neighboringNodes[tmpStart]
            fluidPDFNew[tmpSE, 8] = fluidPDF[indices, 8]
        elif (neighboringNodes[tmpStart] < 0):
            fluidPDFNew[indices, 6] = fluidPDF[indices, 8]
    cuda.syncthreads()
    
"""
Calculate the streaming process 2
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :])')
def calStreaming2GPU(totalNum, xDim, fluidPDFNew, fluidPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
     
    if (indices < totalNum):
        for j in range(1, 9):
            fluidPDF[indices, j] = fluidPDFNew[indices, j]
    cuda.syncthreads()
    
"""
MRT scheme for the first collision
"""
"""
Calculate the first collision part
"""
@cuda.jit('void(int64, int64, float64, float64, float64, float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], \
                float64[:, :], float64[:])')
def calRKCollision1GPU2DMRT(totalNodes, xDim, delta, tauR, tauB, unitEX, unitEY, \
                         constantCR, constantCB, weightsCoeff, physicalVX, physicalVY, \
                         fluidRhoR, fluidRhoB, fluidPDFR, fluidPDFB, transformationM, \
                         inverseTM, collisionS):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
#    sharedEX = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedEY = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedCCR = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedCCB = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
    sharedTM = cuda.shared.array(shape = (9, 9), dtype = float64)
    sharedIM = cuda.shared.array(shape = (9, 9), dtype = float64)
    sharedCS = cuda.shared.array(shape = (9, ), dtype = float64)
    
    localTmpEqR = cuda.local.array(shape = (9, ), dtype = float64)
    localTmpEqB = cuda.local.array(shape = (9, ), dtype = float64)
    localTmpfPDFR = cuda.local.array(shape = (9, ), dtype = float64)
    localTmpfPDFB = cuda.local.array(shape = (9, ), dtype = float64)

    if (indices < totalNodes):
        
        for i in range(9):
#            sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#            sharedCCR[i] = constantCR[i]; sharedCCB[i] = constantCB[i]
#            sharedWeights[i] = weightsCoeff[i]
            sharedCS[i] = collisionS[i]
            for j in range(9):
                sharedTM[i, j] = transformationM[i, j]
                sharedIM[i, j] = inverseTM[i, j]
            
        Phi = (fluidRhoR[indices] - fluidRhoB[indices]) / (fluidRhoR[indices] + \
              fluidRhoB[indices])
        tmpTau = 1.0
        if (Phi > delta):
            tmpTau = tauR
        elif (Phi > 0 and Phi <= delta):
            tmpTau = calTau1AtLocation(Phi, delta, tauR, tauB)
        elif (Phi <= 0 and Phi >= -delta):
            tmpTau = calTau2AtLocation(Phi, delta, tauR, tauB)
        elif (Phi < -delta):
            tmpTau = tauB
        sharedCS[7] = 1./ tmpTau; sharedCS[8] = 1. / tmpTau
        tmpRhoR = fluidRhoR[indices]; tmpRhoB = fluidRhoB[indices]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        for i in range(9):
            tmpCCR = constantCR[i]; tmpCCB = constantCB[i]; tmpEX = unitEX[i]
            tmpEY = unitEY[i]
            localTmpEqR[i] = calEquilibriumRK2DOriginal(tmpRhoR, tmpCCR, weightsCoeff[i], \
                            tmpEX, tmpEY, tmpVX, tmpVY)
            localTmpEqB[i] = calEquilibriumRK2DOriginal(tmpRhoB, tmpCCB, weightsCoeff[i], \
                            tmpEX, tmpEY, tmpVX, tmpVY)
        for i in range(9):
            tmpFR = 0.; tmpFB = 0.; tmpFREQ = 0.; tmpFBEQ = 0.
            for j in range(9):
                tmpFR += sharedTM[i, j] * fluidPDFR[indices, j]
                tmpFB += sharedTM[i, j] * fluidPDFB[indices, j]
                tmpFREQ += sharedTM[i, j] * localTmpEqR[j]
                tmpFBEQ += sharedTM[i, j] * localTmpEqB[j]
            localTmpfPDFR[i] = tmpFR - tmpFREQ; localTmpfPDFB[i] = tmpFB - tmpFBEQ
        
        for i in range(9):
            localTmpfPDFR[i] = localTmpfPDFR[i] * sharedCS[i]
            localTmpfPDFB[i] = localTmpfPDFB[i] * sharedCS[i]
        
        for i in range(9):
            tmpFR = 0.; tmpFB = 0.;
            for j in range(9):
                tmpFR += sharedIM[i, j] * localTmpfPDFR[j]
                tmpFB += sharedIM[i, j] * localTmpfPDFB[j]
            fluidPDFR[indices, i] = fluidPDFR[indices, i] - tmpFR
            fluidPDFB[indices, i] = fluidPDFB[indices, i] - tmpFB
    cuda.syncthreads()
                
"""
Update the fluid distribution function on the outlet
"""
@cuda.jit("void(int64, int64, int64, int64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:, :])")
def copyFluidPDFLastStep(totalNodes, nx, xDim, fluidNodes, fluidPDFR, fluidPDFB, \
                         fluidPDFROld, fluidPDFBOld):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if indices < totalNodes:
        tmpIndex = fluidNodes[indices]
        if tmpIndex < 3 * nx:
            for j in range(9):
                fluidPDFROld[indices, j] = fluidPDFR[indices, j]
                fluidPDFBOld[indices, j] = fluidPDFB[indices, j]
    cuda.syncthreads()

"""
Update the fluid distribution function on the outlet
"""
@cuda.jit("void(int64, int64, int64, int64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:, :])")
def copyFluidPDFRecoverOutlet(totalNodes, nx, xDim, fluidNodes, fluidPDFR, fluidPDFB, \
                         fluidPDFROld, fluidPDFBOld):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if indices < totalNodes:
        tmpIndex = fluidNodes[indices]
        if tmpIndex < 3 * nx:
            for j in range(9):
                fluidPDFR[indices, j] = fluidPDFROld[indices, j]
                fluidPDFB[indices, j] = fluidPDFBOld[indices, j]
    cuda.syncthreads()

"""
Calculate the first collision part
"""
@cuda.jit('void(int64, int64, float64, float64, float64, float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], \
                float64[:, :])')
def calRKCollision1GPU2DSRTNew(totalNodes, xDim, delta, tauR, tauB, unitEX, unitEY, \
                         constantCR, constantCB, weightsCoeff, physicalVX, physicalVY, \
                         fluidRhoR, fluidRhoB, phiValue, fluidPDFR, fluidPDFB, \
                         collisionR1, collisionB1):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        sharedEX = cuda.shared.array(shape = (9,), dtype = float64)
        sharedEY = cuda.shared.array(shape = (9,), dtype = float64)
        sharedCCR = cuda.shared.array(shape = (9,), dtype = float64)
        sharedCCB = cuda.shared.array(shape = (9,), dtype = float64)
        sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
        for i in range(9):
            sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
            sharedCCR[i] = constantCR[i]; sharedCCB[i] = constantCB[i]
            sharedWeights[i] = weightsCoeff[i]
            
        Phi = phiValue[indices]
        tmpTau = 0.5 + 1. / ((1. + Phi)/(2. * (tauR - 0.5)) + (1. - Phi) / (2. * \
                             (tauB - 0.5)))
        tmpRhoR = fluidRhoR[indices]; tmpRhoB = fluidRhoB[indices]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        for i in range(9):
            tmpEquilibriumR = calEquilibriumRK2D(tmpRhoR, sharedWeights[i], \
                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpCollisionR1 = -1./tmpTau * (fluidPDFR[indices, i] - tmpEquilibriumR)
            tmpEquilibriumB = calEquilibriumRK2D(tmpRhoB, sharedWeights[i], \
                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpCollisionB1 = -1./tmpTau * (fluidPDFB[indices, i] - tmpEquilibriumB)
            
            fluidPDFR[indices, i] = fluidPDFR[indices, i] + tmpCollisionR1
            fluidPDFB[indices, i] = fluidPDFB[indices, i] + tmpCollisionB1
#            collisionR1[indices, i] = tmpCollisionR1
#            collisionB1[indices, i] = tmpCollisionB1

"""
Calculate the second and third collisions
"""        
@cuda.jit('void(int64, int64, float64, float64, float64, float64, \
                int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :], \
                float64[:, :], float64[:], float64[:])')
def calRKCollision23GPUNew(totalNodes, xDim, betaCoeff, AkR, AkB, solidPhi, \
                        fluidNodes, neighboringNodes, constantB, weightsCoeff, \
                        unitEX, unitEY, schemeGradient, fluidRhoR, fluidRhoB, phiValue, \
                        constantCR, constantCB, fluidPDFR, fluidPDFB, CGX, CGY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    sharedEX = cuda.shared.array((9,), dtype = float64)
    sharedEY = cuda.shared.array((9,), dtype = float64)
    sharedCB = cuda.shared.array((9,), dtype = float64)
    sharedWeights = cuda.shared.array((9,), dtype = float64)
    sharedScheme = cuda.shared.array((9,), dtype = float64)
    sharedConsCR = cuda.shared.array((9,), dtype = float64)
    sharedConsCB = cuda.shared.array((9,), dtype = float64)
    for i in range(9):
        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
        sharedCB[i] = constantB[i]
        sharedWeights[i] = weightsCoeff[i]
        sharedScheme[i] = schemeGradient[i]
        sharedConsCR[i] = constantCR[i]
        sharedConsCB[i] = constantCB[i]
    if (indices < totalNodes):
#        localPostPerturbationR = cuda.local.array((9,), dtype = float64)
#        localPostPerturbationB = cuda.local.array((9,), dtype = float64)
        #calculate the color gradient around the location
        Phi = phiValue[indices]
        tmpGradientX = 0.; tmpGradientY = 0.
        tmpStart = 8 * indices; tmpIndex = 0
        for i in range(8):
            tmpIndex += 1
            tmpNeighboring = tmpStart + i
            if (neighboringNodes[tmpNeighboring] != -1):
                tmpLocNeighbor = neighboringNodes[tmpNeighboring]
                tmpPhi = (fluidRhoR[tmpLocNeighbor] - fluidRhoB[tmpLocNeighbor]) / \
                         (fluidRhoR[tmpLocNeighbor] + fluidRhoB[tmpLocNeighbor])
                tmpGradientX += 3. * sharedWeights[tmpIndex] * sharedEX[tmpIndex] * \
                                (tmpPhi)
                tmpGradientY += 3. * sharedWeights[tmpIndex] * sharedEY[tmpIndex] * \
                                (tmpPhi)
            else:
                tmpDiff = solidPhi
                tmpGradientX += 3. * sharedWeights[tmpIndex] * sharedEX[tmpIndex] * \
                                tmpDiff
                tmpGradientY += 3. * sharedWeights[tmpIndex] * sharedEY[tmpIndex] * \
                                tmpDiff
#        tmpGradientX += 
        #Calculate the second collision part
        tmpSquareGradient = tmpGradientX * tmpGradientX + tmpGradientY * \
                            tmpGradientY
        tmpNormGradient = math.sqrt(tmpSquareGradient)
        CGX[indices] = tmpGradientX; CGY[indices] = tmpSquareGradient
        CGY[indices] = 0.
        #Calculate the re-coloring part
        tmpRhoSum = fluidRhoR[indices] + fluidRhoB[indices]
        tmpRhoMul = fluidRhoR[indices] * fluidRhoB[indices]
        tmpRhoSumSquare = tmpRhoSum * tmpRhoSum
        for i in range(9):
            tmpSecondCollisionR = 0.; tmpSecondCollisionB = 0.
            if (tmpSquareGradient == 0.):
#            CGY[indices] = tmpSquareGradient
#            if (math.isnan(tmpPartCollision) == 1 or math.isinf(tmpPartCollision) == 1):
                tmpSecondCollisionR = 0.; tmpSecondCollisionB = 0.
            else:
                tmpPartCollision = sharedWeights[i] * ((sharedEX[i] * tmpGradientX + \
                                sharedEY[i] * tmpGradientY) * (sharedEX[i] * tmpGradientX + \
                                sharedEY[i] * tmpGradientY)) / tmpSquareGradient
                tmpSecondCollisionR = AkR * 0.5 * tmpNormGradient * (tmpPartCollision - \
                                    sharedCB[i])
                tmpSecondCollisionB = AkB * 0.5 * tmpNormGradient * (tmpPartCollision - \
                                    sharedCB[i])
            fluidPDFR[indices, i] += tmpSecondCollisionR
            fluidPDFB[indices, i] += tmpSecondCollisionB
#            collisionR1[indices, i] += tmpSecondCollisionR #+ deviceCollisionR[indices, i]
#            collisionB1[indices, i] += tmpSecondCollisionB #+ deviceCollisionR[indices, i]

        for i in range(9):
            tmpUnitSqrt = math.sqrt(sharedEX[i] * sharedEX[i] + sharedEY[i] * \
                        sharedEY[i])
            cosTheta = 0.
            if (tmpUnitSqrt == 0. or tmpNormGradient == 0.):
                cosTheta = 0.
            else:
                cosTheta = (sharedEX[i] * tmpGradientX + sharedEY[i] * tmpGradientY) / \
                        (math.sqrt(sharedEX[i] * sharedEX[i] + sharedEY[i] * \
                        sharedEY[i]) * tmpNormGradient)
#            if (math.isnan(cosTheta) == 1 or math.isinf(cosTheta) == 1):
#                cosTheta = 0.
#            tmpFeqRho = fluidRhoR[indices] * sharedConsCR[i] + fluidRhoB[indices] * \
#                        sharedConsCB[i]
            tmpFeqRho = fluidRhoR[indices] * sharedWeights[i] + fluidRhoB[indices] * \
                        sharedWeights[i]
            tmpSumPDF = fluidPDFR[indices, i] + fluidPDFB[indices, i]
#            tmpSumPDF = collisionR1[indices, i] + collisionB1[indices, i]
#            tmpSumPDF = localPostPerturbationR[i] + localPostPerturbationB[i]
            #re-coloring
            fluidPDFR[indices, i] = fluidRhoR[indices]/tmpRhoSum * tmpSumPDF + \
                            (betaCoeff * tmpRhoMul / tmpRhoSumSquare) * tmpFeqRho * \
                            cosTheta #+ collisionR1[indices, i] + fluidPDFR[indices, i]
            fluidPDFB[indices, i] = fluidRhoB[indices]/tmpRhoSum * tmpSumPDF - \
                            (betaCoeff * tmpRhoMul / tmpRhoSumSquare) * tmpFeqRho * \
                            cosTheta #+ collisionB1[indices, i] + fluidPDFB[indices, i]
    cuda.syncthreads()
    
"""
MRT scheme for the first collision
"""
"""
Calculate the first collision part
"""
@cuda.jit('void(int64, int64, float64, float64, float64, float64, float64, float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], \
                float64[:, :], float64[:])')
def calRKCollision1GPU2DMRTNew(totalNodes, xDim, delta, tauR, tauB, bodyFX, bodyFY, unitEX, unitEY, \
                         constantCR, constantCB, weightsCoeff, physicalVX, physicalVY, \
                         fluidRhoR, fluidRhoB, phiValue, fluidPDFR, fluidPDFB, transformationM, \
                         inverseTM, collisionS):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
#    sharedEX = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedEY = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedCCR = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedCCB = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
    sharedTM = cuda.shared.array(shape = (9, 9), dtype = float64)
    sharedIM = cuda.shared.array(shape = (9, 9), dtype = float64)
    
    sharedCS = cuda.local.array(shape = (9, ), dtype = float64)
    
    localTmpEqR = cuda.local.array(shape = (9, ), dtype = float64)
    localTmpEqB = cuda.local.array(shape = (9, ), dtype = float64)
    localTmpfPDFR = cuda.local.array(shape = (9, ), dtype = float64)
    localTmpfPDFB = cuda.local.array(shape = (9, ), dtype = float64)
    for i in range(9):
        for j in range(9):
            sharedTM[i, j] = transformationM[i, j]
            sharedIM[i, j] = inverseTM[i, j]
    if (indices < totalNodes):
        
        for i in range(9):
#            sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#            sharedCCR[i] = constantCR[i]; sharedCCB[i] = constantCB[i]
#            sharedWeights[i] = weightsCoeff[i]
            sharedCS[i] = collisionS[i]
            
        Phi = phiValue[indices]
        tmpTau = 0.5 + 1. / ((1. + Phi)/(2. * (tauR - 0.5)) + (1. - Phi) / (2. * \
                             (tauB - 0.5)))
        sharedCS[7] = 1./ tmpTau; sharedCS[8] = 1./ tmpTau
        tmpRhoR = fluidRhoR[indices]; tmpRhoB = fluidRhoB[indices]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        for i in range(9):
            tmpEX = unitEX[i]
            tmpEY = unitEY[i]
            localTmpEqR[i] = calEquilibriumRK2D(tmpRhoR, weightsCoeff[i], \
                            tmpEX, tmpEY, tmpVX, tmpVY)
            localTmpEqB[i] = calEquilibriumRK2D(tmpRhoB, weightsCoeff[i], \
                            tmpEX, tmpEY, tmpVX, tmpVY)
        for i in range(9):
            tmpFR = 0.; tmpFB = 0.; tmpFREQ = 0.; tmpFBEQ = 0.
            for j in range(9):
                tmpFR += sharedTM[i, j] * fluidPDFR[indices, j]
                tmpFB += sharedTM[i, j] * fluidPDFB[indices, j]
                tmpFREQ += sharedTM[i, j] * localTmpEqR[j]
                tmpFBEQ += sharedTM[i, j] * localTmpEqB[j]
            localTmpfPDFR[i] = tmpFR - tmpFREQ; localTmpfPDFB[i] = tmpFB - tmpFBEQ
        
        for i in range(9):
            localTmpfPDFR[i] = localTmpfPDFR[i] * sharedCS[i]
            localTmpfPDFB[i] = localTmpfPDFB[i] * sharedCS[i]
        
        for i in range(9):
            tmpFR = 0.; tmpFB = 0.;
            for j in range(9):
                tmpFR += sharedIM[i, j] * localTmpfPDFR[j]
                tmpFB += sharedIM[i, j] * localTmpfPDFB[j]
            fluidPDFR[indices, i] = -tmpFR + 3. * weightsCoeff[i] * (unitEX[i] * \
                        bodyFX + unitEY[i] * bodyFY) + fluidPDFR[indices, i]
            fluidPDFB[indices, i] = -tmpFB + 3. * weightsCoeff[i] * (unitEX[i] * \
                        bodyFX + unitEY[i] * bodyFY) + fluidPDFB[indices, i]
    cuda.syncthreads()

"""
Calculate the phase field value in the domain
"""
@cuda.jit('void(int64, int64, float64[:], float64[:], float64[:])')
def calPhaseFieldPhi(totalNodes, xDim, fluidRhoR, fluidRhoB, phiValue):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        phiValue[indices] = (fluidRhoR[indices] - fluidRhoB[indices]) / (fluidRhoR[indices] + \
                fluidRhoB[indices])

"""
Outlet boundary condition for order parameter: Phi, so the interface can leave 
from the simulation domain.
"""
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:])')
def calNeumannPhiOutlet(totalNodes, xDim, nx, fluidNodes, neighboringNodes, phiValue):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpLoc = fluidNodes[indices]
        if (tmpLoc >= nx and tmpLoc < 2 * nx):
            tmpStart = 8 * indices
            tmpUpper = neighboringNodes[tmpStart + 1]
            tmpLower = neighboringNodes[tmpStart + 3]
            phiValue[indices] = phiValue[tmpUpper]
            phiValue[tmpLower] = phiValue[tmpUpper]
            
                            
"""
Add the modified periodic boundary condition for flow having body force
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :], float64[:, :])')
def calModifiedPeriodicBoundary(totalNodes, nx, ny, xDim, fluidNodes, neighboringNodes, \
                                fluidPDFR, fluidPDFB):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpLoc = fluidNodes[indices]
        if (tmpLoc >= 0 and tmpLoc < nx):
            tmpShift0 = fluidPDFR[indices, 2]; tmpShift1 = fluidPDFR[indices, 5]
            tmpShift2 = fluidPDFR[indices, 6]
            fluidPDFR[indices, 2] = fluidPDFB[indices, 2]
            fluidPDFR[indices, 5] = fluidPDFB[indices, 5]
            fluidPDFR[indices, 6] = fluidPDFB[indices, 6]
            fluidPDFB[indices, 2] = tmpShift0
            fluidPDFB[indices, 5] = tmpShift1
            fluidPDFB[indices, 6] = tmpShift2
        if (tmpLoc >= (ny - 1) * nx and tmpLoc < ny * nx):
            tmpShift0 = fluidPDFR[indices, 4]; tmpShift1 = fluidPDFR[indices, 7]
            tmpShift2 = fluidPDFR[indices, 8]
            fluidPDFR[indices, 4] = fluidPDFB[indices, 4]
            fluidPDFR[indices, 7] = fluidPDFB[indices, 7]
            fluidPDFR[indices, 8] = fluidPDFB[indices, 8]
            fluidPDFB[indices, 4] = tmpShift0
            fluidPDFB[indices, 7] = tmpShift1
            fluidPDFB[indices, 8] = tmpShift2
    cuda.syncthreads()
    
"""
calculate the total distribution
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :])')
def calTotalFluidPDF(totalNodes, xDim, fluidPDFR, fluidPDFB, fluidPDFTotal):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        for i in range(9):
            fluidPDFTotal[indices, i] = fluidPDFR[indices, i] + fluidPDFB[indices, i]
    cuda.syncthreads()
    
"""
Collision process 1 with total distribution function
"""
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])')
def calRKCollision1TotalGPU2DSRT(totalNodes, xDim, tauR, tauB, unitEX, unitEY, \
                         constantCR, constantCB, weightsCoeff, physicalVX, physicalVY, \
                         fluidRhoR, fluidRhoB, phiValue, fluidPDFTotal, collisionTotal1):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    sharedEX = cuda.shared.array(shape = (9,), dtype = float64)
    sharedEY = cuda.shared.array(shape = (9,), dtype = float64)
    sharedCCR = cuda.shared.array(shape = (9,), dtype = float64)
    sharedCCB = cuda.shared.array(shape = (9,), dtype = float64)
    sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
    for i in range(9):
        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
        sharedCCR[i] = constantCR[i]; sharedCCB[i] = constantCB[i]
        sharedWeights[i] = weightsCoeff[i]
        
    if (indices < totalNodes):
        Phi = phiValue[indices]
        tmpTau = 0.5 + 1. / ((1. + Phi)/(2. * (tauR - 0.5)) + (1. - Phi) / (2. * \
                             (tauB - 0.5)))
        tmpRhoR = fluidRhoR[indices]; tmpRhoB = fluidRhoB[indices]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        for i in range(9):
            tmpEquilibriumR = calEquilibriumRK2D(tmpRhoR, sharedWeights[i], \
                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpEquilibriumB = calEquilibriumRK2D(tmpRhoB, sharedWeights[i], \
                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpEquilibriumTotal = tmpEquilibriumR + tmpEquilibriumB
            collisionTotal1[indices, i] = -1./tmpTau * (fluidPDFTotal[indices, i] - \
                           tmpEquilibriumTotal) + fluidPDFTotal[indices, i]
            
"""
Perturbation process for two phase flow
"""
@cuda.jit('void(int64, int64, float64, float64, \
                int64[:], int64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:, :], float64[:], float64[:])')
def calRKCollision2TotalGPUNew(totalNodes, xDim, surfaceTA, solidPhi, \
                        fluidNodes, neighboringNodes, constantB, weightsCoeff, \
                        unitEX, unitEY, phiValue, collisionTotal2, \
                        gradientX, gradientY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    sharedEX = cuda.shared.array((9,), dtype = float64)
    sharedEY = cuda.shared.array((9,), dtype = float64)
    sharedB = cuda.shared.array((9,), dtype = float64)
    sharedWeights = cuda.shared.array((9,), dtype = float64)
    for i in range(9):
        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
        sharedB[i] = constantB[i]
        sharedWeights[i] = weightsCoeff[i]
    
    if (indices < totalNodes):
        tmpGradientX = 0.; tmpGradientY = 0.
        tmpStart = 8 * indices; tmpIndex = 0
        for i in range(8):
            tmpIndex += 1
            tmpNeighboringNode = neighboringNodes[tmpStart + i]
            if tmpNeighboringNode != -1:
                tmpGradientX += sharedWeights[tmpIndex] * phiValue[tmpNeighboringNode] * \
                                sharedEX[tmpIndex]
                tmpGradientY += sharedWeights[tmpIndex] * phiValue[tmpNeighboringNode] * \
                                sharedEY[tmpIndex]
            elif tmpNeighboringNode == -1:
                tmpGradientX += sharedWeights[tmpIndex] * solidPhi * sharedEX[tmpIndex]
                tmpGradientY += sharedWeights[tmpIndex] * solidPhi * sharedEY[tmpIndex]
        tmpGradientX = 3. * tmpGradientX
        tmpGradientY = 3. * tmpGradientY
        tmpGradientSquare = tmpGradientX * tmpGradientX + tmpGradientY * tmpGradientY
        tmpGradientNorm = math.sqrt(tmpGradientSquare)
        gradientX[indices] = tmpGradientX; gradientY[indices] = tmpGradientY
        for i in range(9):
            if tmpGradientSquare == 0:
                collisionTotal2[indices, i] = 0.
            else:
                collisionTotal2[indices, i] = surfaceTA / 2. * tmpGradientNorm * \
                    (sharedWeights[i] * (sharedEX[i] * tmpGradientX + sharedEY[i] * \
                     tmpGradientY) * (sharedEX[i] * tmpGradientX + sharedEY[i] * \
                     tmpGradientY) / tmpGradientSquare - sharedB[i])
            
#"""
#Recoloring step for two components
#"""
@cuda.jit('void(int64, int64, float64, float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:, :], float64[:, :])')
def calRecoloringProcess(totalNodes, xDim, betaValue, weightsCoeff, fluidRhoR, \
                         fluidRhoB, unitEX, unitEY, gradientX, gradientY, collisionTotal1, \
                         collisionTotal2, fluidPDFR, fluidPDFB, fluidPDFTotal):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    sharedEX = cuda.shared.array((9,), dtype = float64)
    sharedEY = cuda.shared.array((9,), dtype = float64)
    sharedWeights = cuda.shared.array((9,), dtype = float64)
    for i in range(9):
        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
        sharedWeights[i] = weightsCoeff[i]
    
    if indices < totalNodes:
        tmpGradientNorm = math.sqrt(gradientX[indices] * gradientX[indices] + \
                                    gradientY[indices] * gradientY[indices])
        costheta = 0.
        totalRho = fluidRhoR[indices] + fluidRhoB[indices]
        for i in range(9):
            tmpUnitNorm = math.sqrt(sharedEX[i] * sharedEX[i] + sharedEY[i] * \
                                    sharedEY[i])
            if tmpGradientNorm == 0. or tmpUnitNorm == 0.:
                costheta = 0.
            elif (tmpGradientNorm > 0. and tmpUnitNorm > 0.):
                costheta = (sharedEX[i] * gradientX[indices] + sharedEY[i] * \
                            gradientY[indices]) / (tmpUnitNorm * tmpGradientNorm)
            tmpTotalPDF = collisionTotal1[indices, i] + collisionTotal2[indices, i] #+ \
                        #fluidPDFTotal[indices, i]
            fluidPDFR[indices, i] = fluidRhoR[indices] / totalRho * tmpTotalPDF + \
                betaValue * fluidRhoR[indices] * fluidRhoB[indices] / totalRho * \
                sharedWeights[i] * costheta + fluidPDFR[indices, i]
            fluidPDFB[indices, i] = fluidRhoB[indices] / totalRho * tmpTotalPDF - \
                betaValue * fluidRhoR[indices] * fluidRhoB[indices] / totalRho * \
                sharedWeights[i] * costheta + fluidPDFB[indices, i]
                
"""
Calculate the color value rho^{N} value on the solid phase
"""
@cuda.jit('void(int64, int64, int64[:], float64[:], float64[:], float64[:])')
def calColorValueOnSolid(totalSolidWetting, xDim, neighboringWettingSolid, \
                         weightsCoeff, colorValueFluid, colorValueSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
#    sharedWeights = cuda.shared.array((9,), dtype = float64)
#    for i in range(9):
#        sharedWeights[i] = weightsCoeff[i]
    if indices < totalSolidWetting:
        tmpStart = 8 * indices; tmpIndices = 0; tmpSumCoeff = 0.; tmpSum = 0.
        for i in range(8):
            tmpIndices += 1
            tmpLoc = neighboringWettingSolid[tmpStart + i]
            if tmpLoc >= 0:
#                tmpSum += sharedWeights[tmpIndices] * colorValueFluid[tmpLoc]
#                tmpSumCoeff += sharedWeights[tmpIndices]
                tmpSum += weightsCoeff[tmpIndices] * colorValueFluid[tmpLoc]
                tmpSumCoeff += weightsCoeff[tmpIndices]
        colorValueSolid[indices] = tmpSum / tmpSumCoeff
    cuda.syncthreads()

@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:])')
def calRKInitialGradient(totalNodes, xDim, numColorSolid, \
                        fluidNodes, neighboringNodes, weightsCoeff, \
                        unitEX, unitEY, colorValueFluid, colorValueSolid, \
                        gradientX, gradientY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array((9,), dtype = float64)
#    sharedEY = cuda.shared.array((9,), dtype = float64)
#    sharedWeights = cuda.shared.array((9,), dtype = float64)
#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]

    if (indices < totalNodes):
        tmpGradientX = 0.; tmpGradientY = 0.
        tmpStart = 8 * indices; tmpIndex = 0
        for i in range(8):
            tmpIndex += 1
            tmpNeighboringNode = neighboringNodes[tmpStart + i]
#            if tmpNeighboringNode >= 0:
#                tmpGradientX += sharedWeights[tmpIndex] * colorValueFluid[tmpNeighboringNode] * \
#                                sharedEX[tmpIndex]
#                tmpGradientY += sharedWeights[tmpIndex] * colorValueFluid[tmpNeighboringNode] * \
#                                sharedEY[tmpIndex]
#            elif tmpNeighboringNode < 0:
#                tmpIndiciesSolid = -tmpNeighboringNode - 2
#                tmpGradientX += sharedWeights[tmpIndex] * colorValueSolid[tmpIndiciesSolid] * \
#                                sharedEX[tmpIndex]
#                tmpGradientY += sharedWeights[tmpIndex] * colorValueSolid[tmpIndiciesSolid] * \
#                                sharedEY[tmpIndex]
            if tmpNeighboringNode >= 0:
                tmpGradientX += weightsCoeff[tmpIndex] * colorValueFluid[tmpNeighboringNode] * \
                                unitEX[tmpIndex]
                tmpGradientY += weightsCoeff[tmpIndex] * colorValueFluid[tmpNeighboringNode] * \
                                unitEY[tmpIndex]
            elif tmpNeighboringNode < 0:
                tmpIndiciesSolid = -tmpNeighboringNode - 2
                tmpGradientX += weightsCoeff[tmpIndex] * colorValueSolid[tmpIndiciesSolid] * \
                                unitEX[tmpIndex]
                tmpGradientY += weightsCoeff[tmpIndex] * colorValueSolid[tmpIndiciesSolid] * \
                                unitEY[tmpIndex]
        tmpGradientX = 3. * tmpGradientX
        tmpGradientY = 3. * tmpGradientY
        gradientX[indices] = tmpGradientX; gradientY[indices] = tmpGradientY
#        unitInitialNx[indices] = tmpGradientX / tmpGradientNorm
#        unitInitialNy[indices] = tmpGradientY / tmpGradientNorm
    cuda.syncthreads()

"""
Update the color gradient value on the nodes neighboring to the solid
"""
@cuda.jit('void(int64, int64, float64, float64, int64[:], float64[:], float64[:], \
                float64[:], float64[:])')
def updateColorGradientOnWetting(totalFluidWettingNodes, xDim, cosTheta, sinTheta, \
                                fluidNodesWetting, unitVectorNsx, unitVectorNsy, \
                                gradientX, gradientY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalFluidWettingNodes:
        tmpN1X = unitVectorNsx[indices] * cosTheta - unitVectorNsy[indices] * sinTheta
        tmpN1Y = unitVectorNsy[indices] * cosTheta + unitVectorNsx[indices] * sinTheta
        tmpN2X = unitVectorNsx[indices] * cosTheta + unitVectorNsy[indices] * sinTheta
        tmpN2Y = unitVectorNsy[indices] * cosTheta - unitVectorNsx[indices] * sinTheta

        #Unit vector of gradient on fluid node
        tmpLoc = fluidNodesWetting[indices]
        tmpGradientNorm = math.sqrt(gradientX[tmpLoc] * gradientX[tmpLoc] + \
                            gradientY[tmpLoc] * gradientY[tmpLoc])
#        if tmpGradientNorm > 0.:
        if tmpGradientNorm > 1.0e-8:
            tmpUnitGradientX = gradientX[tmpLoc] / tmpGradientNorm
            tmpUnitGradientY = gradientY[tmpLoc] / tmpGradientNorm
        else:
            tmpUnitGradientX = 0.; tmpUnitGradientY = 0.
        #calculate the distance between vectors
        tmpDX1 = tmpUnitGradientX - tmpN1X; tmpDY1 = tmpUnitGradientY - tmpN1Y
        tmpDX2 = tmpUnitGradientX - tmpN2X; tmpDY2 = tmpUnitGradientY - tmpN2Y
        tmpDistance1 = math.sqrt(tmpDX1 * tmpDX1 + tmpDY1 * tmpDY1)
        tmpDistance2 = math.sqrt(tmpDX2 * tmpDX2 + tmpDY2 * tmpDY2)
        #Choose the right unit vector for color gradient
        tmpModifiedNX = 0.; tmpModifiedNY = 0.
        if tmpDistance1 < tmpDistance2:
            tmpModifiedNX = tmpN1X; tmpModifiedNY = tmpN1Y
        elif tmpDistance1 > tmpDistance2:
            tmpModifiedNX = tmpN2X; tmpModifiedNY = tmpN2Y
        elif tmpDistance1 == tmpDistance2:
            tmpModifiedNX = unitVectorNsx[indices]
            tmpModifiedNY = unitVectorNsy[indices]
        #Update the color gradient on the fluid nodes near to solid
        gradientX[tmpLoc] = tmpGradientNorm * tmpModifiedNX
        gradientY[tmpLoc] = tmpGradientNorm * tmpModifiedNY
    cuda.syncthreads()

"""
Add the body force (including surface tension)
"""
@cuda.jit('void(int64, int64, float64, int64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:])')
def calForceTermInColorGradient2D(totalNodes, xDim, surfaceTension, neighboringNodes, \
                                 weightsCoeff, unitEX, unitEY, gradientX, gradientY, \
                                 forceX, forceY, KValue):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array((9,), dtype = float64)
#    sharedEY = cuda.shared.array((9,), dtype = float64)
#    sharedWeights = cuda.shared.array((9,), dtype = float64)
#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]
    if indices < totalNodes:
        #Calculate value of K
        tmpStart = 8 * indices
        tmpGradientNorm = math.sqrt(gradientX[indices] * gradientX[indices] + \
                            gradientY[indices] * gradientY[indices])
        if tmpGradientNorm == 0.:
            tmpUnitGX = 0.; tmpUnitGY = 0.
        elif tmpGradientNorm > 0.:
            tmpUnitGX = gradientX[indices] / tmpGradientNorm
            tmpUnitGY = gradientY[indices] / tmpGradientNorm
        tmpPartialYX = 0.; tmpPartialXY = 0.; tmpPartialX = 0.; tmpPartialY = 0.
        tmpIndices = 0
        for i in range(8):
            tmpIndices += 1
            tmpLoc = neighboringNodes[tmpStart + i]
            if tmpLoc >= 0:
                tmpGradientNormNeighbor = math.sqrt(gradientX[tmpLoc] * gradientX[tmpLoc] + \
                            gradientY[tmpLoc] * gradientY[tmpLoc])
                if tmpGradientNormNeighbor > 0.:
                    tmpUnitGXN = gradientX[tmpLoc] / tmpGradientNormNeighbor
                    tmpUnitGYN = gradientY[tmpLoc] / tmpGradientNormNeighbor
                elif tmpGradientNormNeighbor == 0.:
                    tmpUnitGXN = 0.; tmpUnitGYN = 0.
#                tmpPartialYX += 3. * sharedWeights[tmpIndices] * tmpUnitGYN * sharedEX[tmpIndices]
#                tmpPartialXY += 3. * sharedWeights[tmpIndices] * tmpUnitGXN * sharedEY[tmpIndices]
#                tmpPartialX += 3. * sharedWeights[tmpIndices] * tmpUnitGXN * sharedEX[tmpIndices]
#                tmpPartialY += 3. * sharedWeights[tmpIndices] * tmpUnitGYN * sharedEY[tmpIndices]
                tmpPartialYX += 3. * weightsCoeff[tmpIndices] * tmpUnitGYN * unitEX[tmpIndices]
                tmpPartialXY += 3. * weightsCoeff[tmpIndices] * tmpUnitGXN * unitEY[tmpIndices]
                tmpPartialX += 3. * weightsCoeff[tmpIndices] * tmpUnitGXN * unitEX[tmpIndices]
                tmpPartialY += 3. * weightsCoeff[tmpIndices] * tmpUnitGYN * unitEY[tmpIndices]
        KValue[indices] = tmpUnitGX * tmpUnitGY * (tmpPartialYX + tmpPartialXY) - tmpUnitGY * \
                tmpUnitGY * tmpPartialX - tmpUnitGX * tmpUnitGX * tmpPartialY

        forceX[indices] = 0.5 * surfaceTension * KValue[indices] * gradientX[indices]
        forceY[indices] = 0.5 * surfaceTension * KValue[indices] * gradientY[indices]
    cuda.syncthreads()

"""
Calculate the distribution function after adding the force term (SRT)
"""
@cuda.jit('void(int64, int64, int64, float64, float64, float64, float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:, :], float64[:], float64[:])')
def calPerturbationFromForce2D(totalNodes, xDim, optionF, tauR, tauB, deltaValue, \
                               weightsCoeff, unitEX, unitEY, physicalVX, physicalVY, \
                               forceX, forceY, colorValue, fluidTotalPDF, \
                               fluidRhoR, fluidRhoB):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array((9,), dtype = float64)
#    sharedEY = cuda.shared.array((9,), dtype = float64)
#    sharedWeights = cuda.shared.array((9,), dtype = float64)
#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]
    if indices < totalNodes:
        Phi = colorValue[indices]; tmpTau = 1.0
        if Phi > deltaValue:
            tmpTau = tauR
        elif Phi < -deltaValue:
            tmpTau = tauB
        elif math.fabs(Phi) <= deltaValue:
            if optionF == 1:
                tmpTau = 0.5 + 1. / ((1. + Phi)/(2. * (tauR - 0.5)) + (1. - Phi) / (2. * \
                                     (tauB - 0.5)))
            elif optionF == 2:
                ratioR = fluidRhoR[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                ratioB = fluidRhoB[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                tmpMiuR = 3./(tauR - 0.5); tmpMiuB = 3./(tauB - 0.5)
                tmpMiu = 1./(ratioR * tmpMiuR + ratioB * tmpMiuB)
                tmpTau = 3. * tmpMiu + 0.5
        tmpFX = forceX[indices]; tmpFY = forceY[indices]
        for i in range(9):
#            term1 = sharedEX[i] * forceX[indices] * 3.
#            term2 = sharedEY[i] * forceY[indices] * 3.
#            term3 = (sharedEX[i] * sharedEX[i] - 1./3.) * physicalVX[indices] * \
#                    forceX[indices] * 9.
#            term4 = sharedEX[i] * sharedEY[i] * physicalVY[indices] * forceX[indices] * \
#                    9.
#            term5 = sharedEY[i] * sharedEX[i] * physicalVX[indices] * forceY[indices] * \
#                    9.
#            term6 = (sharedEY[i] * sharedEY[i] - 1./3.) * physicalVY[indices] * \
#                    forceY[indices] * 9.
            sourceTerm = weightsCoeff[i] * ((3. * (unitEX[i] - physicalVX[indices]) + \
                                9. * unitEX[i] * (unitEX[i] * physicalVX[indices] + \
                                unitEY[i] * physicalVY[indices])) * tmpFX + (3. * \
                                (unitEY[i] - physicalVY[indices]) + 9. * unitEY[i] * \
                                (unitEX[i] * physicalVX[indices] + unitEY[i] * \
                                 physicalVY[indices])) * tmpFY) * (1. - 1./(2. * tmpTau))
#            sharedWeights[i] * (1. - 1./(2. * tmpTau)) * (term1 + \
#                        term2 + term3 + term4 + term5 + term6)
            fluidTotalPDF[indices, i] = fluidTotalPDF[indices, i] + sourceTerm
    cuda.syncthreads()

"""
Collision process 1 with total distribution function in modified method
"""
@cuda.jit('void(int64, int64, int64, float64, float64, float64, float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :])')
def calRKCollision1TotalGPU2DSRTM(totalNodes, xDim, optionF, tauR, tauB, deltaValue, \
                         unitEX, unitEY, weightsCoeff, physicalVX, physicalVY, \
                         fluidRhoR, fluidRhoB, ColorValue, fluidPDFTotal):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedEY = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]

    if (indices < totalNodes):
        Phi = ColorValue[indices]; tmpTau = 1.
        if Phi > deltaValue:
            tmpTau = tauR
        elif Phi < -deltaValue:
            tmpTau = tauB
        elif math.fabs(Phi) <= deltaValue:
            if optionF == 1:
                tmpTau = 0.5 + 1. / ((1. + Phi)/(2. * (tauR - 0.5)) + (1. - Phi) / (2. * \
                                     (tauB - 0.5)))
            elif optionF == 2:
                ratioR = fluidRhoR[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                ratioB = fluidRhoB[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                tmpMiuR = 3./(tauR - 0.5); tmpMiuB = 3./(tauB - 0.5)
                tmpMiu = 1./(ratioR * tmpMiuR + ratioB * tmpMiuB)
                tmpTau = 3. * tmpMiu + 0.5
        tmpRhoR = fluidRhoR[indices]; tmpRhoB = fluidRhoB[indices]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        for i in range(9):
#            tmpEquilibriumR = calEquilibriumRK2D(tmpRhoR, sharedWeights[i], \
#                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
#            tmpEquilibriumB = calEquilibriumRK2D(tmpRhoB, sharedWeights[i], \
#                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpEquilibriumR = calEquilibriumRK2D(tmpRhoR, weightsCoeff[i], \
                            unitEX[i], unitEY[i], tmpVX, tmpVY)
            tmpEquilibriumB = calEquilibriumRK2D(tmpRhoB, weightsCoeff[i], \
                            unitEX[i], unitEY[i], tmpVX, tmpVY)
            tmpEquilibriumTotal = tmpEquilibriumR + tmpEquilibriumB
            fluidPDFTotal[indices, i] = -1./tmpTau * (fluidPDFTotal[indices, i] - \
                           tmpEquilibriumTotal) + fluidPDFTotal[indices, i]
    cuda.syncthreads()

"""
#Recoloring step for two components in modified method
"""
@cuda.jit('void(int64, int64, float64, float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], \
                float64[:, :], float64[:, :], float64[:, :])')
def calRecoloringProcessM(totalNodes, xDim, betaValue, weightsCoeff, fluidRhoR, \
                         fluidRhoB, unitEX, unitEY, gradientX, gradientY, \
                         fluidPDFR, fluidPDFB, fluidPDFTotal):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array((9,), dtype = float64)
#    sharedEY = cuda.shared.array((9,), dtype = float64)
#    sharedWeights = cuda.shared.array((9,), dtype = float64)
#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]

    if indices < totalNodes:
        tmpGradientNorm = math.sqrt(gradientX[indices] * gradientX[indices] + \
                                    gradientY[indices] * gradientY[indices])
        costheta = 0.
        totalRho = fluidRhoR[indices] + fluidRhoB[indices]
        for i in range(9):
#            tmpUnitNorm = math.sqrt(sharedEX[i] * sharedEX[i] + sharedEY[i] * \
#                                    sharedEY[i])
            tmpUnitNorm = math.sqrt(unitEX[i] * unitEX[i] + unitEY[i] * \
                                    unitEY[i])
            if (tmpGradientNorm > 1.0e-8 and tmpUnitNorm > 1.0e-8):
                costheta = (unitEX[i] * gradientX[indices] + unitEY[i] * \
                            gradientY[indices]) / (tmpUnitNorm * tmpGradientNorm)
#            if tmpGradientNorm == 0. or tmpUnitNorm == 0.:
            else:
                costheta = 0.
            tmpTotalPDF = fluidPDFTotal[indices, i]
#            fluidPDFR[indices, i] = fluidRhoR[indices] / totalRho * tmpTotalPDF + \
#                betaValue * fluidRhoR[indices] * fluidRhoB[indices] / totalRho * \
#                sharedWeights[i] * costheta * tmpUnitNorm
#            fluidPDFB[indices, i] = fluidRhoB[indices] / totalRho * tmpTotalPDF - \
#                betaValue * fluidRhoR[indices] * fluidRhoB[indices] / totalRho * \
#                sharedWeights[i] * costheta * tmpUnitNorm
            fluidPDFR[indices, i] = fluidRhoR[indices] / totalRho * tmpTotalPDF + \
                betaValue * fluidRhoR[indices] * fluidRhoB[indices] / totalRho * \
                weightsCoeff[i] * costheta * tmpUnitNorm
            fluidPDFB[indices, i] = fluidRhoB[indices] / totalRho * tmpTotalPDF - \
                betaValue * fluidRhoR[indices] * fluidRhoB[indices] / totalRho * \
                weightsCoeff[i] * costheta * tmpUnitNorm
    cuda.syncthreads()

"""
Calculate the macro-scale velocity
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:])')
def calPhysicalVelocityRKGPU2DM(totalNodes, xDim, fluidPDFTotal, fluidRhoR, \
                               fluidRhoB, physicalVX, physicalVY, forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        tmpVX = 0.; tmpVY = 0.
        tmpRhoSum = fluidRhoB[indices] + fluidRhoR[indices]
        tmpVX = fluidPDFTotal[indices, 1] - fluidPDFTotal[indices, 3] + \
                fluidPDFTotal[indices, 5] - fluidPDFTotal[indices, 6] - \
                fluidPDFTotal[indices, 7] + fluidPDFTotal[indices, 8] + 0.5 * \
                forceX[indices]
        physicalVX[indices] = tmpVX / tmpRhoSum
        tmpVY = fluidPDFTotal[indices, 2] - fluidPDFTotal[indices, 4] + \
                fluidPDFTotal[indices, 5] + fluidPDFTotal[indices, 6] - \
                fluidPDFTotal[indices, 7] - fluidPDFTotal[indices, 8] + 0.5 * \
                forceY[indices]
        physicalVY[indices] = tmpVY / tmpRhoSum
    cuda.syncthreads()

"""
Collision process 1 with total distribution function in modified method
"""
@cuda.jit('void(int64, int64, int64, float64, float64, float64, float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], \
                float64[:])')
def calRKCollision1TotalGPU2DMRTM(totalNodes, xDim, optionF, tauR, tauB, deltaValue, \
                         unitEX, unitEY, weightsCoeff, physicalVX, physicalVY, \
                         fluidRhoR, fluidRhoB, ColorValue, fluidPDFTotal, transformationM, \
                         inverseTM, collisionS):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedEY = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
#
#    sharedTM = cuda.shared.array(shape = (9, 9), dtype = float64)
#    sharedIM = cuda.shared.array(shape = (9, 9), dtype = float64)

    localCollisionS = cuda.shared.array(shape = (9,), dtype = float64)
    localSingleCollision = cuda.local.array(shape = (9,), dtype = float64)
    localTransformation = cuda.local.array(shape = (9,), dtype = float64)

#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]
#        for j in range(9):
#            sharedTM[i, j] = transformationM[i, j]
#            sharedIM[i, j] = inverseTM[i, j]
    for i in range(9):
        localCollisionS[i] = collisionS[i]

    if (indices < totalNodes):
        Phi = ColorValue[indices]; tmpTau = 1.
        if Phi > deltaValue:
            tmpTau = tauR
        elif Phi < -deltaValue:
            tmpTau = tauB
        elif math.fabs(Phi) <= deltaValue:
            if optionF == 1:
                tmpTau = 0.5 + 1. / ((1. + Phi)/(2. * (tauR - 0.5)) + (1. - Phi) / (2. * \
                                     (tauB - 0.5)))
            elif optionF == 2:
                ratioR = fluidRhoR[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                ratioB = fluidRhoB[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                tmpMiuR = 3./(tauR - 0.5); tmpMiuB = 3./(tauB - 0.5)
                tmpMiu = 1./(ratioR * tmpMiuR + ratioB * tmpMiuB)
                tmpTau = 3. * tmpMiu + 0.5
        localCollisionS[7] = 1./tmpTau; localCollisionS[8] = 1./tmpTau

        tmpRhoR = fluidRhoR[indices]; tmpRhoB = fluidRhoB[indices]
        tmpVX = physicalVX[indices]; tmpVY = physicalVY[indices]
        for i in range(9):
#            tmpEquilibriumR = calEquilibriumRK2D(tmpRhoR, sharedWeights[i], \
#                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
#            tmpEquilibriumB = calEquilibriumRK2D(tmpRhoB, sharedWeights[i], \
#                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpEquilibriumR = calEquilibriumRK2D(tmpRhoR, weightsCoeff[i], \
                            unitEX[i], unitEY[i], tmpVX, tmpVY)
            tmpEquilibriumB = calEquilibriumRK2D(tmpRhoB, weightsCoeff[i], \
                            unitEX[i], unitEY[i], tmpVX, tmpVY)
#            tmpEquilibriumR = calEquilibriumRK2DOriginal(tmpRhoR, constCR[i], sharedWeights[i], \
#                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
#            tmpEquilibriumB = calEquilibriumRK2DOriginal(tmpRhoB, constCB[i], sharedWeights[i], \
#                            sharedEX[i], sharedEY[i], tmpVX, tmpVY)
            tmpEquilibriumTotal = tmpEquilibriumR + tmpEquilibriumB
            localSingleCollision[i] = (fluidPDFTotal[indices, i] - tmpEquilibriumTotal)
        #start MRT part
        for i in range(9):
            tmpSum = 0.
            for j in range(9):
#                tmpSum += sharedTM[i, j] * localSingleCollision[j]
                tmpSum += transformationM[i, j] * localSingleCollision[j]
            localTransformation[i] = tmpSum

        for i in range(9):
            localTransformation[i] = localTransformation[i] * localCollisionS[i]

        for i in range(9):
            tmpSum = 0.
            for j in range(9):
#                tmpSum += sharedIM[i, j] * localTransformation[j]
                tmpSum += inverseTM[i, j] * localTransformation[j]
            fluidPDFTotal[indices, i] = -tmpSum + fluidPDFTotal[indices, i]
    cuda.syncthreads()

"""
Calculate the distribution function after adding the force term (MRT)
"""
@cuda.jit('void(int64, int64, int64, float64, float64, float64, float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:], \
                float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:], \
                float64[:])')
def calPerturbationFromForce2DMRT(totalNodes, xDim, optionF, tauR, tauB, deltaValue, \
                               weightsCoeff, unitEX, unitEY, physicalVX, physicalVY, \
                               forceX, forceY, colorValue, fluidTotalPDF, transformationM, \
                               inverseTM, collisionS, fluidRhoR, fluidRhoB):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array((9,), dtype = float64)
#    sharedEY = cuda.shared.array((9,), dtype = float64)
#    sharedWeights = cuda.shared.array((9,), dtype = float64)
#    sharedTM = cuda.shared.array(shape = (9, 9), dtype = float64)
#    sharedIM = cuda.shared.array(shape = (9, 9), dtype = float64)

    localCollisionS = cuda.shared.array(shape = (9,), dtype = float64)
    localSource = cuda.local.array(shape = (9,), dtype = float64)
    localTransform = cuda.local.array(shape = (9,), dtype = float64)
#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]
#        for j in range(9):
#            sharedTM[i, j] = transformationM[i, j]
#            sharedIM[i, j] = inverseTM[i, j]
    for i in range(9):
        localCollisionS[i] = 1. - 0.5 * collisionS[i]
    if indices < totalNodes:
        Phi = colorValue[indices]; tmpTau = 1.
        if Phi > deltaValue:
            tmpTau = tauR
        elif Phi < -deltaValue:
            tmpTau = tauB
        elif math.fabs(Phi) <= deltaValue:
            if optionF == 1:
                tmpTau = 0.5 + 1. / ((1. + Phi)/(2. * (tauR - 0.5)) + (1. - Phi) / (2. * \
                                     (tauB - 0.5)))
            elif optionF == 2:
                ratioR = fluidRhoR[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                ratioB = fluidRhoB[indices] / (fluidRhoR[indices] + fluidRhoB[indices])
                tmpMiuR = 3./(tauR - 0.5); tmpMiuB = 3./(tauB - 0.5)
                tmpMiu = 1./(ratioR * tmpMiuR + ratioB * tmpMiuB)
                tmpTau = 3. * tmpMiu + 0.5

        localCollisionS[7] = 1. - 0.5 * 1./tmpTau; localCollisionS[8] = 1. - 0.5 * 1./tmpTau
        tmpFX = forceX[indices]; tmpFY = forceY[indices]
        for i in range(9):
#            term1 = sharedEX[i] * forceX[indices] * 3.
#            term2 = sharedEY[i] * forceY[indices] * 3.
#            term3 = (sharedEX[i] * sharedEX[i] - 1./3.) * physicalVX[indices] * \
#                    forceX[indices] * 9.
#            term4 = sharedEX[i] * sharedEY[i] * physicalVY[indices] * forceX[indices] * \
#                    9.
#            term5 = sharedEY[i] * sharedEX[i] * physicalVX[indices] * forceY[indices] * \
#                    9.
#            term6 = (sharedEY[i] * sharedEY[i] - 1./3.) * physicalVY[indices] * \
#                    forceY[indices] * 9.
#            sourceTerm = sharedWeights[i] * (term1 + \
#                        term2 + term3 + term4 + term5 + term6)
            term1 = unitEX[i] * forceX[indices] * 3.
            term2 = unitEY[i] * forceY[indices] * 3.
            term3 = (unitEX[i] * unitEX[i] - 1./3.) * physicalVX[indices] * \
                    forceX[indices] * 9.
            term4 = unitEX[i] * unitEY[i] * physicalVY[indices] * forceX[indices] * \
                    9.
            term5 = unitEY[i] * unitEX[i] * physicalVX[indices] * forceY[indices] * \
                    9.
            term6 = (unitEY[i] * unitEY[i] - 1./3.) * physicalVY[indices] * \
                    forceY[indices] * 9.
            sourceTerm = weightsCoeff[i] * (term1 + \
                        term2 + term3 + term4 + term5 + term6)
            localSource[i] = sourceTerm
        #Start MRT part
        for i in range(9):
            tmpSum = 0.
            for j in range(9):
#                tmpSum += sharedTM[i, j] * localSource[j]
                tmpSum += transformationM[i, j] * localSource[j]
            localTransform[i] = tmpSum

        for i in range(9):
            localTransform[i] = localCollisionS[i] * localTransform[i]

        for i in range(9):
            tmpSum = 0.
            for j in range(9):
#                tmpSum += sharedIM[i, j] * localTransform[j]
                tmpSum += inverseTM[i, j] * localTransform[j]
            fluidTotalPDF[indices, i] = fluidTotalPDF[indices, i] + tmpSum
    cuda.syncthreads()

"""
Calculate Von Neumann boundary condition with Zou-He method, but no flow for the 
other fluid
"""
@cuda.jit('void(int64, int64, int64, int64, float64, float64, int64[:], \
                int64[:], float64[:], float64[:], float64[:, :], float64[:, :])')
def constantVelocityZHBoundaryHigherNewRK(totalNodes, nx, ny, xDim, \
                                        specificVYR, specificVYB, fluidNodes, \
                                        neighboringNodes, fluidRhoR, \
                                        fluidRhoB, fluidPDFR, fluidPDFB):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    if (indices < totalNodes - nx and indices >= totalNodes - 2 * nx):
    if indices < totalNodes:
        tmpStart = 8 * indices
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < (ny - 1) * nx and tmpIndex >= (ny - 2) * nx):
            fluidRhoR[indices] = (fluidPDFR[indices, 0] + fluidPDFR[indices, 1] + \
                    fluidPDFR[indices, 3] + 2. * (fluidPDFR[indices, 2] + \
                    fluidPDFR[indices, 5] + fluidPDFR[indices, 6])) / \
                    (1. + specificVYR)
            fluidPDFR[indices, 4] = fluidPDFR[indices, 2] - 2./3. * \
                    fluidRhoR[indices] * specificVYR
            fluidPDFR[indices, 7] = fluidPDFR[indices, 5] + \
                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) / 2. - \
                    1./6. * fluidRhoR[indices] * specificVYR
            fluidPDFR[indices, 8] = fluidPDFR[indices, 6] - \
                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) / 2. - \
                    1./6. * fluidRhoR[indices] * specificVYR

            #for retreating fluid
            tmpUpper = neighboringNodes[tmpStart + 1]
            fluidPDFB[indices, 4] = fluidPDFB[tmpUpper, 2]
            tmpFor7 = neighboringNodes[tmpUpper * 8]
            tmpFor8 = neighboringNodes[tmpUpper * 8 + 2]
            if tmpFor7 >= 0:
                fluidPDFB[indices, 7] = fluidPDFB[tmpFor7, 5]
            if tmpFor8 >= 0:
                fluidPDFB[indices, 8] = fluidPDFB[tmpFor8, 6]
    cuda.syncthreads()

"""
Update the color gradient value on the nodes neighboring to the solid, Takashi 2017
"""
@cuda.jit('void(int64, int64, float64, float64, int64[:], float64[:], float64[:], \
                float64[:], float64[:])')
def updateColorGradientOnWettingNew(totalFluidWettingNodes, xDim, cosTheta, sinTheta, \
                                fluidNodesWetting, unitVectorNsx, unitVectorNsy, \
                                gradientX, gradientY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalFluidWettingNodes:
#        tmpN1X = unitVectorNsx[indices] * cosTheta - unitVectorNsy[indices] * sinTheta
#        tmpN1Y = unitVectorNsy[indices] * cosTheta + unitVectorNsx[indices] * sinTheta
#        tmpN2X = unitVectorNsx[indices] * cosTheta + unitVectorNsy[indices] * sinTheta
#        tmpN2Y = unitVectorNsy[indices] * cosTheta - unitVectorNsx[indices] * sinTheta

        #Unit vector of gradient on fluid node
        tmpLoc = fluidNodesWetting[indices]
        tmpGradientNorm = math.sqrt(gradientX[tmpLoc] * gradientX[tmpLoc] + \
                            gradientY[tmpLoc] * gradientY[tmpLoc])
#        if tmpGradientNorm > 0.:
        if tmpGradientNorm > 1.0e-8:
            tmpUnitGradientX = -gradientX[tmpLoc] / tmpGradientNorm
            tmpUnitGradientY = -gradientY[tmpLoc] / tmpGradientNorm
        else:
            tmpUnitGradientX = 0.; tmpUnitGradientY = 0.
        tmpAngleGS = tmpUnitGradientX * unitVectorNsx[indices] + tmpUnitGradientY * \
                    unitVectorNsy[indices]
        thetaGS = math.acos(tmpAngleGS)
        #calculate the distance between vectors
        tmpCoeffGS1 = 0.; tmpCoeffGS2 = 0.; tmpCoeffGS3 = 0.; tmpCoeffGS4 = 0.
#        if math.sin(thetaGS) != 0.:
        if math.fabs(math.sin(thetaGS)) > 1.0e-9:
            tmpCoeffGS1 = sinTheta * math.cos(thetaGS) / math.sin(thetaGS)
            tmpCoeffGS2 = sinTheta / math.sin(thetaGS)
            tmpCoeffGS3 = -sinTheta * math.cos(thetaGS) / math.sin(thetaGS)
            tmpCoeffGS4 = -sinTheta / math.sin(thetaGS)

        tmpNX1 = (cosTheta - tmpCoeffGS1) * unitVectorNsx[indices] + tmpCoeffGS2 * \
                tmpUnitGradientX
        tmpNY1 = (cosTheta - tmpCoeffGS1) * unitVectorNsy[indices] + tmpCoeffGS2 * \
                tmpUnitGradientY
        tmpNX2 = (cosTheta - tmpCoeffGS3) * unitVectorNsx[indices] + tmpCoeffGS4 * \
                tmpUnitGradientX
        tmpNY2 = (cosTheta - tmpCoeffGS3) * unitVectorNsy[indices] + tmpCoeffGS4 * \
                tmpUnitGradientY
        tmpDX1 = tmpNX1 - tmpUnitGradientX; tmpDY1 = tmpNY1 - tmpUnitGradientY
        tmpDX2 = tmpNX2 - tmpUnitGradientX; tmpDY2 = tmpNY2 - tmpUnitGradientY
        tmpDistance1 = math.sqrt(tmpDX1 * tmpDX1 + tmpDY1 * tmpDY1)
        tmpDistance2 = math.sqrt(tmpDX2 * tmpDX2 + tmpDY2 * tmpDY2)
        #Choose the right unit vector for color gradient
        tmpModifiedNX = 0.; tmpModifiedNY = 0.
        if tmpDistance1 < tmpDistance2:
            tmpModifiedNX = tmpNX1; tmpModifiedNY = tmpNY1
            gradientX[tmpLoc] = -tmpGradientNorm * tmpModifiedNX
            gradientY[tmpLoc] = -tmpGradientNorm * tmpModifiedNY
        elif tmpDistance1 > tmpDistance2:
            tmpModifiedNX = tmpNX2; tmpModifiedNY = tmpNY2
            gradientX[tmpLoc] = -tmpGradientNorm * tmpModifiedNX
            gradientY[tmpLoc] = -tmpGradientNorm * tmpModifiedNY
#        elif tmpDistance1 == tmpDistance2:
#            tmpModifiedNX = unitVectorNsx[indices]
#            tmpModifiedNY = unitVectorNsy[indices]
        #Update the color gradient on the fluid nodes near to solid

    cuda.syncthreads()

"""
Add the body force (including surface tension). Takashi 2018
"""
@cuda.jit('void(int64, int64, float64, int64[:], float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:])')
def calForceTermInColorGradientNew2D(totalNodes, xDim, surfaceTension, neighboringNodes, \
                                 weightsCoeff, unitEX, unitEY, gradientX, gradientY, \
                                 forceX, forceY, KValue):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

#    sharedEX = cuda.shared.array((9,), dtype = float64)
#    sharedEY = cuda.shared.array((9,), dtype = float64)
#    sharedWeights = cuda.shared.array((9,), dtype = float64)
#    for i in range(9):
#        sharedEX[i] = unitEX[i]; sharedEY[i] = unitEY[i]
#        sharedWeights[i] = weightsCoeff[i]
    if indices < totalNodes:
        #Calculate value of K
        tmpStart = 8 * indices
        tmpGradientNorm = math.sqrt(gradientX[indices] * gradientX[indices] + \
                            gradientY[indices] * gradientY[indices])
#        if tmpGradientNorm > 0.:
        if tmpGradientNorm > 1.0e-8:
            tmpUnitGX = -gradientX[indices] / tmpGradientNorm
            tmpUnitGY = -gradientY[indices] / tmpGradientNorm
#        elif tmpGradientNorm == 0.:
        else:
            tmpUnitGX = 0.; tmpUnitGY = 0.
        tmpPartialYX = 0.; tmpPartialXY = 0.; tmpPartialX = 0.; tmpPartialY = 0.
        tmpIndices = 0
        for i in range(8):
            tmpIndices += 1
            tmpLoc = neighboringNodes[tmpStart + i]
            if tmpLoc >= 0:
                tmpGradientNormNeighbor = math.sqrt(gradientX[tmpLoc] * gradientX[tmpLoc] + \
                            gradientY[tmpLoc] * gradientY[tmpLoc])
#                if tmpGradientNormNeighbor > 0.:
                if tmpGradientNormNeighbor > 1.0e-8:
                    tmpUnitGXN = -gradientX[tmpLoc] / tmpGradientNormNeighbor
                    tmpUnitGYN = -gradientY[tmpLoc] / tmpGradientNormNeighbor
#                elif tmpGradientNormNeighbor == 0.:
                else:
                    tmpUnitGXN = 0.; tmpUnitGYN = 0.
#                tmpPartialYX += 3. * sharedWeights[tmpIndices] * tmpUnitGYN * sharedEX[tmpIndices]
#                tmpPartialXY += 3. * sharedWeights[tmpIndices] * tmpUnitGXN * sharedEY[tmpIndices]
#                tmpPartialX += 3. * sharedWeights[tmpIndices] * tmpUnitGXN * sharedEX[tmpIndices]
#                tmpPartialY += 3. * sharedWeights[tmpIndices] * tmpUnitGYN * sharedEY[tmpIndices]
                tmpPartialYX += 3. * weightsCoeff[tmpIndices] * tmpUnitGYN * unitEX[tmpIndices]
                tmpPartialXY += 3. * weightsCoeff[tmpIndices] * tmpUnitGXN * unitEY[tmpIndices]
                tmpPartialX += 3. * weightsCoeff[tmpIndices] * tmpUnitGXN * unitEX[tmpIndices]
                tmpPartialY += 3. * weightsCoeff[tmpIndices] * tmpUnitGYN * unitEY[tmpIndices]
        KValue[indices] = tmpUnitGX * tmpUnitGY * (tmpPartialYX + tmpPartialXY) - tmpUnitGY * \
                tmpUnitGY * tmpPartialX - tmpUnitGX * tmpUnitGX * tmpPartialY

        forceX[indices] = -0.5 * surfaceTension * KValue[indices] * gradientX[indices]
        forceY[indices] = -0.5 * surfaceTension * KValue[indices] * gradientY[indices]
    cuda.syncthreads()