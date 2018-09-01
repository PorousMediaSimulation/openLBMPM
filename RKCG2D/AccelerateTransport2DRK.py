"""
Module for using CUDA in numba to accelerate the D2Q5 LBM for transport
"""

import sys, os

import numpy as np
import scipy as sp

import math

from numba import cuda, int64, float64
from numba import cuda, jit

#from accelerate import cuda as acuda
#from accelerate import numba as anumba

#"""
#Calculate the neighboring nodes in transport scheme D2Q9
#"""
#@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:, :], int64[:])')
#def fillNeighboringNodesTransport(totalNodes, nx, ny, xDim, fluidNodes, domainNewIndex, \
#                                  neighboringNodes):
#    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
#    by = cuda.blockIdx.y
#    indices = by * xDim + bx * bDimX + tx
#    
#    if (indices < totalNodes):
#        tmpStart = 4 * indices
#        tmpLoc = fluidNodes[indices]
#        i = int(tmpLoc / nx); j = tmpLoc % nx
#        tmpF = j + 1 if j < nx - 1 else 0
#        tmpB = j - 1 if j > 0 else (nx - 1)
#        tmpU = i + 1 if i < ny - 1 else 0
#        tmpL = i - 1 if i > 0 else (ny - 1)
#        #Eastern node
#        neighboringNodes[tmpStart] = domainNewIndex[i, tmpF]
#        #Northern node
#        tmpStart += 1
#        neighboringNodes[tmpStart] = domainNewIndex[tmpU, j]
#        #Western node
#        tmpStart += 1
#        neighboringNodes[tmpStart] = domainNewIndex[i, tmpB]
#        #Southern node
#        tmpStart += 1
#        neighboringNodes[tmpStart] = domainNewIndex[tmpL, j]
"""
Calculate the neighboring nodes in transport scheme D2Q5
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:, :], int64[:])')
def fillNeighboringNodesTransport(totalNodes, nx, ny, xDim, fluidNodes, domainNewIndex, \
                                  neighboringNodes):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpStart = 4 * indices
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
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpB]
        #Western node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, j]
        #Southern node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, j]
    
@cuda.jit('void(int64, int64, int64, int64, float64[:, :], float64[:, :, :])')
def calConcentrationGPU(totalNodes, numTracers, xDim, numSchemes, tracerConc, \
                        tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        for i in range(numTracers):
            tracerConc[i, indices] = 0.
            for j in range(numSchemes):
                tracerConc[i, indices] += tracerPDF[i, indices, j]

"""
Reaction part for tracer in bulk fluid
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:, :], \
                float64[:, :], float64[:, :, :])')
def calReactionTracersGPU(totalNodes, numTracers, xDim, reactionRate, diffJcoeffs, \
                       tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpReactionS = cuda.local.array(shape = (3,), dtype = float64)
        tmpReactionS[0] = -reactionRate[0] * tracerConc[0, indices] * tracerConc[1, indices]
        tmpReactionS[1] = -reactionRate[0] * tracerConc[0, indices] * tracerConc[1, indices]
        tmpReactionS[2] = reactionRate[0] * tracerConc[0, indices] * tracerConc[1, indices]
        for i in range(numTracers):
            for j in range(5):
                tracerPDF[i, indices, j] = tracerPDF[i, indices, j] + diffJcoeffs[i, j] * \
                                    tmpReactionS[i]
        

"""
Collision process of transport phenomena without reaction
"""
@cuda.jit('void(int64, int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :], \
                float64[:, :, :], float64[:, :, :])')
def calCollisionTransportGPU(totalNodes, xDim, numTracers, numScheme, unitVX, unitVY, \
                          velocityVX, velocityVY, tauTransport, valueJDE, tracerConc, \
                          tracerPDF, tracerPDFNew):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        for i in range(numTracers):
            for j in range(numScheme):
                tmpTracerEq = tracerConc[i, indices] * (valueJDE[i, j] + 1./2. * \
                           (unitVX[j] * velocityVX[indices] + unitVY[j] * \
                            velocityVY[indices]))
                tracerPDF[i, indices, j] = tracerPDF[i, indices, j] - 1./tauTransport[i] * \
                            (tracerPDF[i, indices, j] - tmpTracerEq)
                            
"""
Streaming process of transport phenomena
"""        
@cuda.jit('void(int64, int64, int64, int64[:], float64[:, :, :], \
                float64[:, :, :])')
def calStreamingTransportGPU(totalNodes, xDim, numTracers, neighboringNodes, \
                             tracerPDF, tracerPDFNew):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpStart = 4 * indices
        if neighboringNodes[tmpStart] != -1:
            tmpE = neighboringNodes[tmpStart]
            for j in range(numTracers):
                tracerPDFNew[j, tmpE, 1] = tracerPDF[j, indices, 1]
        elif neighboringNodes[tmpStart] == -1:
            for j in range(numTracers):
                tracerPDFNew[j,  indices, 2] = tracerPDF[j, indices, 1]
        tmpStart += 1
        if neighboringNodes[tmpStart] != -1:
            tmpW = neighboringNodes[tmpStart]
            for j in range(numTracers):
                tracerPDFNew[j, tmpW, 2] = tracerPDF[j, indices, 2]
        elif neighboringNodes[tmpStart] == -1:
            for j in range(numTracers):
                tracerPDFNew[j, indices, 1] = tracerPDF[j, indices, 2]
        tmpStart += 1
        if neighboringNodes[tmpStart] != -1:
            tmpN = neighboringNodes[tmpStart]
            for j in range(numTracers):
                tracerPDFNew[j, tmpN, 3] = tracerPDF[j, indices, 3]
        elif neighboringNodes[tmpStart] == -1:
            for j in range(numTracers):
                tracerPDFNew[j, indices, 4] = tracerPDF[j, indices, 3]
        tmpStart += 1
        if neighboringNodes[tmpStart] != -1:
            tmpS = neighboringNodes[tmpStart]
            for j in range(numTracers):
                tracerPDFNew[j, tmpS, 4] = tracerPDF[j, indices, 4]
        elif neighboringNodes[tmpStart] == -1:
            for j in range(numTracers):
                tracerPDFNew[j, indices, 3] = tracerPDF[j, indices, 4]
    cuda.syncthreads()
    
"""
Calculate the streaming process 2
"""
@cuda.jit('void(int64, int64, int64, float64[:, :, :], float64[:, :, :])')
def calStreamingTransport2GPU(totalNum, numTracers, xDim, tracerPDFNew, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if indices < totalNum:
        for i in range(numTracers):
            for j in range(1, 5):
                tracerPDF[i, indices, j] = tracerPDFNew[i, indices, j]
"""
Update the fluid distribution region
"""
@cuda.jit('void(int64, int64, float64, float64[:], boolean[:])')
def calUpdateDistributionGPU(totalNodes, xDim, criteriaFluid, fluidRhoR, \
                             distriField):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        if (fluidRhoR[indices] < criteriaFluid):
            distriField[indices] = 1
        else:
            distriField[indices] = 0
                       


"""
Update the concentration on new fluid nodes with the method in Kang. et.al 2007 
"""
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:, :], \
                boolean[:])')
def calUpdateConcOnNewNodesGPU(totalNodes, xDim, numTracers, newFluidList, \
                               surroundingNodes, tracerConc, distrField):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        for tmpNew in newFluidList:
            if indices == tmpNew:
                for i in range(numTracers):
                    tmpTotalC = 0.; totalCount = 0
                    tmpStart = indices * 8
                    for j in range(8):
                        tmpSurrounding = surroundingNodes[tmpStart]
                        if (distrField[tmpSurrounding] == True and tmpSurrounding >= 0):
                            tmpAddNeighboring = 0
                            for m in newFluidList:
                                if tmpSurrounding == m:
                                    tmpAddNeighboring += 1
                            if tmpAddNeighboring == 0:
                                tmpTotalC += tracerConc[i, tmpSurrounding]
                                totalCount += 1
                        tmpStart += 1
                    tracerConc[i, indices] = tmpTotalC / totalCount
                      
"""
Update the concentration on the old fluid nodes when displacement happens
"""
@cuda.jit('void(int64, int64, int64, int64[:], float64[:, :], float64[:, :, :])')
def calUpdateConcOnOldNodesGPU(totalNodes, xDim, numTracers, oldFluidList, \
                               tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        for tmpOld in oldFluidList:
            if indices == tmpOld:
                for i in range(numTracers):
                    tracerConc[i, indices] = 0.
                    for j in range(5):
                        tracerPDF[i, indices, j] = 0.
#        if (transportDomain[indices] == False):
#            for i in range(numTracers):
#                tracerConc[i, indices] = 0.
#                for j in range(5):
#                    tracerPDF[i, indices, j] = 0.
"""
Update the concentration on the old fluid nodes when displacement happens
"""
@cuda.jit('void(int64, int64, int64, boolean[:], float64[:, :], float64[:, :, :])')
def calUpdateConcOnAllNewNodesGPU(totalNodes, xDim, numTracers, transportDomain, \
                               tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        if (transportDomain[indices] == False):
            for i in range(numTracers):
                tracerConc[i, indices] = 0.
                for j in range(5):
                    tracerPDF[i, indices, j] = 0.

"""
Update the concentration in the whole domain when the interface is moving
"""
@cuda.jit('void(int64, int64, int64, int64, float64, int64[:], float64[:], float64[:], float64[:], \
                float64[:, :], float64[:, :], boolean[:])')
def calUpdateConcWholeDomainGPU(totalNodes, nx, xDim, numTracers, randomPert, \
                                fluidNodes, sumOldConc, sumOldList, sumNewList, \
                                tracerConcNew, tracerConc, distrField):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
#        if (distrField[indices] == 1):
        for i in range(numTracers):
            tmpLeave = 0.
#            for j in range(nx):
#                if fluidNodes[j] < nx:
#                    tmpLeave += tracerConc[i, j]
            tmpConc = (1. + randomPert) * tracerConc[i, indices] * \
                         (sumOldConc[i] / (sumNewList[i] + sumOldConc[i] - \
                          sumOldList[i] + tmpLeave))
            tracerConcNew[i, indices] = tmpConc
            
        
"""
Deal with the interface boundary
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], float64[:], float64[:], \
                float64[:, :], float64[:, :, :], boolean[:])')
def calTransportInterfaceGPU(totalNodes, xDim, numTracers, numScheme, neighboringNodes, \
                             velocityVX, velocityVY, tracerConc, tracerPDF, distriField):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        if distriField[indices] == 1:
            for i in range(numTracers):
                tmpStart = 4 * indices
                tmpDirection = 0
                for j in range(1, numScheme):
                    tmpDirection += 1
                    tmpIndices = neighboringNodes[tmpStart]
                    if (tmpIndices >= 0 and distriField[tmpIndices] == 0):
                        if tmpDirection == 1:
#                            tmpNeighboring = neighboringNodes[tmpStart + 1]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection + 2] = -velocityVX[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection + 1] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection + 1] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection + 1] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 2:
#                            tmpNeighboring = neighboringNodes[tmpStart + 2]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection + 2] = -velocityVY[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection - 1] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 3:
#                            tmpNeighboring = neighboringNodes[tmpStart - 2]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection - 2] = velocityVX[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection + 1] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 4:
#                            tmpNeighboring = neighboringNodes[tmpStart - 2]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection - 2] = velocityVY[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection - 1] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                    tmpStart += 1
#        cuda.syncthreads()
"""
Update the distribution function for new nodes in transport domain
"""
@cuda.jit('void(int64, int64, int64, int64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:, :, :], boolean[:])')
def calUpdatedPDFWithNewRho(totalNodes, xDim, numTracers, newList, \
                            unitX, unitY, velocityX, velocityY, tracerConc, \
                            tracerConcNew, valueJDE, tracerPDF, distrField):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        if (distrField[indices] == 1):
            for tmpNew in newList:
                if indices == tmpNew:
                    for i in range(numTracers):
                        for j in range(5):
                            tracerPDF[i, indices, j] = tracerConcNew[i, indices] * \
                                     (valueJDE[i, j] + 1./2. * (unitX[j] * velocityX[indices] + \
                                      unitY[j] * velocityY[indices]))
                else:
                    for i in range(numTracers):
                        tmpConcDiff = tracerConcNew[i, indices] - tracerConc[i, indices]
                        for j in range(5):
                            tmpConcRatio = tracerPDF[i, indices, j] / tracerConc[i, indices]
                            tmpPDF = tracerPDF[i, indices, j] + tmpConcDiff * tmpConcRatio
                            tracerPDF[i, indices, j] = tmpPDF
        cuda.syncthreads()
        
"""
Free flow boundary condition for tracer
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :], \
                float64[:, :, :])')
def calFreeConcBoundary1(totalNodes, numTracers, nx, xDim, fluidNodes, neighboringNodes, \
                         tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        tmpIndices = fluidNodes[indices]
        if (tmpIndices <  3 * nx  and tmpIndices >= 2 * nx):
            tmpNeighbor = neighboringNodes[4 * indices + 2]
            for i in range(numTracers):
#                tracerConc[i, indices] = 0.
                for j in range(5):
                    tracerPDF[i, indices, j] = tracerPDF[i, tmpNeighbor, j]
    cuda.syncthreads()
    
"""
Free flow boundary condition for tracer
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :], \
                float64[:, :, :])')
def calFreeConcBoundary2(totalNodes, numTracers, nx, xDim, fluidNodes, neighboringNodes, \
                         tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpIndices = fluidNodes[indices]
        if (tmpIndices <  2 * nx  and tmpIndices >= nx):
            tmpNeighbor = neighboringNodes[4 * indices + 2]
            for i in range(numTracers):
#                tracerConc[i, indices] = 0.
                for j in range(5):
                    tracerPDF[i, indices, j] = tracerPDF[i, tmpNeighbor, j]
    cuda.syncthreads()
    
"""
Free flow boundary condition for tracer
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :], \
                float64[:, :, :])')
def calFreeConcBoundary3(totalNodes, numTracers, nx, xDim, fluidNodes, neighboringNodes, \
                         tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpIndices = fluidNodes[indices]
        if (tmpIndices <  nx  and tmpIndices >= 0):
            tmpNeighbor = neighboringNodes[4 * indices + 2]
            for i in range(numTracers):
#                tracerConc[i, indices] = 0.
                for j in range(5):
                    tracerPDF[i, indices, j] = tracerPDF[i, tmpNeighbor, j]
#                    tracerConc[i, indices] += tracerPDF[i, indices, j]
    cuda.syncthreads()
    
@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], float64[:, :], \
                float64[:,:,:], int64[:])')
def calZeroConcenBoundary(totalNodes, numTracers, nx, ny, xDim, fluidNodes, tracerConc, \
                          tracerPDF, neighboringNodes):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpIndices = fluidNodes[indices]
        if (tmpIndices >= (ny - 2) * nx and tmpIndices < (ny - 1) * nx):
            tmpStart = 4 * indices + 3
            tmpL = neighboringNodes[tmpStart]
            
            for i in range(numTracers):
                tracerConc[i, indices] = 0.
                for j in range(5):
                    tracerPDF[i, indices, j] = tracerPDF[i, tmpL, j]
                    tracerConc[i, indices] += tracerPDF[i, indices, j]
                    
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], boolean[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:, :], float64[:, :, :])')
def calUpdateConcInTransportDomainByV(totalNodes, numTracers, xDim, totalTracer, totalOld, \
                                      transportDomain, physicalVX, physicalVY, \
                                      unitVX, unitVY, weightsCoeff, tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    sharedWeights = cuda.shared.array(shape = (5,), dtype = float64)
    for i in range(5):
        sharedWeights[i] = weightsCoeff[i]
    if (indices < totalNodes):
        tmpVNorm = math.sqrt(physicalVX[indices] * physicalVX[indices] + \
                             physicalVY[indices] * physicalVY[indices])
        if (transportDomain[indices] == 1 and tmpVNorm > 1e-10):
            for i in range(numTracers):
                tmpExtraConc = tracerConc[i, indices] * \
                        totalOld[i] / totalTracer[i]
#                if tracerConc[i, indices] > 1.0e-7:
                tracerConc[i, indices] += tmpExtraConc#totalOld[i]
                for j in range(5):
                    tracerPDF[i, indices, j] = tracerConc[i, indices] * sharedWeights[j] * (1. + \
                      3. * (unitVX[j] * physicalVX[indices] + unitVY[j] * \
                      physicalVY[indices]))
#        if (transportDomain[indices] == 1):
#            for i in range(numTracers):
#                if tracerConc[i, indices] < 0.:
#                    tracerConc[i, indices] = 0.
#                    for j in range(5):
#                        tracerPDF[i, indices, j] = 0.
                
"""
Implement MRT scheme to the transport part (linear equilibrium function)
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:, :], float64[:, :, :], \
                float64[:, :], float64[:, :, :], float64[:])')
def calCollisionTransportLinearEqlMRTGPU(totalNodes, xDim, numTracers, unitVX, unitVY, \
                          velocityVX, velocityVY, tracerConc, tracerPDF, transportM, \
                          inverseRelaxationMS, weightsCoeff):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
#    sharedTM = cuda.shared.array(shape=(5, 5), dtype = float64)
#    sharedWeights = cuda.shared.array(shape = (5,), dtype = float64)
#    sharedInverseTM = cuda.shared.array(shape = (5, 5), dtype = float64)
#    sharedCollisionS = cuda.shared.array(shape = (5, 5), dtype = float64)
    
#    for i in range(5):
#        for j in range(5):
#            sharedTM[i, j] = transportM[i, j]
#    for i in range(5):
#        sharedWeights[i] = weightsCoeff[i]
#            sharedInverseTM = inverseTransportM[i, j]
    if (indices < totalNodes):
        tmpEql = cuda.local.array(shape = (5,), dtype = float64)
#        tmpMEql = cuda.local.array(shape = (5, ), dtype = float64)
        tmpIMEql = cuda.local.array(shape = (5,), dtype = float64)
#        tmpPDFM = cuda.local.array(shape = (5,), dtype = float64)
#        tmpPDFIM = cuda.local.array(shape = (5,), dtype = float64)
        tmpDiff = cuda.local.array(shape = (5,), dtype = float64)
        for i in range(numTracers):
            for j in range(5):
#                tmpEql[j] = tracerConc[i, indices] * sharedWeights[j] * (1. + \
#                      3. * (unitVX[j] * velocityVX[indices] + unitVY[j] * \
#                      velocityVY[indices]))
                tmpEql[j] = tracerConc[i, indices] * weightsCoeff[j] * (1. + \
                      3. * (unitVX[j] * velocityVX[indices] + unitVY[j] * \
                      velocityVY[indices]))

            for j in range(5):
                tmpValueEql = 0.; tmpValuePDF = 0.
                for k in range(5):
#                    tmpValueEql += sharedTM[j, k] * tmpEql[k]
#                    tmpValuePDF += tracerPDF[i, indices, k] * sharedTM[j, k]
                    tmpValueEql += transportM[j, k] * tmpEql[k]
                    tmpValuePDF += tracerPDF[i, indices, k] * transportM[j, k]
#                tmpMEql[j] = tmpValueEql; tmpPDFM[j] = tmpValuePDF
                tmpDiff[j] = tmpValuePDF - tmpValueEql
                
            for j in range(5):
#                tmpValueEql = 0
                tmpValuePDF = 0.
                for k in range(5):
#                    tmpValueEql += inverseRelaxationMS[i, j, k] * tmpMEql[k]
#                    tmpValuePDF += inverseRelaxationMS[i, j, k] * tmpPDFM[k]
                    tmpValuePDF += inverseRelaxationMS[i, j, k] * tmpDiff[k]
                tmpIMEql[j] = tmpValuePDF
            for j in range(5):
                tracerPDF[i, indices, j] = tracerPDF[i, indices, j] + tmpIMEql[j]

"""
Implement MRT scheme to the transport part (linear equilibrium function)
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:, :], float64[:, :, :], float64[:, :], \
                float64[:, :, :], float64[:])')
def calCollisionTransportQuadraticEqlMRTGPU(totalNodes, xDim, numTracers, unitVX, unitVY, \
                          velocityVX, velocityVY, tracerConc, tracerPDF, transportM, \
                          inverseRelaxationMS, weightsCoeff):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    sharedTM = cuda.shared.array(shape=(5, 5), dtype = float64)
    sharedWeights = cuda.shared.array(shape = (5,), dtype = float64)
#    sharedInverseTM = cuda.shared.array(shape = (5, 5), dtype = float64)
#    sharedCollisionS = cuda.shared.array(shape = (5, 5), dtype = float64)
    
    for i in range(5):
        for j in range(5):
            sharedTM[i, j] = transportM[i, j]
    for i in range(5):
        sharedWeights[i] = weightsCoeff[i]
#            sharedInverseTM = inverseTransportM[i, j]
    if (indices < totalNodes):
        tmpVX = 0.0
        tmpEql = cuda.local.array(shape = (5,), dtype = float64)
        tmpMEql = cuda.local.array(shape = (5, ), dtype = float64)
        tmpIMEql = cuda.local.array(shape = (5,), dtype = float64)
        tmpPDFM = cuda.local.array(shape = (5,), dtype = float64)
        tmpPDFIM = cuda.local.array(shape = (5,), dtype = float64)
        for i in range(numTracers):
            for j in range(5):
#                tmpEql[j] = tracerConc[i, indices] * sharedWeights[j] * (1. + \
#                      3. * (unitVX[j] * velocityVX[indices] + unitVY[j] * \
#                      velocityVY[indices]) + 4.5 * (unitVX[j] * velocityVX[indices] + \
#                      unitVY[indices] * velocityVY[indices]) * (unitVX[j] * \
#                      velocityVX[indices] + unitVY[j] * velocityVY[indices]) - 1.5 * \
#                      (velocityVX[indices] * velocityVX[indices] + velocityVY[indices] * \
#                       velocityVY[indices]))
                tmpEql[j] = tracerConc[i, indices] * sharedWeights[j] * (1. + \
                      3. * (unitVX[j] * tmpVX + unitVY[j] * \
                      velocityVY[indices]) + 4.5 * (unitVX[j] * tmpVX + \
                      unitVY[indices] * velocityVY[indices]) * (unitVX[j] * \
                      tmpVX + unitVY[j] * velocityVY[indices]) - 1.5 * \
                      (tmpVX * tmpVX + velocityVY[indices] * \
                       velocityVY[indices]))
#                (valueJDE[i, j] + 1./2. * \
#                           (unitVX[j] * velocityVX[indices] + unitVY[j] * \
#                            velocityVY[indices]))
            for j in range(5):
                tmpValueEql = 0.; tmpValuePDF = 0.
                for k in range(5):
                    tmpValueEql += sharedTM[j, k] * tmpEql[k]
                    tmpValuePDF += tracerPDF[i, indices, k] * sharedTM[j, k]
                tmpMEql[j] = tmpValueEql; tmpPDFM[j] = tmpValuePDF
            
            for j in range(5):
                tmpValueEql = 0.; tmpValuePDF = 0.
                for k in range(5):
                    tmpValueEql += inverseRelaxationMS[i, j, k] * tmpMEql[k]
                    tmpValuePDF += inverseRelaxationMS[i, j, k] * tmpPDFM[k]
                tmpIMEql[j] = tmpValueEql; tmpPDFIM[j] = tmpValuePDF
                tracerPDF[i, indices, j] = -tmpIMEql[j] + tmpPDFIM[j] + tracerPDF[i, \
                         indices, j]   
                
"""
Implement constant concentration boundary with anti-bounce back mehod
"""
@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], int64[:], float64[:],\
                float64[:], float64[:, :, :])')
def calAntiCollisionConcBoundary(totalNodes, xDim, numTracers, ny, nx, fluidNodes, \
                                 neighboringTRNodes, concBoundary, weightsCoeff,\
                                 tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpLoc = fluidNodes[indices]
        if (tmpLoc >=(ny - 2) * nx and tmpLoc < (ny - 1) * nx):
            for i in range(numTracers):
                tmpNewPDF = -tracerPDF[i, indices, 3] + 2. * weightsCoeff[3] * \
                            concBoundary[i]
                upperNeighbor = neighboringTRNodes[4 * indices + 2]
                tracerPDF[i, upperNeighbor, 4] = tmpNewPDF
                
"""
Implement constant concentration boundary condition with Inamuro's method
"""
@cuda.jit('void(int64, int64, int64, int64, int64, int64[:], int64[:], float64[:], \
                float64[:], float64[:, :, :])')
def calInamuroConstConcBoundary(totalNodes, xDim, numTracers, ny, nx, fluidNodes, \
                                neighboringTRNodes, concBoundary, weightsCoeff, \
                                tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpLoc = fluidNodes[indices]
        if (tmpLoc >=(ny - 1) * nx and tmpLoc < ny * nx):
            for i in range(numTracers):
                tmpSumPDF = tracerPDF[i, indices, 0] + tracerPDF[i, indices, 1] + \
                        tracerPDF[i, indices, 2] + tracerPDF[i, indices, 3]
                tmpUnknownConc = (concBoundary[i] - tmpSumPDF) / weightsCoeff[4]
                tracerPDF[i, indices, 4] = weightsCoeff[4] * tmpUnknownConc

                            
"""
Implement D2Q9 for transport
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :, :], float64[:])')
def calCollisionQ9(totalNodes, xDim, numTracers, unitVX, unitVY, \
                          velocityVX, velocityVY, tauDiff, tracerConc, tracerPDF, weightsCoeff):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
    for i in range(9):
        sharedWeights[i] = weightsCoeff[i]
#    sharedInverseTM = cuda.shared.array(shape = (5, 5), dtype = float64)
#    sharedCollisionS = cuda.shared.array(shape = (5, 5), dtype = float64)
    
#            sharedInverseTM = inverseTransportM[i, j]
    if (indices < totalNodes):
        for i in range(numTracers):
            for j in range(9):
                tmpEql = tracerConc[i, indices] * sharedWeights[j] * (1. + \
                      3. * (unitVX[j] * velocityVX[indices] + unitVY[j] * \
                      velocityVY[indices]))
#                      + 4.5 * (unitVX[j] * velocityVX[indices] + \
#                      unitVY[indices] * velocityVY[indices]) * (unitVX[j] * \
#                      velocityVX[indices] + unitVY[j] * velocityVY[indices]) - 1.5 * \
#                      (velocityVX[indices] * velocityVX[indices] + velocityVY[indices] * \
#                       velocityVY[indices]))
                tracerPDF[i, indices, j] = -(tracerPDF[i, indices, j] - tmpEql)/tauDiff[i] + \
                                        tracerPDF[i, indices, j] 
                
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
Deal with the interface boundary
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], float64[:], float64[:], \
                float64[:, :], float64[:, :, :], boolean[:])')
def calTransportInterfaceQ9GPU(totalNodes, xDim, numTracers, numScheme, neighboringNodes, \
                             velocityVX, velocityVY, tracerConc, tracerPDF, distriField):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        if distriField[indices] == 1:
            for i in range(numTracers):
                tmpStart = 8 * indices
                tmpDirection = 0
                for j in range(1, numScheme):
                    tmpDirection += 1
                    tmpIndices = neighboringNodes[tmpStart]
                    if (tmpIndices >= 0 and distriField[tmpIndices] == 0):
                        if tmpDirection == 1:
#                            tmpNeighboring = neighboringNodes[tmpStart + 1]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection + 2] = -velocityVX[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection + 1] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection + 1] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 2:
#                            tmpNeighboring = neighboringNodes[tmpStart + 2]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection + 2] = -velocityVY[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 3:
#                            tmpNeighboring = neighboringNodes[tmpStart - 2]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection - 2] = velocityVX[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 4:
#                            tmpNeighboring = neighboringNodes[tmpStart - 2]
#                            if (tmpNeighboring >= 0 and distriField[tmpNeighboring] != 0):
#                                tracerConc[i, indices] = tracerConc[i, tmpNeighboring]
#                                tracerPDF[i, indices, tmpDirection - 2] = velocityVY[indices] * \
#                                         tracerConc[i, tmpNeighboring] + tracerPDF[i, indices, tmpDirection]
                            tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
#                            elif (tmpNeighboring >= 0 and distriField[tmpNeighboring] == 0):
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
#                            elif tmpNeighboring < 0:
#                                tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
#                                tracerPDF[i, tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 5:
                            tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 6:
                            tracerPDF[i, indices, tmpDirection + 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 7:
                            tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
                        elif tmpDirection == 8:
                            tracerPDF[i, indices, tmpDirection - 2] = tracerPDF[i, tmpIndices, tmpDirection]
                            tracerPDF[i,  tmpIndices, tmpDirection] = 0.
                    tmpStart += 1
    cuda.syncthreads()

@cuda.jit('void(int64, int64, int64, float64[:], float64[:], boolean[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:, :], float64[:, :, :])')
def calUpdateConcInTransportDomainByVQ9(totalNodes, numTracers, xDim, totalTracer, totalOld, \
                                      transportDomain, physicalVX, physicalVY, \
                                      unitVX, unitVY, weightsCoeff, tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    sharedWeights = cuda.shared.array(shape = (5,), dtype = float64)
    for i in range(9):
        sharedWeights[i] = weightsCoeff[i]
    if (indices < totalNodes):
        tmpVNorm = math.sqrt(physicalVX[indices] * physicalVX[indices] + \
                             physicalVY[indices] * physicalVY[indices])
        if (transportDomain[indices] == 1 and tmpVNorm > 1e-10):
            for i in range(numTracers):
                tmpExtraConc = tracerConc[i, indices] * \
                        totalOld[i] / totalTracer[i]
                tracerConc[i, indices] += tmpExtraConc
                for j in range(9):
                    tracerPDF[i, indices, j] = tracerConc[i, indices] * sharedWeights[j] * (1. + \
                      3. * (unitVX[j] * physicalVX[indices] + unitVY[j] * \
                      physicalVY[indices]) + 4.5 * (unitVX[j] * physicalVX[indices] + \
                      unitVY[indices] * physicalVY[indices]) * (unitVX[j] * \
                      physicalVX[indices] + unitVY[j] * physicalVY[indices]) - 1.5 * \
                      (physicalVX[indices] * physicalVX[indices] + physicalVY[indices] * \
                       physicalVY[indices]))
                
"""
Update the domain for the transport
"""
@cuda.jit('void(int64, int64, float64, float64[:], float64[:])')
def calValueTransportDomain(totalNodes, xDim, critiriaValue, valueTransportDomain, \
                            fluidRhoR):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpRho = fluidRhoR[indices] 
        if (tmpRho > critiriaValue):
            valueTransportDomain[indices] = -(1. - 1.) 
        else:
            valueTransportDomain[indices] = -(1. - 0.)
            
"""
Update the tracer PDF due to the interface effect
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:, :], \
                float64[:, :, :])')
def calTransportWithInterfaceD2Q5(totalNodes, xDim, numTracers, betaTracer, 
                              valueTransportDomain, unitEX, unitEY, gradientX,\
                              gradientY, weightsCoeff, tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    sharedEX = cuda.shared.array(shape = (4,), dtype = float64)
    sharedEY = cuda.shared.array(shape = (4,), dtype = float64)
    
    sharedEX[0] = unitEX[1]; sharedEX[1] = unitEX[3]; sharedEX[2] = unitEX[2]
    sharedEX[3] = unitEX[4]
    sharedEY[0] = unitEY[1]; sharedEY[1] = unitEY[3]; sharedEY[2] = unitEY[2]
    sharedEY[3] = unitEY[4]
    if (indices < totalNodes):

        tmpGradientNorm = math.sqrt(gradientX[indices] * gradientX[indices] + \
                                    gradientY[indices] * gradientY[indices])
        if tmpGradientNorm > 1.0e-8:
            unitGX = -gradientX[indices] / tmpGradientNorm
            unitGY = -gradientY[indices] / tmpGradientNorm
            unitGradNorm = math.sqrt(unitGX * unitGX + unitGY * unitGY)
        else:
            unitGX = 0.; unitGY = 0.; unitGradNorm = 0.
        for i in range(numTracers):
            for j in range(4):
                tmpEq = weightsCoeff[j + 1] * tracerConc[i, indices]
                tmpUnitNorm = math.sqrt(sharedEX[j] * sharedEX[j] + sharedEY[j] * \
                            sharedEY[j])
                if (unitGradNorm > 1.0e-8 and tmpUnitNorm > 1.0e-8):
                    cosTheta = (sharedEX[j] * unitGX + sharedEY[j] * unitGY) / \
                                (tmpUnitNorm * unitGradNorm)
                else:
                    cosTheta = 0.
                tracerPDF[i, indices, j + 1] = tracerPDF[i, indices, j + 1] + betaTracer[i] * \
                    valueTransportDomain[indices] * tmpEq * cosTheta
    
"""
Update the tracer PDF due to the interface effect
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:], float64[:], float64[:], float64[:, :], \
                float64[:, :, :])')
def calTransportWithInterfaceD2Q9(totalNodes, xDim, numTracers, betaTracer, 
                              valueTransportDomain, unitEX, unitEY, gradientX,\
                              gradientY, weightsCoeff, tracerConc, tracerPDF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpGradientNorm = math.sqrt(gradientX[indices] * gradientX[indices] + \
                                    gradientY[indices] * gradientY[indices])
        if tmpGradientNorm > 1.0e-8:
            unitGX = -gradientX[indices] / tmpGradientNorm
            unitGY = -gradientY[indices] / tmpGradientNorm
            unitGradNorm = math.sqrt(unitGX * unitGX + unitGY * unitGY)
        else:
            unitGX = 0.; unitGY = 0.; unitGradNorm = 0.
        for i in range(numTracers):
            for j in range(1, 9):
                tmpEq = weightsCoeff[j] * tracerConc[i, indices]
                tmpUnitNorm = math.sqrt(unitEX[j] * unitEX[j] + unitEY[j] * unitEY[j])
                if (unitGradNorm > 1.0e-8 and tmpUnitNorm > 1.0e-8):
                    cosTheta = (unitEX[j] * unitGX + unitEY[j] * unitGY) / \
                        (tmpUnitNorm * unitGradNorm)
                else:
                    cosTheta = 0.
                tracerPDF[i, indices, j] = tracerPDF[i, indices, j] + betaTracer[i] * \
                    valueTransportDomain[indices] * tmpEq * cosTheta
                    
"""
Implement MRT scheme to the transport part (linear equilibrium function)
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:], float64[:, :], float64[:, :, :], \
                float64[:, :], float64[:, :, :], float64[:])')
def calCollisionTransportLinearEqlMRTGPUD2Q9(totalNodes, xDim, numTracers, unitVX, unitVY, \
                          velocityVX, velocityVY, tracerConc, tracerPDF, transportM, \
                          inverseRelaxationMS, weightsCoeff):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
#    sharedTM = cuda.shared.array(shape=(9, 9), dtype = float64)
#    sharedWeights = cuda.shared.array(shape = (9,), dtype = float64)
#    sharedInverseTM = cuda.shared.array(shape = (5, 5), dtype = float64)
#    sharedCollisionS = cuda.shared.array(shape = (5, 5), dtype = float64)
    
#    for i in range(9):
#        for j in range(9):
#            sharedTM[i, j] = transportM[i, j]
#    for i in range(9):
#        sharedWeights[i] = weightsCoeff[i]
#            sharedInverseTM = inverseTransportM[i, j]
    if (indices < totalNodes):
        tmpEql = cuda.local.array(shape = (9,), dtype = float64)
#        tmpMEql = cuda.local.array(shape = (5, ), dtype = float64)
        tmpIMEql = cuda.local.array(shape = (9,), dtype = float64)
#        tmpPDFM = cuda.local.array(shape = (5,), dtype = float64)
#        tmpPDFIM = cuda.local.array(shape = (5,), dtype = float64)
        tmpDiff = cuda.local.array(shape = (9,), dtype = float64)
        for i in range(numTracers):
            for j in range(9):
#                tmpEql[j] = tracerConc[i, indices] * sharedWeights[j] * (1. + \
#                      3. * (unitVX[j] * velocityVX[indices] + unitVY[j] * \
#                      velocityVY[indices]))
                tmpEql[j] = tracerConc[i, indices] * weightsCoeff[j] * (1. + \
                      3. * (unitVX[j] * velocityVX[indices] + unitVY[j] * \
                      velocityVY[indices]))

            for j in range(9):
                tmpValueEql = 0.; tmpValuePDF = 0.
                for k in range(9):
#                    tmpValueEql += sharedTM[j, k] * tmpEql[k]
#                    tmpValuePDF += tracerPDF[i, indices, k] * sharedTM[j, k]
#                tmpMEql[j] = tmpValueEql; tmpPDFM[j] = tmpValuePDF
                    tmpValueEql += transportM[j, k] * tmpEql[k]
                    tmpValuePDF += tracerPDF[i, indices, k] * transportM[j, k]
                tmpDiff[j] = tmpValuePDF - tmpValueEql
                
            for j in range(9):
#                tmpValueEql = 0
                tmpValuePDF = 0.
                for k in range(9):
#                    tmpValueEql += inverseRelaxationMS[i, j, k] * tmpMEql[k]
#                    tmpValuePDF += inverseRelaxationMS[i, j, k] * tmpPDFM[k]
                    tmpValuePDF += inverseRelaxationMS[i, j, k] * tmpDiff[k]
                tmpIMEql[j] = tmpValuePDF
            for j in range(9):
                tracerPDF[i, indices, j] = tracerPDF[i, indices, j] + tmpIMEql[j]