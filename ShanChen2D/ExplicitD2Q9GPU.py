"""
File for explicit force scheme of D2Q9 Shan-Chen model LBM. The isotropy here 
can be 4, 8 and 10. MRT scheme is also included
"""

import numpy as np
import scipy as sp
from math import sqrt

from numba import cuda, int64, float64
from numba import cuda, jit


"""
The calculation for the pressures in the domain.
"""
@cuda.jit('void(int64, int64, int64, float64[:, :], float64[:, :], float64[:, :], \
                float64[:])')
def calMacroPressureEX(totalNodes, numFluids, xDim, interactionCoeff, fluidRho, \
                       fluidPotential, fluidPressure):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        tmpPart1 = 0.; tmpPart2 = 0.
        for i in range(numFluids):
            tmpPart1 += fluidRho[i, indices]
            for j in range(numFluids):
                tmpPart2 = interactionCoeff[i, j] * fluidPotential[i, indices] * \
                                           fluidPotential[j, indices]
        fluidPressure[indices] = 1./3. * tmpPart1 +  3. * tmpPart2
                     
"""
P-R equation of state for effective mass
"""
@cuda.jit('void(int64, int64, int64, float64, float64[:, :], float64[:, :], float64[:, :])')
def calEffectiveMassPR(totalNodes, numFluids, xDim, temperature, interactionCoeff, \
                       fluidRho, fluidPsi):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    

"""
The explicit scheme for isotropy is 4
"""            
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:], float64[:, :], \
               float64[:], float64[:, :], float64[:, :], float64[:, :])')
def calExplicit4thOrderScheme(totalNodes, numFluids, xDim, fluidNodes, \
                              neighboringNodes, weightInter, interactionCoeff, \
                              interactionSolid, fluidPotential, forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx 
    
    if (indices < totalNodes):
        for i in range(numFluids):
            forceX[i, indices] = 0.; tmpGradX = 0.; tmpFXS = 0.; sumWeightX = 0.
            tmpFXF = 0.; sumWeightSX = 0.
            forceY[i, indices] = 0.; tmpGradY = 0.; tmpFYS = 0.; sumWeightY = 0
            tmpFYF = 0.; sumWeightSY = 0.
            tmpStart = indices * 8
            if (neighboringNodes[tmpStart] != -1):
                tmpE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradX += weightInter[0] *(fluidPotential[j, tmpE] - \
                                fluidPotential[j, indices]) * (1.) * interactionCoeff[i, j]
                    sumWeightX += weightInter[0]
            elif (neighboringNodes[tmpStart] == -1):
                tmpFXS += -weightInter[0] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
#                tmpFXS += -1./9. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (1.)
                sumWeightSX += weightInter[0]
            #Northern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpN = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradY += weightInter[1] * (fluidPotential[j, tmpN] - \
                                fluidPotential[j, indices]) * (1.) * interactionCoeff[i, j]
                    sumWeightY += weightInter[1]
            elif (neighboringNodes[tmpStart] == -1):
                tmpFYS += -weightInter[1] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
#                tmpFYS += -1./9. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (1.)
                sumWeightSY += weightInter[1]
            #Western Point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradX +=  weightInter[2] * (fluidPotential[j, tmpW] - \
                                fluidPotential[j, indices]) * (-1.) * interactionCoeff[i, j]
                    sumWeightX += weightInter[2]
            elif(neighboringNodes[tmpStart] == -1):
                tmpFXS += -weightInter[2] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
#                tmpFXS += -1./9. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (-1.)
                sumWeightSX += weightInter[2]
            #Southern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpS = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradY +=  weightInter[3] * (fluidPotential[j, tmpS] - \
                                fluidPotential[j, indices]) * (-1.) * interactionCoeff[i, j]
                    sumWeightY += weightInter[3]
            elif (neighboringNodes[tmpStart] == -1):
                tmpFYS += -weightInter[3] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
#                tmpFYS += -1./9. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (-1.)
                sumWeightSY += weightInter[3]
            #Northeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradX += weightInter[4] * (fluidPotential[j, tmpNE] - \
                                fluidPotential[j, indices]) * (1.) * interactionCoeff[i, j]
                    tmpGradY += weightInter[4] * (fluidPotential[j, tmpNE] - \
                                fluidPotential[j, indices]) * (1.) * interactionCoeff[i, j]
                    sumWeightX += weightInter[4]
                    sumWeightY += weightInter[4]
            elif(neighboringNodes[tmpStart] == -1):
                tmpFXS += -weightInter[4] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFYS += -weightInter[4] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
#                tmpFXS += -1./36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (1.)
#                tmpFYS += -1./36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (1.)                      
                sumWeightSX += weightInter[4]
                sumWeightSY += weightInter[4]
            #Northwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpNW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradX += weightInter[5] * (fluidPotential[j, tmpNW] - \
                                fluidPotential[j, indices]) * (-1.) * interactionCoeff[i, j]
                    tmpGradY += weightInter[5] * (fluidPotential[j, tmpNW] - \
                                fluidPotential[j, indices]) * (1.) * interactionCoeff[i, j]
                    sumWeightX += weightInter[5]
                    sumWeightY += weightInter[5]
            elif (neighboringNodes[tmpStart] == -1):
                tmpFXS += -weightInter[5] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFYS += -weightInter[5] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)  
#                tmpFXS += -1/36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (-1.)
#                tmpFYS += -1./36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (1.)
                sumWeightSX += weightInter[5]
                sumWeightSY += weightInter[5]
            #Southwestern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSW = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradX += weightInter[6] * (fluidPotential[j, tmpSW] - \
                                fluidPotential[j, indices]) * (-1.) * interactionCoeff[i, j]
                    tmpGradY += weightInter[6] * (fluidPotential[j, tmpSW] - \
                                fluidPotential[j, indices]) * (-1.) * interactionCoeff[i, j]
                    sumWeightX += weightInter[6]
                    sumWeightY += weightInter[6]
            elif (neighboringNodes[tmpStart] == -1):
                tmpFXS += -weightInter[6] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                tmpFYS += -weightInter[6] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
#                tmpFXS += -1./36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (-1.)
#                tmpFYS += -1./36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (-1.)
                sumWeightSX += weightInter[6]
                sumWeightSY += weightInter[6]
            #Southeastern point
            tmpStart += 1
            if (neighboringNodes[tmpStart] != -1):
                tmpSE = neighboringNodes[tmpStart]
                for j in range(numFluids):
                    tmpGradX += weightInter[7] * (fluidPotential[j, tmpSE] - \
                            fluidPotential[j, indices]) * (1.) * interactionCoeff[i, j]
                    tmpGradY += weightInter[7] * (fluidPotential[j, tmpSE] - \
                            fluidPotential[j, indices]) * (-1.) * interactionCoeff[i, j]
                    sumWeightX += weightInter[7]
                    sumWeightY += weightInter[7]
            elif (neighboringNodes[tmpStart] == -1):
                tmpFXS += -weightInter[7] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                tmpFYS += -weightInter[7] * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
#                tmpFXS += -1./36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (1.)
#                tmpFYS += -1./36. * interactionSolid[i] * \
#                      fluidPotential[i, indices] * (-1.)
                sumWeightSX += weightInter[7]
                sumWeightSY += weightInter[7]
#            if (sumWeightX > 1e-12):
            tmpFXF = -6.0 * fluidPotential[i, indices] * tmpGradX #/ sumWeightX
            forceX[i, indices] +=  tmpFXF
#            if (sumWeightY > 1e-12):
            tmpFYF = -6.0 * fluidPotential[i, indices] * tmpGradY #/ sumWeightY
            forceY[i, indices] += tmpFYF
#            if (sumWeightSX > 1e-12):
            forceX[i, indices] += tmpFXS
#            if (sumWeightSY > 1e-12):
            forceY[i, indices] += tmpFYS
    cuda.syncthreads()
    
"""
Calculate the equilibrium function in each direction
"""
"""
Calculate the equilibrium function for collision process (most common f_eq)
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], float64[:, :], float64[:], \
        float64[:], float64[:, :, :])')
def calEquilibriumFuncEFGPU(totalNum, numFluids, xDim, weightCoeff, EX, EY, \
                            fluidRho, equilibriumVX, equilibriumVY, fEq):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    sharedWeightCoeff = cuda.shared.array(shape = (9, ), dtype = float64)
    tmpEX = cuda.shared.array(shape = (9,), dtype = float64)
    tmpEY = cuda.shared.array(shape = (9,), dtype = float64)
    for i in range(9):
        sharedWeightCoeff[i] = weightCoeff[i]
        tmpEX[i] = EX[i]
        tmpEY[i] = EY[i]
    if (indices < totalNum):
        for i in range(numFluids):
            for j in range(9):
                fEq[i, indices, j] = sharedWeightCoeff[j] * fluidRho[i, indices] * \
                   (1. + 3. * (tmpEX[j] * equilibriumVX[indices] + tmpEY[j] * \
                    equilibriumVY[indices]) + 9./2. * ((tmpEX[j] * equilibriumVX[indices] + \
                    tmpEY[j] * equilibriumVY[indices]) * (tmpEX[j] * equilibriumVX[indices] + \
                    tmpEY[j] * equilibriumVY[indices])) - 3./2. * (equilibriumVX[indices] * \
                    equilibriumVX[indices] + equilibriumVY[indices] * equilibriumVY[indices]))
    cuda.syncthreads()
"""
Calculate the PDF of force term
""" 
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], float64[:], \
                float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], \
                float64[:, :, :])')
def calForceDistrGPU(totalNodes, numFluids, xDim, EX, EY, equilibriumVX, \
                     equilibriumVY, fluidRho, forceX, forceY, fEq, fForce):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    tmpEX = cuda.shared.array(shape = (9,), dtype = float64)
    tmpEY = cuda.shared.array(shape = (9,), dtype = float64)
    for i in range(9):
        tmpEX[i] = EX[i]
        tmpEY[i] = EY[i]
    if (indices < totalNodes):
        for i in range(numFluids):
            for j in range(9):
                fForce[i, indices, j] = ((forceX[i, indices] * (tmpEX[j] - \
                    equilibriumVX[indices])) + (forceY[i, indices] * (tmpEY[j] - \
                    equilibriumVY[indices]))) * fEq[i, indices, j] / (1./3. * \
                    fluidRho[i, indices])
    cuda.syncthreads()
    
"""
Transform the original fluid PDF
"""
@cuda.jit('void(int64, int64, int64, float64[:, :, :], float64[:, :, :])')
def transformPDFGPU(totalNodes, numFluids, xDim, fluidPDF, fForce):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        for i in range(numFluids):
            for j in range(9):
                fluidPDF[i, indices, j] = fluidPDF[i, indices, j] - 1./2. * \
                        fForce[i, indices, j]

"""
Calculate collision process in the explicit scheme
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:, :, :], float64[:, :, :], \
                float64[:, :, :])')
def calCollisionEXGPU(totalNodes, numFluids, xDim, tau, fluidPDF, fEq, fForce):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    
    if (indices < totalNodes):
        for i in range(numFluids):
            for j in range(9):
                fluidPDF[i, indices, j] = fluidPDF[i, indices, j] + 1./tau[i] * \
                        (fEq[i, indices, j] - fluidPDF[i, indices, j] - 1./2. * \
                         fForce[i, indices, j]) + 1. * fForce[i, indices, j]

"""
Calculate the velocity of each component in the domain
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:],  float64[:, :], \
                float64[:, :], float64[:, :, :], float64[:], float64[:])')
def calTotalVelocityGPU(totalNodes, numFluids, xDim, EX, EY, forceX, \
                        forceY, fluidPDF, totalVX, totalVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    tmpEX = cuda.shared.array(shape = (9,), dtype = float64)
    tmpEY = cuda.shared.array(shape = (9,), dtype = float64)
    for i in range(9):
        tmpEX[i] = EX[i]
        tmpEY[i] = EY[i]
    if (indices < totalNodes):
        tmpMomentumX = 0.; tmpMomentumY = 0.; tmpRho = 0.
        for i in range(numFluids):
            for j in range(9):
                tmpMomentumX += fluidPDF[i, indices, j] * tmpEX[j]
                tmpMomentumY += fluidPDF[i, indices, j] * tmpEY[j]
                tmpRho += fluidPDF[i, indices, j]
            tmpMomentumX += 1./2. * forceX[i, indices]
            tmpMomentumY += 1./2. * forceY[i, indices]
#            tmpRho += fluidRho[i, indices]
        totalVX[indices] = tmpMomentumX / tmpRho
        totalVY[indices] = tmpMomentumY / tmpRho

"""
Calculate the equilibrium velocity in explicit scheme
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], float64[:], \
                float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], \
                float64[:], float64[:])')
def calEquilibriumVEFGPU(totalNodes, numFluids, xDim, tau, EX, EY, fluidRho, \
                         forceX, forceY, fluidPDF, eqVX, eqVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx 
    tmpEX = cuda.shared.array(shape = (9,), dtype = float64)
    tmpEY = cuda.shared.array(shape = (9,), dtype = float64)
    for i in range(9):
        tmpEX[i] = EX[i]
        tmpEY[i] = EY[i]
    if (indices < totalNodes):
        tmpMomentumX = 0.; tmpMomentumY = 0.; tmpRhoT = 0.
        for i in range(numFluids):
            eachMX = 0.; eachMY = 0.; tmpRho = fluidRho[i, indices]; tmpTau = tau[i]
            for j in range(9):
                eachMX += fluidPDF[i, indices, j] * tmpEX[j]
                eachMY += fluidPDF[i, indices, j] * tmpEY[j]
            eachMX += 1./2. * forceX[i, indices]
            eachMY += 1./2. * forceY[i, indices]
            tmpMomentumX += eachMX / tmpTau
            tmpMomentumY += eachMY / tmpTau
            tmpRhoT = tmpRhoT + tmpRho / tmpTau
        eqVX[indices] = tmpMomentumX / tmpRhoT
        eqVY[indices] = tmpMomentumY / tmpRhoT


"""
Calculate the local pressure in the explicit scheme
"""
@cuda.jit('void(int64, int64, int64, float64[:, :], float64[:, :], float64[:, :], \
                float64[:])')
def calPressureExpGPU(totalNodes, numFluids, xDim, interCoeff, fluidRho, \
                      fluidPotential, fluidPressure):
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
                tmpPressure += 6./2. * interCoeff[i, j] * fluidPotential[i, indices] * \
                            fluidPotential[j, indices]
        fluidPressure[indices] = tmpPressure
                     
"""
Calculate the neighboring and the next neighboring nodes in isotropy = 8
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:, :], int64[:])')
def fillNeighboringNodesISO8(totalNodes, nx, ny, xDim, fluidNodes, domainNewIndex, \
                         neighboringNodes):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
#    ty = cuda.threadIdx.y; by = cuda.blockIdx.y; bDimY = cuda.blockDim.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        tmpStart = 24 * indices
        tmpLoc = fluidNodes[indices]
        i = int(tmpLoc / nx); j = tmpLoc % nx
        tmpF = j + 1 if j < nx - 1 else 0
        tmpB = j - 1 if j > 0 else (nx - 1)
        tmpU = i + 1 if i < ny - 1 else 0
        tmpL = i - 1 if i > 0 else (ny - 1)
        
        tmpF2 = j + 2 if (j < nx - 2) else (j - nx + 2)
        tmpB2 = j - 2 if (j > 1) else (nx - 2 + j)
        tmpU2 = i + 2 if (i < ny - 2) else (i - ny + 2)
        tmpL2 = i - 2 if (i > 1) else (ny - 2 + i)
        
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
        #Eastern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpF2]
        #Northern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, j]
        #Western node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpB2]
        #Southern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, j]
        #Northeastern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpF2]
        #Northwestern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpB2]
        #Southwestern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpB2]
        #Southeastern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpF2]
        #North1Eastern2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpF2]
        #North2Eastern1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpF]
        #North2Western1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpB]
        #North1Western2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpB2]
        #South1Western2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpB2]
        #South2Western1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpB]
        #South2Eastern1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpF]
        #South1Eastern2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpF2]
        
"""
Calculate the neighboring and the next neighboring nodes in isotropy = 10
"""  
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:, :], int64[:])')
def fillNeighboringNodesISO10(totalNodes, nx, ny, xDim, fluidNodes, domainNewIndex, \
                         neighboringNodes):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
#    ty = cuda.threadIdx.y; by = cuda.blockIdx.y; bDimY = cuda.blockDim.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        tmpStart = 36 * indices
        tmpLoc = fluidNodes[indices]
        i = int(tmpLoc / nx); j = tmpLoc % nx
        tmpF = j + 1 if j < nx - 1 else 0
        tmpB = j - 1 if j > 0 else (nx - 1)
        tmpU = i + 1 if i < ny - 1 else 0
        tmpL = i - 1 if i > 0 else (ny - 1)
        
        tmpF2 = j + 2 if (j < nx - 2) else (j - nx + 2)
        tmpB2 = j - 2 if (j > 1) else (nx - 2 + j)
        tmpU2 = i + 2 if (i < ny - 2) else (i - ny + 2)
        tmpL2 = i - 2 if (i > 1) else (ny - 2 + i)
        
        tmpF3 = j + 3 if (j < nx - 3) else (j - nx + 3)
        tmpB3 = j - 3 if (j > 2) else (nx - 3 + j)
        tmpU3 = i + 3 if (i < ny - 3) else (i - ny + 3)
        tmpL3 = i - 3 if (i > 2) else (ny - 3 + i)
        
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
        #Eastern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpF2]
        #Northern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, j]
        #Western node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpB2]
        #Southern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, j]
        #Northeastern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpF2]
        #Northwestern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpB2]
        #Southwestern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpB2]
        #Southeastern node 2
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpF2]
        #North1Eastern2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpF2]
        #North2Eastern1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpF]
        #North2Western1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU2, tmpB]
        #North1Western2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpB2]
        #South1Western2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpB2]
        #South2Western1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpB]
        #South2Eastern1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL2, tmpF]
        #South1Eastern2 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpF2]
        #Eastern node 3
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpF3]
        #Northern node 3
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU3, j]
        #Western node 3
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[i, tmpB3]
        #Southern node 3
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL3, j]
        #North1eastern3 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpF3]
        #North3eastern1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU3, tmpF]
        #North3western1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU3, tmpB]
        #North1western3 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpU, tmpB3]
        #South1western3 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpB3]
        #South3western1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL3, tmpB]
        #South3eastern1 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL3, tmpF]
        #South1eastern3 node
        tmpStart += 1
        neighboringNodes[tmpStart] = domainNewIndex[tmpL, tmpF3]
        

"""
The explicit scheme for isotropy is 8
"""            
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:], float64[:, :], \
               float64[:], float64[:, :], float64[:, :], float64[:, :])')
def calExplicit8thOrderScheme(totalNodes, numFluids, xDim, fluidNodes, \
                              neighboringNodes, weightInter, interactionCoeff, \
                              interactionSolid, fluidPotential, forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
#    ty = cuda.threadIdx.y; by = cuda.blockIdx.y; bDimY = cuda.blockDim.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        for i in range(numFluids):
            forceX[i, indices] = 0.0; forceY[i, indices] = 0.0
        tmpStart = indices * 24
        tmpStart0 = tmpStart
        tmpStart1 = tmpStart + 1
        tmpStart2 = tmpStart + 2
        tmpStart3 = tmpStart + 3
        tmpStart4 = tmpStart + 4
        tmpStart5 = tmpStart + 5
        tmpStart6 = tmpStart + 6
        tmpStart7 = tmpStart + 7
        tmpStart8 = tmpStart + 8
        tmpStart9 = tmpStart + 9
        tmpStart10 = tmpStart + 10
        tmpStart11 = tmpStart + 11
        tmpStart12 = tmpStart + 12
        tmpStart13 = tmpStart + 13
        tmpStart14 = tmpStart + 14
        tmpStart15 = tmpStart + 15
        tmpStart16 = tmpStart + 16
        tmpStart17 = tmpStart + 17
        tmpStart18 = tmpStart + 18
        tmpStart19 = tmpStart + 19
        tmpStart20 = tmpStart + 20
        tmpStart21 = tmpStart + 21
        tmpStart22 = tmpStart + 22
        tmpStart23 = tmpStart + 23
        
        if (neighboringNodes[tmpStart0] != -1):
            tmpE = neighboringNodes[tmpStart0]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[0] * interactionCoeff[i, j] * \
                              fluidPotential[i, indices] * (fluidPotential[j, tmpE] - \
                              fluidPotential[j, indices]) * (1.)
        elif (neighboringNodes[tmpStart0] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Northern point
        if (neighboringNodes[tmpStart1] != -1):
            tmpN = neighboringNodes[tmpStart1]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -6.0 * weightInter[1] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN] - \
                          fluidPotential[j, indices])* (1.)
        elif (neighboringNodes[tmpStart1] == -1):
            for i in range(numFluids):
                forceY[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Western Point
        if (neighboringNodes[tmpStart2] != -1):
            tmpW = neighboringNodes[tmpStart2]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[2] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpW] - \
                          fluidPotential[j, indices]) * (-1.)
        elif(neighboringNodes[tmpStart2] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Southern point
        if (neighboringNodes[tmpStart3] != -1):
            tmpS = neighboringNodes[tmpStart3]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -6.0 * weightInter[3] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS] - \
                          fluidPotential[j, indices]) * (-1.)
        elif (neighboringNodes[tmpStart3] == -1):
            for i in range(numFluids):
                forceY[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Northeastern point
        if (neighboringNodes[tmpStart4] != -1):
            tmpNE = neighboringNodes[tmpStart4]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[4] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpNE] - \
                          fluidPotential[j, indices]) * (1.)
                    forceY[i, indices] += -6.0 * weightInter[4] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpNE] - \
                          fluidPotential[j, indices]) * (1.)
        elif(neighboringNodes[tmpStart4] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Northwestern point
        if (neighboringNodes[tmpStart5] != -1):
            tmpNW = neighboringNodes[tmpStart5]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[5] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpNW] - \
                          fluidPotential[j, indices]) * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[5] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpNW] - \
                          fluidPotential[j, indices]) * (1.)
        elif (neighboringNodes[tmpStart5] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Southwestern point
        if (neighboringNodes[tmpStart6] != -1):
            tmpSW = neighboringNodes[tmpStart6]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[6] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpSW] - \
                          fluidPotential[j, indices]) * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[6] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpSW] - \
                          fluidPotential[j, indices]) * (-1.)
        elif (neighboringNodes[tmpStart6] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Southeastern point
        if (neighboringNodes[tmpStart7] != -1):
            tmpSE = neighboringNodes[tmpStart7]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[7] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpSE] - \
                          fluidPotential[j, indices]) * (1.)
                    forceY[i, indices] += -6.0 * weightInter[7] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpSE] - \
                          fluidPotential[j, indices]) * (-1.)
        elif (neighboringNodes[tmpStart7] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Easter point 2
        if (neighboringNodes[tmpStart8] != -1 and neighboringNodes[tmpStart] != -1):
            tmpE2 = neighboringNodes[tmpStart8]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[8] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpE2] - \
                          fluidPotential[j, indices]) * (1.)
        #Northern point 2
        if (neighboringNodes[tmpStart9] != -1 and neighboringNodes[tmpStart1] != -1):
            tmpN2 = neighboringNodes[tmpStart9]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -2. * 6.0 * weightInter[9] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN2] - \
                          fluidPotential[j, indices]) * (1.)
        #Western point 2
        if (neighboringNodes[tmpStart10] != -1 and neighboringNodes[tmpStart2] != -1):
            tmpW2 = neighboringNodes[tmpStart10]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[10] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpW2] - \
                          fluidPotential[j, indices]) * (-1.)
        #Southern poitn 2
        if (neighboringNodes[tmpStart11] != -1 and neighboringNodes[tmpStart3] != -1):
            tmpS2 = neighboringNodes[tmpStart11]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -2. * 6.0 * weightInter[11] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS2] - \
                          fluidPotential[j, indices]) * (-1.)
        #Notheastern 2
        if (neighboringNodes[tmpStart12] != -1 and neighboringNodes[tmpStart4] != -1):
            tmpE2N2 = neighboringNodes[tmpStart12]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[12] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpE2N2] - \
                          fluidPotential[j, indices])* (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[12] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpE2N2] - \
                          fluidPotential[j, indices])* (1.)
        #Northwestern 2
        if (neighboringNodes[tmpStart13] != -1 and neighboringNodes[tmpStart5] != -1):
            tmpW2N2 = neighboringNodes[tmpStart13]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[13] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpW2N2] - \
                          fluidPotential[j, indices]) * (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[13] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpW2N2] - \
                          fluidPotential[j, indices]) * (1.)
        #Southwestern 2
        if (neighboringNodes[tmpStart14] != -1 and neighboringNodes[tmpStart6] != -1):
            tmpW2S2 = neighboringNodes[tmpStart14]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[14] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpW2S2] - \
                          fluidPotential[j, indices])* (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[14] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpW2S2] - \
                          fluidPotential[j, indices]) * (-1.)
        #Southeastern 2
        if (neighboringNodes[tmpStart15] != -1 and neighboringNodes[tmpStart7] != -1):
            tmpE2S2 = neighboringNodes[tmpStart15]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[15] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpE2S2] - \
                          fluidPotential[j, indices]) * (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[15] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpE2S2] - \
                          fluidPotential[j, indices]) * (-1.)
        #North1Eastern2 
        if (neighboringNodes[tmpStart16] != -1 and (neighboringNodes[tmpStart] != -1 or \
            neighboringNodes[tmpStart4] != -1)):
            tmpN1E2 = neighboringNodes[tmpStart16]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[16] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN1E2] - \
                          fluidPotential[j, indices]) * (1.)
                    forceY[i, indices] += -6.0 * weightInter[16] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN1E2] - \
                          fluidPotential[j, indices]) * (1.)
        #North2Eastern1
        if (neighboringNodes[tmpStart17] != -1 and (neighboringNodes[tmpStart1] != -1 or \
            neighboringNodes[tmpStart4] != -1)):
            tmpN2E1 = neighboringNodes[tmpStart17]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[17] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN2E1] - \
                          fluidPotential[j, indices]) * (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[17] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN2E1] - \
                          fluidPotential[j, indices]) * (1.)
        #North2Western1
        if (neighboringNodes[tmpStart18] != -1 and (neighboringNodes[tmpStart1] != -1 or \
            neighboringNodes[tmpStart5] != -1)):
            tmpN2W1 = neighboringNodes[tmpStart18]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[18] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN2W1] - \
                          fluidPotential[j, indices]) * (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[18] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN2W1] - \
                          fluidPotential[j, indices]) * (1.)
        #North1Western2
        if (neighboringNodes[tmpStart19] != -1 and (neighboringNodes[tmpStart2] != -1 or \
            neighboringNodes[tmpStart5] != -1)):
            tmpN1W2 = neighboringNodes[tmpStart19]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[19] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN1W2] - \
                          fluidPotential[j, indices]) * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[19] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpN1W2] - \
                          fluidPotential[j, indices]) * (1.)
        #South1Western2
        if (neighboringNodes[tmpStart20] != -1 and (neighboringNodes[tmpStart2] != -1 or \
            neighboringNodes[tmpStart6] != -1)):
            tmpS1W2 = neighboringNodes[tmpStart20]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[20] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS1W2] - \
                          fluidPotential[j, indices]) * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[20] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS1W2] - \
                          fluidPotential[j, indices]) * (-1.)
        #South2Western1
        if (neighboringNodes[tmpStart21] != -1 and (neighboringNodes[tmpStart3] != -1 or \
            neighboringNodes[tmpStart6] != -1)):
            tmpS2W1 = neighboringNodes[tmpStart21]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[21] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS2W1] - \
                          fluidPotential[j, indices]) * (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[21] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS2W1] - \
                          fluidPotential[j, indices]) * (-1.)
        #South2Eastern1
        if (neighboringNodes[tmpStart22] != -1 and (neighboringNodes[tmpStart3] != -1 or \
            neighboringNodes[tmpStart7] != -1)):
            tmpS2E1 = neighboringNodes[tmpStart22]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[22] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS2E1] - \
                          fluidPotential[j, indices]) * (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[22] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS2E1] - \
                          fluidPotential[j, indices]) * (-1.)
        #South1Eastern2
        if (neighboringNodes[tmpStart23] != -1 and (neighboringNodes[tmpStart] != -1 or\
            neighboringNodes[tmpStart7] != -1)):
            tmpS1E2 = neighboringNodes[tmpStart23]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[23] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS1E2] - \
                          fluidPotential[j, indices]) * (1.)
                    forceY[i, indices] += -6.0 * weightInter[23] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * (fluidPotential[j, tmpS1E2] - \
                          fluidPotential[j, indices]) * (-1.)
    cuda.syncthreads()
                          
"""
The explicit scheme for isotropy = 10
"""
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:], float64[:, :], \
               float64[:], float64[:, :], float64[:, :], float64[:, :])')
def calExplicit10thOrderScheme(totalNodes, numFluids, xDim, fluidNodes, \
                              neighboringNodes, weightInter, interactionCoeff, \
                              interactionSolid, fluidPotential, forceX, forceY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if (indices < totalNodes):
        for i in range(numFluids):
            forceX[i, indices] = 0.0; forceY[i, indices] = 0.0
        tmpStart = indices * 36
        tmpStart0 = tmpStart
        tmpStart1 = tmpStart + 1
        tmpStart2 = tmpStart + 2
        tmpStart3 = tmpStart + 3
        tmpStart4 = tmpStart + 4
        tmpStart5 = tmpStart + 5
        tmpStart6 = tmpStart + 6
        tmpStart7 = tmpStart + 7
        tmpStart8 = tmpStart + 8
        tmpStart9 = tmpStart + 9
        tmpStart10 = tmpStart + 10
        tmpStart11 = tmpStart + 11
        tmpStart12 = tmpStart + 12
        tmpStart13 = tmpStart + 13
        tmpStart14 = tmpStart + 14
        tmpStart15 = tmpStart + 15
        tmpStart16 = tmpStart + 16
        tmpStart17 = tmpStart + 17
        tmpStart18 = tmpStart + 18
        tmpStart19 = tmpStart + 19
        tmpStart20 = tmpStart + 20
        tmpStart21 = tmpStart + 21
        tmpStart22 = tmpStart + 22
        tmpStart23 = tmpStart + 23
        tmpStart24 = tmpStart + 24
        tmpStart25 = tmpStart + 25
        tmpStart26 = tmpStart + 26
        tmpStart27 = tmpStart + 27
        tmpStart28 = tmpStart + 28
        tmpStart29 = tmpStart + 29
        tmpStart30 = tmpStart + 30
        tmpStart31 = tmpStart + 31
        tmpStart32 = tmpStart + 32
        tmpStart33 = tmpStart + 33
        tmpStart34 = tmpStart + 34
        tmpStart35 = tmpStart + 35
        
        if (neighboringNodes[tmpStart0] != -1):
            tmpE = neighboringNodes[tmpStart0]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[0] * interactionCoeff[i, j] * \
                              fluidPotential[i, indices] * fluidPotential[j, tmpE] * (1.)
        elif (neighboringNodes[tmpStart0] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Northern point
        if (neighboringNodes[tmpStart1] != -1):
            tmpN = neighboringNodes[tmpStart1]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -6.0 * weightInter[1] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN] * (1.)
        elif (neighboringNodes[tmpStart1] == -1):
            for i in range(numFluids):
                forceY[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Western Point
        if (neighboringNodes[tmpStart2] != -1):
            tmpW = neighboringNodes[tmpStart2]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[2] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW] * (-1.)
        elif(neighboringNodes[tmpStart2] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Southern point
        if (neighboringNodes[tmpStart3] != -1):
            tmpS = neighboringNodes[tmpStart3]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -6.0 * weightInter[3] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS] * (-1.)
        elif (neighboringNodes[tmpStart3] == -1):
            for i in range(numFluids):
                forceY[i, indices] += -1./9. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Northeastern point
        if (neighboringNodes[tmpStart4] != -1):
            tmpNE = neighboringNodes[tmpStart4]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[4] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
                    forceY[i, indices] += -6.0 * weightInter[4] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNE] * (1.)
        elif(neighboringNodes[tmpStart4] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Northwestern point
        if (neighboringNodes[tmpStart5] != -1):
            tmpNW = neighboringNodes[tmpStart5]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[5] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[5] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpNW] * (1.)
        elif (neighboringNodes[tmpStart5] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
        #Southwestern point
        if (neighboringNodes[tmpStart6] != -1):
            tmpSW = neighboringNodes[tmpStart6]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[6] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[6] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSW] * (-1.)
        elif (neighboringNodes[tmpStart6] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Southeastern point
        if (neighboringNodes[tmpStart7] != -1):
            tmpSE = neighboringNodes[tmpStart7]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[7] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (1.)
                    forceY[i, indices] += -6.0 * weightInter[7] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpSE] * (-1.)
        elif (neighboringNodes[tmpStart7] == -1):
            for i in range(numFluids):
                forceX[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (1.)
                forceY[i, indices] += -1./36. * interactionSolid[i] * \
                      fluidPotential[i, indices] * (-1.)
        #Easter point 2
        if (neighboringNodes[tmpStart8] != -1 and neighboringNodes[tmpStart] != -1):
            tmpE2 = neighboringNodes[tmpStart8]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[8] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpE2] * (1.)
        #Northern point 2
        if (neighboringNodes[tmpStart9] != -1 and neighboringNodes[tmpStart1] != -1):
            tmpN2 = neighboringNodes[tmpStart9]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -2. * 6.0 * weightInter[9] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN2] * (1.)
        #Western point 2
        if (neighboringNodes[tmpStart10] != -1 and neighboringNodes[tmpStart2] != -1):
            tmpW2 = neighboringNodes[tmpStart10]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[10] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW2] * (-1.)
        #Southern poitn 2
        if (neighboringNodes[tmpStart11] != -1 and neighboringNodes[tmpStart3] != -1):
            tmpS2 = neighboringNodes[tmpStart11]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -2. * 6.0 * weightInter[11] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS2] * (-1.)
        #Notheastern 2
        if (neighboringNodes[tmpStart12] != -1 and neighboringNodes[tmpStart4] != -1):
            tmpE2N2 = neighboringNodes[tmpStart12]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[12] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpE2N2] * (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[12] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpE2N2] * (1.)
        #Northwestern 2
        if (neighboringNodes[tmpStart13] != -1 and neighboringNodes[tmpStart5] != -1):
            tmpW2N2 = neighboringNodes[tmpStart13]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[13] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW2N2] * (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[13] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW2N2] * (1.)
        #Southwestern 2
        if (neighboringNodes[tmpStart14] != -1 and neighboringNodes[tmpStart6] != -1):
            tmpW2S2 = neighboringNodes[tmpStart14]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[14] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW2S2] * (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[14] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW2S2] * (-1.)
        #Southeastern 2
        if (neighboringNodes[tmpStart15] != -1 and neighboringNodes[tmpStart7] != -1):
            tmpE2S2 = neighboringNodes[tmpStart15]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[15] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpE2S2] * (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[15] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpE2S2] * (-1.)
        #North1Eastern2 
        if (neighboringNodes[tmpStart16] != -1 and (neighboringNodes[tmpStart] != -1 or \
            neighboringNodes[tmpStart4] != -1)):
            tmpN1E2 = neighboringNodes[tmpStart16]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[16] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1E2] * (1.)
                    forceY[i, indices] += -6.0 * weightInter[16] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1E2] * (1.)
        #North2Eastern1
        if (neighboringNodes[tmpStart17] != -1 and (neighboringNodes[tmpStart1] != -1 or \
            neighboringNodes[tmpStart4] != -1)):
            tmpN2E1 = neighboringNodes[tmpStart17]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[17] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN2E1] * (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[17] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN2E1] * (1.)
        #North2Western1
        if (neighboringNodes[tmpStart18] != -1 and (neighboringNodes[tmpStart1] != -1 or \
            neighboringNodes[tmpStart5] != -1)):
            tmpN2W1 = neighboringNodes[tmpStart18]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[18] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN2W1] * (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[18] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN2W1] * (1.)
        #North1Western2
        if (neighboringNodes[tmpStart19] != -1 and (neighboringNodes[tmpStart2] != -1 or \
            neighboringNodes[tmpStart5] != -1)):
            tmpN1W2 = neighboringNodes[tmpStart19]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[19] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1W2] * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[19] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1W2] * (1.)
        #South1Western2
        if (neighboringNodes[tmpStart20] != -1 and (neighboringNodes[tmpStart2] != -1 or \
            neighboringNodes[tmpStart6] != -1)):
            tmpS1W2 = neighboringNodes[tmpStart20]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[20] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1W2] * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[20] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1W2] * (-1.)
        #South2Western1
        if (neighboringNodes[tmpStart21] != -1 and (neighboringNodes[tmpStart3] != -1 or \
            neighboringNodes[tmpStart6] != -1)):
            tmpS2W1 = neighboringNodes[tmpStart21]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[21] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS2W1] * (-1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[21] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS2W1] * (-1.)
        #South2Eastern1
        if (neighboringNodes[tmpStart22] != -1 and (neighboringNodes[tmpStart3] != -1 or \
            neighboringNodes[tmpStart7] != -1)):
            tmpS2E1 = neighboringNodes[tmpStart22]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[22] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS2E1] * (1.)
                    forceY[i, indices] += -2. * 6.0 * weightInter[22] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS2E1] * (-1.)
        #South1Eastern2
        if (neighboringNodes[tmpStart23] != -1 and (neighboringNodes[tmpStart] != -1 or\
            neighboringNodes[tmpStart7] != -1)):
            tmpS1E2 = neighboringNodes[tmpStart23]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -2. * 6.0 * weightInter[23] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1E2] * (1.)
                    forceY[i, indices] += -6.0 * weightInter[23] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1E2] * (-1.)
                          
        #Eastern 3
        if ((neighboringNodes[tmpStart] != -1 and neighboringNodes[tmpStart8] != -1) and \
            neighboringNodes[tmpStart24] != -1):
            tmpE3 = neighboringNodes[tmpStart24]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -3. * 6.0 * weightInter[24] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpE3] * (1.)
        #Northern 3
        if ((neighboringNodes[tmpStart1] != -1 and neighboringNodes[tmpStart9] != -1) and \
            neighboringNodes[tmpStart25] != -1):
            tmpN3 = neighboringNodes[tmpStart25]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -3. * 6.0 * weightInter[25] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN3] * (1.)
        #Western 3
        if ((neighboringNodes[tmpStart2] != -1 and neighboringNodes[tmpStart10] != -1) and \
            neighboringNodes[tmpStart26] != -1):
            tmpW3 = neighboringNodes[tmpStart26]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -3. * 6.0 * weightInter[26] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpW3] * (-1.)
        #Southern 3
        if ((neighboringNodes[tmpStart3] != -1 and neighboringNodes[tmpStart11] != -1) and \
            neighboringNodes[tmpStart27] != -1):
            tmpS3 = neighboringNodes[tmpStart27]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceY[i, indices] += -3. * 6.0 * weightInter[27] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS3] * (-1.)
        #North1Eastern3
        if (neighboringNodes[tmpStart28] != -1 and ((neighboringNodes[tmpStart4] != -1 \
            and neighboringNodes[tmpStart16] != -1) or (neighboringNodes[tmpStart] != -1 \
            and neighboringNodes[tmpStart8] != -1))):
            tmpN1E3 = neighboringNodes[tmpStart28]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -3. * 6.0 * weightInter[28] * interactionCoeff[i, j]  * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1E3] * (1.)
                    forceY[i, indices] += -6.0 * weightInter[28] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1E3] * (1.)
        #North3Eastern1
        if (neighboringNodes[tmpStart29] != -1 and ((neighboringNodes[tmpStart1] != -1 \
            and neighboringNodes[tmpStart9] != -1) or (neighboringNodes[tmpStart4] != -1 \
            and neighboringNodes[tmpStart17] != -1))):
            tmpN3E1 = neighboringNodes[tmpStart29]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[29] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN3E1] * (1.)
                    forceY[i, indices] += -3. * 6.0 * weightInter[29] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN3E1] * (1.)
        #North3Western1
        if (neighboringNodes[tmpStart30] != -1 and ((neighboringNodes[tmpStart1] != -1 \
            and neighboringNodes[tmpStart9] != -1) or (neighboringNodes[tmpStart5] != -1 \
            and neighboringNodes[tmpStart18] != -1))):
            tmpN3W1 = neighboringNodes[tmpStart30]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[30] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN3W1] * (-1.)
                    forceY[i, indices] += -3. * 6.0 * weightInter[30] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN3W1] * (1.)
        #North1Western3
        if (neighboringNodes[tmpStart31] != -1 and ((neighboringNodes[tmpStart2] != -1 \
            and neighboringNodes[tmpStart10] != -1) or (neighboringNodes[tmpStart5] != -1 \
            and neighboringNodes[tmpStart19] != -1))):
            tmpN1W3 = neighboringNodes[tmpStart31]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -3. * 6.0 * weightInter[31] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1W3] * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[31] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpN1W3] * (1.)
        #South1Western3
        if (neighboringNodes[tmpStart32] != -1 and ((neighboringNodes[tmpStart2] != -1 \
            and neighboringNodes[tmpStart10] != -1) or (neighboringNodes[tmpStart6] != -1 \
            and neighboringNodes[tmpStart20] != -1))): 
            tmpS1W3 = neighboringNodes[tmpStart32]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -3. * 6.0 * weightInter[32] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1W3] * (-1.)
                    forceY[i, indices] += -6.0 * weightInter[32] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1W3] * (-1.)
        #South3Western1 
        if (neighboringNodes[tmpStart33] != -1 and ((neighboringNodes[tmpStart3] != -1 \
            and neighboringNodes[tmpStart11] != -1) or (neighboringNodes[tmpStart6] != -1 \
            and neighboringNodes[tmpStart21] != -1))):
            tmpS3W1 = neighboringNodes[tmpStart33]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[33] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS3W1] * (-1.)
                    forceY[i, indices] += -3. * 6.0 * weightInter[33] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS3W1] * (-1.)
        #South3Eastern1
        if (neighboringNodes[tmpStart34] != -1 and ((neighboringNodes[tmpStart3] != -1 \
            and neighboringNodes[tmpStart11] != -1) or (neighboringNodes[tmpStart7] != -1 \
            and neighboringNodes[tmpStart22] != -1))):
            tmpS3E1 = neighboringNodes[tmpStart34]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -6.0 * weightInter[34] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS3E1] * (1.)
                    forceY[i, indices] += -3. * 6.0 * weightInter[34] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS3E1] * (-1.)
        #South1Eastern3
        if (neighboringNodes[tmpStart35] != -1 and ((neighboringNodes[tmpStart] != -1 \
            and neighboringNodes[tmpStart8] != -1) or (neighboringNodes[tmpStart7] != -1 \
            and neighboringNodes[tmpStart23] != -1))):
            tmpS1E3 = neighboringNodes[tmpStart35]
            for i in range(numFluids):
                for j in range(numFluids):
                    forceX[i, indices] += -3. * 6.0 * weightInter[35] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1E3] * (1.)
                    forceY[i, indices] += -6.0 * weightInter[35] * interactionCoeff[i, j] * \
                          fluidPotential[i, indices] * fluidPotential[j, tmpS1E3] * (-1.)    
    cuda.syncthreads()
        
"""
Convert original distribution f and f_eq to m and m_eq by collision matrix
"""
@cuda.jit('void(int64, int64, int64, float64[:, :, :], float64[:, :, :], float64[:, :, :], \
        float64[:, :, :])')
def transformPDFandEquil(totalNodes, numFluids, xDim, fluidPDF, fEq, collisionMatrix,\
                         fluidPDFM):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    tmpFluidPDF = cuda.local.array(shape = 9, dtype = float64)
    tmpfEq = cuda.local.array(shape = 9, dtype = float64)
    if indices < totalNodes:
        for i in range(numFluids):
            for j in range(9):
                tmpfPDF = 0.; tmpEq = 0.
                for k in range(9):
                    tmpfPDF += collisionMatrix[i, j, k] * fluidPDF[i, indices, k]
                    tmpEq += collisionMatrix[i, j, k] * fEq[i, indices, k]
                tmpFluidPDF[j] = tmpfPDF
                tmpfEq[j] = tmpEq
            for j in range(9):
                fluidPDFM[i, indices, j] = tmpFluidPDF[j]
                fEq[i, indices, j] = tmpfEq[j]
                
"""
Convert force term to m_force by collision matrix
"""
@cuda.jit('void(int64, int64, int64, float64[:, :, :], float64[:, :, :], \
        float64[:, :,:])')
def transfromForceTerm(totalNodes, numFluids, xDim, fForce, collisionMatrix, \
                       fForceM):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    tmpForceTerm = cuda.local.array(shape = 9, dtype = float64)
    if indices < totalNodes:
        for i in range(numFluids):
            for j in range(9):
                tmpF = 0.
                for k in range(9):
                    tmpF += collisionMatrix[i, j, k] * fForce[i, indices, k]
                tmpForceTerm[j] = tmpF
            for j in range(9):
                fForceM[i, indices, j] = tmpForceTerm[j]

"""
Convert u_eq to MRT way for conserving momentum
"""
@cuda.jit('void(int64, int64, int64, float64[:], float64[:], \
                float64[:, :], float64[:, :], float64[:, :], float64[:, :, :], \
                float64[:], float64[:], float64[:])')
def transformEquilibriumVelocity(totalNodes, numFluids, xDim, EX, EY, fluidRho, \
                         forceX, forceY, fluidPDF, conserveS, eqVX, eqVY):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx 
    tmpEX = cuda.shared.array(shape = (9,), dtype = float64)
    tmpEY = cuda.shared.array(shape = (9,), dtype = float64)
    for i in range(9):
        tmpEX[i] = EX[i]
        tmpEY[i] = EY[i]
    if (indices < totalNodes):
        tmpMomentumX = 0.; tmpMomentumY = 0.; tmpRhoT = 0.
        for i in range(numFluids):
            eachMX = 0.; eachMY = 0.; tmpRho = fluidRho[i, indices]
            for j in range(9):
                eachMX += fluidPDF[i, indices, j] * tmpEX[j]
                eachMY += fluidPDF[i, indices, j] * tmpEY[j]
            eachMX += 1./2. * forceX[i, indices]
            eachMY += 1./2. * forceY[i, indices]
            tmpMomentumX += eachMX * conserveS[i]
            tmpMomentumY += eachMY * conserveS[i]
            tmpRhoT += tmpRho * conserveS[i]
        eqVX[indices] = tmpMomentumX / tmpRhoT
        eqVY[indices] = tmpMomentumY / tmpRhoT
    
    
"""
The last step of MRT transformation
"""
@cuda.jit('void(int64, int64, int64, float64[:, :, :], float64[:, :, :], \
            float64[:, :, :], float64[:, :, :], float64[:, :, :])')
def calAfterCollisionMRT(totalNodes, numFluids, xDim, fluidPDF, fForce, fEq, \
                         fluidPDFM, fForceM):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    if indices < totalNodes:
        for i in range(numFluids):
            for j in range(9):
                tmpCollision = 0.
                tmpCollision = (fEq[i, indices, j] - fluidPDFM[i, indices, j] - \
                                1./2. * fForceM[i, indices, j])
                fluidPDF[i, indices, j] = fluidPDF[i, indices, j] + tmpCollision + \
                        1. * fForce[i, indices, j]
      
"""
Calculate the outlet boundary with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :], float64[:, :, :], float64[:, :, :])')
def convectiveOutletGPUEFS(totalNodes, numFluids, nx, xDim, fluidNodes, neighboringNodes, \
                        fluidPDFNew, fluidRho, fForce, fEq):
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
                    fForce[i, indices, j] = fForce[i, tmpIndices, j]
                    fEq[i, indices, j] = fEq[i, tmpIndices, j]
                    fluidRho[i, indices] += fluidPDFNew[i, tmpIndices, j]
    cuda.syncthreads()
     
"""
Calculate the outlet boundary ghost nodes in second layer with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :], float64[:, :, :], float64[:, :, :])')
def convectiveOutletGhost2GPUEFS(totalNodes, numFluids, nx, xDim, \
                             fluidNodes, neighboringNodes, fluidPDFNew, fluidRho, \
                             fForce, fEq):
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
                    fForce[i, indices, j] = fForce[i, tmpIndices, j]
                    fEq[i, indices, j] = fEq[i, tmpIndices, j]
                    fluidRho[i, indices] += fluidPDFNew[i, tmpIndices, j]
    cuda.syncthreads()
     
"""
Calculate the outlet boundary ghost nodes in first layer with convective flow method. 
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], float64[:, :, :], \
                float64[:, :], float64[:, :, :], float64[:, :, :])')
def convectiveOutletGhost3GPUEFS(totalNodes, numFluids, nx, xDim, \
                             fluidNodes, neighboringNodes, fluidPDFNew, fluidRho, \
                             fForce, fEq):
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
                    fForce[i, indices, j] = fForce[i, tmpIndices, j]
                    fEq[i, indices, j] = fEq[i, tmpIndices, j]
                    fluidRho[i, indices] += fluidPDFNew[i, tmpIndices, j]
    cuda.syncthreads()       

      