"""
Use Accelerate of anaconda to parallelize computation of LBM in 2D. It uses 
CUDA from Nvidia 
"""

import sys, os, math

import numpy as np
import scipy as sp

from numba import cuda, int64, float64
#from accelerate import cuda as acuda
#from accelerate import numba as anumba

#cuda.select_device(0)
#Calculate the macro mass in each lattice


@cuda.jit('void(int64, int64, float64[:, :], float64[:, :])')
def savePDFofLastStep(nx, ny, fluidDistrOld, fluidDistrNew):
    """
    Save the values of PDF from the last step
    """
    bIdx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + bIdx
    id1D = by * nx + idX
    totalNum = nx * ny
    for i in range(9):
        fluidDistrOld[i, id1D] = fluidDistrNew[i, id1D]

@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :], \
        float64[:, :, :], float64[:], float64[:], boolean[:, :])')
def calmacroDensityAndVelocityGPU(nx, ny, fluidDensity, velocityY, velocityX, \
    fluidDistr, microVY, microVX, isDomain):
    bIdx = cuda.blockIdx.x; bIdy = cuda.blockIdx.y
    bDimx = cuda.blockDim.x; bDimy = cuda.blockDim.y
    tIdx = cuda.threadIdx.x; tIdy = cuda.threadIdx.y
    idX = bIdx * bDimx + tIdx; idY = bIdy * bDimy + tIdy
    if (idX < nx and idY < ny):
        fluidDensity[idY, idX] = 0.; velocityY[idY, idX] = 0.0; velocityX[idY, idX] = 0.0
        if (isDomain[idY, idX] == True):
            for i in range(0, 9):
                tmpValue = fluidDistr[i, idY, idX]
                fluidDensity[idY, idX] = fluidDensity[idY, idX] + tmpValue
            cuda.syncthreads
            for i in range(0, 9):
                velocityY[idY, idX] += (fluidDistr[i, idY, idX] * microVY[i])
                velocityX[idY, idX] += (fluidDistr[i, idY, idX] * microVX[i])
            velocityY[idY, idX] = velocityY[idY, idX] / fluidDensity[idY, idX]
            velocityX[idY, idX] = velocityX[idY, idX] / fluidDensity[idY, idX]
    
@cuda.jit('void(int64, int64, float64[:], float64[:, :], float64[:, :], boolean[:])')
def calMacroDensityGPU1D(nx, ny, fluidDensity, fluidDistrC, \
    fluidDistrN, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    
    if (id1D < totalNum):
#        if (idX < nx and by < ny):
        fluidDistrN[0, id1D] = fluidDistrC[0, id1D]
        fluidDistrN[1, id1D] = fluidDistrC[1, id1D]
        fluidDistrN[2, id1D] = fluidDistrC[2, id1D]
        fluidDistrN[3, id1D] = fluidDistrC[3, id1D]
        fluidDistrN[4, id1D] = fluidDistrC[4, id1D]
        fluidDistrN[5, id1D] = fluidDistrC[5, id1D]
        fluidDistrN[6, id1D] = fluidDistrC[6, id1D]
        fluidDistrN[7, id1D] = fluidDistrC[7, id1D]
        fluidDistrN[8, id1D] = fluidDistrC[8, id1D]
        if (isDomain[id1D] == True):
            fluidDensity[id1D] = fluidDistrC[0, id1D] + fluidDistrC[1, id1D] + \
        fluidDistrC[2, id1D] + fluidDistrC[3, id1D] + fluidDistrC[4, id1D] + \
        fluidDistrC[5, id1D] + fluidDistrC[6, id1D] + fluidDistrC[7, id1D] + \
        fluidDistrC[8, id1D]

@cuda.jit('void(int64, int64, float64[:], float64[:], float64[:],  float64[:, :], boolean[:])')
def calMacroVelocityGPU1D(nx, ny, fluidVelocityX, fluidVelocityY, fluidDensity, \
                         fluidDistr, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by *  nx + idX
    totalNum = nx * ny
    
    if (id1D < totalNum):
        fluidVelocityX[id1D] = (fluidDistr[1, id1D] - fluidDistr[3, id1D] + \
            fluidDistr[5, id1D] - fluidDistr[6, id1D] - fluidDistr[7, id1D] + \
            fluidDistr[8, id1D]) / fluidDensity[id1D]
        fluidVelocityY[id1D] = (fluidDistr[2, id1D] - fluidDistr[4, id1D] + \
            fluidDistr[5, id1D] + fluidDistr[6, id1D] - fluidDistr[7, id1D] - \
            fluidDistr[8, id1D])

"""
Calculate the pressure value by Huang and Sukop (2007)
"""
@cuda.jit('void(int64, int64, float64, float64[:], float64[:], float64[:], boolean[:])')
def calMacroPressureHuang1D(nx, ny, interactionStrength, densityFluid0, densityFluid1, \
                            macroPressure, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (idX < nx and id1D < totalNum):
        if (isDomain[id1D] == True):
            macroPressure[id1D] = (densityFluid0[id1D] + densityFluid1[id1D]) / 3. + \
                1./3. * (interactionStrength) * densityFluid0[id1D] * \
                densityFluid1[id1D]

"""
Calculate the pressure value through Shan and Doolen method (1995)
"""
@cuda.jit('void(int64, int64, float64, float64[:], float64[:], float64[:], boolean[:])')
def calMacroPressureShan1D(nx, ny, interactionStrength, densityFluid0, densityFluid1, \
    macroPressure, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (idX < nx and id1D < totalNum):
        if (isSolid[id1D] == True):
            macroPressure[id1D] = (densityFluid0[id1D] + densityFluid1[id1D]) / 3. + \
                (3./2.) * (1./3.) * interactionStrength * (densityFluid1[id1D] * \
                densityFluid0[id1D])

"""
Calculate the pressure value through Kang et.al method (2002) interaction strength here 
is 9 times of that in Shan's
"""
@cuda.jit('void(int64, int64, float64, float64[:], float64[:], float64[:], boolean[:])')
def calMacroPressureKang1D(nx, ny, interactionStrength, densityFluid0, densityFluid1, \
    macroPressure, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (idX < nx and id1D < totalNum):
        if (isSolid[id1D] == True):
            macroPressure[id1D] = (densityFluid0[id1D] + densityFluid1[id1D]) / 3. + \
                (3./2.) * (1./3.) * (interactionStrength * densityFluid1[id1D] * \
                densityFluid0[id1D])

"""
Constant density/pressure boundary condition with rho_i / (rho_i + rho_j)
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:, :], float64[:], \
        float64[:], boolean[:], boolean[:])')
def calConstantDensityBoundaryHGPU(nx, ny, specificDensityH, fluidDistr0, \
    fluidDistr1, fluidDensity0, fluidDensity1, isSolid, isBoundaryFluid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (isSolid[id1D] == False):
        if (isBoundaryFluid[id1D] == True and id1D > 2 * nx):
            tmpRho0 = (fluidDensity0[id1D] / (fluidDensity1[id1D] + fluidDensity0[id1D])) * \
                      specificDensityH
            tmpRho1 = (fluidDensity1[id1D] / (fluidDensity1[id1D] + fluidDensity0[id1D])) * \
                      specificDensityH
            #for fluid0 and fluid1
            velocityY0 = -1. + (fluidDistr0[0, id1D] + \
                    fluidDistr0[1, id1D] + fluidDistr0[3, id1D] + \
                    2. * (fluidDistr0[2, id1D] + fluidDistr0[5, id1D] + \
                    fluidDistr0[6, id1D])) / tmpRho0
            velocityY1 = -1. + (fluidDistr1[0, id1D] + \
                    fluidDistr1[1, id1D] + fluidDistr1[3, id1D] + \
                    2. * (fluidDistr1[2, id1D] + fluidDistr1[5, id1D] + \
                    fluidDistr1[6, id1D])) / tmpRho1
            fluidDistr0[4, id1D] = fluidDistr0[2, id1D] - 2. / 3. * tmpRho0 *\
                       velocityY0
            fluidDistr1[4, id1D] = fluidDistr1[2, id1D] - 2. / 3. * tmpRho1 * \
                       velocityY1
            fluidDistr0[7, id1D] = fluidDistr0[5, id1D] + 1./2. * \
                    (fluidDistr0[1, id1D] - fluidDistr0[3, id1D]) - \
                    1./6. * tmpRho0 * velocityY0
            fluidDistr1[7, id1D] = fluidDistr1[5, id1D] + 1./2. * \
                    (fluidDistr1[1, id1D] - fluidDistr1[3, id1D]) - \
                    1./6. * tmpRho1 * velocityY1
            fluidDistr0[8, id1D] = fluidDistr0[6, id1D] - 1./2. * \
                    (fluidDistr0[1, id1D] - fluidDistr0[3, id1D]) - \
                    1./6. * tmpRho0 * velocityY0
            fluidDistr1[8, id1D] = fluidDistr1[6, id1D] - 1./2. * \
                    (fluidDistr1[1, id1D] - fluidDistr1[3, id1D]) - \
                    1./6. * tmpRho1 * velocityY1
            fluidDensity0[id1D] = tmpRho0
            fluidDensity1[id1D] = tmpRho1
    
"""
Constant density/pressure boundary condition with rho_i / (rho_i + rho_j)
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:, :], float64[:], \
        float64[:], boolean[:], boolean[:])')
def calConstantDensityBoundaryLGPU(nx, ny, specificDensityL, fluidDistr0, \
    fluidDistr1, fluidDensity0, fluidDensity1, isSolid, isBoundaryFluid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (isSolid[id1D] == False):
        if (isBoundaryFluid[id1D] == True and id1D < 2 * nx):
            tmpRho0 = (fluidDensity0[id1D] / (fluidDensity1[id1D] + fluidDensity0[id1D])) * \
                      specificDensityL
            tmpRho1 = (fluidDensity1[id1D] / (fluidDensity1[id1D] + fluidDensity0[id1D])) * \
                      specificDensityL
            velocityY0 = 1. - (fluidDistr0[0, id1D] + fluidDistr0[1, id1D] + \
                        fluidDistr0[3, id1D] + 2.* (fluidDistr0[4, id1D] + \
                        fluidDistr0[7, id1D] + fluidDistr0[8, id1D])) / \
                        tmpRho0
            velocityY1 = 1. - (fluidDistr1[0, id1D] + fluidDistr1[1, id1D] + \
                        fluidDistr1[3, id1D] + 2.* (fluidDistr1[4, id1D] + \
                        fluidDistr1[7, id1D] + fluidDistr1[8, id1D])) / \
                        tmpRho1
            fluidDistr0[2, id1D] = fluidDistr0[4, id1D] + 2./3. * velocityY0 * \
                       tmpRho0
            fluidDistr1[2, id1D] = fluidDistr1[4, id1D] + 2./3. * velocityY1 * \
                       tmpRho1
            fluidDistr0[5, id1D] = fluidDistr0[7, id1D ] + \
                    1./2. * (fluidDistr0[3, id1D] - fluidDistr0[1, id1D]) + \
                    1./6. * tmpRho0 * velocityY0
            fluidDistr1[5, id1D] = fluidDistr1[7, id1D ] + \
                    1./2. * (fluidDistr1[3, id1D] - fluidDistr1[1, id1D]) + \
                    1./6. * tmpRho1 * velocityY1
            fluidDistr0[6, id1D] = fluidDistr0[8, id1D] - 1./2. * \
                    (fluidDistr0[3, id1D] - fluidDistr0[1, id1D]) + \
                    1./6. * tmpRho0 * velocityY0
            fluidDistr1[6, id1D] = fluidDistr1[8, id1D] - 1./2. * \
                    (fluidDistr1[3, id1D] - fluidDistr1[1, id1D]) + \
                    1./6. * tmpRho1 * velocityY1
            fluidDensity0[id1D] = tmpRho0
            fluidDensity1[id1D] = tmpRho1
    

@cuda.jit('void(int64, int64, float64, float64, float64[:, :], float64[:, :], \
        float64[:], float64[:], boolean[:], boolean[:], boolean[:])')
def calConstantPressureBoundaryGPU(nx, ny, specificHigher, specificLow, \
    fluidDistr0, fluidDistr1, fluidDensity0, fluidDensity1, isBoundaryF0, isBoundaryF1, \
    isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    velocityY0 = 0.; velocityY1 = 0.
    if (idX < nx and id1D < totalNum):
        if (isBoundaryF0[id1D] == True and isSolid[id1D] == False):
            velocityY0 = 1. - (fluidDistr0[0, id1D] + fluidDistr0[1, id1D] + \
                        fluidDistr0[3, id1D] + 2.* (fluidDistr0[4, id1D] + \
                        fluidDistr0[7, id1D] + fluidDistr0[8, id1D])) / specificLow
            #three unknown distribution
            fluidDistr0[2, id1D] = fluidDistr0[4, id1D] + \
                    2./3. * velocityY0 * specificLow
            fluidDistr0[5, id1D] = fluidDistr0[7, id1D ] + \
                    1./2. * (fluidDistr0[3, id1D] - fluidDistr0[1, id1D]) + \
                    1./6. * specificLow * velocityY0
            fluidDistr0[6, id1D] = fluidDistr0[8, id1D] - 1./2. * \
                    (fluidDistr0[3, id1D] - fluidDistr0[1, id1D]) + \
                    1./6. * specificLow * velocityY0
            fluidDensity0[id1D] = specificLow
    
    if (id1D < totalNum and id1D >= totalNum - nx):
        if (isBoundaryF1[id1D] == True and isSolid[id1D] == False):
            velocityY1 = -1. + (fluidDistr1[0, id1D] + \
                    fluidDistr1[1, id1D] + fluidDistr1[3, id1D] + \
                    2. * (fluidDistr1[2, id1D] + fluidDistr1[5, id1D] + \
                    fluidDistr1[6, id1D])) / specificHigher
            #three unknown distribution functions
            fluidDistr1[4, id1D] = fluidDistr1[2, id1D] - \
                    2. / 3. * specificHigher * velocityY1
            fluidDistr1[7, id1D] = fluidDistr1[5, id1D] + 1./2. * \
                    (fluidDistr1[1, id1D] - fluidDistr1[3, id1D]) - \
                    1./6. * specificHigher * velocityY1
            fluidDistr1[8, id1D] = fluidDistr1[6, id1D] - 1./2. * \
                    (fluidDistr1[1, id1D] - fluidDistr1[3, id1D]) - \
                    1./6. * specificHigher * velocityY1
            fluidDensity1[id1D] = specificHigher


@cuda.jit('void(int64, int64, float64, float64, float64[:, :], float64[:, :], \
        float64[:], float64[:], boolean[:], boolean[:], boolean[:])')
def calConstantPressureBoundaryRevGPU(nx, ny, specificHigher, specificLower, \
    fluidDistr0, fluidDistr1, fluidDensity0, fluidDensity1, isBoundaryF0, isBoundaryF1, \
    isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (idX < nx and id1D < totalNum):
        if (isBoundaryF1[id1D] == True and isSolid[id1D] == False):
            velocityY1 = 1. - (fluidDistr1[0, id1D] + fluidDistr1[1, id1D] + \
                        fluidDistr1[3, id1D] + 2.* (fluidDistr1[4, id1D] + \
                        fluidDistr1[7, id1D] + fluidDistr1[8, id1D])) / specificLower
            #three unknown distribution
            fluidDistr1[2, id1D] = fluidDistr1[4, id1D] + \
                    2./3. * velocityY1 * specificLower
            fluidDistr1[5, id1D] = fluidDistr1[7, id1D ] + \
                    1./2. * (fluidDistr1[3, id1D] - fluidDistr1[1, id1D]) + \
                    1./6. * specificLower * velocityY1
            fluidDistr1[6, id1D] = fluidDistr1[8, id1D] - 1./2. * \
                    (fluidDistr1[3, id1D] - fluidDistr1[1, id1D]) + \
                    1./6. * specificLower * velocityY1
            fluidDensity1[id1D] = specificLower
    if (id1D >= totalNum - nx and id1D < totalNum):
        if (isBoundaryF0[id1D] == True and isSolid[id1D] == False):
            velocityY0 = -1. + (fluidDistr0[0, id1D] + \
                    fluidDistr0[1, id1D] + fluidDistr0[3, id1D] + \
                    2. * (fluidDistr0[2, id1D] + fluidDistr0[5, id1D] + \
                    fluidDistr0[6, id1D])) / specificHigher
            #three unknown distribution functions
            fluidDistr0[4, id1D] = fluidDistr0[2, id1D] - \
                    2. / 3. * specificHigher * velocityY0
            fluidDistr0[7, id1D] = fluidDistr0[5, id1D] + 1./2. * \
                    (fluidDistr0[1, id1D] - fluidDistr0[3, id1D]) - \
                    1./6. * specificHigher * velocityY0
            fluidDistr0[8, id1D] = fluidDistr0[6, id1D] - 1./2. * \
                    (fluidDistr0[1, id1D] - fluidDistr0[3, id1D]) - \
                    1./6. * specificHigher * velocityY0
            fluidDensity0[id1D] = specificHigher
"""
Constant pressure boundary condition on the lower part in Zou-He method
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:], boolean[:], \
        boolean[:])')
def calConstantPressureLowerFGPU(nx, ny, specificRhoLower, fluidDistr, fluidDensity, \
                                 isBoundaryF, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D < nx and id1D < totalNum):
        if (isBoundaryF[id1D] == True and isSolid[id1D] == False):
            velocityY = 1. - (fluidDistr[0, id1D] + fluidDistr[1, id1D] + \
                        fluidDistr[3, id1D] + 2.* (fluidDistr[4, id1D] + \
                        fluidDistr[7, id1D] + fluidDistr[8, id1D])) / specificRhoLower
            #three unknown distribution
            fluidDistr[2, id1D] = fluidDistr[4, id1D] + \
                    2./3. * velocityY * specificRhoLower
            fluidDistr[5, id1D] = fluidDistr[7, id1D ] + \
                    1./2. * (fluidDistr[3, id1D] - fluidDistr[1, id1D]) + \
                    1./6. * specificRhoLower * velocityY
            fluidDistr[6, id1D] = fluidDistr[8, id1D] - 1./2. * \
                    (fluidDistr[3, id1D] - fluidDistr[1, id1D]) + \
                    1./6. * specificRhoLower * velocityY
            fluidDensity[id1D] = specificRhoLower
    cuda.syncthreads()
            
"""
Constant pressure boundary condition on the higher part in Zou-He method
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:], boolean[:], \
        boolean[:])')
def calConstantPressureHigherFGPU(nx, ny, specificRhoHigher, fluidDistr, fluidDensity, \
                                 isBoundaryF, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D >= totalNum - nx and id1D < totalNum):
        if (isBoundaryF[id1D] == True and isSolid[id1D] == False):
            velocityY = -1. + (fluidDistr[0, id1D] + \
                    fluidDistr[1, id1D] + fluidDistr[3, id1D] + \
                    2. * (fluidDistr[2, id1D] + fluidDistr[5, id1D] + \
                    fluidDistr[6, id1D])) / specificRhoHigher
            #three unknown distribution functions
            fluidDistr[4, id1D] = fluidDistr[2, id1D] - \
                    2. / 3. * specificRhoHigher * velocityY
            fluidDistr[7, id1D] = fluidDistr[5, id1D] + 1./2. * \
                    (fluidDistr[1, id1D] - fluidDistr[3, id1D]) - \
                    1./6. * specificRhoHigher * velocityY
            fluidDistr[8, id1D] = fluidDistr[6, id1D] - 1./2. * \
                    (fluidDistr[1, id1D] - fluidDistr[3, id1D]) - \
                    1./6. * specificRhoHigher * velocityY
            fluidDensity[id1D] = specificRhoHigher
    cuda.syncthreads()

"""
Constant flux/velocity boundary condition in Zou-He method (Fluid0, lower boundary)
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], \
        float64[:], boolean[:], boolean[:])')
def calConstantVelocityBoundaryFLGPU(nx, ny, specificVelocityLF, fluidDistr, \
    fluidDensity, isBoundaryF, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D < nx and id1D < totalNum):
        if (isBoundaryF[id1D] == True and isSolid[id1D] == False):
            fluidDensity[id1D] = (fluidDistr[0, id1D] + fluidDistr[1, id1D] + \
                    fluidDistr[3, id1D] + 2. * (fluidDistr[4, id1D] + \
                    fluidDistr[7, id1D] + fluidDistr[8, id1D])) / (1. - specificVelocityLF)
            fluidDistr[2, id1D] = fluidDistr[4, id1D] + 2./3. * fluidDensity[id1D] * \
                    specificVelocityLF
            fluidDistr[5, id1D] = fluidDistr[7, id1D] - (fluidDistr[1, id1D] - \
                    fluidDistr[3, id1D]) / 2. + 1./2. * fluidDensity[id1D] * 0. + \
                    1./6. * fluidDensity[id1D] * specificVelocityLF
            fluidDistr[6, id1D] = fluidDistr[8, id1D] + (fluidDistr[1, id1D] - \
                    fluidDistr[3, id1D]) / 2. - 1./2. * fluidDensity[id1D] * 0. + \
                    1./6. * fluidDensity[id1D] * specificVelocityLF
                    
"""
Constant flux/velocity boundary condition in Zou-He method (Fluid1, lower boundary)
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:], boolean[:], \
        boolean[:])')
def calConstantVelocityBoundaryF1LGPU(nx, ny, specificVelocityLF1, fluidDistr1, \
    fluidDensity1, isBoundaryF1, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (idX < nx and id1D < totalNum):
        if (isBoundaryF1[id1D] == True and isSolid[id1D] == False):
            fluidDensity1[id1D] = (fluidDistr1[0, id1D] + fluidDistr1[1, id1D] + \
                    fluidDistr1[3, id1D] + 2. * (fluidDistr1[4, id1D] + \
                    fluidDistr1[7, id1D] + fluidDistr1[8, id1D])) / (1. - specificVelocityLF1)
            fluidDistr1[2, id1D] = fluidDistr1[4, id1D] + 2./3. * fluidDensity1[id1D] * \
                    specificVelocityLF1
            fluidDistr1[5, id1D] = fluidDistr1[7, id1D] - (fluidDistr1[1, id1D] - \
                    fluidDistr1[3, id1D]) / 2. + 1./2. * fluidDensity1[id1D] * 0. + \
                    1./6. * fluidDensity1[id1D] * specificVelocityLF1
            fluidDistr1[6, id1D] = fluidDistr1[8, id1D] + (fluidDistr1[1, id1D] - \
                    fluidDistr1[3, id1D]) / 2. - 1./2. * fluidDensity1[id1D] * 0. + \
                    1./6. * fluidDensity1[id1D] * specificVelocityLF1
                    
"""
Constant flux/velocity boundary condition in Zou-He method (Fluid0, upper-boundary)
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:], boolean[:], \
        boolean[:])')
def calConstantVelocityBoundaryFHGPU(nx, ny, specificVelocityHF, fluidDistr, \
    fluidDensity, isBoundaryF, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D >= totalNum - nx and id1D < totalNum):
        if (isBoundaryF[id1D] == True and isSolid[id1D] == False):
            fluidDensity[id1D] = (fluidDistr[0, id1D] + fluidDistr[1, id1D] + \
                    fluidDistr[3, id1D] + 2. * (fluidDistr[2, id1D] + \
                    fluidDistr[5, id1D] + fluidDistr[6, id1D])) / (1. + specificVelocityHF)
            fluidDistr[4, id1D] = fluidDistr[2, id1D] - 2./3. * fluidDensity[id1D] * \
                    specificVelocityHF
            fluidDistr[7, id1D] = fluidDistr[5, id1D] + (fluidDistr[1, id1D] - \
                    fluidDistr[3, id1D]) / 2. - 1./6. * fluidDensity[id1D] * \
                    specificVelocityHF
            fluidDistr[8, id1D] = fluidDistr[6, id1D] - (fluidDistr[1, id1D] - \
                    fluidDistr[3, id1D]) / 2. - 1./6. * fluidDensity[id1D] * \
                    specificVelocityHF
                    
"""
Constant flux/velocity boundary condition in Zou-He method (Fluid1, lower-boundary)
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:], boolean[:], \
        boolean[:])')
def calConstantVelocityBoundaryF1HGPU(nx, ny, specificVelocityHF1, fluidDistr1, \
    fluidDensity1, isBoundaryF1, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D >= totalNum - nx and id1D < totalNum):
        if (isBoundaryF1[id1D] == True and isSolid[id1D] == False):
            fluidDensity1[id1D] = (fluidDistr1[0, id1D] + fluidDistr1[1, id1D] + \
                    fluidDistr1[3, id1D] + 2. * (fluidDistr1[2, id1D] + \
                    fluidDistr1[5, id1D] + fluidDistr1[6, id1D])) / (1. + specificVelocityHF1)
            fluidDistr1[4, id1D] = fluidDistr1[2, id1D] - 2./3. * fluidDensity1[id1D] * \
                    specificVelocityHF1
            fluidDistr1[7, id1D] = fluidDistr1[5, id1D] + (fluidDistr1[1, id1D] - \
                    fluidDistr1[3, id1D]) / 2. - 1./6. * fluidDensity1[id1D] * \
                    specificVelocityHF1
            fluidDistr1[8, id1D] = fluidDistr1[6, id1D] - (fluidDistr1[1, id1D] - \
                    fluidDistr1[3, id1D]) / 2. - 1./6. * fluidDensity1[id1D] * \
                    specificVelocityHF1
"""
Implement the 'corrector' boundary method with specific velocity through the method
in 
"""                 
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:, :], float64[:], \
                float64[:], float64[:], boolean[:], boolean[:])')
def calCorrectorBoundaryVelocityHigherGPU(nx, ny, specificVY, fluidDistrNew, \
    fluidDistrOld, fluidDensity, forceX, forceY, isBoundaryF, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    
    if (id1D < totalNum and id1D >= totalNum - nx):
        if (isBoundaryF[id1D] == True and isSolid[id1D] == False):
            fluidDensity[id1D] = (fluidDistrNew[0, id1D] + fluidDistrNew[1, id1D] + \
                fluidDistrNew[3, id1D] + 2. * (fluidDistrNew[2, id1D] + \
                fluidDistrNew[5, id1D] + fluidDistrNew[6, id1D]) + \
                1./2. * forceY[id1D]) / (1. + specificVY)
            fluidDistrNew[4, id1D] = fluidDistrOld[4, id1D] - 2./3. * (fluidDensity[id1D] * \
                specificVY + fluidDistrOld[4, id1D] + fluidDistrOld[7, id1D] + \
                fluidDistrOld[8, id1D]) + 2./3. * (fluidDistrNew[2, id1D] + \
                fluidDistrNew[5, id1D] + fluidDistrNew[6, id1D] + 1./2. * forceY[id1D])
            fluidDistrNew[7, id1D] = fluidDistrOld[7, id1D] + 1./2. * \
                (fluidDistrNew[1, id1D] - fluidDistrNew[3, id1D]) + 1./6. * \
                (fluidDistrNew[2, id1D] - fluidDistrOld[4, id1D]) + 2./3. * \
                (fluidDistrNew[5, id1D] - fluidDistrOld[7, id1D]) - 1./3. * \
                (fluidDistrNew[6, id1D] - fluidDistrOld[8, id1D]) + 1./4. * \
                forceX[id1D] - 1./12. * forceY[id1D] - 1./6. * fluidDensity[id1D] * \
                specificVY
            fluidDistrNew[8, id1D] = fluidDistrOld[8, id1D] - 1./6. * \
                fluidDensity[id1D] * specificVY - 1./2. * (fluidDistrNew[1, id1D] - \
                fluidDistrNew[3, id1D]) + 1./6. * (fluidDistrNew[2, id1D] - \
                fluidDistrOld[4, id1D]) - 1./3. * (fluidDistrNew[5, id1D] - \
                fluidDistrOld[7, id1D]) + 2./3. * (fluidDistrNew[6, id1D] - \
                fluidDistrOld[8, id1D]) - 1./4. * forceX[id1D] + 1./12. * forceY[id1D]
                
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:, :], float64[:], \
                float64[:], float64[:], boolean[:], boolean[:])')
def calCorrectorBoundaryVelocityLowerGPU(nx, ny, specificVY, fluidDistrNew, \
    fluidDistrOld, fluidDensity, forceX, forceY, isBoundaryF, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D < nx and id1D < totalNum):
        if (isBoundaryF[id1D] == True and isSolid[id1D]):
            fluidDensity[id1D] = (fluidDistrNew[0, id1D] + fluidDistrNew[1, id1D] + \
                fluidDistrNew[3, id1D] + 2. * (fluidDistrNew[4, id1D] + \
                fluidDistrNew[7, id1D] + fluidDistrNew[8, id1D]) - 1./2. * forceY[id1D]) / \
                (1. - specificVY)
            fluidDistrNew[2, id1D] = fluidDistrOld[2, id1D] + 2./3. * (fluidDensity[id1D] * \
                specificVY + fluidDistrNew[4, id1D] + fluidDistrNew[7, id1D] + \
                fluidDistrNew[8, id1D] - fluidDistrOld[2, id1D] - \
                fluidDistrOld[5, id1D] - fluidDistrOld[6, id1D] - 1./2. * forceY[id1D])
            fluidDistrNew[5, id1D] = 1./6. * fluidDensity[id1D] * specificVY + \
                1./2. * (fluidDistrNew[3, id1D] - fluidDistrNew[1, id1D]) + 2./3. * \
                fluidDistrNew[7, id1D] - 1./3. * fluidDistrNew[8, id1D] + 1./3. * \
                fluidDistrOld[5, id1D] - 1./6. * fluidDistrOld[2, id1D]  + 1./3. * \
                fluidDistrOld[6, id1D] + 1./6. * fluidDistrNew[4, id1D] - 1./4. * \
                forceX[id1D] - 1./12. * forceY[id1D]
            fluidDistrNew[6, id1D] = 1./3. * fluidDistrOld[6, id1D] + 1./6. * \
                fluidDensity[id1D] * specificVY + 1./6. * fluidDistrNew[4, id1D] - \
                1./3. * fluidDistrNew[7, id1D] + 2./3. * fluidDistrNew[8, id1D] - \
                1.6 * fluidDistrOld[2, id1D] + 1./3. * fluidDistrOld[5, id1D] - \
                1./2. * (fluidDistrNew[3, id1D] - fluidDistrNew[1, id1D]) - 1./12. * \
                forceY[id1D] + 1./4. * forceX[id1D]
    

@cuda.jit('void(int64, int64, float64[:, :], boolean[:])')
def calOutletGhostPoints1(nx, ny, fluidDistr, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D >= nx and id1D < 2 * nx):
        if (isSolid[id1D] == False):
            fluidDistr[0, id1D] = fluidDistr[0, id1D + nx]
            fluidDistr[1, id1D] = fluidDistr[1, id1D + nx]
            fluidDistr[2, id1D] = fluidDistr[2, id1D + nx]
            fluidDistr[3, id1D] = fluidDistr[3, id1D + nx]
            fluidDistr[4, id1D] = fluidDistr[4, id1D + nx]
            fluidDistr[5, id1D] = fluidDistr[5, id1D + nx]
            fluidDistr[6, id1D] = fluidDistr[6, id1D + nx]
            fluidDistr[7, id1D] = fluidDistr[7, id1D + nx]
            fluidDistr[8, id1D] = fluidDistr[8, id1D + nx]
    cuda.syncthreads()

@cuda.jit('void(int64, int64, float64[:, :], boolean[:])')
def calOutletGhostPoints0(nx, ny, fluidDistr, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D >= 0 and id1D < nx):
        if (isSolid[id1D] == False):
            fluidDistr[0, id1D] = fluidDistr[0, id1D + nx]
            fluidDistr[1, id1D] = fluidDistr[1, id1D + nx]
            fluidDistr[2, id1D] = fluidDistr[2, id1D + nx]
            fluidDistr[3, id1D] = fluidDistr[3, id1D + nx]
            fluidDistr[4, id1D] = fluidDistr[4, id1D + nx]
            fluidDistr[5, id1D] = fluidDistr[5, id1D + nx]
            fluidDistr[6, id1D] = fluidDistr[6, id1D + nx]
            fluidDistr[7, id1D] = fluidDistr[7, id1D + nx]
            fluidDistr[8, id1D] = fluidDistr[8, id1D + nx]
    cuda.syncthreads()
    
"""
Imlement the convective boundary condition
"""
@cuda.jit('void(int64, int64, float64, float64[:, :], float64[:, :], boolean[:], \
        boolean[:])')
def calOutletBoundayConvective(nx, ny, averageV, fluidDistrOld, fluidDistrNew, \
                               isBoundaryF,  isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    id1X = bx * bDimX + tx
    id1D = by * nx + id1X
    totalNum = nx * ny
    if (isBoundaryF[id1D] == True and id1D < 3 * nx):
        for i in range(9):
            fluidDistrNew[i, id1D] = (fluidDistrOld[i, id1D] + averageV * \
                         fluidDistrNew[i, id1D + nx]) / (1. + averageV)
    cuda.syncthreads()
    if (id1D >= nx  and id1D < nx * 2):
        for i in range(9):
            fluidDistrNew[i, id1D] = (fluidDistrOld[i, id1D] + averageV * \
                         fluidDistrNew[i, id1D + nx]) / (1 + averageV)
    cuda.syncthreads()
    if (id1D >= 0 and id1D < nx):
        for i in range(9):
            fluidDistrNew[i, id1D] = (fluidDistrOld[i, id1D] + averageV * \
                         fluidDistrNew[i, id1D + nx]) / (1 + averageV)
    cuda.syncthreads()
    
@cuda.jit('void(int64, int64, float64[:, :],  boolean[:], boolean[:])')
def calOutletBoundaryLower(nx, ny, fluidDistr, isSolid, isBoundaryF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D >= nx and id1D < 2 * nx):
#    if (id1D >= 2 * nx and id1D < 3 * nx):
        if (isBoundaryF[id1D] == True and isSolid[id1D] == False):
            fluidDistr[0, id1D] = fluidDistr[0, id1D + nx]
            fluidDistr[1, id1D] = fluidDistr[1, id1D + nx]
            fluidDistr[2, id1D] = fluidDistr[2, id1D + nx]
            fluidDistr[3, id1D] = fluidDistr[3, id1D + nx]
            fluidDistr[4, id1D] = fluidDistr[4, id1D + nx]
            fluidDistr[5, id1D] = fluidDistr[5, id1D + nx]
            fluidDistr[6, id1D] = fluidDistr[6, id1D + nx]
            fluidDistr[7, id1D] = fluidDistr[7, id1D + nx]
            fluidDistr[8, id1D] = fluidDistr[8, id1D + nx]
    cuda.syncthreads()
            
@cuda.jit('void(int64, int64, float64[:, :], boolean[:], boolean[:])')
def calOutletBoundaryHigher(nx, ny, fluidDistr, isSolid, isBoundaryF):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D < totalNum and id1D >= totalNum - nx):
#    if (id1D < totalNum and id1D >= totalNum - nx):
        if (isBoundaryF[id1D] == True and isSolid[id1D] == False):
            fluidDistr[0, id1D] = fluidDistr[0, id1D - nx]
            fluidDistr[1, id1D] = fluidDistr[1, id1D - nx]
            fluidDistr[2, id1D] = fluidDistr[2, id1D - nx]
            fluidDistr[3, id1D] = fluidDistr[3, id1D - nx]
            fluidDistr[4, id1D] = fluidDistr[4, id1D - nx]
            fluidDistr[5, id1D] = fluidDistr[5, id1D - nx]
            fluidDistr[6, id1D] = fluidDistr[6, id1D - nx]
            fluidDistr[7, id1D] = fluidDistr[7, id1D - nx]
            fluidDistr[8, id1D] = fluidDistr[8, id1D - nx]
    cuda.syncthreads()

"""
update the value in the ghost points with constant pressure/density boundary
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:], boolean[:])')
def updateGhostPoints(nx, ny, fluidDistr, fluidDensity, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (id1D < nx and isSolid[id1D] == False):
        fluidDistr[0, id1D] = fluidDistr[0, id1D + nx]
        fluidDistr[1, id1D] = fluidDistr[1, id1D + nx]
        fluidDistr[2, id1D] = fluidDistr[2, id1D + nx]
        fluidDistr[3, id1D] = fluidDistr[3, id1D + nx]
        fluidDistr[4, id1D] = fluidDistr[4, id1D + nx]
        fluidDistr[5, id1D] = fluidDistr[5, id1D + nx]
        fluidDistr[6, id1D] = fluidDistr[6, id1D + nx]
        fluidDistr[7, id1D] = fluidDistr[7, id1D + nx]
        fluidDistr[8, id1D] = fluidDistr[8, id1D + nx]
        fluidDensity[id1D] = fluidDistr[0, id1D] + fluidDistr[1, id1D] + \
                    fluidDistr[2, id1D] + fluidDistr[3, id1D] + fluidDistr[4, id1D] + \
                    fluidDistr[5, id1D] + fluidDistr[6, id1D] + fluidDistr[7, id1D] + \
                    fluidDistr[8, id1D]
    if (id1D < totalNum and id1D >= totalNum -nx):
        if (isSolid[id1D] == False):
            fluidDistr[0, id1D] = fluidDistr[0, id1D - nx]
            fluidDistr[1, id1D] = fluidDistr[1, id1D - nx]
            fluidDistr[2, id1D] = fluidDistr[2, id1D - nx]
            fluidDistr[3, id1D] = fluidDistr[3, id1D - nx]
            fluidDistr[4, id1D] = fluidDistr[4, id1D - nx]
            fluidDistr[5, id1D] = fluidDistr[5, id1D - nx]
            fluidDistr[6, id1D] = fluidDistr[6, id1D - nx]
            fluidDistr[7, id1D] = fluidDistr[7, id1D - nx]
            fluidDistr[8, id1D] = fluidDistr[8, id1D - nx]
            fluidDensity[id1D] = fluidDistr[0, id1D] + fluidDistr[1, id1D] + \
                    fluidDistr[2, id1D] + fluidDistr[3, id1D] + fluidDistr[4, id1D] + \
                    fluidDistr[5, id1D] + fluidDistr[6, id1D] + fluidDistr[7, id1D] + \
                    fluidDistr[8, id1D]
    cuda.syncthreads
                
@cuda.jit('void(int64, int64, float64, float64, float64, float64, float64, \
            float64[:], \
            float64[:], \
            float64[:, :], float64[:, :], float64[:], float64[:], \
            float64[:], float64[:], \
            boolean[:], boolean[:], boolean[:], int64)')
def calEquilibriumVelocity1DGPU(nx, ny, tau0, tau1, interactionSolid0, interactionSolid1, \
    interactionFluid, fluidDensity0, fluidDensity1, fluidDistr0, fluidDistr1, \
    equilibriumVX0, equilibriumVY0, equilibriumVX1, equilibriumVY1, isDomain, \
    isSolid, isFluidBoundary, boundaryType):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
#    if (id1D < totalNum and (idX < nx and by < ny)):
    if (isDomain[id1D ]== True):
        tmpVx0 = 0.; tmpVx1 = 0.; tmpVy0 = 0.; tmpVy1 = 0.
        velocityPrimeX = 0.; velocityPrimeY = 0.
        tmpVx0 = fluidDistr0[0, id1D] * 0. + fluidDistr0[1, id1D] - \
                fluidDistr0[2, id1D] * 0. - fluidDistr0[3, id1D] + 0. * \
                fluidDistr0[4, id1D] + fluidDistr0[5, id1D] - fluidDistr0\
                [6, id1D] - fluidDistr0[7, id1D] + fluidDistr0[8, id1D]
        tmpVx1 = fluidDistr1[0, id1D] * 0. + fluidDistr1[1, id1D] - \
                fluidDistr1[2, id1D] * 0. - fluidDistr1[3, id1D] + 0. * \
                fluidDistr1[4, id1D] + fluidDistr1[5, id1D] - fluidDistr1\
                [6, id1D] - fluidDistr1[7, id1D] + fluidDistr1[8, id1D]
        velocityPrimeX = (tmpVx0 / tau0 + tmpVx1 / tau1) / (fluidDensity0\
                [id1D] / tau0 + fluidDensity1[id1D] / tau1)

        tmpVy0 = fluidDistr0[0, id1D] * 0. + 0. * fluidDistr0[1, id1D] + \
                fluidDistr0[2, id1D] + 0. * fluidDistr0[3, id1D] - \
                fluidDistr0[4, id1D] + fluidDistr0[5, id1D] + fluidDistr0\
                [6, id1D] - fluidDistr0[7, id1D] - fluidDistr0[8, id1D]
        tmpVy1 = fluidDistr1[0, id1D] * 0. + 0. * fluidDistr1[1, id1D] + \
                fluidDistr1[2, id1D] + 0. * fluidDistr1[3, id1D] - \
                fluidDistr1[4, id1D] + fluidDistr1[5, id1D] + fluidDistr1\
                [6, id1D] - fluidDistr1[7, id1D] - fluidDistr1[8, id1D]
        velocityPrimeY = (tmpVy0 / tau0 + tmpVy1 / tau1) / (fluidDensity0\
                [id1D] / tau0 + fluidDensity1[id1D] / tau1)
        #force with solid phase
        #locate neighboring nodes
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx) 
        tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
        tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
        tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
        tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
        #fore with fluid interaction
        FxFluid0 = 0.; FyFluid0 = 0.; FxFluid1 = 0.; FyFluid1 = 0.
        if (boundaryType != 1):
            if (isDomain[tmpIdF] == True and isFluidBoundary[tmpIdF] != True):
                FxFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdF]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdF]
                FxFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdF]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdF]
            if (isDomain[tmpIdU] == True and isFluidBoundary[tmpIdU] != True):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdU]
                FyFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdU]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdU]
                FyFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdU]
            if (isDomain[tmpIdB] == True and isFluidBoundary[tmpIdB] != True):
                FxFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdB]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdB]
                FxFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdB]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdB]
            if (isDomain[tmpIdL] == True and isFluidBoundary[tmpIdL] != True):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdL]
                FyFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdL]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdL]
                FyFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdL]
            if (isDomain[tmpIdFU] == True and isFluidBoundary[tmpIdFU] != True):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
            if (isDomain[tmpIdBU] == True and isFluidBoundary[tmpIdBU] != True):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdBU]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdBU]
            if (isDomain[tmpIdBL] == True and isFluidBoundary[tmpIdBL] != True):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
            if (isDomain[tmpIdFL] == True and isFluidBoundary[tmpIdFL] != True):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdFL]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdFL]
                
#            if (isDomain[tmpIdF] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdF]
#                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdF]
#                FxFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdF]
#                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdF]
#            if (isDomain[tmpIdU] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdU]
#                FyFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdU]
#                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdU]
#                FyFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdU]
#            if (isDomain[tmpIdB] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdB]
#                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdB]
#                FxFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdB]
#                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdB]
#            if (isDomain[tmpIdL] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdL]
#                FyFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdL]
#                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdL]
#                FyFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdL]
#            if (isDomain[tmpIdFU] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
#                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
#                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
#                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
#            if (isDomain[tmpIdBU] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBU]
#                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdBU]
#                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBU]
#                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdBU]
#            if (isDomain[tmpIdBL] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
#                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
#                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
#                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
#            if (isDomain[tmpIdFL] == True and isFluidBoundary[id1D] != True):
#                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFL]
#                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdFL]
#                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFL]
#                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdFL]
#            cuda.syncthreads()
        if (boundaryType == 1):
            if (isDomain[tmpIdF] == True):
                FxFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdF]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdF]
                FxFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdF]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdF]
            if (isDomain[tmpIdU] == True):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdU]
                FyFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdU]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdU]
                FyFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdU]
            if (isDomain[tmpIdB] == True):
                FxFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdB]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdB]
                FxFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdB]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdB]
            if (isDomain[tmpIdL] == True):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdL]
                FyFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdL]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdL]
                FyFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdL]
            if (isDomain[tmpIdFU] == True):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
            if (isDomain[tmpIdBU] == True):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdBU]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdBU]
            if (isDomain[tmpIdBL] == True):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
            if (isDomain[tmpIdFL] == True ):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdFL]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdFL]
#            cuda.syncthreads()
        potential0 = fluidDensity0[id1D]; potential1 = fluidDensity1[id1D]
        forceFx0 = 0.; forceFy0 = 0.; forceFx1 = 0.; forceFy1 = 0.
        forceFx1 = -interactionFluid * potential1 * FxFluid0
        forceFx0 = -interactionFluid * potential0 * FxFluid1
        forceFy1 = -interactionFluid * potential1 * FyFluid0
        forceFy0 = -interactionFluid * potential0 * FyFluid1
        tmpSolidX = 0.; tmpSolidY = 0.
        FxSolid0 = 0.; FySolid0 = 0.; FxSolid1 = 0.; FySolid1 = 0.
        if (isSolid[tmpIdF] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9.; tmpSolidY += 1./9. * 0.

        if (isSolid[tmpIdU] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9. * 0.; tmpSolidY += 1./9.
        if (isSolid[tmpIdB] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9. * (-1.); tmpSolidY += 1./9. * 0.
        if (isSolid[tmpIdL] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9. * 0.; tmpSolidY += 1./9. * (-1.)
        if (isSolid[tmpIdFU] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * 1.; tmpSolidY += 1./36. * 1.
        if (isSolid[tmpIdBU] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * (-1.); tmpSolidY += 1./36. * (1.)
        if (isSolid[tmpIdBL] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * (-1.); tmpSolidY += 1./36. * (-1.)
        if (isSolid[tmpIdFL] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * (1.); tmpSolidY += 1./36. * (-1.)

        #interaction between fluids
#        cuda.syncthreads
        FxSolid0 = -interactionSolid0 * fluidDensity0[id1D] * tmpSolidX
        FySolid0 = -interactionSolid0 * fluidDensity0[id1D] * tmpSolidY
        FxSolid1 = -interactionSolid1 * fluidDensity1[id1D] * tmpSolidX
        FySolid1 = -interactionSolid1 * fluidDensity1[id1D] * tmpSolidY

            
        totalForceX0 = forceFx0 + FxSolid0; totalForceY0 = forceFy0 + FySolid0
        totalForceX1 = forceFx1 + FxSolid1; totalForceY1 = forceFy1 + FySolid1
        equilibriumVX0[id1D] = 0.0; equilibriumVY0[id1D] = 0.0
        equilibriumVX1[id1D] = 0.0; equilibriumVY1[id1D] = 0.0
        equilibriumVX0[id1D] = velocityPrimeX + tau0 * totalForceX0 / \
                fluidDensity0[id1D]

        equilibriumVY0[id1D] = velocityPrimeY + tau0 * totalForceY0 / \
                fluidDensity0[id1D]
        equilibriumVX1[id1D] = velocityPrimeX + tau1 * totalForceX1 / \
                fluidDensity1[id1D]
        equilibriumVY1[id1D] = velocityPrimeY + tau1 * totalForceY1 / \
                fluidDensity1[id1D]

"""
Calculate the whole fluid velocity in origin Shan-Chen model
"""
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], \
        float64[:, :], float64[:, :], float64[:], float64[:], boolean[:])')
def calWholeVelocity(nx, ny, tau0, tau1, fluidDensity0, fluidDensity1, \
                     fluidDistr0, fluidDistr1, wholeVelocityX, wholeVelocityY, \
                     isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        wholeVelocityX[id1D] = ((fluidDistr0[1, id1D] - fluidDistr0[3, id1D] + \
                    fluidDistr0[5, id1D] - fluidDistr0[6, id1D] - \
                    fluidDistr0[7, id1D] + fluidDistr0[8, id1D]) / tau0 + \
                    (fluidDistr1[1, id1D] - fluidDistr1[3, id1D] + fluidDistr1[5, id1D] - \
                     fluidDistr1[6, id1D] - fluidDistr1[7, id1D] + fluidDistr1[8, id1D]) / \
                     tau1) / (fluidDensity0[id1D] / tau0 + fluidDensity1[id1D] /tau1)
        wholeVelocityY[id1D] = ((fluidDistr0[2, id1D] - fluidDistr0[4, id1D] + \
                    fluidDistr0[5, id1D] + fluidDistr0[6, id1D] - fluidDistr0[7, id1D]  -\
                    fluidDistr0[8, id1D]) / tau0 + (fluidDistr1[2, id1D] - \
                    fluidDistr1[4, id1D] + fluidDistr1[5, id1D] + fluidDistr1[6, id1D] - \
                    fluidDistr1[7, id1D] - fluidDistr1[8, id1D]) / tau1) / (\
                    fluidDensity0[id1D] / tau0 + fluidDensity1[id1D] / tau1)
                
@cuda.jit('void(int64, int64, float64, float64, float64, float64, float64, \
            float64, float64, float64, float64[:], float64[:], float64[:], \
            float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:], \
            float64[:], float64[:], float64[:], float64[:], boolean[:], \
            boolean[:], boolean[:], int64)')
def calEquilibriumVelocityCycles1DGPU(nx, ny, tau0, tau1, tau2, interactionSolid0, \
    interactionSolid1, interactionSolid2, interactionFluid, \
    interactionFluid1, fluidDensity0, fluidDensity1, fluidDensity2, fluidDistr0, fluidDistr1, \
    fluidDistr2, equilibriumVX0, equilibriumVY0, equilibriumVX1, equilibriumVY1, \
    equilibriumVX2, equilibriumVY2, isDomain, \
    isSolid, isFluidBoundary, boundaryType):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
#    if (id1D < totalNum and (idX < nx and by < ny)):
    if (isDomain[id1D ]== True):
        tmpVx0 = 0.; tmpVx1 = 0.; tmpVy0 = 0.; tmpVy1 = 0.; tmpVx2 = 0.; tempVY2 = 0.
        velocityPrimeX = 0.; velocityPrimeY = 0.
        tmpVx0 = fluidDistr0[0, id1D] * 0. + fluidDistr0[1, id1D] - \
                fluidDistr0[2, id1D] * 0. - fluidDistr0[3, id1D] + 0. * \
                fluidDistr0[4, id1D] + fluidDistr0[5, id1D] - fluidDistr0\
                [6, id1D] - fluidDistr0[7, id1D] + fluidDistr0[8, id1D]
        tmpVx1 = fluidDistr1[0, id1D] * 0. + fluidDistr1[1, id1D] - \
                fluidDistr1[2, id1D] * 0. - fluidDistr1[3, id1D] + 0. * \
                fluidDistr1[4, id1D] + fluidDistr1[5, id1D] - fluidDistr1\
                [6, id1D] - fluidDistr1[7, id1D] + fluidDistr1[8, id1D]
        tmpVx2 = fluidDistr2[0, id1D] * 0. + fluidDistr2[1, id1D] - \
                fluidDistr2[2, id1D] * 0. - fluidDistr2[3, id1D] + 0. * \
                fluidDistr2[4, id1D] + fluidDistr2[5, id1D] - fluidDistr2\
                [6, id1D] - fluidDistr2[7, id1D] + fluidDistr2[8, id1D]
        velocityPrimeX = (tmpVx0 / tau0 + tmpVx1 / tau1 + tmpVx2 / tau2) / (fluidDensity0\
                [id1D] / tau0 + fluidDensity1[id1D] / tau1 + fluidDensity2[id1D] / tau2)

        tmpVy0 = fluidDistr0[0, id1D] * 0. + 0. * fluidDistr0[1, id1D] + \
                fluidDistr0[2, id1D] + 0. * fluidDistr0[3, id1D] - \
                fluidDistr0[4, id1D] + fluidDistr0[5, id1D] + fluidDistr0\
                [6, id1D] - fluidDistr0[7, id1D] - fluidDistr0[8, id1D]
        tmpVy1 = fluidDistr1[0, id1D] * 0. + 0. * fluidDistr1[1, id1D] + \
                fluidDistr1[2, id1D] + 0. * fluidDistr1[3, id1D] - \
                fluidDistr1[4, id1D] + fluidDistr1[5, id1D] + fluidDistr1\
                [6, id1D] - fluidDistr1[7, id1D] - fluidDistr1[8, id1D]
        tmpVy2 = fluidDistr2[0, id1D] * 0. + 0. * fluidDistr2[1, id1D] + \
                fluidDistr2[2, id1D] + 0. * fluidDistr2[3, id1D] - \
                fluidDistr2[4, id1D] + fluidDistr2[5, id1D] + fluidDistr2\
                [6, id1D] - fluidDistr2[7, id1D] - fluidDistr2[8, id1D]
        velocityPrimeY = (tmpVy0 / tau0 + tmpVy1 / tau1 + tmpVy2 / tau2) / (fluidDensity0\
                [id1D] / tau0 + fluidDensity1[id1D] / tau1 + fluidDensity2[id1D] / tau2)
        #force with solid phase
        #locate neighboring nodes
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx) 
        tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
        tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
        tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
        tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
        #fore with fluid interaction
        FxFluid0 = 0.; FyFluid0 = 0.; FxFluid1 = 0.; FyFluid1 = 0.
        FxFluid2 = 0.; FyFluid2 = 0.
        if (boundaryType != 1):
            if (isDomain[tmpIdF] == True and isFluidBoundary[tmpIdF] != True):
                FxFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdF]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdF]
                FxFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdF]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdF]
                FxFluid2 += 1./9. * (1.0) * fluidDensity2[tmpIdF]
                FyFluid2 += 0.
            if (isDomain[tmpIdU] == True and isFluidBoundary[tmpIdU] != True):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdU]
                FyFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdU]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdU]
                FyFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdU]
                FxFluid2 += 0.
                FyFluid2 += 1./9. * (1.0) * fluidDensity2[tmpIdU]
            if (isDomain[tmpIdB] == True and isFluidBoundary[tmpIdB] != True):
                FxFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdB]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdB]
                FxFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdB]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdB]
                FxFluid2 += 1./9. * (-1.0) * fluidDensity2[tmpIdB]
                FyFluid2 += 0.
            if (isDomain[tmpIdL] == True and isFluidBoundary[tmpIdL] != True):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdL]
                FyFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdL]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdL]
                FyFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdL]
                FxFluid2 += 0.
                FyFluid2 += 1./9. * (-1.0) * fluidDensity2[tmpIdL]
            if (isDomain[tmpIdFU] == True and isFluidBoundary[tmpIdFU] != True):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
                FxFluid2 += 1./36. * (1.0) * fluidDensity2[tmpIdFU]
                FyFluid2 += 1./36. * (1.0) * fluidDensity2[tmpIdFU]
            if (isDomain[tmpIdBU] == True and isFluidBoundary[tmpIdBU] != True):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdBU]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdBU]
                FxFluid2 += 1./36. * (-1.0) * fluidDensity2[tmpIdBU]
                FyFluid2 += 1./36. * (1.0) * fluidDensity2[tmpIdBU]
            if (isDomain[tmpIdBL] == True and isFluidBoundary[tmpIdBL] != True):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
                FxFluid2 += 1./36. * (-1.0) * fluidDensity2[tmpIdBL]
                FyFluid2 += 1./36. * (-1.0) * fluidDensity2[tmpIdBL]
            if (isDomain[tmpIdFL] == True and isFluidBoundary[tmpIdFL] != True):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdFL]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdFL]
                FxFluid2 += 1./36. * (1.0) * fluidDensity2[tmpIdFL]
                FyFluid2 += 1./36. * (-1.0) * fluidDensity2[tmpIdFL]
#            cuda.syncthreads()
        if (boundaryType == 1):
            if (isDomain[tmpIdF] == True and (tmpRow < ny and tmpColF < nx)):
                FxFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdF]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdF]
                FxFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdF]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdF]
            if (isDomain[tmpIdU] == True and (tmpRowU < ny and tmpCol < nx)):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdU]
                FyFluid0 += 1./9. * (1.0) * fluidDensity0[tmpIdU]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdU]
                FyFluid1 += 1./9. * (1.0) * fluidDensity1[tmpIdU]
            if (isDomain[tmpIdB] == True and (tmpRow < ny and tmpColB < nx)):
                FxFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdB]
                FyFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdB]
                FxFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdB]
                FyFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdB]
            if (isDomain[tmpIdL] == True and (tmpRowL < ny and tmpCol < nx)):
                FxFluid0 += 1./9. * (0.0) * fluidDensity0[tmpIdL]
                FyFluid0 += 1./9. * (-1.0) * fluidDensity0[tmpIdL]
                FxFluid1 += 1./9. * (0.0) * fluidDensity1[tmpIdL]
                FyFluid1 += 1./9. * (-1.0) * fluidDensity1[tmpIdL]
            if (isDomain[tmpIdFU] == True and (tmpRowU < ny and tmpColF < nx)):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFU]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFU]
            if (isDomain[tmpIdBU] == True and (tmpRowU < ny and tmpColB < nx)):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBU]
                FyFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdBU]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBU]
                FyFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdBU]
            if (isDomain[tmpIdBL] == True and (tmpRowL < ny and tmpColB < nx)):
                FxFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdBL]
                FxFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdBL]
            if (isDomain[tmpIdFL] == True and (tmpRowL < ny and tmpColF < nx)):
                FxFluid0 += 1./36. * (1.0) * fluidDensity0[tmpIdFL]
                FyFluid0 += 1./36. * (-1.0) * fluidDensity0[tmpIdFL]
                FxFluid1 += 1./36. * (1.0) * fluidDensity1[tmpIdFL]
                FyFluid1 += 1./36. * (-1.0) * fluidDensity1[tmpIdFL]
#            cuda.syncthreads()
        potential0 = fluidDensity0[id1D]; potential1 = fluidDensity1[id1D]
        potential2 = fluidDensity2[id1D]
        forceFx0 = 0.; forceFy0 = 0.; forceFx1 = 0.; forceFy1 = 0.
        forceFx02 = 0.; forceFy02 = 0.;  
        forceFx1 = -interactionFluid * potential1 * FxFluid0
        forceFx0 = -interactionFluid * potential0 * FxFluid1
        
        forceFx02 = -interactionFluid1 * potential0 * FxFluid2
        forceFx20 = -interactionFluid1 * potential2 * FxFluid0
        forceFx12 = -interactionFluid * potential1 * FxFluid2
        forceFx21 = -interactionFluid * potential2 * FxFluid1
        
        forceFy1 = -interactionFluid * potential1 * FyFluid0
        forceFy0 = -interactionFluid * potential0 * FyFluid1
        
        forceFy02 = -interactionFluid1 * potential0 * FyFluid2
        forceFy20 = -interactionFluid1 * potential2 * FyFluid0
        forceFy12 = -interactionFluid * potential1 * FyFluid2
        forceFy21 = -interactionFluid * potential2 * FyFluid1
        
        tmpSolidX = 0.; tmpSolidY = 0.
        FxSolid0 = 0.; FySolid0 = 0.; FxSolid1 = 0.; FySolid1 = 0.
        if (isSolid[tmpIdF] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9.; tmpSolidY += 1./9. * 0.
        if (isSolid[tmpIdU] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9. * 0.; tmpSolidY += 1./9.
        if (isSolid[tmpIdB] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9. * (-1.); tmpSolidY += 1./9. * 0.
        if (isSolid[tmpIdL] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./9. * 0.; tmpSolidY += 1./9. * (-1.)
        if (isSolid[tmpIdFU] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * 1.; tmpSolidY += 1./36. * 1.
        if (isSolid[tmpIdBU] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * (-1.); tmpSolidY += 1./36. * (1.)
        if (isSolid[tmpIdBL] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * (-1.); tmpSolidY += 1./36. * (-1.)
        if (isSolid[tmpIdFL] == True and isFluidBoundary[id1D] != True):
            tmpSolidX += 1./36. * (1.); tmpSolidY += 1./36. * (-1.)

        #interaction between fluids
#        cuda.syncthreads
        FxSolid0 = -interactionSolid0 * fluidDensity0[id1D] * tmpSolidX
        FySolid0 = -interactionSolid0 * fluidDensity0[id1D] * tmpSolidY
        FxSolid1 = -interactionSolid1 * fluidDensity1[id1D] * tmpSolidX
        FySolid1 = -interactionSolid1 * fluidDensity1[id1D] * tmpSolidY
        FxSolid2 = -interactionSolid2 * fluidDensity2[id1D] * tmpSolidX
        FySolid2 = -interactionSolid2 * fluidDensity2[id1D] * tmpSolidY
            
        totalForceX0 = forceFx0 + FxSolid0 + forceFx02
        totalForceY0 = forceFy0 + FySolid0 + forceFy02
        totalForceX1 = forceFx1 + FxSolid1 + forceFx12
        totalForceY1 = forceFy1 + FySolid1 + forceFy12
        totalForceX2 = forceFx20 + FxSolid2 + forceFx21
        totalForceY2 = forceFy20 + FySolid2 + forceFy21
        
        equilibriumVX0[id1D] = 0.0; equilibriumVY0[id1D] = 0.0
        equilibriumVX1[id1D] = 0.0; equilibriumVY1[id1D] = 0.0
        equilibriumVX2[id1D] = 0.0; equilibriumVY2[id1D] = 0.0
        
        equilibriumVX0[id1D] = velocityPrimeX + tau0 * totalForceX0 / \
                fluidDensity0[id1D]
        equilibriumVY0[id1D] = velocityPrimeY + tau0 * totalForceY0 / \
                fluidDensity0[id1D]
        equilibriumVX1[id1D] = velocityPrimeX + tau1 * totalForceX1 / \
                fluidDensity1[id1D]
        equilibriumVY1[id1D] = velocityPrimeY + tau1 * totalForceY1 / \
                fluidDensity1[id1D]
                
        equilibriumVX2[id1D] = velocityPrimeX + tau2 * totalForceX2 / \
                fluidDensity2[id1D]
        equilibriumVY2[id1D] = velocityPrimeY + tau2 * totalForceY2 / \
                fluidDensity2[id1D]

@cuda.jit('void(int64, int64, float64, float64[:], float64[:], float64[:, :], \
        boolean[:], boolean[:])')
def calCollisionGPU(nx, ny, tau, equilibriumVX, equilibriumVY, \
    fluidDistrOld, isDomain, isSolid):
    totalNum = nx * ny
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; by = cuda.blockIdx.y
    bDimX = cuda.blockDim.x
    yStart = by; xStart = tx + bx * bDimX
    indicesK = nx * yStart + xStart
    #start  to calculate collision in the domain
    if (isDomain[indicesK] == True and indicesK < totalNum):
        fluidDensity = fluidDistrOld[0, indicesK] + fluidDistrOld[1, indicesK] + \
            fluidDistrOld[2, indicesK] + fluidDistrOld[3, indicesK] + \
            fluidDistrOld[4, indicesK] + fluidDistrOld[5, indicesK] + \
            fluidDistrOld[6, indicesK] + fluidDistrOld[7, indicesK] + \
            fluidDistrOld[8, indicesK]
        squareV = 1.5 * (equilibriumVX[indicesK] * equilibriumVX[indicesK] + \
            equilibriumVY[indicesK] * equilibriumVY[indicesK])
        fEq0 = 4./9. * fluidDensity * (1. - squareV)
        fluidDensity = fluidDensity * 1./9.
        fEq1 = fluidDensity * (1. + 3. * equilibriumVX[indicesK] + 4.5 * \
            equilibriumVX[indicesK] * equilibriumVX[indicesK] - squareV)
        fEq3 = fEq1 - 6. * equilibriumVX[indicesK] * fluidDensity
        fEq2 = fluidDensity * (1. + 3. * equilibriumVY[indicesK] + 4.5 * \
            equilibriumVY[indicesK] * equilibriumVY[indicesK] - squareV)
        fEq4 = fEq2 - 6. * equilibriumVY[indicesK] * fluidDensity
        fluidDensity = fluidDensity * 1./4.
        fEq5 = fluidDensity * (1. + 3. * (equilibriumVX[indicesK] + equilibriumVY[indicesK]) + \
            4.5 * (equilibriumVX[indicesK] + equilibriumVY[indicesK]) * \
            (equilibriumVX[indicesK] + equilibriumVY[indicesK]) - squareV)
        fEq6 = fluidDensity * (1. + 3. * (-equilibriumVX[indicesK] + equilibriumVY[indicesK]) + \
            4.5 * (-equilibriumVX[indicesK] + equilibriumVY[indicesK]) * \
            (-equilibriumVX[indicesK] + equilibriumVY[indicesK]) - squareV)
        fEq7 = fluidDensity * (1. + 3. * (-equilibriumVX[indicesK] - equilibriumVY[indicesK]) + \
            4.5 * (-equilibriumVX[indicesK] - equilibriumVY[indicesK]) * \
            (-equilibriumVX[indicesK] - equilibriumVY[indicesK]) - squareV)
        fEq8 = fluidDensity * (1. + 3. * (equilibriumVX[indicesK] - equilibriumVY[indicesK]) + \
            4.5 * (equilibriumVX[indicesK] - equilibriumVY[indicesK]) * \
            (equilibriumVX[indicesK] - equilibriumVY[indicesK]) - squareV)
        fluidDistrOld[0, indicesK] = fluidDistrOld[0, indicesK] + (fEq0 - \
            fluidDistrOld[0, indicesK]) / tau
        fluidDistrOld[1, indicesK] = fluidDistrOld[1, indicesK] + (fEq1 - \
            fluidDistrOld[1, indicesK]) / tau
        fluidDistrOld[2, indicesK] = fluidDistrOld[2, indicesK] + (fEq2 - \
            fluidDistrOld[2, indicesK]) / tau
        fluidDistrOld[3, indicesK] = fluidDistrOld[3, indicesK] + (fEq3 - \
            fluidDistrOld[3, indicesK]) / tau
        fluidDistrOld[4, indicesK] = fluidDistrOld[4, indicesK] + (fEq4 - \
            fluidDistrOld[4, indicesK]) / tau
        fluidDistrOld[5, indicesK] = fluidDistrOld[5, indicesK] + (fEq5 - \
            fluidDistrOld[5, indicesK]) / tau
        fluidDistrOld[6, indicesK] = fluidDistrOld[6, indicesK] + (fEq6 - \
            fluidDistrOld[6, indicesK]) / tau
        fluidDistrOld[7, indicesK] = fluidDistrOld[7, indicesK] + (fEq7 - \
            fluidDistrOld[7, indicesK]) / tau
        fluidDistrOld[8, indicesK] = fluidDistrOld[8, indicesK] + (fEq8 - \
            fluidDistrOld[8, indicesK]) / tau
            
@cuda.jit('void(int64, int64, int64, float64[:, :], float64[:, :])')
def calStreamingGPU(nx, ny, numberThreads, fluidDistrOld, fluidDistrNew):
    totalNum = nx * ny
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; by = cuda.blockIdx.y 
    xStart = tx + bx * numberThreads
    yStart = by
    indicesK = nx * yStart + xStart
    #define shared memory
    distrFOut0 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut1 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut2 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut3 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut4 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut5 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut6 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut7 = cuda.shared.array(shape = (32,), dtype=float64)
    distrFOut8 = cuda.shared.array(shape = (32,), dtype=float64)

    if (xStart < nx and yStart < ny):
        distrFOut0[tx] = fluidDistrOld[0, indicesK]
        distrFOut2[tx] = fluidDistrOld[2, indicesK]
        distrFOut4[tx] = fluidDistrOld[4, indicesK]    
        if (tx == 0):
            distrFOut1[tx + 1] = fluidDistrOld[1, indicesK]
            distrFOut3[numberThreads - 1] = fluidDistrOld[3, indicesK]
            distrFOut5[tx + 1] = fluidDistrOld[5, indicesK]
            distrFOut6[numberThreads - 1] = fluidDistrOld[6, indicesK]
            distrFOut7[numberThreads - 1] = fluidDistrOld[7, indicesK]
            distrFOut8[tx + 1] = fluidDistrOld[8, indicesK]
            
        if (tx == numberThreads - 1):
            distrFOut1[0] = fluidDistrOld[1, indicesK]
            distrFOut3[tx - 1] = fluidDistrOld[3, indicesK]
            distrFOut5[0] = fluidDistrOld[5, indicesK]
            distrFOut6[tx - 1] = fluidDistrOld[6, indicesK]
            distrFOut7[tx - 1] = fluidDistrOld[7, indicesK]
            distrFOut8[0] = fluidDistrOld[8, indicesK]
        
        if (tx > 0 and tx < numberThreads - 1):
            distrFOut1[tx + 1] = fluidDistrOld[1, indicesK]
            distrFOut3[tx - 1] = fluidDistrOld[3, indicesK]
            distrFOut5[tx + 1] = fluidDistrOld[5, indicesK]
            distrFOut6[tx - 1] = fluidDistrOld[6, indicesK]
            distrFOut7[tx - 1] = fluidDistrOld[7, indicesK]
            distrFOut8[tx + 1] = fluidDistrOld[8, indicesK]
    cuda.syncthreads()
    
    fluidDistrNew[0, indicesK] = distrFOut0[tx]
    fluidDistrNew[1, indicesK] = distrFOut1[tx]
    fluidDistrNew[3, indicesK] = distrFOut3[tx]
        
    if (by < ny - 1):
        indicesK = nx * (yStart + 1) + xStart
        fluidDistrNew[2, indicesK] = distrFOut2[tx]
        fluidDistrNew[5, indicesK] = distrFOut5[tx]
        fluidDistrNew[6, indicesK] = distrFOut6[tx]
    
#    if (by == ny - 1):
#        indicesK = nx * 0 + xStart
#        fluidDistrNew[2, indicesK] = distrFOut2[tx]
#        fluidDistrNew[5, indicesK] = distrFOut5[tx]
#        fluidDistrNew[6, indicesK] = distrFOut6[tx]
        
    if (by > 0):
        indicesK = nx * (yStart - 1) + xStart
        fluidDistrNew[4, indicesK] = distrFOut4[tx]
        fluidDistrNew[7, indicesK] = distrFOut7[tx]
        fluidDistrNew[8, indicesK] = distrFOut8[tx]
    
#    if (by == 0):
#        indicesK = nx * (ny - 1) + xStart
#        fluidDistrNew[4, indicesK] = distrFOut4[tx]
#        fluidDistrNew[7, indicesK] = distrFOut7[tx]
#        fluidDistrNew[8, indicesK] = distrFOut8[tx]

@cuda.jit('void(int64, int64, float64[:, :], float64[:, :])')
def calStreamingStep1(nx, ny, fluidDistrOld, fluidDistrMiddle):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
#    if (isDomain[id1D] == True):
    tmpRow = int(id1D / nx); tmpCol = (id1D % nx)
    tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
    tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
    tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
    tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
    tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
    tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
    tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
    tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
    fluidDistrMiddle[0, id1D] = fluidDistrOld[0, id1D]
    if (tmpRow < ny and tmpColF < nx):
        fluidDistrMiddle[1, tmpIdF] = fluidDistrOld[1, id1D]
    if (tmpRow < ny and tmpColB < nx):
        fluidDistrMiddle[3, tmpIdB] = fluidDistrOld[3, id1D]
    if (tmpRowU < ny and tmpCol < nx):
        fluidDistrMiddle[2, tmpIdU] = fluidDistrOld[2, id1D]
    if (tmpRowL < ny and tmpCol < nx):
        fluidDistrMiddle[4, tmpIdL] = fluidDistrOld[4, id1D]
    if (tmpRowU < ny and tmpColF < nx):
        fluidDistrMiddle[5, tmpIdFU] = fluidDistrOld[5, id1D]
    if (tmpRowU < ny and tmpColB < nx):
        fluidDistrMiddle[6, tmpIdBU] = fluidDistrOld[6, id1D]
    if (tmpRowL < ny and tmpColB < nx):
        fluidDistrMiddle[7, tmpIdBL] = fluidDistrOld[7, id1D]
    if (tmpRowL < ny and tmpColF < nx):
        fluidDistrMiddle[8, tmpIdFL] = fluidDistrOld[8, id1D]

@cuda.jit('void(int64, int64, float64[:, :], float64[:, :])')
def calStreamingStep2(nx, ny, fluidDistrNew, fluidDistrMiddle):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
#    if (isDomain[id1D] == True):
    for i in range(0, 9):
        fluidDistrNew[i, id1D] = fluidDistrMiddle[i, id1D]
            

"""
Function for calculating force between fluids in explicit forcing model. Currently,
the iosotrpy is 4 in this model, so only the potential values of  nearest and 
next-nearest nodes are considered.(There are 4, 8 and 10 options)
"""
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], \
        float64[:], float64[:], float64[:], float64[:],  boolean[:], \
        boolean[:])')
def calInteractionForceEFGPU(nx, ny, constC, interactionFluids, potentialFluid0, \
    potentialFluid1, externalForce0X, externalForce0Y, externalForce1X,\
    externalForce1Y, isDomain1D, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny; eps = 1.e-12
    if (isDomain1D[id1D] == True):
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx)
        tmpPotentialDiffX0 = 0.; tmpPotentialDiffY0 = 0.
        tmpPotentialDiffX1 = 0.; tmpPotentialDiffY1 = 0. 
        weightSumX = 0.; weightSumY = 0.
        tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
        tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
        tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
        tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF       
        #isotopy = 4
        if (isDomain1D[tmpIdF] == True and (tmpRow < ny and tmpColF < nx)):
            tmpPotentialDiffX0 += 1./3. * (potentialFluid0[tmpIdF] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 1./3. * (potentialFluid1[tmpIdF] - potentialFluid1\
            [id1D])
            weightSumX += 1./3.
        if (isDomain1D[tmpIdFU] == True and (tmpRowU < ny and tmpColF < nx)):
            tmpPotentialDiffX0 += 1./12. * (potentialFluid0[tmpIdFU] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 1./12. * (potentialFluid1[tmpIdFU] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 1./12. * (potentialFluid0[tmpIdFU] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 1./12. * (potentialFluid1[tmpIdFU] - potentialFluid1\
            [id1D])
            weightSumX += 1./12.; weightSumY += 1./12.
        if (isDomain1D[tmpIdFL] == True and (tmpRowL < ny and tmpColF < nx)):
            tmpPotentialDiffX0 += 1./12. * (potentialFluid0[tmpIdFL] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 1./12. * (potentialFluid1[tmpIdFL] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -1./12. * (potentialFluid0[tmpIdFL] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -1./12. * (potentialFluid1[tmpIdFL] - potentialFluid1\
            [id1D])
            weightSumX += 1./12.; weightSumY += 1./12.
        if (isDomain1D[tmpIdB] == True and (tmpRow < ny and tmpColB < nx)):
            tmpPotentialDiffX0 += -1./3. * (potentialFluid0[tmpIdB] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -1./3. * (potentialFluid1[tmpIdB] - potentialFluid1\
            [id1D])
            weightSumX += 1./3.
        if (isDomain1D[tmpIdBU] == True and (tmpRowU < ny and tmpColB < nx)):
            tmpPotentialDiffX0 += -1./12. * (potentialFluid0[tmpIdBU] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -1./12. * (potentialFluid1[tmpIdBU] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 1./12. * (potentialFluid0[tmpIdBU] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 1./12. * (potentialFluid1[tmpIdBU] - potentialFluid1\
            [id1D])
            weightSumX += 1./12.; weightSumY += 1./12.
        if (isDomain1D[tmpIdBL] == True and (tmpRowL < ny and tmpColB < nx)):
            tmpPotentialDiffX0 += -1./12. * (potentialFluid0[tmpIdBL] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -1./12. * (potentialFluid1[tmpIdBL] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -1./12. * (potentialFluid0[tmpIdBL] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -1./12. * (potentialFluid1[tmpIdBL] - potentialFluid1\
            [id1D])
            weightSumX += 1./12.; weightSumY += 1./12.
        if (isDomain1D[tmpIdU] == True and (tmpRowU < ny and tmpCol < nx)):
            tmpPotentialDiffY0 += 1./3. * (potentialFluid0[tmpIdU] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 1./3. * (potentialFluid1[tmpIdU] - potentialFluid1\
            [id1D])
            weightSumY += 1./3.
        if (isDomain1D[tmpIdL] == True and (tmpRowL < ny and tmpCol < nx)):
            tmpPotentialDiffY0 += -1./3. * (potentialFluid0[tmpIdL] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -1./3. * (potentialFluid1[tmpIdL] - potentialFluid1\
            [id1D])
            weightSumY += 1./3.
        #calculate the force
        externalForce0X[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
        tmpPotentialDiffX1
        externalForce0Y[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
        tmpPotentialDiffY1
        externalForce1X[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
        tmpPotentialDiffX0
        externalForce1Y[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
        tmpPotentialDiffY0
#        if (weightSumX > eps):
#            externalForce0X[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
#            tmpPotentialDiffX1 / (weightSumX)
#            externalForce1X[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
#            tmpPotentialDiffX0 / (weightSumX)
#        if (weightSumY > eps):
#            externalForce0Y[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
#            tmpPotentialDiffY1 / weightSumY
#            externalForce1Y[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
#            tmpPotentialDiffY0 / weightSumY
#        tmpSolidX = 0.; tmpSolidY= 0.
#        if (isSolid[tmpIdF] == True):
#            tmpSolidX += 1./3.; tmpSolidY += 1./3. * 0.
#        if (isSolid[tmpIdU] == True):
#            tmpSolidX += 1./3. * 0.; tmpSolidY += 1./3.
#        if (isSolid[tmpIdB] == True):
#            tmpSolidX += 1./3. * (-1.); tmpSolidY += 1./3. * 0.
#        if (isSolid[tmpIdL] == True):
#            tmpSolidX += 1./3. * 0.; tmpSolidY += 1./3. * (-1.)
#        if (isSolid[tmpIdFU] == True):
#            tmpSolidX += 1./12. * 1.; tmpSolidY += 1./12. * 1.
#        if (isSolid[tmpIdBU] == True):
#            tmpSolidX += 1./12. * (-1.); tmpSolidY += 1./12. * (1.)
#        if (isSolid[tmpIdBL] == True):
#            tmpSolidX += 1./12. * (-1.); tmpSolidY += 1./12. * (-1.)
#        if (isSolid[tmpIdFL] == True):
#            tmpSolidX += 1./12. * (1.); tmpSolidY += 1./12. * (-1.)
#        externalForce0X[id1D] += -interactionS0 * potentialFluid0[id1D] * \
#                                tmpSolidX
#        externalForce0Y[id1D] += -interactionS0 * potentialFluid0[id1D] * \
#                                tmpSolidY
#        externalForce1X[id1D] += -interactionS1 * potentialFluid1[id1D] * \
#                                tmpSolidX
#        externalForce1Y[id1D] += -interactionS1 * potentialFluid1[id1D] * \
#                                tmpSolidY

"""

"""
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], \
        float64[:], float64[:], float64[:], float64[:],  boolean[:], \
        boolean[:])')
def calInteractionForceEFGPUIS8(nx, ny, constC, interactionFluids, potentialFluid0, \
    potentialFluid1, externalForce0X, externalForce0Y, externalForce1X,\
    externalForce1Y, isDomain1D, isSolid1D):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain1D[id1D] == True):
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx)
        #nearest and next-nearst
        tmpRowL1 = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU1 = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB1 = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF1 = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF1 = tmpRow * nx + tmpColF1; tmpIdB1 = tmpRow * nx + tmpColB1
        tmpIdU1 = tmpRowU1 * nx + tmpCol; tmpIdL1 = tmpRowL1 * nx + tmpCol
        tmpIdFU1 = tmpRowU1 * nx + tmpColF1; tmpIdBU1 = tmpRowU1 * nx + tmpColB1
        tmpIdBL1 = tmpRowL1 * nx + tmpColB1; tmpIdFL1 = tmpRowL1 * nx + tmpColF1
        #2nd nearest and 2nd-nearst (4 & 8)
        tmpRowL2 = tmpRow - 2 if (tmpRow > 1) else (ny - 2 + tmpRow)
        tmpRowU2 = tmpRow + 2 if (tmpRow < ny - 2) else (tmpRow - ny + 2)
        tmpColB2 = tmpCol - 2 if (tmpCol > 1) else (nx - 2 + tmpCol)
        tmpColF2 = tmpCol + 2 if (tmpCol < nx - 2) else (tmpCol - nx + 2)
        tmpIdF2 = tmpRow * nx + tmpColF2; tmpIdB2 = tmpRow * nx + tmpColB2
        tmpIdU2 = tmpRowU2  * nx + tmpCol; tmpIdL2 = tmpRowL2 * nx + tmpCol
        tmpIdFU2 = tmpRowU2 * nx + tmpColF2; tmpIdBU2 = tmpRowU2 * nx + tmpColB2
        tmpIdFL2 = tmpRowL2 * nx + tmpColF2; tmpIdBL2 = tmpRowL2 * nx + tmpColB2
        #for 5
        tmpIdF2U1 = tmpRowU1 * nx + tmpColF2 
        tmpIdF2L1 = tmpRowL1 * nx + tmpColF2
        tmpIdB2U1 = tmpRowU1 * nx + tmpColB2
        tmpIdB2L1 = tmpRowL1 * nx + tmpColB2
        tmpIdF1U2 = tmpRowU2 * nx + tmpColF1
        tmpIdF1L2 = tmpRowL2 * nx + tmpColF1
        tmpIdB1U2 = tmpRowU2 * nx + tmpColB1
        tmpIdB1L2 = tmpRowL2 * nx + tmpColB1
        
        tmpPotentialDiffX0 = 0.; tmpPotentialDiffY0 = 0.
        tmpPotentialDiffX1 = 0.; tmpPotentialDiffY1 = 0.
        weightSumX = 0.; weightSumY = 0.
        if (isDomain1D[tmpIdF1] == True):
            tmpPotentialDiffX0 += 4./21. * (potentialFluid0[tmpIdF1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 4./21. * (potentialFluid1[tmpIdF1] - potentialFluid1\
            [id1D])
            weightSumX += 4./21.
        if (isDomain1D[tmpIdFU1] == True):
            tmpPotentialDiffX0 += 4./45. * (potentialFluid0[tmpIdFU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 4./45. * (potentialFluid1[tmpIdFU1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 4./45. * (potentialFluid0[tmpIdFU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 4./45. * (potentialFluid1[tmpIdFU1] - potentialFluid1\
            [id1D])
            weightSumX += 4./45.; weightSumY += 4./45.
        if (isDomain1D[tmpIdFL1] == True):
            tmpPotentialDiffX0 += 4./45. * (potentialFluid0[tmpIdFL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 4./45. * (potentialFluid1[tmpIdFL1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -4./45. * (potentialFluid0[tmpIdFL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -4./45. * (potentialFluid1[tmpIdFL1] - potentialFluid1\
            [id1D])
            weightSumX += 4./45.; weightSumY += 4./45.
        if (isDomain1D[tmpIdB1] == True):
            tmpPotentialDiffX0 += -4./21. * (potentialFluid0[tmpIdB1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -4./21. * (potentialFluid1[tmpIdB1] - potentialFluid1\
            [id1D])
            weightSumX += 4./21.
        if (isDomain1D[tmpIdBU1] == True):
            tmpPotentialDiffX0 += -4./45. * (potentialFluid0[tmpIdBU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -4./45. * (potentialFluid1[tmpIdBU1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 4./45. * (potentialFluid0[tmpIdBU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 4./45. * (potentialFluid1[tmpIdBU1] - potentialFluid1\
            [id1D])
            weightSumX += 4./45.; weightSumY += 4./45.
        if (isDomain1D[tmpIdBL1] == True):
            tmpPotentialDiffX0 += -4./45. * (potentialFluid0[tmpIdBL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -4./45. * (potentialFluid1[tmpIdBL1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -4./45. * (potentialFluid0[tmpIdBL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -4./45. * (potentialFluid1[tmpIdBL1] - potentialFluid1\
            [id1D])
            weightSumX += 4./45.; weightSumY += 4./45.
        if (isDomain1D[tmpIdU1] == True):
            tmpPotentialDiffY0 += 4./21. * (potentialFluid0[tmpIdU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 4./21. * (potentialFluid1[tmpIdU1] - potentialFluid1\
            [id1D])
            weightSumY += 4./21.
        if (isDomain1D[tmpIdL1] == True):
            tmpPotentialDiffY0 += -4./21. * (potentialFluid0[tmpIdL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -4./21. * (potentialFluid1[tmpIdL1] - potentialFluid1\
            [id1D])
            weightSumY += 4./21.
        #2nd near and 2nd nearest 4 & 8
        if (isDomain1D[tmpIdF1] == True and isDomain1D[tmpIdF2] == True):
            tmpPotentialDiffX0 += 2./60. * (potentialFluid0[tmpIdF2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 2./60. * (potentialFluid1[tmpIdF2] - potentialFluid1\
            [id1D])
            weightSumX += 1./60. * 4.
        if (isDomain1D[tmpIdB1] == True and isDomain1D[tmpIdB2] == True):
            tmpPotentialDiffX0 += -2./60. * (potentialFluid0[tmpIdB2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -2./60. * (potentialFluid1[tmpIdB2] - potentialFluid1\
            [id1D])
            weightSumX += 1./60. * 4.
            
        if (isDomain1D[tmpIdFU1] == True and isDomain1D[tmpIdFU2] == True):
            tmpPotentialDiffX0 += 2./5040. * (potentialFluid0[tmpIdFU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 2./5040. * (potentialFluid1[tmpIdFU2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 2./5040. * (potentialFluid0[tmpIdFU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 2./5040. * (potentialFluid1[tmpIdFU2] - potentialFluid1\
            [id1D])
            weightSumX += 1./5040. * 4.; weightSumY += 1./5040. * 4.
        if (isDomain1D[tmpIdFL1] == True and isDomain1D[tmpIdFL2] == True):
            tmpPotentialDiffX0 += 2./5040. * (potentialFluid0[tmpIdFL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 2./5040. * (potentialFluid1[tmpIdFL2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -2./5040. * (potentialFluid0[tmpIdFL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -2./5040. * (potentialFluid1[tmpIdFL2] - potentialFluid1\
            [id1D])
            weightSumX += 1./5040. * 4; weightSumY += 1./5040. * 4.
        if (isDomain1D[tmpIdBU1] == True and isDomain1D[tmpIdBU2] == True):
            tmpPotentialDiffX0 += -2./5040. * (potentialFluid0[tmpIdBU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -2./5040. * (potentialFluid1[tmpIdBU2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 2./5040. * (potentialFluid0[tmpIdBU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 2./5040. * (potentialFluid1[tmpIdBU2] - potentialFluid1\
            [id1D])
            weightSumX += 1./5040. * 4.; weightSumY += 1./5040. * 4.
        if (isDomain1D[tmpIdBL1] == True and isDomain1D[tmpIdBL2] == True):
            tmpPotentialDiffX0 += -2./5040. * (potentialFluid0[tmpIdBL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -2./5040. * (potentialFluid1[tmpIdBL2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -2./5040. * (potentialFluid0[tmpIdBL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -2./5040. * (potentialFluid1[tmpIdBL2] - potentialFluid1\
            [id1D])
            weightSumX += 1./5040. * 4.; weightSumY += 1./5040. * 4.
        if (isDomain1D[tmpIdU1] == True and isDomain1D[tmpIdU2] == True):
            tmpPotentialDiffY0 += 2./60. * (potentialFluid0[tmpIdU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 2./60. * (potentialFluid1[tmpIdU2] - potentialFluid1\
            [id1D])
            weightSumY += 1./60. * 4.
        if (isDomain1D[tmpIdL1] == True and isDomain1D[tmpIdL2] == True):
            tmpPotentialDiffY0 += -2./60. * (potentialFluid0[tmpIdL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -2./60. * (potentialFluid1[tmpIdL2] - potentialFluid1\
            [id1D])
            weightSumY += 1./60. * 4.
        # 2nd nearest node 5
        if (isDomain1D[tmpIdF2U1] == True and (isDomain1D[tmpIdF1] == True or \
            isDomain1D[tmpIdFU1] == True)):
            tmpPotentialDiffX0 += 4./ 315. * (potentialFluid0[tmpIdF2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 4./ 315. * (potentialFluid1[tmpIdF2U1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 2./ 315. * (potentialFluid0[tmpIdF2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 2./ 315. * (potentialFluid1[tmpIdF2U1] - \
            potentialFluid1[id1D])
            weightSumX += 2./315. * 4. ; weightSumY += 2./315.
        if (isDomain1D[tmpIdF1U2] == True and (isDomain1D[tmpIdU1] == True or \
            isDomain1D[tmpIdFU1] == True)):
            tmpPotentialDiffX0 += 2./ 315. * (potentialFluid0[tmpIdF1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 2./ 315. * (potentialFluid1[tmpIdF1U2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 4./ 315. * (potentialFluid0[tmpIdF1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 4./ 315. * (potentialFluid1[tmpIdF1U2] - \
            potentialFluid1[id1D])
            weightSumX += 2./315.; weightSumY += 2./315.* 4.
        if (isDomain1D[tmpIdB1U2] == True and (isDomain1D[tmpIdU1] == True or \
            isDomain1D[tmpIdBU1] == True)):
            tmpPotentialDiffX0 += -2./ 315. * (potentialFluid0[tmpIdB1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -2./315. * (potentialFluid1[tmpIdB1U2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 4./315. * (potentialFluid0[tmpIdB1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 4./315. * (potentialFluid1[tmpIdB1U2] - \
            potentialFluid1[id1D])
            weightSumX += 2./315; weightSumY += 2./315. * 4.
        if (isDomain1D[tmpIdB2U1] == True and (isDomain1D[tmpIdB1] == True or \
            isDomain1D[tmpIdBU1] == True)):
            tmpPotentialDiffX0 += -4./315 * (potentialFluid0[tmpIdB2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -4./315 * (potentialFluid1[tmpIdB2U1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 2./315. * (potentialFluid0[tmpIdB2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 2./315. * (potentialFluid1[tmpIdB2U1] - \
            potentialFluid1[id1D])
            weightSumX += 2./315. * 4.; weightSumY += 2./315.
        if (isDomain1D[tmpIdF2L1] == True and (isDomain1D[tmpIdF1] == True or \
            isDomain1D[tmpIdFL1] == True)):
            tmpPotentialDiffX0 += 4./315. * (potentialFluid0[tmpIdF2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 4./315. * (potentialFluid1[tmpIdF2L1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -2./315. * (potentialFluid0[tmpIdF2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -2./315. * (potentialFluid1[tmpIdF2L1] - \
            potentialFluid1[id1D])
            weightSumX += 2./315. * 4.; weightSumY += 2./315.
        if (isDomain1D[tmpIdF1L2] == True and (isDomain1D[tmpIdL1] == True or \
            isDomain1D[tmpIdFL1] == True)):
            tmpPotentialDiffX0 += 2./315. * (potentialFluid0[tmpIdF1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 2./315. * (potentialFluid1[tmpIdF1L2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -4./315. * (potentialFluid0[tmpIdF1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -4./315. * (potentialFluid1[tmpIdF1L2] - \
            potentialFluid1[id1D])
            weightSumX += 2./315.; weightSumY += 2./315. * 4.
        if (isDomain1D[tmpIdB1L2] == True and (isDomain1D[tmpIdL1] == True or \
            isDomain1D[tmpIdBL1] == True)):
            tmpPotentialDiffX0 += -2./315. * (potentialFluid0[tmpIdB1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -2./315. * (potentialFluid1[tmpIdB1L2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -4./315. * (potentialFluid0[tmpIdB1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -4./315. * (potentialFluid1[tmpIdB1L2] - \
            potentialFluid1[id1D])
            weightSumX += 2./315.; weightSumY += 2./315. * 4
        if (isDomain1D[tmpIdB2L1] == True and (isDomain1D[tmpIdB1] == True or \
            isDomain1D[tmpIdBL1] == True)):
            tmpPotentialDiffX0 += -4./315. * (potentialFluid0[tmpIdB2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -4./315. * (potentialFluid1[tmpIdB2L1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -2./315. * (potentialFluid0[tmpIdB2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -2./315. * (potentialFluid1[tmpIdB2L1] - \
            potentialFluid1[id1D])
            weightSumX += 2./315. * 4.; weightSumY += 2./315.
        externalForce0X[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
        tmpPotentialDiffX1
        externalForce0Y[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
        tmpPotentialDiffY1
        externalForce1X[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
        tmpPotentialDiffX0
        externalForce1Y[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
        tmpPotentialDiffY0
        
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], \
        float64[:], float64[:], float64[:], float64[:],  boolean[:], \
        boolean[:])')
def calInteractionForceEFGPUIS10(nx, ny, constC, interactionFluids, potentialFluid0, \
    potentialFluid1, externalForce0X, externalForce0Y, externalForce1X,\
    externalForce1Y, isDomain1D, isSolid1D):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain1D[id1D] == True):
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx)
        #nearest and next-nearst
        tmpRowL1 = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU1 = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB1 = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF1 = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF1 = tmpRow * nx + tmpColF1; tmpIdB1 = tmpRow * nx + tmpColB1
        tmpIdU1 = tmpRowU1 * nx + tmpCol; tmpIdL1 = tmpRowL1 * nx + tmpCol
        tmpIdFU1 = tmpRowU1 * nx + tmpColF1; tmpIdBU1 = tmpRowU1 * nx + tmpColB1
        tmpIdBL1 = tmpRowL1 * nx + tmpColB1; tmpIdFL1 = tmpRowL1 * nx + tmpColF1
        #2nd nearest and 2nd-nearst (4 & 8)
        tmpRowL2 = tmpRow - 2 if (tmpRow > 1) else (ny - 2 + tmpRow)
        tmpRowU2 = tmpRow + 2 if (tmpRow < ny - 2) else (tmpRow - ny + 2)
        tmpColB2 = tmpCol - 2 if (tmpCol > 1) else (nx - 2 + tmpCol)
        tmpColF2 = tmpCol + 2 if (tmpCol < nx - 2) else (tmpCol - nx + 2)
        tmpIdF2 = tmpRow * nx + tmpColF2; tmpIdB2 = tmpRow * nx + tmpColB2
        tmpIdU2 = tmpRowU2 * nx + tmpCol; tmpIdL2 = tmpRowL2 * nx + tmpCol
        tmpIdFU2 = tmpRowU2 * nx + tmpColF2; tmpIdBU2 = tmpRowU2 * nx + tmpColB2
        tmpIdFL2 = tmpRowL2 * nx + tmpColF2; tmpIdBL2 = tmpRowL2 * nx + tmpColB2
        #for 5
        tmpIdF2U1 = tmpRowU1 * nx + tmpColF2 
        tmpIdF2L1 = tmpRowL1 * nx + tmpColF2
        tmpIdB2U1 = tmpRowU1 * nx + tmpColB2
        tmpIdB2L1 = tmpRowL1 * nx + tmpColB2
        tmpIdF1U2 = tmpRowU2 * nx + tmpColF1
        tmpIdF1L2 = tmpRowL2 * nx + tmpColF1
        tmpIdB1U2 = tmpRowU2 * nx + tmpColB1
        tmpIdB1L2 = tmpRowL2 * nx + tmpColB1
        
        tmpRowL3 = tmpRow - 3 if (tmpRow > 2) else (ny - 3 + tmpRow)
        tmpRowU3 = tmpRow + 3 if (tmpRow < ny - 3) else (tmpRow - ny + 3)
        tmpColB3 = tmpCol - 3 if (tmpCol > 2) else (nx - 3 + tmpCol)
        tmpColF3 = tmpCol + 3 if (tmpCol < nx - 3) else (tmpCol - nx + 3)
        tmpIdF3 = tmpRow * nx + tmpColF3; tmpIdB3 = tmpRow * nx + tmpColB3
        tmpIdU3 = tmpRowU3 * nx + tmpCol; tmpIdL3 = tmpRowL3 * nx + tmpCol
        
        tmpIdF3U1 = tmpRowU1 * nx + tmpColF3
        tmpIdF1U3 = tmpRowU3 * nx + tmpColF1
        tmpIdB1U3 = tmpRowU3 * nx + tmpColB1
        tmpIdB3U1 = tmpRowU1 * nx + tmpColB3
        tmpIdB3L1 = tmpRowL1 * nx + tmpColB3
        tmpIdB1L3 = tmpRowL3 * nx + tmpColB1
        tmpIdF1L3 = tmpRowL3 * nx + tmpColF1
        tmpIdF3L1 = tmpRowL1 * nx + tmpColF3
        
        tmpPotentialDiffX0 = 0.; tmpPotentialDiffY0 = 0.
        tmpPotentialDiffX1 = 0.; tmpPotentialDiffY1 = 0.
        weightSumX = 0.; weightSumY = 0.
        if (isDomain1D[tmpIdF1] == True):
            tmpPotentialDiffX0 += 262./1785. * (potentialFluid0[tmpIdF1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 262./1785. * (potentialFluid1[tmpIdF1] - potentialFluid1\
            [id1D])
            weightSumX += 262./1785.
        if (isDomain1D[tmpIdFU1] == True):
            tmpPotentialDiffX0 += 93./1190. * (potentialFluid0[tmpIdFU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 93./1190. * (potentialFluid1[tmpIdFU1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 93./1190. * (potentialFluid0[tmpIdFU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 93./1190. * (potentialFluid1[tmpIdFU1] - potentialFluid1\
            [id1D])
            weightSumX += 93./1190.; weightSumY += 93./1190.
        if (isDomain1D[tmpIdFL1] == True):
            tmpPotentialDiffX0 += 93./1190. * (potentialFluid0[tmpIdFL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 93./1190. * (potentialFluid1[tmpIdFL1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -93./1190. * (potentialFluid0[tmpIdFL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -93./1190. * (potentialFluid1[tmpIdFL1] - potentialFluid1\
            [id1D])
            weightSumX += 93./1190.; weightSumY += 93./1190.
        if (isDomain1D[tmpIdB1] == True):
            tmpPotentialDiffX0 += -262./1785. * (potentialFluid0[tmpIdB1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -262./1785. * (potentialFluid1[tmpIdB1] - potentialFluid1\
            [id1D])
            weightSumX += 262./1785.
        if (isDomain1D[tmpIdBU1] == True):
            tmpPotentialDiffX0 += -93./1190. * (potentialFluid0[tmpIdBU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -93./1190. * (potentialFluid1[tmpIdBU1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 93./1190. * (potentialFluid0[tmpIdBU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 93./1190. * (potentialFluid1[tmpIdBU1] - potentialFluid1\
            [id1D])
            weightSumX += 93./1190.; weightSumY += 93./1190.
        if (isDomain1D[tmpIdBL1] == True):
            tmpPotentialDiffX0 += -93./1190. * (potentialFluid0[tmpIdBL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -93./1190. * (potentialFluid1[tmpIdBL1] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -93./1190. * (potentialFluid0[tmpIdBL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -93./1190. * (potentialFluid1[tmpIdBL1] - potentialFluid1\
            [id1D])
            weightSumX += 93./1190.; weightSumY += 93./1190.
        if (isDomain1D[tmpIdU1] == True):
            tmpPotentialDiffY0 += 262./1785. * (potentialFluid0[tmpIdU1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 262./1785. * (potentialFluid1[tmpIdU1] - potentialFluid1\
            [id1D])
            weightSumY += 262./1785.
        if (isDomain1D[tmpIdL1] == True):
            tmpPotentialDiffY0 += -262./1785. * (potentialFluid0[tmpIdL1] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -262./1785. * (potentialFluid1[tmpIdL1] - potentialFluid1\
            [id1D])
            weightSumY += 262./1785.
            
        #2nd near and 2nd nearest 4 & 8
        if (isDomain1D[tmpIdF1] == True and isDomain1D[tmpIdF2] == True):
            tmpPotentialDiffX0 += 14./340. * (potentialFluid0[tmpIdF2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 14./340. * (potentialFluid1[tmpIdF2] - potentialFluid1\
            [id1D])
            weightSumX += 7./340. * 4.
        if (isDomain1D[tmpIdB1] == True and isDomain1D[tmpIdB2] == True):
            tmpPotentialDiffX0 += -14./340. * (potentialFluid0[tmpIdB2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -14./340. * (potentialFluid1[tmpIdB2] - potentialFluid1\
            [id1D])
            weightSumX += 7./340. * 4.
        if (isDomain1D[tmpIdFU1] == True and isDomain1D[tmpIdFU2] == True):
            tmpPotentialDiffX0 += 18./9520. * (potentialFluid0[tmpIdFU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 18./9520. * (potentialFluid1[tmpIdFU2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 18./9520. * (potentialFluid0[tmpIdFU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 18./9520. * (potentialFluid1[tmpIdFU2] - potentialFluid1\
            [id1D])
            weightSumX += 9./9520. * 4.; weightSumY +=  9./9520. * 4.
        if (isDomain1D[tmpIdFL1] == True and isDomain1D[tmpIdFL2] == True):
            tmpPotentialDiffX0 += 18./9520. * (potentialFluid0[tmpIdFL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += 18./9520. * (potentialFluid1[tmpIdFL2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -18./9520. * (potentialFluid0[tmpIdFL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -18./9520. * (potentialFluid1[tmpIdFL2] - potentialFluid1\
            [id1D])
            weightSumX += 9./9520. * 4.; weightSumY +=  9./9520. * 4.
        if (isDomain1D[tmpIdBU1] == True and isDomain1D[tmpIdBU2] == True):
            tmpPotentialDiffX0 += -18./9520. * (potentialFluid0[tmpIdBU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -18./9520. * (potentialFluid1[tmpIdBU2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += 18./9520. * (potentialFluid0[tmpIdBU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 18./9520. * (potentialFluid1[tmpIdBU2] - potentialFluid1\
            [id1D])
            weightSumX += 9./9520. * 4.; weightSumY +=  9./9520. * 4.
        if (isDomain1D[tmpIdBL1] == True and isDomain1D[tmpIdBL2] == True):
            tmpPotentialDiffX0 += -18./9520. * (potentialFluid0[tmpIdBL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffX1 += -18./9520. * (potentialFluid1[tmpIdBL2] - potentialFluid1\
            [id1D])
            tmpPotentialDiffY0 += -18./9520. * (potentialFluid0[tmpIdBL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -18./9520. * (potentialFluid1[tmpIdBL2] - potentialFluid1\
            [id1D])
            weightSumX += 9./9520. * 4.; weightSumY +=  9./9520. * 4.
        if (isDomain1D[tmpIdU1] == True and isDomain1D[tmpIdU2] == True):
            tmpPotentialDiffY0 += 14./340. * (potentialFluid0[tmpIdU2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += 14./340. * (potentialFluid1[tmpIdU2] - potentialFluid1\
            [id1D])
            weightSumY += 7./340. * 4.
        if (isDomain1D[tmpIdL1] == True and isDomain1D[tmpIdL2] == True):
            tmpPotentialDiffY0 += -14./340. * (potentialFluid0[tmpIdL2] - potentialFluid0\
            [id1D])
            tmpPotentialDiffY1 += -14./340. * (potentialFluid1[tmpIdL2] - potentialFluid1\
            [id1D])
            weightSumY += 7./340. * 4.
        # 2nd nearest node 5
        if (isDomain1D[tmpIdF2U1] == True and (isDomain1D[tmpIdF1] == True or \
            isDomain1D[tmpIdFU1] == True)):
            tmpPotentialDiffX0 += 12./595. * (potentialFluid0[tmpIdF2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 12./595. * (potentialFluid1[tmpIdF2U1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 6./595. * (potentialFluid0[tmpIdF2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 6./595. * (potentialFluid1[tmpIdF2U1] - \
            potentialFluid1[id1D])
            weightSumX += 6./595. * 4.; weightSumY += 6./595.
        if (isDomain1D[tmpIdF1U2] == True and (isDomain1D[tmpIdU1] == True or \
            isDomain1D[tmpIdFU1] == True)):
            tmpPotentialDiffX0 += 6./595. * (potentialFluid0[tmpIdF1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 6./595. * (potentialFluid1[tmpIdF1U2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 12./595. * (potentialFluid0[tmpIdF1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 12./595. * (potentialFluid1[tmpIdF1U2] - \
            potentialFluid1[id1D])
            weightSumX += 6./595.; weightSumY += 6./595. * 4.
        if (isDomain1D[tmpIdB1U2] == True and (isDomain1D[tmpIdU1] == True or \
            isDomain1D[tmpIdBU1] == True)):
            tmpPotentialDiffX0 += -6./595. * (potentialFluid0[tmpIdB1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -6./595. * (potentialFluid1[tmpIdB1U2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 12./595. * (potentialFluid0[tmpIdB1U2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 12./595. * (potentialFluid1[tmpIdB1U2] - \
            potentialFluid1[id1D])
            weightSumX += 6./595.; weightSumY += 6./595. * 4.
        if (isDomain1D[tmpIdB2U1] == True and (isDomain1D[tmpIdB1] == True or \
            isDomain1D[tmpIdBU1] == True)):
            tmpPotentialDiffX0 += -12./595. * (potentialFluid0[tmpIdB2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -12./595. * (potentialFluid1[tmpIdB2U1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 6./595. * (potentialFluid0[tmpIdB2U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 6./595. * (potentialFluid1[tmpIdB2U1] - \
            potentialFluid1[id1D])
            weightSumX += 6./595. * 4.; weightSumY += 6./595.
        if (isDomain1D[tmpIdF2L1] == True and (isDomain1D[tmpIdF1] == True or \
            isDomain1D[tmpIdFL1] == True)):
            tmpPotentialDiffX0 += 12./595. * (potentialFluid0[tmpIdF2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 12./595. * (potentialFluid1[tmpIdF2L1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -6./595. * (potentialFluid0[tmpIdF2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -6./595. * (potentialFluid1[tmpIdF2L1] - \
            potentialFluid1[id1D])
            weightSumX += 6./595. * 4.; weightSumY += 6./595.
        if (isDomain1D[tmpIdF1L2] == True and (isDomain1D[tmpIdL1] == True or \
            isDomain1D[tmpIdFL1] == True)):
            tmpPotentialDiffX0 += 6./595. * (potentialFluid0[tmpIdF1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 6./595. * (potentialFluid1[tmpIdF1L2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -12./595. * (potentialFluid0[tmpIdF1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -12./595. * (potentialFluid1[tmpIdF1L2] - \
            potentialFluid1[id1D])
            weightSumX += 6./595.; weightSumY += 6./595. * 4.
        if (isDomain1D[tmpIdB1L2] == True and (isDomain1D[tmpIdL1] == True or \
            isDomain1D[tmpIdBL1] == True)):
            tmpPotentialDiffX0 += -6./595. * (potentialFluid0[tmpIdB1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -6./595. * (potentialFluid1[tmpIdB1L2] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -12./595. * (potentialFluid0[tmpIdB1L2] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -12./595. * (potentialFluid1[tmpIdB1L2] - \
            potentialFluid1[id1D])
            weightSumX += 6./595.; weightSumY += 6./595. * 4.
        if (isDomain1D[tmpIdB2L1] == True and (isDomain1D[tmpIdB1] == True or \
            isDomain1D[tmpIdBL1] == True)):
            tmpPotentialDiffX0 += -12./595. * (potentialFluid0[tmpIdB2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -12./595. * (potentialFluid1[tmpIdB2L1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -6./595. * (potentialFluid0[tmpIdB2L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -6./595. * (potentialFluid1[tmpIdB2L1] - \
            potentialFluid1[id1D])
            weightSumX += 6./595. * 4.; weightSumY += 6./595.
        #3rd nearest and next-nearst
        if ((isDomain1D[tmpIdF1] == True and isDomain1D[tmpIdF2] == True) and \
            isDomain1D[tmpIdF3] == True):
            tmpPotentialDiffX0 += 6./5355. * (potentialFluid0[tmpIdF3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 6./5355. * (potentialFluid1[tmpIdF3] - \
            potentialFluid1[id1D])
        if ((isDomain1D[tmpIdB1] == True and isDomain1D[tmpIdB2] == True) and \
            isDomain1D[tmpIdB3] == True):
            tmpPotentialDiffX0 += -6./5355. * (potentialFluid0[tmpIdB3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -6./5355. * (potentialFluid1[tmpIdB3] - \
            potentialFluid1[id1D])
        if ((isDomain1D[tmpIdU1] == True and isDomain1D[tmpIdU2] == True) and \
            isDomain1D[tmpIdU3] == True):
            tmpPotentialDiffY0 += 6./5355. * (potentialFluid0[tmpIdU3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 6./5355. * (potentialFluid1[tmpIdU3] - \
            potentialFluid1[id1D])
        if ((isDomain1D[tmpIdL1] == True and isDomain1D[tmpIdL2] == True) and \
            isDomain1D[tmpIdL3] == True):
            tmpPotentialDiffY0 += -6./5355. * (potentialFluid0[tmpIdL3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -6./5355. * (potentialFluid1[tmpIdL3] - \
            potentialFluid1[id1D])
        
        if (isDomain1D[tmpIdF3U1] == True and (isDomain1D[tmpIdFU1] == True and \
            isDomain1D[tmpIdF2U1] == True) or (isDomain1D[tmpIdF1] == True and \
            isDomain1D[tmpIdF2] == True)):
            tmpPotentialDiffX0 += 3./7140. * (potentialFluid0[tmpIdF3U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 3./7140. * (potentialFluid1[tmpIdF3U1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 1./7140. * (potentialFluid0[tmpIdF3U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 1./7140. * (potentialFluid1[tmpIdF3U1] - \
            potentialFluid1[id1D])
        if (isDomain1D[tmpIdF1U3] == True and ((isDomain1D[tmpIdU2] == True and \
            isDomain1D[tmpIdU1] == True) or (isDomain1D[tmpIdF1U2] == True and \
            isDomain1D[tmpIdFU1]== True))):
            tmpPotentialDiffX0 += 1./7140. * (potentialFluid0[tmpIdF1U3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 1./7140. * (potentialFluid1[tmpIdF1U3] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 3./7140. * (potentialFluid0[tmpIdF1U3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 3./7140. * (potentialFluid1[tmpIdF1U3] - \
            potentialFluid1[id1D])
        if (isDomain1D[tmpIdF3L1] == True and (isDomain1D[tmpIdF2] == True and \
            isDomain1D[tmpIdF1] == True) or (isDomain1D[tmpIdF2L1] == True and \
            isDomain1D[tmpIdFL1] == True)):
            tmpPotentialDiffX0 += 3./7140. * (potentialFluid0[tmpIdF3L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 3./7140. * (potentialFluid1[tmpIdF3L1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -1./7140. * (potentialFluid0[tmpIdF3L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -1./7140. * (potentialFluid1[tmpIdF3L1] - \
            potentialFluid1[id1D])
        if (isDomain1D[tmpIdF1L3] == True and ((isDomain1D[tmpIdL1] == True and \
            isDomain1D[tmpIdL2] == True) or (isDomain1D[tmpIdFL1] == True or \
            isDomain1D[tmpIdF1L2] == True))):
            tmpPotentialDiffX0 += 1./7140. * (potentialFluid0[tmpIdF1L3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += 1./7140. * (potentialFluid1[tmpIdF1L3] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -3./7140. * (potentialFluid0[tmpIdF1L3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -3./7140. * (potentialFluid1[tmpIdF1L3] - \
            potentialFluid1[id1D])
        if (isDomain1D[tmpIdB1U3] == True and ((isDomain1D[tmpIdU2] == True or \
            isDomain1D[tmpIdU1] == True) or (isDomain1D[tmpIdB1U2] == True or \
            isDomain1D[tmpIdBU1] == True))):
            tmpPotentialDiffX0 += -1./7140. * (potentialFluid0[tmpIdB1U3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -1./7140. * (potentialFluid1[tmpIdB1U3] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 3./7140. * (potentialFluid0[tmpIdB1U3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 3./7140. * (potentialFluid1[tmpIdB1U3] - \
            potentialFluid1[id1D])
        if (isDomain1D[tmpIdB3U1] == True and ((isDomain1D[tmpIdB2] == True and \
            isDomain1D[tmpIdB1] == True) or (isDomain1D[tmpIdB2U1] == True and \
            isDomain1D[tmpIdBU1] == True))):
            tmpPotentialDiffX0 += -3./7140. * (potentialFluid0[tmpIdB3U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -3./7140. * (potentialFluid1[tmpIdB3U1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += 1./7140. * (potentialFluid0[tmpIdB3U1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += 1./7140. * (potentialFluid1[tmpIdB3U1] - \
            potentialFluid1[id1D])
        if (isDomain1D[tmpIdB3L1] == True and ((isDomain1D[tmpIdB2] == True and \
            isDomain1D[tmpIdB1] == True) or (isDomain1D[tmpIdB2L1] == True and \
            isDomain1D[tmpIdBL1] == True))):
            tmpPotentialDiffX0 += -3./7140. * (potentialFluid0[tmpIdB3L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -3./7140. * (potentialFluid1[tmpIdB3L1] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -1./7140. * (potentialFluid0[tmpIdB3L1] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -1./7140. * (potentialFluid1[tmpIdB3L1] - \
            potentialFluid1[id1D])
        if (isDomain1D[tmpIdB1L3] == True and ((isDomain1D[tmpIdL2] == True and \
            isDomain1D[tmpIdL1] == True) or (isDomain1D[tmpIdB1L2] == True and \
            isDomain1D[tmpIdBL1] == True))):
            tmpPotentialDiffX0 += -1./7140. * (potentialFluid0[tmpIdB1L3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffX1 += -1./7140. * (potentialFluid1[tmpIdB1L3] - \
            potentialFluid1[id1D])
            tmpPotentialDiffY0 += -3./7140. * (potentialFluid0[tmpIdB1L3] - \
            potentialFluid0[id1D])
            tmpPotentialDiffY1 += -3./7140. * (potentialFluid1[tmpIdB1L3] - \
            potentialFluid1[id1D])
        externalForce0X[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
        tmpPotentialDiffX1
        externalForce0Y[id1D] = -constC * potentialFluid0[id1D] * interactionFluids * \
        tmpPotentialDiffY1
        externalForce1X[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
        tmpPotentialDiffX0
        externalForce1Y[id1D] = -constC * potentialFluid1[id1D] * interactionFluids * \
        tmpPotentialDiffY0
        
"""
Function for calculating the interaction force with solid phase in explicit 
forcing method
"""        
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], float64[:], \
        float64[:], float64[:], float64[:], boolean[:])')
def calExternalForceSolid(nx, ny, interactionS0, interactionS1, potentialFluid0, \
    potentialFluid1, externalForce0X, externalForce0Y, externalForce1X, \
    externalForce1Y, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isSolid[id1D] == False):
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx) 
        tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
        tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
        tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
        tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
        tmpSolidX = 0.; tmpSolidY= 0.
        if (isSolid[tmpIdF] == True):
            tmpSolidX += 1./9.; tmpSolidY += 1./9. * 0.

        if (isSolid[tmpIdU] == True):
            tmpSolidX += 1./9. * 0.; tmpSolidY += 1./9.
        if (isSolid[tmpIdB] == True):
            tmpSolidX += 1./9. * (-1.); tmpSolidY += 1./9. * 0.
        if (isSolid[tmpIdL] == True):
            tmpSolidX += 1./9. * 0.; tmpSolidY += 1./9. * (-1.)
        if (isSolid[tmpIdFU] == True):
            tmpSolidX += 1./36. * 1.; tmpSolidY += 1./36. * 1.
        if (isSolid[tmpIdBU] == True):
            tmpSolidX += 1./36. * (-1.); tmpSolidY += 1./36. * (1.)
        if (isSolid[tmpIdBL] == True):
            tmpSolidX += 1./36. * (-1.); tmpSolidY += 1./36. * (-1.)
        if (isSolid[tmpIdFL] == True):
            tmpSolidX += 1./36. * (1.); tmpSolidY += 1./36. * (-1.)
        externalForce0X[id1D] += -interactionS0 * potentialFluid0[id1D] * \
                                tmpSolidX
        externalForce0Y[id1D] += -interactionS0 * potentialFluid0[id1D] * \
                                tmpSolidY
        externalForce1X[id1D] += -interactionS1 * potentialFluid1[id1D] * \
                                tmpSolidX
        externalForce1Y[id1D] += -interactionS1 * potentialFluid1[id1D] * \
                                tmpSolidY

@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], float64[:], \
        float64[:], float64[:], float64[:], boolean[:], boolean[:])')
def calExternalForceSolidEF(nx, ny, interactionS0, interactionS1, potentialFluid0, \
    potentialFluid1, externalForce0X, externalForce0Y, externalForce1X, \
    externalForce1Y, isDomain, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx) 
        tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
        tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
        tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
        tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
        tmpSolidX = 0.; tmpSolidY= 0.
        if (isSolid[tmpIdF] == True):
            tmpSolidX += 1./3.; tmpSolidY += 1./3. * 0.

        if (isSolid[tmpIdU] == True):
            tmpSolidX += 1./3. * 0.; tmpSolidY += 1./3.
        if (isSolid[tmpIdB] == True):
            tmpSolidX += 1./3. * (-1.); tmpSolidY += 1./3. * 0.
        if (isSolid[tmpIdL] == True):
            tmpSolidX += 1./3. * 0.; tmpSolidY += 1./3. * (-1.)
        if (isSolid[tmpIdFU] == True):
            tmpSolidX += 1./12. * 1.; tmpSolidY += 1./12. * 1.
        if (isSolid[tmpIdBU] == True):
            tmpSolidX += 1./12. * (-1.); tmpSolidY += 1./12. * (1.)
        if (isSolid[tmpIdBL] == True):
            tmpSolidX += 1./12. * (-1.); tmpSolidY += 1./12. * (-1.)
        if (isSolid[tmpIdFL] == True):
            tmpSolidX += 1./12. * (1.); tmpSolidY += 1./12. * (-1.)
        externalForce0X[id1D] += -interactionS0 * potentialFluid0[id1D] * \
                                tmpSolidX
        externalForce0Y[id1D] += -interactionS0 * potentialFluid0[id1D] * \
                                tmpSolidY
        externalForce1X[id1D] += -interactionS1 * potentialFluid1[id1D] * \
                                tmpSolidX
        externalForce1Y[id1D] += -interactionS1 * potentialFluid1[id1D] * \
                                tmpSolidY

"""
Function for effective velocity in explicit forcing method
"""
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], \
            float64[:], float64[:], float64[:], float64[:], float64[:], \
            float64[:], boolean[:])')
def calEffectiveVGPU(nx, ny, tau0, tau1, fluidDensity0, fluidDensity1, velocityX0, \
    velocityY0, velocityX1, velocityY1, effectiveVX, effectiveVY, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        effectiveVX[id1D] = (fluidDensity0[id1D] * velocityX0[id1D] / tau0 + \
        fluidDensity1[id1D] * velocityX1[id1D] / tau1) / (fluidDensity0[id1D] / \
        tau0 + fluidDensity1[id1D] / tau1)
        effectiveVY[id1D] = (fluidDensity0[id1D] * velocityY0[id1D] / tau0 + \
        fluidDensity1[id1D] * velocityY1[id1D] / tau1) / (fluidDensity0[id1D] / \
        tau0 + fluidDensity1[id1D] / tau1)
        
"""
Function for effective velocity for explicit forcing scheme - MRT.
conserveS0 and conserveS1 are from relaxation matrix in MRT (S_c^{k})
"""
@cuda.jit('void(int64, int64, float64, float64, float64[:], float64[:], \
            float64[:], float64[:], float64[:], float64[:], float64[:], \
            float64[:], boolean[:])')
def calEffectiveVGPUMRT(nx, ny, conserveS0, conserveS1, fluidDensity0, fluidDensity1, velocityX0, \
    velocityY0, velocityX1, velocityY1, effectiveVX, effectiveVY, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        effectiveVX[id1D] = (fluidDensity0[id1D] * velocityX0[id1D] * conserveS0 + \
        fluidDensity1[id1D] * velocityX1[id1D] * conserveS1) / (fluidDensity0[id1D] * \
        conserveS0 + fluidDensity1[id1D] * conserveS1)
        effectiveVY[id1D] = (fluidDensity0[id1D] * velocityY0[id1D] * conserveS0 + \
        fluidDensity1[id1D] * velocityY1[id1D] * conserveS1) / (fluidDensity0[id1D] * \
        conserveS0 + fluidDensity1[id1D] * conserveS1)
        
"""
Function for equilibrium distribution function values in explicit forcing method.
u_eq^k is different from original SC model
"""
@cuda.jit('void(int64, int64, float64[:], float64[:], float64[:], float64[:, :], \
        boolean[:])')
def calEquilibriumFuncEFGPU(nx, ny, fluidDensity, effectiveVX, effectiveVY, \
    equilibriumFunc, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
#        squareV = 1.5 * (effectiveVX[id1D] * effectiveVX[id1D] + effectiveVY[id1D] * \
#            effectiveVY[id1D])
        squareV = (effectiveVX[id1D] * effectiveVX[id1D] + effectiveVY[id1D] * \
            effectiveVY[id1D])
#        equilibriumFunc[0, id1D] = 4./9. * fluidDensity[id1D] * (1. - squareV)
        equilibriumFunc[0, id1D] = fluidDensity[id1D] * (1./6. - 2. * squareV / 3.)
#        equilibriumFunc[1, id1D] = 1./9. * fluidDensity[id1D] * (1. + 3. * \
#            effectiveVX[id1D] + 4.5 * effectiveVX[id1D] * effectiveVX[id1D] - squareV)
#        equilibriumFunc[3, id1D] = equilibriumFunc[1, id1D] - 1./9. * 6. * \
#            effectiveVX[id1D] * fluidDensity[id1D]
        equilibriumFunc[1, id1D] = 1./9. * fluidDensity[id1D] * (1.5 + 3. * \
            effectiveVX[id1D] + 4.5 *  effectiveVX[id1D] * effectiveVX[id1D] - \
            squareV / (2.* 1./3.))
        equilibriumFunc[2, id1D] = 1./9. * fluidDensity[id1D] * (1.5 + 3. * \
            effectiveVY[id1D] + 4.5 *  effectiveVY[id1D] * effectiveVY[id1D] - \
            squareV / (2.* 1./3.))
        equilibriumFunc[3, id1D] = 1./9. * fluidDensity[id1D] * (1.5 + 3. * \
            (-effectiveVX[id1D]) + 4.5 *  (-effectiveVX[id1D]) * (-effectiveVX[id1D]) - \
            squareV / (2.* 1./3.))
        equilibriumFunc[4, id1D] = 1./9. * fluidDensity[id1D] * (1.5 + 3. * \
            (-effectiveVY[id1D]) + 4.5 *  (-effectiveVY[id1D]) * (-effectiveVY[id1D]) - \
            squareV / (2.* 1./3.))
        equilibriumFunc[5, id1D] = 1./36. * fluidDensity[id1D] * (1.5 + 3. * \
            (effectiveVX[id1D] + effectiveVY[id1D]) + 4.5 * (effectiveVX[id1D] + \
            effectiveVY[id1D]) * (effectiveVX[id1D] + effectiveVY[id1D]) - squareV/(2. * 1./3.))
        equilibriumFunc[6, id1D] = 1./36. * fluidDensity[id1D] * (1.5 + 3. * \
            (-effectiveVX[id1D] + effectiveVY[id1D]) + 4.5 * (-effectiveVX[id1D] + \
            effectiveVY[id1D]) * (-effectiveVX[id1D] + effectiveVY[id1D]) - squareV/(2. * 1./3.))
        equilibriumFunc[7, id1D] = 1./36. * fluidDensity[id1D] * (1.5 + 3. * \
            (-effectiveVX[id1D] - effectiveVY[id1D]) + 4.5 * (-effectiveVX[id1D] - \
            effectiveVY[id1D]) * (-effectiveVY[id1D] - effectiveVY[id1D]) - squareV/(2. * 1./3.))
        equilibriumFunc[8, id1D] = 1./36. * fluidDensity[id1D] * (1.5 + 3. * \
            (effectiveVX[id1D] - effectiveVY[id1D]) + 4.5 * (effectiveVX[id1D] - \
            effectiveVY[id1D]) * (effectiveVX[id1D] - effectiveVY[id1D]) - squareV/(2. * 1./3.))
        
"""
function for forcing term distribution distribution function in explicit forcing method
"""
@cuda.jit('void(int64, int64, float64[:], float64[:], float64[:], float64[:], \
        float64[:], float64[:, :], float64[:, :], boolean[:])')
def calForcingTermEFGPU(nx, ny, fluidDensity, externalForceX, externalForceY, \
    effectiveVX, effectiveVY, equilibriumFunc, forcingTerm, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        forcingTerm[0, id1D] = (externalForceX[id1D] * (-effectiveVX[id1D]) + \
            externalForceY[id1D] * (-effectiveVY[id1D])) * equilibriumFunc[0, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[1, id1D] = (externalForceX[id1D] * (1. - effectiveVX[id1D]) + \
            externalForceY[id1D] * (-effectiveVY[id1D])) * equilibriumFunc[1, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[2, id1D] = (externalForceX[id1D] * (-effectiveVX[id1D]) + \
            externalForceY[id1D] * (1. - effectiveVY[id1D])) * equilibriumFunc[2, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[3, id1D] = (externalForceX[id1D] * (-1. - effectiveVX[id1D]) + \
            externalForceY[id1D] * (-effectiveVY[id1D])) * equilibriumFunc[3, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[4, id1D] = (externalForceX[id1D] * (-effectiveVX[id1D]) + \
            externalForceY[id1D] * (-1. - effectiveVY[id1D])) * equilibriumFunc[4, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[5, id1D] = (externalForceX[id1D] * (1. - effectiveVX[id1D]) + \
            externalForceY[id1D] * (1. - effectiveVY[id1D])) * equilibriumFunc[5, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[6, id1D] = (externalForceX[id1D] * (-1. - effectiveVX[id1D]) + \
            externalForceY[id1D] * (1. - effectiveVY[id1D])) * equilibriumFunc[6, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[7, id1D] = (externalForceX[id1D] * (-1. - effectiveVX[id1D]) + \
            externalForceY[id1D] * (-1. - effectiveVY[id1D])) * equilibriumFunc[7, id1D] /\
            (1./3. * fluidDensity[id1D])
        forcingTerm[8, id1D] = (externalForceX[id1D] * (1. - effectiveVX[id1D]) + \
            externalForceY[id1D] * (-1. - effectiveVY[id1D])) * equilibriumFunc[8, id1D] /\
            (1./3. * fluidDensity[id1D])
            
"""
function for the transformation with forcing term
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], boolean[:])')
def calTransformedDistrFuncGPU(nx, ny, fluidDistr, forcingTerm, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        for i in range(0, 9):
            fluidDistr[i, id1D] = fluidDistr[i, id1D] - 1./2. * forcingTerm[i, id1D]
            
"""
Calculate the macro-velocity of each fluid for explicit forcing method.
"""
@cuda.jit('void(int64, int64, float64[:], float64[:], float64[:], float64[:, :], \
    float64[:], float64[:], boolean[:])')
def calMacroVelocityEFGPU(nx, ny, fluidDensity, externalFX, externalFY, distrFunc, \
    velocityX, velocityY, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        velocityX[id1D] = ((distrFunc[1, id1D] - distrFunc[3, id1D] + \
            distrFunc[5, id1D] - distrFunc[6, id1D] + distrFunc[8, id1D] -\
            distrFunc[7, id1D]) + 1./2. * externalFX[id1D]) / fluidDensity[id1D]
        velocityY[id1D] = ((distrFunc[2, id1D] - distrFunc[4, id1D] + \
            distrFunc[5, id1D] + distrFunc[6, id1D] - distrFunc[7, id1D] - \
            distrFunc[8, id1D]) + 1./2. * externalFY[id1D]) / fluidDensity[id1D]
#        velocityX[id1D] = ((distrFunc[1, id1D] - distrFunc[3, id1D] + \
#            distrFunc[5, id1D] - distrFunc[6, id1D] + distrFunc[8, id1D] -\
#            distrFunc[7, id1D])) / fluidDensity[id1D]
#        velocityY[id1D] = ((distrFunc[2, id1D] - distrFunc[4, id1D] + \
#            distrFunc[5, id1D] + distrFunc[6, id1D] - distrFunc[7, id1D] - \
#            distrFunc[8, id1D])) / fluidDensity[id1D]

"""
function for the collision of explicit forcing method
"""
@cuda.jit('void(int64, int64, float64, float64[:, :],  float64[:, :], float64[:, :], \
        boolean[:])')
def calCollisionEFGPU(nx, ny, tau, fluidDistrOld, equilibriumFunc, forcingTerm, \
    isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    #total number of nodes
    totalNum = nx * ny
    if (isDomain[id1D] == True):
        for i in range(0, 9):
            fluidDistrOld[i, id1D] = fluidDistrOld[i, id1D] + 1./tau * (\
                equilibriumFunc[i, id1D] - fluidDistrOld[i, id1D] - \
                1./2. * forcingTerm[i, id1D]) + 1. * forcingTerm[i, id1D]
                
"""
Save the results in the periodic boundary condition on eastern and western side
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :])')
def savePeriodicCollisionDataLR(nx, ny, fluidDistr, dataLeft, dataRight):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    
    tmpRow = int(id1D / nx); tmpCol = id1D % nx
    if (tmpCol == 0):
        dataLeft[0, tmpRow] = fluidDistr[3, id1D]
        dataLeft[1, tmpRow] = fluidDistr[6, id1D]
        dataLeft[2, tmpRow] = fluidDistr[7, id1D]
    if (tmpCol == nx - 1):
        dataRight[0, tmpRow] = fluidDistr[1, id1D]
        dataRight[1, tmpRow] = fluidDistr[5, id1D]
        dataRight[2, tmpRow] = fluidDistr[8, id1D]
        
"""
Save the results in the periodic boundary condition on northern and southern side
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :])')
def savePeriodicCollisionDataTB(nx, ny, fluidDistr, dataBottom, dataTop):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    tmpRow = int(id1D / nx); tmpCol = id1D % nx
    if (tmpRow == 0):
        dataBottom[0, tmpCol] = fluidDistr[4, id1D]
        dataBottom[1, tmpCol] = fluidDistr[7, id1D]
        dataBottom[2, tmpCol] = fluidDistr[8, id1D]
    if (tmpRow == ny - 1):
        dataTop[0, tmpCol] = fluidDistr[2, id1D]
        dataTop[1, tmpCol] = fluidDistr[5, id1D]
        dataTop[2, tmpCol] = fluidDistr[6, id1D]

        
"""
after streaming process, replace the values on the periodic boundary of eastern 
and western sides
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :], \
    boolean[:])')
def calPeriodicBoundaryLR(nx, ny, dataLeft, dataRight, fluidDistr, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    tmpRow = int(id1D / nx); tmpCol = (id1D % nx)
    tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
    tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
    tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
    tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
    tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
    tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
    tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
    tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
    if (tmpCol == 0 and isDomain[id1D] == True):
#        if (isDomain[tmpIdB] == True):
        fluidDistr[1, id1D] = dataRight[0, tmpRow]
#        if (isDomain[tmpIdBU] == True):
        fluidDistr[8, id1D] = dataRight[2, tmpRowU]
#        if (isDomain[tmpIdBL] == True):
        fluidDistr[5, id1D] = dataRight[1, tmpRowL]
    if (tmpCol == nx - 1 and isDomain[id1D] == True):
#        if (isDomain[tmpIdF] == True):
        fluidDistr[3, id1D] = dataLeft[0, tmpRow]
#        if (isDomain[tmpIdFU] == True):
        fluidDistr[7, id1D] = dataLeft[2, tmpRowU]
#        if (isDomain[tmpIdFL] == True):
        fluidDistr[6, id1D] = dataLeft[1, tmpRowL]
        
"""
after streaming process, replace the values on the periodic boundary of northern
and southern sides
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :], \
    boolean[:])')
def calPeriodicBoundaryTB(nx, ny, dataBottom, dataTop, fluidDistr, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    tmpRow = int(id1D / nx); tmpCol = (id1D % nx)
    tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
    tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
    tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
    tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
    tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
    tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
    tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
    tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
    if (tmpRow == 0 and isDomain[id1D] == True):
        fluidDistr[2, id1D] = dataTop[0, tmpCol]
        fluidDistr[5, id1D] = dataTop[1, tmpColB]
        fluidDistr[6, id1D] = dataTop[2, tmpColF]
    if (tmpRow == ny - 1 and isDomain[id1D] == True):
        fluidDistr[4, id1D] = dataBottom[0, tmpCol]
        fluidDistr[7, id1D] = dataBottom[1, tmpColF]
        fluidDistr[8, id1D] = dataBottom[2, tmpColB]

"""
Implement Multi-Relaxation Time algorithm. Transform f and f_eq to m and m_eq
"""
@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :], \
        float64[:, :], boolean[:])')
def calTransformFandFeq(nx, ny, fluidDistr, equilibriumFunc, collisionMatrix, \
    fluidDistrM, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    tmpFluidDistr = cuda.local.array(shape = 9, dtype = float64)
    tmpFluidEq = cuda.local.array(shape = 9, dtype = float64)
    if (isDomain[id1D] == True):
        for i in range(9):
            tmpResult = 0.
            tmpEq = 0.
            for j in range(9):
                tmpResult += collisionMatrix[i, j] * fluidDistr[j, id1D]
                tmpEq += collisionMatrix[i, j] * equilibriumFunc[j, id1D] 
            tmpFluidDistr[i] = tmpResult
            tmpFluidEq[i] = tmpEq
        for i in range(9):
            fluidDistrM[i, id1D] = tmpFluidDistr[i]
            equilibriumFunc[i, id1D] = tmpFluidEq[i]

@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :], \
    boolean[:])')
def calTransformForceTerm(nx, ny, forceTerm, collisionMatrix, forceTermM, \
    isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    tmpForceArray = cuda.local.array(shape = 9, dtype = float64)
    if (isDomain[id1D] == True):
        for i in range(9):
            tmpTrans = 0.
            for j in range(9):
                tmpTrans += collisionMatrix[i, j] * forceTerm[j, id1D]
            tmpForceArray[i] = tmpTrans
        for i in range(9):
            forceTermM[i, id1D] = tmpForceArray[i]

@cuda.jit('void(int64, int64, float64[:, :], float64[:, :], float64[:, :], \
    float64[:, :], float64[:, :], boolean[:])')        
def calFinalTransformEFMRT(nx, ny, fluidDistr, forceTerm, equilibriumFunc, \
    fluidDistrM,forceTermM, isDomain):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    totalNum = nx * ny
    
    if (isDomain[id1D] == True):
        for i in range(9):
            tmpCollision = 0.
            tmpCollision = (equilibriumFunc[i, id1D] - \
                fluidDistrM[i, id1D] - 1./2. * forceTermM[i, id1D])
            fluidDistr[i, id1D] = fluidDistr[i, id1D] + tmpCollision + 1. * \
            forceTerm[i, id1D]
                

#"""
#delete the effect of force
#"""
#@cuda.jit('void')

@cuda.jit('void(int64, int64, float64[:, :], boolean[:])')
def calBounceBack1D(nx, ny, fluidDistr, isSolid):
    totalNum = nx * ny
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; by = cuda.blockIdx.y
    bDimX = cuda.blockDim.x 
    xStart = tx + bx * bDimX
    yStart = by
    indicesK = nx * yStart + xStart
    if (isSolid[indicesK] == True  and indicesK < totalNum):
        if (xStart < nx and yStart < ny):
            tmp1 = fluidDistr[1, indicesK]; fluidDistr[1, indicesK] = fluidDistr[3, indicesK]
            fluidDistr[3, indicesK] = tmp1
            tmp2 = fluidDistr[2, indicesK]; fluidDistr[2, indicesK] = fluidDistr[4, indicesK]
            fluidDistr[4, indicesK] = tmp2
            tmp3 = fluidDistr[5, indicesK]; fluidDistr[5, indicesK] = fluidDistr[7, indicesK]
            fluidDistr[7, indicesK] = tmp3
            tmp4 = fluidDistr[6, indicesK]; fluidDistr[6, indicesK] = fluidDistr[8, indicesK]
            fluidDistr[8, indicesK] = tmp4
            
"""
half-wall boundary condition to reach second order accuracy
"""
@cuda.jit('void(int64, int64, float64[:, :], boolean[:], boolean[:])')
def calHalfWallBounceBack(nx, ny, fluidDistr, isDomain, isSolid):
    tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    idX = bx * bDimX + tx
    id1D = by * nx + idX
    if (isDomain[id1D] == True):
        tmpRow = int(id1D / nx); tmpCol = (id1D % nx)
        tmpRowL = tmpRow - 1 if (tmpRow > 0) else (ny - 1)
        tmpRowU = tmpRow + 1 if (tmpRow < ny - 1) else 0
        tmpColB = tmpCol - 1 if (tmpCol > 0) else (nx - 1)
        tmpColF = tmpCol + 1 if (tmpCol < nx - 1) else 0
        tmpIdF = tmpRow * nx + tmpColF; tmpIdB = tmpRow * nx + tmpColB
        tmpIdU = tmpRowU * nx + tmpCol; tmpIdL = tmpRowL * nx + tmpCol
        tmpIdFU = tmpRowU * nx + tmpColF; tmpIdBU = tmpRowU * nx + tmpColB
        tmpIdBL = tmpRowL * nx + tmpColB; tmpIdFL = tmpRowL * nx + tmpColF
        if (isSolid[tmpIdF] == True):
            fluidDistr[3, tmpIdF] = fluidDistr[1, id1D]
        if (isSolid[tmpIdU] == True):
            fluidDistr[4, tmpIdU] = fluidDistr[2, id1D]
        if (isSolid[tmpIdB] == True):
            fluidDistr[1, tmpIdB] = fluidDistr[3, id1D]
        if (isSolid[tmpIdL] == True):
            fluidDistr[2, tmpIdL] = fluidDistr[4, id1D]
        if (isSolid[tmpIdFU] == True):
            fluidDistr[7, tmpIdFU] = fluidDistr[5, id1D]
        if (isSolid[tmpIdBU] == True):
            fluidDistr[8, tmpIdBU] = fluidDistr[6, id1D]
        if (isSolid[tmpIdBL] == True):
            fluidDistr[5, tmpIdBL] = fluidDistr[7, id1D]
        if (isSolid[tmpIdFL] == True):
            fluidDistr[6, tmpIdFL] = fluidDistr[8, id1D]
        


@cuda.jit('void(int64, int64, int64, float64[:, :])')
def exchangeDistrGlobalGPU(nx, ny, numberThreads, fluidDistr):
    totalNum = nx * ny
    nbx = int(nx / numberThreads) + 1
#    nbx = int(math.ceil(nx / numberThreads))
    numberThread1 = cuda.blockDim.x
    by = cuda.blockIdx.y; tx = cuda.threadIdx.x; ty = cuda.threadIdx.y
    for i in range(0, nbx - 1):
        xStart = i * numberThreads
        xStartW = xStart + 2 * numberThreads - 1
        xTargetW = xStartW - numberThreads
        yStart = by * numberThread1 + tx
        kStartW = nx * yStart + xStartW
        kTargetW = nx * yStart + xTargetW
#        if (kTargetW < totalNum and kTargetW < totalNum):
        if (yStart < ny and (xStartW < nx and xTargetW < nx)):
            fluidDistr[3, kTargetW] = fluidDistr[3, kStartW]
            fluidDistr[6, kTargetW] = fluidDistr[6, kStartW]
            fluidDistr[7, kTargetW] = fluidDistr[7, kStartW]
#        if (kTargetW ==)
        
    for j in range(0, nbx - 1):
        tmpValue = nbx - 2
        i = tmpValue - j            
        xStart = i * numberThreads
        xStartE = xStart
        xTargetE = xStartE + numberThreads
        yStart = by * numberThread1 + tx
        kStartE = nx * yStart + xStartE
        kTargetE = nx * yStart + xTargetE
#        if (kStartE < totalNum and kTargetE < totalNum):
        if (yStart < ny and (xStartE < nx and xTargetE < nx)):
            fluidDistr[1, kTargetE] = fluidDistr[1, kStartE]
            fluidDistr[5, kTargetE] = fluidDistr[5, kStartE]
            fluidDistr[8, kTargetE] = fluidDistr[8, kStartE]
    #exchange across boundaries
        
#@cuda.jit()