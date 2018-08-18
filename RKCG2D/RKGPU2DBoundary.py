"""
File includes the boundaries for the inlet and outlet of the domain, except bounce-back boundary condition. Currently,
half bounce-back boundary condition is included in the streaming step.
"""

"""
Calculate Neumann boundary condition with Zou-He method
"""
@cuda.jit('void(int64, int64, int64, int64, float64, float64, int64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])')
def constantVelocityZHBoundaryHigherRK(totalNodes, nx, ny, xDim, \
                                       specificVYR, specificVYB, fluidNodes, fluidRhoR, \
                                       fluidRhoB, fluidPDFR, fluidPDFB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < (ny - 1) * nx and tmpIndex >= (ny - 2) * nx):
            fluidRhoR[indices] = (fluidPDFR[indices, 0] + fluidPDFR[indices, 1] + \
                                  fluidPDFR[indices, 3] + 2. * (fluidPDFR[indices, 2] + \
                                                                fluidPDFR[indices, 5] + fluidPDFR[indices, 6])) / \
                                 (1. + specificVYR)
            fluidPDFR[indices, 4] = fluidPDFR[indices, 2] - 2. / 3. * \
                                    fluidRhoR[indices] * specificVYR
            fluidPDFR[indices, 7] = fluidPDFR[indices, 5] + \
                                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) / 2. - \
                                    1. / 6. * fluidRhoR[indices] * specificVYR
            fluidPDFR[indices, 8] = fluidPDFR[indices, 6] - \
                                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) / 2. - \
                                    1. / 6. * fluidRhoR[indices] * specificVYR

            fluidRhoB[indices] = (fluidPDFB[indices, 0] + fluidPDFB[indices, 1] + \
                                  fluidPDFB[indices, 3] + 2. * (fluidPDFB[indices, 2] + \
                                                                fluidPDFB[indices, 5] + fluidPDFB[indices, 6])) / \
                                 (1. + specificVYB)
            fluidPDFB[indices, 4] = fluidPDFB[indices, 2] - 2. / 3. * \
                                    fluidRhoB[indices] * specificVYB
            fluidPDFB[indices, 7] = fluidPDFB[indices, 5] + \
                                    (fluidPDFB[indices, 1] - fluidPDFB[indices, 3]) / 2. - \
                                    1. / 6. * fluidRhoB[indices] * specificVYB
            fluidPDFB[indices, 8] = fluidPDFB[indices, 6] - \
                                    (fluidPDFB[indices, 1] - fluidPDFB[indices, 3]) / 2. - \
                                    1. / 6. * fluidRhoB[indices] * specificVYB
    cuda.syncthreads()


"""
Update the ghost points with constant velocity boundary
"""


@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])')
def ghostPointsConstantVelocityRK(totalNodes, nx, ny, xDim, fluidNodes, \
                                  neighboringNodes, fluidRhoR, fluidRhoB, fluidPDFR, \
                                  fluidPDFB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < ny * nx and tmpIndex >= (ny - 1) * nx):
            tmpStart = 8 * indices + 3
            tmpL = neighboringNodes[tmpStart]

            fluidPDFR[indices, 0] = fluidPDFR[tmpL, 0]
            fluidPDFR[indices, 1] = fluidPDFR[tmpL, 1]
            fluidPDFR[indices, 2] = fluidPDFR[tmpL, 2]
            fluidPDFR[indices, 3] = fluidPDFR[tmpL, 3]
            fluidPDFR[indices, 4] = fluidPDFR[tmpL, 4]
            fluidPDFR[indices, 5] = fluidPDFR[tmpL, 5]
            fluidPDFR[indices, 6] = fluidPDFR[tmpL, 6]
            fluidPDFR[indices, 7] = fluidPDFR[tmpL, 7]
            fluidPDFR[indices, 8] = fluidPDFR[tmpL, 8]
            fluidRhoR[indices] = fluidPDFR[indices, 0] + fluidPDFR[indices, 1] + \
                                 fluidPDFR[indices, 2] + fluidPDFR[indices, 3] + \
                                 fluidPDFR[indices, 4] + fluidPDFR[indices, 5] + \
                                 fluidPDFR[indices, 6] + fluidPDFR[indices, 7] + \
                                 fluidPDFR[indices, 8]

            fluidPDFB[indices, 0] = fluidPDFB[tmpL, 0]
            fluidPDFB[indices, 1] = fluidPDFB[tmpL, 1]
            fluidPDFB[indices, 2] = fluidPDFB[tmpL, 2]
            fluidPDFB[indices, 3] = fluidPDFB[tmpL, 3]
            fluidPDFB[indices, 4] = fluidPDFB[tmpL, 4]
            fluidPDFB[indices, 5] = fluidPDFB[tmpL, 5]
            fluidPDFB[indices, 6] = fluidPDFB[tmpL, 6]
            fluidPDFB[indices, 7] = fluidPDFB[tmpL, 7]
            fluidPDFB[indices, 8] = fluidPDFB[tmpL, 8]
            fluidRhoB[indices] = fluidPDFB[indices, 0] + fluidPDFB[indices, 1] + \
                                 fluidPDFB[indices, 2] + fluidPDFB[indices, 3] + \
                                 fluidPDFB[indices, 4] + fluidPDFB[indices, 5] + \
                                 fluidPDFB[indices, 6] + fluidPDFB[indices, 7] + \
                                 fluidPDFB[indices, 8]
    cuda.syncthreads()


"""
Calculate the outlet boundary with convective flow method. 
"""


@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:, :], float64[:, :], \
                float64[:], float64[:])')
def convectiveOutletGPU(totalNodes, nx, xDim, fluidNodes, neighboringNodes, \
                        fluidPDFR, fluidPDFB, fluidRhoR, fluidRhoB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    # calculate the average velocity
    tmpSumV = 0.
    tmpStart = 3 * nx

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 3 * nx and tmpIndex >= 2 * nx):
            tmpNeighbor = neighboringNodes[8 * indices + 1]
            fluidRhoR[indices] = 0.;
            fluidRhoB[indices] = 0.
            for j in range(9):
                fluidPDFR[indices, j] = fluidPDFR[tmpNeighbor, j]
                fluidPDFB[indices, j] = fluidPDFB[tmpNeighbor, j]
                #                fluidPDFR[indices, j] = fluidPDFR[indices + nx - 2, j]
                #                fluidPDFB[indices, j] = fluidPDFB[indices + nx - 2, j]
                #                               (fluidPDFOld[i, indices, j] + \
                #                            averageV * fluidPDFNew[i, indices + nx, j]) / (1. + \
                #                            averageV)
                fluidRhoR[indices] += fluidPDFR[indices, j]
                fluidRhoB[indices] += fluidPDFB[indices, j]
    cuda.syncthreads()


"""
Calculate the outlet boundary ghost nodes in second layer with convective flow method. 
"""


@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:, :], float64[:, :], \
                float64[:], float64[:])')
def convectiveOutletGhost2GPU(totalNodes, nx, xDim, fluidNodes, neighboringNodes, \
                              fluidPDFR, fluidPDFB, fluidRhoR, fluidRhoB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    # calculate the average velocity
    tmpSumV = 0.
    tmpStart = 3 * nx

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 2 * nx and tmpIndex >= nx):
            tmpNeighbor = neighboringNodes[8 * indices + 1]
            fluidRhoR[indices] = 0.;
            fluidRhoB[indices] = 0.
            for j in range(9):
                fluidPDFR[indices, j] = fluidPDFR[tmpNeighbor, j]
                fluidPDFB[indices, j] = fluidPDFB[tmpNeighbor, j]
                #                fluidPDFR[indices, j] = fluidPDFR[indices + nx -2, j]
                ##                               (fluidPDFOld[i, indices, j] + \
                ##                            averageV * fluidPDFNew[i, indices + nx, j]) / (1. + \
                ##                            averageV)
                #                fluidPDFB[indices, j] = fluidPDFB[indices + nx -2, j]
                fluidRhoR[indices] += fluidPDFR[indices, j]
                fluidRhoB[indices] += fluidPDFB[indices, j]
    cuda.syncthreads()


#
"""
Calculate the outlet boundary ghost nodes in first layer with convective flow method. 
"""


@cuda.jit('void(int64, int64, int64, int64[:], int64[:], float64[:, :], float64[:, :], \
                float64[:], float64[:])')
def convectiveOutletGhost3GPU(totalNodes, nx, xDim, fluidNodes, neighboringNodes, \
                              fluidPDFR, fluidPDFB, fluidRhoR, fluidRhoB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx
    # calculate the average velocity
    tmpSumV = 0.
    tmpStart = 3 * nx

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < nx and tmpIndex >= 0):
            tmpNeighbor = neighboringNodes[8 * indices + 1]
            fluidRhoR[indices] = 0.;
            fluidRhoB[indices] = 0.
            for j in range(9):
                fluidPDFR[indices, j] = fluidPDFR[tmpNeighbor, j]
                fluidPDFB[indices, j] = fluidPDFB[tmpNeighbor, j]
                #                               (fluidPDFOld[i, indices, j] + \
                #                            averageV * fluidPDFNew[i, indices + nx, j]) / (1. + \
                #                            averageV)
                fluidRhoR[indices] += fluidPDFR[indices, j]
                fluidRhoB[indices] += fluidPDFB[indices, j]
    cuda.syncthreads()


"""
Calculate the boundary flow from convective-average method
"""


@cuda.jit("void(int64, int64, int64, int64[:], int64[:], float64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:, :])")
def convectiveAverageBoundaryGPU(totalNodes, nx, xDim, fluidNodes, neighboringNodes, normalVelocity, \
                                 fluidPDFR, fluidPDFB, fluidPDFROld, fluidPDFBOld):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalNodes:
        #        tmpSumVNorm = 0.
        #        for i in range(nx - 2):
        #            tmpSumVNorm += normalVelocity[3 * (nx - 2) + i]
        #        averageNormalV = tmpSumVNorm / (nx - 2)
        # for Nth layer
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 3 * nx and tmpIndex >= 2 * nx):
            tmpNeighbor = neighboringNodes[8 * indices + 1]
            averageNormalV = abs(normalVelocity[tmpNeighbor])
            for j in range(9):
                fluidPDFR[indices, j] = (fluidPDFROld[indices, j] + averageNormalV * \
                                         fluidPDFR[tmpNeighbor, j]) / (1. + averageNormalV)
                fluidPDFB[indices, j] = (fluidPDFBOld[indices, j] + averageNormalV * \
                                         fluidPDFB[tmpNeighbor, j]) / (1. + averageNormalV)
            normalVelocity[indices] = normalVelocity[tmpNeighbor]
    cuda.syncthreads()


"""
Calculate the boundary flow from convective-average method
"""
@cuda.jit("void(int64, int64, int64, int64[:], int64[:], float64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:, :])")
def convectiveAverageBoundaryGPU2(totalNodes, nx, xDim, fluidNodes, neighboringNodes, normalVelocity, \
                                  fluidPDFR, fluidPDFB, fluidPDFROld, fluidPDFBOld):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalNodes:
        #        tmpSumVNorm = 0.
        #        for i in range(nx - 2):
        #            tmpSumVNorm += normalVelocity[3 * (nx - 2) + i]
        #        averageNormalV = tmpSumVNorm / (nx - 2)
        #        #for Nth layer
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < 2 * nx and tmpIndex >= nx):
            tmpNeighbor1 = neighboringNodes[8 * indices + 1]
            tmpNeighbor = neighboringNodes[8 * tmpNeighbor1 + 1]
            #            averageNormalV = abs(normalVelocity[indices + 2 * (nx - 2)])
            averageNormalV = abs(normalVelocity[tmpNeighbor])
            for j in range(9):
                fluidPDFR[indices, j] = (fluidPDFROld[indices, j] + averageNormalV * \
                                         fluidPDFR[tmpNeighbor1, j]) / (1. + averageNormalV)
                fluidPDFB[indices, j] = (fluidPDFBOld[indices, j] + averageNormalV * \
                                         fluidPDFB[tmpNeighbor1, j]) / (1. + averageNormalV)
            #            normalVelocity[indices] = normalVelocity[indices + 2 * (nx - 2)]
            normalVelocity[indices] = normalVelocity[tmpNeighbor]
    cuda.syncthreads()


"""
Calculate the boundary flow from convective-average method
"""
@cuda.jit("void(int64, int64, int64, int64[:], int64[:], float64[:], float64[:, :], float64[:, :], \
                float64[:, :], float64[:, :])")
def convectiveAverageBoundaryGPU3(totalNodes, nx, xDim, fluidNodes, neighboringNodes, normalVelocity, \
                                  fluidPDFR, fluidPDFB, fluidPDFROld, fluidPDFBOld):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalNodes:
        #        tmpSumVNorm = 0.
        #        for i in range(nx - 2):
        #            tmpSumVNorm += normalVelocity[3 * (nx - 2) + i]
        #        averageNormalV = tmpSumVNorm / (nx - 2)
        # for Nth layer
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < nx and tmpIndex >= 0):
            tmpNeighbor2 = neighboringNodes[8 * indices + 1]
            tmpNeighbor1 = neighboringNodes[8 * tmpNeighbor2 + 1]
            tmpNeighbor = neighboringNodes[8 * tmpNeighbor1 + 1]
            #            averageNormalV = abs(normalVelocity[indices + 3 * (nx - 2)])
            averageNormalV = abs(normalVelocity[tmpNeighbor])
            for j in range(9):
                fluidPDFR[indices, j] = (fluidPDFROld[indices, j] + averageNormalV * \
                                         fluidPDFR[tmpNeighbor2, j]) / (1. + averageNormalV)
                fluidPDFB[indices, j] = (fluidPDFBOld[indices, j] + averageNormalV * \
                                         fluidPDFB[tmpNeighbor2, j]) / (1. + averageNormalV)
            #            normalVelocity[indices] = normalVelocity[indices + 3 * (nx - 2)]
            normalVelocity[indices] = normalVelocity[tmpNeighbor]
    cuda.syncthreads()


"""
Constant pressure boundary condition at the inlet of the domain
"""
@cuda.jit("void(int64, int64, int64, int64, float64, float64, int64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])")
def calConstPressureInletGPU(totalNodes, nx, ny, xDim, constPHB, constPHR, fluidNodes, \
                             fluidRhoB, fluidRhoR, fluidPDFB, fluidPDFR):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalNodes:
        tmpIndex = fluidNodes[indices]
        if tmpIndex >= (ny - 2) * nx and tmpIndex < (ny - 1) * nx:
            # for fluid B
            tmpVB = -1. + (fluidPDFB[indices, 0] + \
                           fluidPDFB[indices, 1] + fluidPDFB[indices, 3] + \
                           2. * (fluidPDFB[indices, 2] + fluidPDFB[indices, 5] + \
                                 fluidPDFB[indices, 6])) / constPHB
            fluidPDFB[indices, 4] = fluidPDFB[indices, 2] - 2. / 3. * \
                                    constPHB * tmpVB
            fluidPDFB[indices, 7] = fluidPDFB[indices, 5] + 1. / 2. * \
                                    (fluidPDFB[indices, 1] - fluidPDFB[indices, 3]) - \
                                    1. / 6. * constPHB * tmpVB
            fluidPDFB[indices, 8] = fluidPDFB[indices, 6] - 1. / 2. * \
                                    (fluidPDFB[indices, 1] - fluidPDFB[indices, 3]) - \
                                    1. / 6. * constPHB * tmpVB
            fluidRhoB[indices] = constPHB
            # for fluid R
            tmpVR = -1. + (fluidPDFR[indices, 0] + \
                           fluidPDFR[indices, 1] + fluidPDFR[indices, 3] + \
                           2. * (fluidPDFR[indices, 2] + fluidPDFR[indices, 5] + \
                                 fluidPDFR[indices, 6])) / constPHR
            fluidPDFR[indices, 4] = fluidPDFR[indices, 2] - 2. / 3. * \
                                    constPHR * tmpVR
            fluidPDFR[indices, 7] = fluidPDFR[indices, 5] + 1. / 2. * \
                                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) - \
                                    1. / 6. * constPHR * tmpVR
            fluidPDFR[indices, 8] = fluidPDFR[indices, 6] - 1. / 2. * \
                                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) - \
                                    1. / 6. * constPHR * tmpVR
            fluidRhoR[indices] = constPHR


"""
Ghost nodes on the inlet boundary for the constant pressure condition
"""
@cuda.jit('void(int64, int64, int64, int64, int64[:], int64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])')
def ghostPointsConstPressureInletRK(totalNodes, nx, ny, xDim, fluidNodes, \
                                    neighboringNodes, fluidRhoR, fluidRhoB, fluidPDFR, \
                                    fluidPDFB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if tmpIndex >= (ny - 1) * nx and tmpIndex < ny * nx:
            tmpStart = 8 * indices + 3
            tmpH = neighboringNodes[tmpStart]
            fluidPDFR[indices, 0] = fluidPDFR[tmpH, 0]
            fluidPDFR[indices, 1] = fluidPDFR[tmpH, 1]
            fluidPDFR[indices, 2] = fluidPDFR[tmpH, 2]
            fluidPDFR[indices, 3] = fluidPDFR[tmpH, 3]
            fluidPDFR[indices, 4] = fluidPDFR[tmpH, 4]
            fluidPDFR[indices, 5] = fluidPDFR[tmpH, 5]
            fluidPDFR[indices, 6] = fluidPDFR[tmpH, 6]
            fluidPDFR[indices, 7] = fluidPDFR[tmpH, 7]
            fluidPDFR[indices, 8] = fluidPDFR[tmpH, 8]
            fluidRhoR[indices] = fluidRhoR[tmpH]

            fluidPDFB[indices, 0] = fluidPDFB[tmpH, 0]
            fluidPDFB[indices, 1] = fluidPDFB[tmpH, 1]
            fluidPDFB[indices, 2] = fluidPDFB[tmpH, 2]
            fluidPDFB[indices, 3] = fluidPDFB[tmpH, 3]
            fluidPDFB[indices, 4] = fluidPDFB[tmpH, 4]
            fluidPDFB[indices, 5] = fluidPDFB[tmpH, 5]
            fluidPDFB[indices, 6] = fluidPDFB[tmpH, 6]
            fluidPDFB[indices, 7] = fluidPDFB[tmpH, 7]
            fluidPDFB[indices, 8] = fluidPDFB[tmpH, 8]
            fluidRhoB[indices] = fluidRhoB[tmpH]
    cuda.syncthreads()


"""
Constant pressure boundary condition at the outlet of the domain
"""
@cuda.jit("void(int64, int64, int64, float64, float64, int64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])")
def calConstPressureLowerGPU(totalNodes, nx, xDim, constPLB, constPLR, fluidNodes, \
                             fluidRhoB, fluidRhoR, fluidPDFB, fluidPDFR):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalNodes:
        tmpIndex = fluidNodes[indices]
        if tmpIndex >= nx and tmpIndex < 2 * nx:
            # for fluid B
            tmpVB = 1. - 1. / constPLB * (fluidPDFB[indices, 0] + fluidPDFB[indices, 1] + \
                                          fluidPDFB[indices, 3] + 2. * (fluidPDFB[indices, 4] + \
                                                                        fluidPDFB[indices, 7] + fluidPDFB[indices, 8]))
            fluidPDFB[indices, 2] = fluidPDFB[indices, 4] + 2. / 3. * (constPLB * tmpVB)
            fluidPDFB[indices, 5] = fluidPDFB[indices, 7] + 0.5 * (fluidPDFB[indices, 3] - \
                                                                   fluidPDFB[indices, 1]) + 1. / 6. * constPLB * tmpVB
            fluidPDFB[indices, 6] = fluidPDFB[indices, 8] + 0.5 * (fluidPDFB[indices, 1] - \
                                                                   fluidPDFB[indices, 3]) + 1. / 6. * constPLB * tmpVB
            fluidRhoB[indices] = constPLB
            # for fluid R
            tmpVR = 1. - 1. / constPLR * (fluidPDFR[indices, 0] + fluidPDFR[indices, 1] + \
                                          fluidPDFR[indices, 3] + 2. * (fluidPDFR[indices, 4] + \
                                                                        fluidPDFR[indices, 7] + fluidPDFR[indices, 8]))
            fluidPDFR[indices, 2] = fluidPDFR[indices, 4] + 2. / 3. * constPLR * tmpVR
            fluidPDFR[indices, 5] = fluidPDFR[indices, 7] + 0.5 * (fluidPDFR[indices, 3] - \
                                                                   fluidPDFR[indices, 1]) + 1. / 6. * constPLR * tmpVR
            fluidPDFR[indices, 6] = fluidPDFR[indices, 8] + 0.5 * (fluidPDFR[indices, 1] - \
                                                                   fluidPDFR[indices, 3]) + 1. / 6. * constPLR * tmpVR
            fluidRhoR[indices] = constPLR


"""
Ghost nodes on the lower boundary for the constant pressure condition
"""
@cuda.jit('void(int64, int64, int64, int64[:], int64[:], \
                float64[:], float64[:], float64[:, :], float64[:, :])')
def ghostPointsConstPressureLowerRK(totalNodes, nx, xDim, fluidNodes, \
                                    neighboringNodes, fluidRhoR, fluidRhoB, fluidPDFR, \
                                    fluidPDFB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        tmpIndex = fluidNodes[indices]
        if tmpIndex < nx:
            tmpStart = 8 * indices + 1
            tmpL = neighboringNodes[tmpStart]

            fluidPDFR[indices, 0] = fluidPDFR[tmpL, 0]
            fluidPDFR[indices, 1] = fluidPDFR[tmpL, 1]
            fluidPDFR[indices, 2] = fluidPDFR[tmpL, 2]
            fluidPDFR[indices, 3] = fluidPDFR[tmpL, 3]
            fluidPDFR[indices, 4] = fluidPDFR[tmpL, 4]
            fluidPDFR[indices, 5] = fluidPDFR[tmpL, 5]
            fluidPDFR[indices, 6] = fluidPDFR[tmpL, 6]
            fluidPDFR[indices, 7] = fluidPDFR[tmpL, 7]
            fluidPDFR[indices, 8] = fluidPDFR[tmpL, 8]
            fluidRhoR[indices] = fluidRhoR[tmpL]

            fluidPDFB[indices, 0] = fluidPDFB[tmpL, 0]
            fluidPDFB[indices, 1] = fluidPDFB[tmpL, 1]
            fluidPDFB[indices, 2] = fluidPDFB[tmpL, 2]
            fluidPDFB[indices, 3] = fluidPDFB[tmpL, 3]
            fluidPDFB[indices, 4] = fluidPDFB[tmpL, 4]
            fluidPDFB[indices, 5] = fluidPDFB[tmpL, 5]
            fluidPDFB[indices, 6] = fluidPDFB[tmpL, 6]
            fluidPDFB[indices, 7] = fluidPDFB[tmpL, 7]
            fluidPDFB[indices, 8] = fluidPDFB[tmpL, 8]
            fluidRhoB[indices] = fluidRhoB[tmpL]
    cuda.syncthreads()


"""
Constant pressure boundary condition on the inlet of the domain
"""
@cuda.jit("void(int64, int64, int64, int64, float64, float64, int64[:], \
                float64[:, :], float64[:, :])")
def calConstPressureHighGPU(totalNodes, nx, ny, xDim, constPHB, constPHR, fluidNodes, \
                            fluidPDFB, fluidPDFR):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if indices < totalNodes:
        tmpIndex = fluidNodes[indices]
        if tmpIndex >= (ny - 1) * nx and tmpIndex < ny * nx:
            # for fluid B
            tmpVB = -1. + 1. / constPHB * (fluidPDFB[indices, 0] + fluidPDFB[indices, 1] + \
                                           fluidPDFB[indices, 3] + 2. * (fluidPDFB[indices, 2] + \
                                                                         fluidPDFB[indices, 5] + fluidPDFB[indices, 6]))
            fluidPDFB[indices, 4] = fluidPDFB[indices, 2] - 2. / 3. * constPHB * \
                                    tmpVB
            fluidPDFB[indices, 7] = fluidPDFB[indices, 5] - 0.5 * (fluidPDFB[indices, 3] - \
                                                                   fluidPDFB[indices, 1]) - 1. / 6. * constPHB * tmpVB
            fluidPDFB[indices, 8] = fluidPDFB[indices, 6] - 0.5 * (fluidPDFB[indices, 1] - \
                                                                   fluidPDFB[indices, 3]) - 1. / 6. * constPHB * tmpVB
            # for fluid R
            tmpVR = -1. + 1. / constPHR * (fluidPDFR[indices, 0] + fluidPDFR[indices, 1] + \
                                           fluidPDFR[indices, 3] + 2. * (fluidPDFR[indices, 2] + \
                                                                         fluidPDFR[indices, 5] + fluidPDFR[indices, 6]))
            fluidPDFR[indices, 4] = fluidPDFR[indices, 2] - 2. / 3. * constPHR * \
                                    tmpVR
            fluidPDFR[indices, 7] = fluidPDFR[indices, 5] - 0.5 * (fluidPDFR[indices, 3] - \
                                                                   fluidPDFR[indices, 1]) - 1. / 6. * constPHR * tmpVR
            fluidPDFR[indices, 8] = fluidPDFR[indices, 6] - 0.5 * (fluidPDFR[indices, 1] - \
                                                                   fluidPDFR[indices, 3]) - 1. / 6. * constPHR * tmpVR


"""
Calculate Neumann boundary condition with Zou-He method, but no flow for the 
other fluid
"""
@cuda.jit('void(int64, int64, int64, int64, float64, float64, int64[:], \
                int64[:], float64[:], float64[:], float64[:, :], float64[:, :])')
def constantVelocityZHBoundaryHigherNewRK(totalNodes, nx, ny, xDim, \
                                          specificVYR, specificVYB, fluidNodes, \
                                          neighboringNodes, fluidRhoR, \
                                          fluidRhoB, fluidPDFR, fluidPDFB):
    tx = cuda.threadIdx.x;
    bx = cuda.blockIdx.x;
    bDimX = cuda.blockDim.x
    by = cuda.blockIdx.y
    indices = by * xDim + bx * bDimX + tx

    if (indices < totalNodes):
        tmpStart = 8 * indices
        tmpIndex = fluidNodes[indices]
        if (tmpIndex < (ny - 1) * nx and tmpIndex >= (ny - 2) * nx):
            fluidRhoR[indices] = (fluidPDFR[indices, 0] + fluidPDFR[indices, 1] + \
                                  fluidPDFR[indices, 3] + 2. * (fluidPDFR[indices, 2] + \
                                                                fluidPDFR[indices, 5] + fluidPDFR[indices, 6])) / \
                                 (1. + specificVYR)
            fluidPDFR[indices, 4] = fluidPDFR[indices, 2] - 2. / 3. * \
                                    fluidRhoR[indices] * specificVYR
            fluidPDFR[indices, 7] = fluidPDFR[indices, 5] + \
                                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) / 2. - \
                                    1. / 6. * fluidRhoR[indices] * specificVYR
            fluidPDFR[indices, 8] = fluidPDFR[indices, 6] - \
                                    (fluidPDFR[indices, 1] - fluidPDFR[indices, 3]) / 2. - \
                                    1. / 6. * fluidRhoR[indices] * specificVYR

            # for retreating fluid
            tmpUpper = neighboringNodes[tmpStart + 1]
            fluidPDFB[indices, 4] = fluidPDFB[tmpUpper, 2]
            tmpFor7 = neighboringNodes[tmpUpper * 8]
            tmpFor8 = neighboringNodes[tmpUpper * 8 + 2]
            if tmpFor7 >= 0:
                fluidPDFB[indices, 7] = fluidPDFB[tmpFor7, 5]
            if tmpFor8 >= 0:
                fluidPDFB[indices, 8] = fluidPDFB[tmpFor8, 6]