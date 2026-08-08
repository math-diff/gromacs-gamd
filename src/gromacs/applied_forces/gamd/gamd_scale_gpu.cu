/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 */

#include "gmxpre.h"

#include "gamd_scale_gpu.h"

#include "gromacs/gpu_utils/cudautils.cuh"

namespace gmx
{

namespace
{

__global__ void gamdProductionScaleKernel(float*       gm_gamdState,
                                          const int    igamd,
                                          const int    stage,
                                          const double thresholdP,
                                          const double kP,
                                          const double thresholdD,
                                          const double kD)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const double totalEnergy    = gm_gamdState[0];
        const double dihedralEnergy = gm_gamdState[1];
        double       scaleP         = 1.0;
        double       scaleD         = 1.0;
        double       boostP         = 0.0;
        double       boostD         = 0.0;

        if (stage >= 3)
        {
            if ((igamd == 2 || igamd == 3) && dihedralEnergy < thresholdD)
            {
                const double deltaD = thresholdD - dihedralEnergy;
                boostD              = 0.5 * kD * deltaD * deltaD;
                scaleD              = 1.0 - kD * deltaD;
            }

            const double totalEnergyForP = totalEnergy + (igamd == 3 ? boostD : 0.0);
            if ((igamd == 1 || igamd == 3) && totalEnergyForP < thresholdP)
            {
                const double deltaP = thresholdP - totalEnergyForP;
                boostP              = 0.5 * kP * deltaP * deltaP;
                scaleP              = 1.0 - kP * deltaP;
            }
        }

        gm_gamdState[2] = static_cast<float>(fmax(0.01, scaleP));
        gm_gamdState[3] = static_cast<float>(fmax(0.01, scaleD));
        gm_gamdState[4] = static_cast<float>(boostP);
        gm_gamdState[5] = static_cast<float>(boostD);
    }
}

} // namespace

void launchGamdProductionScaleKernel(DeviceBuffer<float> gamdState,
                                     int                 igamd,
                                     int                 stage,
                                     double              thresholdP,
                                     double              kP,
                                     double              thresholdD,
                                     double              kD,
                                     const DeviceStream& deviceStream)
{
    KernelLaunchConfig config;
    config.blockSize[0]     = 1;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = 1;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel     = gamdProductionScaleKernel;
    const auto kernelArgs = prepareGpuKernelArguments(
            kernel, config, &gamdState, &igamd, &stage, &thresholdP, &kP, &thresholdD, &kD);
    launchGpuKernel(kernel, config, deviceStream, nullptr, "Evaluate GaMD production scale", kernelArgs);
}

} // namespace gmx
