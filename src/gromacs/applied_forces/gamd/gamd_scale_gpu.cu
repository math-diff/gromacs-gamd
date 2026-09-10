/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

#include "gmxpre.h"

#include "gamd_scale_gpu.h"

#include "gromacs/gpu_utils/cudautils.cuh"

#include "gamdmath.h"

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
        const GaMDBiasParameters parameters{ igamd, stage, thresholdP, kP, thresholdD, kD };
        // 直接在设备端调用同一份公式，保持原有缓冲区和内核启动顺序。
        const GaMDStepResult result = evaluateGaMDBias(parameters, gm_gamdState[0], gm_gamdState[1]);

        gm_gamdState[2] = static_cast<float>(fmax(c_gaMDScaleFloor, result.scaleP));
        gm_gamdState[3] = static_cast<float>(fmax(c_gaMDScaleFloor, result.scaleD));
        gm_gamdState[4] = static_cast<float>(result.boostP);
        gm_gamdState[5] = static_cast<float>(result.boostD);
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
