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
/*! \internal \file
 * \brief CUDA evaluation of the shared GaMD bias and diagnostic force overrides.
 *
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "gamd_scale_gpu.h"

#include "gromacs/gpu_utils/cudautils.cuh"

#include "gamdmath.h"

//! \cond internal
namespace gmx
{

namespace
{

/*! \internal \brief Evaluates the shared bias and writes the six-float device state. */
__global__ void gamdProductionScaleKernel(float*                   gm_gamdState,
                                          const GaMDBiasParameters parameters,
                                          const GaMDForceOverrides forceOverrides)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        GaMDStepResult result;
        if (forceOverrides.enabled && parameters.boostType != 0)
        {
            // CPU 和 CUDA 共用诊断配置，让力、维里和输出都读取同一份结果。
            result.scaleP = forceOverrides.totalScale;
            result.scaleD = forceOverrides.dihedralScale;
            result.boostP = forceOverrides.potentialBoost;
            result.boostD = forceOverrides.dihedralBoost;
        }
        else
        {
            // 普通 GaMD 继续使用共享公式，保持原有计算顺序。
            result = evaluateGaMDBias(
                    parameters, gm_gamdState[c_gaMDPotentialEnergy], gm_gamdState[c_gaMDDihedralEnergy]);
        }

        gm_gamdState[c_gaMDPotentialScale] = static_cast<float>(fmax(c_gaMDScaleFloor, result.scaleP));
        gm_gamdState[c_gaMDDihedralScale] = static_cast<float>(fmax(c_gaMDScaleFloor, result.scaleD));
        gm_gamdState[c_gaMDPotentialBoost] = static_cast<float>(result.boostP);
        gm_gamdState[c_gaMDDihedralBoost]  = static_cast<float>(result.boostD);
    }
}

} // namespace

void launchGamdProductionScaleKernel(DeviceBuffer<float>       gamdState,
                                     const GaMDBiasParameters& parameters,
                                     const GaMDForceOverrides& forceOverrides,
                                     const DeviceStream&       deviceStream)
{
    KernelLaunchConfig config;
    config.blockSize[0]     = 1;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = 1;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel = gamdProductionScaleKernel;
    const auto kernelArgs =
            prepareGpuKernelArguments(kernel, config, &gamdState, &parameters, &forceOverrides);
    launchGpuKernel(kernel, config, deviceStream, nullptr, "Evaluate GaMD production scale", kernelArgs);
}

} // namespace gmx
//! \endcond
