/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2018- The GROMACS Authors
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
 * \brief CUDA force, energy, virial and kinetic-history operations for GaMD.
 *
 * \author Jon Vincent <jvincent@nvidia.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include <math_constants.h>

#include <cassert>

#include <algorithm>

#include "gromacs/applied_forces/gamd/gamd_scale_gpu.h"
#include "gromacs/applied_forces/gamd/gamdgpu_impl.h"
#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/typecasts_cuda_hip.h"
#include "gromacs/gpu_utils/vectype_ops_cuda.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdlib/gpuforcereduction_impl_internal.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"

//! \cond internal
namespace gmx
{
namespace
{

//! Copies PME energy and the symmetric virial into the GaMD staging layout.
__global__ void stageGamdReciprocalEnergyKernel(const float* gm_virialAndEnergy, float* gm_destination)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        gm_destination[0]                 = 0.5F * gm_virialAndEnergy[6];
        gm_destination[1 + XX * DIM + XX] = 0.25F * gm_virialAndEnergy[0];
        gm_destination[1 + YY * DIM + YY] = 0.25F * gm_virialAndEnergy[1];
        gm_destination[1 + ZZ * DIM + ZZ] = 0.25F * gm_virialAndEnergy[2];
        gm_destination[1 + XX * DIM + YY] = 0.25F * gm_virialAndEnergy[3];
        gm_destination[1 + YY * DIM + XX] = 0.25F * gm_virialAndEnergy[3];
        gm_destination[1 + XX * DIM + ZZ] = 0.25F * gm_virialAndEnergy[4];
        gm_destination[1 + ZZ * DIM + XX] = 0.25F * gm_virialAndEnergy[4];
        gm_destination[1 + YY * DIM + ZZ] = 0.25F * gm_virialAndEnergy[5];
        gm_destination[1 + ZZ * DIM + YY] = 0.25F * gm_virialAndEnergy[5];
    }
}


// 这些核在 update stream 上记录约束维里和动能历史，避免 GaMD 状态散落在 update 模块中。
/*! \internal \brief Appends a constraint-virial sample using the device-side slot counter. */
__global__ void recordConstraintVirialKernel(const float* gm_lincsVirial,
                                             const bool   haveLincs,
                                             const float* gm_settleVirial,
                                             const bool   haveSettles,
                                             const float  scaleFactor,
                                             int*         gm_historyCount,
                                             const int    historyCapacity,
                                             float*       gm_history)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const int sampleIndex = atomicAdd(gm_historyCount, 1);
        if (sampleIndex >= historyCapacity)
        {
            return;
        }
        float compact[6] = {};
        for (int component = 0; component < 6; ++component)
        {
            if (haveLincs)
            {
                compact[component] += gm_lincsVirial[component];
            }
            if (haveSettles)
            {
                compact[component] += gm_settleVirial[component];
            }
            compact[component] *= scaleFactor;
        }
        float* sample         = gm_history + sampleIndex * DIM * DIM;
        sample[XX * DIM + XX] = compact[0];
        sample[XX * DIM + YY] = compact[1];
        sample[XX * DIM + ZZ] = compact[2];
        sample[YY * DIM + XX] = compact[1];
        sample[YY * DIM + YY] = compact[3];
        sample[YY * DIM + ZZ] = compact[4];
        sample[ZZ * DIM + XX] = compact[2];
        sample[ZZ * DIM + YY] = compact[4];
        sample[ZZ * DIM + ZZ] = compact[5];
    }
}

/*! \internal \brief Reduces half-step kinetic tensors and temperature-group energies. */
__global__ void accumulateHalfStepKineticKernel(const int             numAtoms,
                                                const int             numGroups,
                                                const float3*         gm_velocities,
                                                const float*          gm_masses,
                                                const unsigned short* gm_temperatureGroups,
                                                float*                gm_halfKinetic)
{
    const int               atom          = blockIdx.x * blockDim.x + threadIdx.x;
    const int               numComponents = 6 + numGroups;
    extern __shared__ float sm_warpSums[];
    float                   values[8] = {};
    int                     group     = 0;
    if (atom < numAtoms)
    {
        group                 = gm_temperatureGroups[atom];
        const float3 v        = gm_velocities[atom];
        const float  halfMass = 0.5F * gm_masses[atom];
        values[0]             = halfMass * v.x * v.x;
        values[1]             = halfMass * v.x * v.y;
        values[2]             = halfMass * v.x * v.z;
        values[3]             = halfMass * v.y * v.y;
        values[4]             = halfMass * v.y * v.z;
        values[5]             = halfMass * v.z * v.z;
        values[6 + group]     = values[0] + values[3] + values[5];
    }

    const unsigned int activeMask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        for (int component = 0; component < numComponents; ++component)
        {
            values[component] += __shfl_down_sync(activeMask, values[component], offset);
        }
    }
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    if (lane == 0)
    {
        for (int component = 0; component < numComponents; ++component)
        {
            sm_warpSums[warp * numComponents + component] = values[component];
        }
    }
    __syncthreads();

    const int numWarps = blockDim.x / warpSize;
    if (threadIdx.x < numComponents)
    {
        float blockSum = 0.0F;
        for (int warpIndex = 0; warpIndex < numWarps; ++warpIndex)
        {
            blockSum += sm_warpSums[warpIndex * numComponents + threadIdx.x];
        }
        atomicAdd(&gm_halfKinetic[threadIdx.x], blockSum);
    }
}

/*! \internal \brief Combines adjacent half steps and optionally records a full-step sample. */
__global__ void finalizeKineticHistoryKernel(const int    numGroups,
                                             const bool   recordSample,
                                             const float* gm_currentHalf,
                                             float*       gm_previousHalf,
                                             int*         gm_historyCount,
                                             const int    historyCapacity,
                                             float*       gm_history)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int sampleIndex = -1;
        if (recordSample)
        {
            sampleIndex = atomicAdd(gm_historyCount, 1);
        }
        const int   historyStride  = DIM * DIM + numGroups;
        const float fullCompact[6] = { 0.5F * (gm_currentHalf[0] + gm_previousHalf[0]),
                                       0.5F * (gm_currentHalf[1] + gm_previousHalf[1]),
                                       0.5F * (gm_currentHalf[2] + gm_previousHalf[2]),
                                       0.5F * (gm_currentHalf[3] + gm_previousHalf[3]),
                                       0.5F * (gm_currentHalf[4] + gm_previousHalf[4]),
                                       0.5F * (gm_currentHalf[5] + gm_previousHalf[5]) };
        if (recordSample && sampleIndex < historyCapacity)
        {
            float* history         = gm_history + sampleIndex * historyStride;
            history[XX * DIM + XX] = fullCompact[0];
            history[XX * DIM + YY] = fullCompact[1];
            history[XX * DIM + ZZ] = fullCompact[2];
            history[YY * DIM + XX] = fullCompact[1];
            history[YY * DIM + YY] = fullCompact[3];
            history[YY * DIM + ZZ] = fullCompact[4];
            history[ZZ * DIM + XX] = fullCompact[2];
            history[ZZ * DIM + YY] = fullCompact[4];
            history[ZZ * DIM + ZZ] = fullCompact[5];
            for (int group = 0; group < numGroups; ++group)
            {
                history[DIM * DIM + group] =
                        0.5F * (gm_currentHalf[6 + group] + gm_previousHalf[6 + group]);
            }
        }
        for (int component = 0; component < 6 + numGroups; ++component)
        {
            gm_previousHalf[component] = gm_currentHalf[component];
        }
    }
}

/*! \internal \brief Reduces the previous half-step kinetic tensor for each temperature group. */
__global__ void accumulateGroupHalfKineticTensorKernel(const int             numAtoms,
                                                       const int             numGroups,
                                                       const float3*         gm_velocities,
                                                       const float*          gm_masses,
                                                       const unsigned short* gm_temperatureGroups,
                                                       float*                gm_groupTensors)
{
    extern __shared__ float sm_groupKinetic[];
    for (int component = threadIdx.x; component < numGroups * DIM * DIM; component += blockDim.x)
    {
        sm_groupKinetic[component] = 0.0F;
    }
    __syncthreads();
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom < numAtoms)
    {
        const int    group    = gm_temperatureGroups[atom];
        const float3 v        = gm_velocities[atom];
        const float  value[3] = { v.x, v.y, v.z };
        const float  halfMass = 0.5F * gm_masses[atom];
        for (int row = 0; row < DIM; ++row)
        {
            for (int column = 0; column < DIM; ++column)
            {
                atomicAdd(&sm_groupKinetic[group * DIM * DIM + row * DIM + column],
                          halfMass * value[row] * value[column]);
            }
        }
    }
    __syncthreads();
    for (int component = threadIdx.x; component < numGroups * DIM * DIM; component += blockDim.x)
    {
        atomicAdd(&gm_groupTensors[component], sm_groupKinetic[component]);
    }
}

/*! \internal \brief Accumulates raw potential and dihedral-partition energies in device order. */
__global__ void reduceGamdRawEnergiesKernel(const float* gm_nbLJEnergy,
                                            const float* gm_nbElecEnergy,
                                            const float* gm_pmeEnergy,
                                            const float* gm_listedEnergies,
                                            float*       gm_gamdEnergies)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        float totalEnergy = 0.0F;
        for (int fType = 0; fType < F_EPOT; ++fType)
        {
            if (fType != F_DISRESVIOL && fType != F_ORIRESDEV)
            {
                float energy = gm_listedEnergies[fType];
                if (fType == F_COUL_SR)
                {
                    energy = gm_nbElecEnergy[0];
                }
                else if (fType == F_LJ)
                {
                    energy = gm_nbLJEnergy[0];
                }
                else if (fType == F_COUL_RECIP)
                {
                    energy = gm_pmeEnergy[0];
                }
                totalEnergy += energy;
            }
        }

        const float dihedralEnergy = gm_listedEnergies[F_PDIHS] + gm_listedEnergies[F_RBDIHS]
                                     + gm_listedEnergies[F_FOURDIHS] + gm_listedEnergies[F_CMAP];
        gm_gamdEnergies[c_gaMDPotentialEnergy] = totalEnergy;
        gm_gamdEnergies[c_gaMDDihedralEnergy]  = dihedralEnergy;
    }
}

/*! \internal \brief Appends a complete energy-component sample within the fixed capacity. */
__global__ void recordGamdEnergySampleKernel(const float* gm_nbLJEnergy,
                                             const float* gm_nbElecEnergy,
                                             const float* gm_pmeEnergy,
                                             const float* gm_listedEnergies,
                                             const float* gm_gamdState,
                                             int*         gm_historyCount,
                                             const int    historyCapacity,
                                             float*       gm_energyHistory)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const int sampleIndex = atomicAdd(gm_historyCount, 1);
        if (sampleIndex >= historyCapacity)
        {
            return;
        }
        float* sample = gm_energyHistory + sampleIndex * F_NRE;
        for (int fType = 0; fType < F_NRE; ++fType)
        {
            sample[fType] = 0.0F;
        }
        for (int fType = 0; fType < F_EPOT; ++fType)
        {
            float energy = gm_listedEnergies[fType];
            if (fType == F_COUL_SR)
            {
                energy = gm_nbElecEnergy[0];
            }
            else if (fType == F_LJ)
            {
                energy = gm_nbLJEnergy[0];
            }
            else if (fType == F_COUL_RECIP)
            {
                energy = gm_pmeEnergy[0];
            }
            sample[fType] = energy;
        }
        sample[F_EPOT] = gm_gamdState[c_gaMDPotentialEnergy] + gm_gamdState[c_gaMDPotentialBoost]
                         + gm_gamdState[c_gaMDDihedralBoost];
    }
}

/*! \internal \brief Appends force virial with the same weights as the force correction. */
__global__ void recordGamdForceVirialSampleKernel(const float* gm_shortRangeVirial,
                                                  const float* gm_dihedralVirial,
                                                  const float* gm_pmeEnergyAndVirial,
                                                  const float* gm_gamdState,
                                                  const int*   gm_historyCount,
                                                  const int    historyCapacity,
                                                  float*       gm_forceVirialHistory)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const int sampleIndex = gm_historyCount[0] - 1;
        if (sampleIndex < 0 || sampleIndex >= historyCapacity)
        {
            return;
        }
        const float scaleP                  = gm_gamdState[c_gaMDPotentialScale];
        const float dihedralCorrectionScale = scaleP * (gm_gamdState[c_gaMDDihedralScale] - 1.0F);
        float*      sample                  = gm_forceVirialHistory + sampleIndex * DIM * DIM;
        for (int component = 0; component < DIM * DIM; ++component)
        {
            sample[component] =
                    scaleP * (gm_shortRangeVirial[component] + gm_pmeEnergyAndVirial[1 + component])
                    + dihedralCorrectionScale * gm_dihedralVirial[component];
        }
    }
}

/*! \internal \brief Combines raw and dihedral forces with the current GaMD force weights. */
__global__ void gamdForceCorrectionKernel(const int     numAtoms,
                                          const float   totalScale,
                                          const float   dihedralCorrectionScale,
                                          const float*  gm_gamdState,
                                          const bool    useDeviceScale,
                                          const float3* gm_dihedralForce,
                                          float3*       gm_force)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom < numAtoms)
    {
        float appliedTotalScale              = totalScale;
        float appliedDihedralCorrectionScale = dihedralCorrectionScale;
        if (useDeviceScale)
        {
            appliedTotalScale = gm_gamdState[c_gaMDPotentialScale];
            appliedDihedralCorrectionScale =
                    gm_gamdState[c_gaMDPotentialScale] * (gm_gamdState[c_gaMDDihedralScale] - 1.0F);
        }
        const float3 rawForce      = gm_force[atom];
        const float3 dihedralForce = gm_dihedralForce[atom];
        gm_force[atom]             = make_float3(
                appliedTotalScale * rawForce.x + appliedDihedralCorrectionScale * dihedralForce.x,
                appliedTotalScale * rawForce.y + appliedDihedralCorrectionScale * dihedralForce.y,
                appliedTotalScale * rawForce.z + appliedDihedralCorrectionScale * dihedralForce.z);
    }
}

/*! \internal \brief Reduces direct and periodic-shift contributions to a virial tensor. */
__global__ void reduceGamdDihedralVirialKernel(const int     numAtoms,
                                               const float4* gm_xq,
                                               const float3* gm_dihedralForce,
                                               const float3* gm_dihedralShiftForce,
                                               const PbcAiuc pbcAiuc,
                                               float*        gm_virial)
{
    const int index                  = blockIdx.x * blockDim.x + threadIdx.x;
    float     localVirial[DIM * DIM] = {};
    if (index < numAtoms)
    {
        const float4 x = gm_xq[index];
        const float3 f = gm_dihedralForce[index];
        localVirial[0] = -0.5F * x.x * f.x;
        localVirial[1] = -0.5F * x.x * f.y;
        localVirial[2] = -0.5F * x.x * f.z;
        localVirial[3] = -0.5F * x.y * f.x;
        localVirial[4] = -0.5F * x.y * f.y;
        localVirial[5] = -0.5F * x.y * f.z;
        localVirial[6] = -0.5F * x.z * f.x;
        localVirial[7] = -0.5F * x.z * f.y;
        localVirial[8] = -0.5F * x.z * f.z;
    }
    if (index < gmx::c_numShiftVectors)
    {
        constexpr int c_nBoxX = 2 * gmx::c_dBoxX + 1;
        constexpr int c_nBoxY = 2 * gmx::c_dBoxY + 1;
        const int     sx      = index % c_nBoxX - gmx::c_dBoxX;
        const int     yzIndex = index / c_nBoxX;
        const int     sy      = yzIndex % c_nBoxY - gmx::c_dBoxY;
        const int     sz      = yzIndex / c_nBoxY - gmx::c_dBoxZ;
        const float3 shift = make_float3(sx * pbcAiuc.boxXX + sy * pbcAiuc.boxYX + sz * pbcAiuc.boxZX,
                                         sy * pbcAiuc.boxYY + sz * pbcAiuc.boxZY,
                                         sz * pbcAiuc.boxZZ);
        const float3 f = gm_dihedralShiftForce[index];
        localVirial[0] += -0.5F * shift.x * f.x;
        localVirial[1] += -0.5F * shift.x * f.y;
        localVirial[2] += -0.5F * shift.x * f.z;
        localVirial[3] += -0.5F * shift.y * f.x;
        localVirial[4] += -0.5F * shift.y * f.y;
        localVirial[5] += -0.5F * shift.y * f.z;
        localVirial[6] += -0.5F * shift.z * f.x;
        localVirial[7] += -0.5F * shift.z * f.y;
        localVirial[8] += -0.5F * shift.z * f.z;
    }

    const unsigned int activeMask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        for (float& component : localVirial)
        {
            component += __shfl_down_sync(activeMask, component, offset);
        }
    }

    extern __shared__ float sm_warpVirial[];
    const int               lane = threadIdx.x % warpSize;
    const int               warp = threadIdx.x / warpSize;
    if (lane == 0)
    {
        for (int component = 0; component < DIM * DIM; ++component)
        {
            sm_warpVirial[warp * DIM * DIM + component] = localVirial[component];
        }
    }
    __syncthreads();

    const int numWarps = blockDim.x / warpSize;
    if (threadIdx.x < DIM * DIM)
    {
        float blockVirial = 0.0F;
        for (int warpIndex = 0; warpIndex < numWarps; ++warpIndex)
        {
            blockVirial += sm_warpVirial[warpIndex * DIM * DIM + threadIdx.x];
        }
        atomicAdd(&gm_virial[threadIdx.x], blockVirial);
    }
}


/*-------------------------------- End CUDA kernels-----------------------------*/

} // namespace

void launchGaMDPmeEnergyStaging(DeviceBuffer<float>   source,
                                DeviceBuffer<float>   destination,
                                const DeviceStream&   pmeStream,
                                GpuEventSynchronizer* readyEvent)
{
    GMX_ASSERT(destination != nullptr && readyEvent != nullptr,
               "GaMD reciprocal-energy staging requires output storage and an event");

    KernelLaunchConfig config;
    config.blockSize[0]     = 1;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = 1;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel     = stageGamdReciprocalEnergyKernel;
    const auto kernelArgs = prepareGpuKernelArguments(kernel, config, &source, &destination);
    launchGpuKernel(kernel, config, pmeStream, nullptr, "Stage GaMD reciprocal energy", kernelArgs);
    readyEvent->markEvent(pmeStream);
}


void GaMDGpuExecutor::Impl::recordUpdateConstraintVirial(DeviceBuffer<float> lincsVirial,
                                                         const bool          haveLincs,
                                                         DeviceBuffer<float> settleVirial,
                                                         const bool          haveSettles,
                                                         const real          scaleFactor)
{
    GMX_RELEASE_ASSERT(updateStream_ != nullptr,
                       "GaMD constraint-virial history requested before initialization");

    KernelLaunchConfig config;
    config.blockSize[0]     = 1;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = 1;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel     = recordConstraintVirialKernel;
    const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &lincsVirial,
                                                      &haveLincs,
                                                      &settleVirial,
                                                      &haveSettles,
                                                      &scaleFactor,
                                                      &d_updateConstraintVirialHistoryCount_,
                                                      &sc_historyCapacity,
                                                      &d_updateConstraintVirialHistory_);
    launchGpuKernel(kernel, config, *updateStream_, nullptr, "Record deferred constraint virial", kernelArgs);
}

void GaMDGpuExecutor::Impl::recordUpdateKineticEnergy(const bool recordSample)
{
    GMX_RELEASE_ASSERT(updateStream_ != nullptr && updateVelocities_ != nullptr
                               && d_updateMasses_ != nullptr && d_updateTemperatureGroups_ != nullptr,
                       "GaMD kinetic history requires bound update buffers");
    if (updateNumAtoms_ == 0)
    {
        return;
    }

    clearDeviceBufferAsync(&d_updateCurrentHalfKinetic_, 0, 6 + updateNumTemperatureGroups_, *updateStream_);
    constexpr int      c_kineticThreadsPerBlock = 256;
    KernelLaunchConfig kineticConfig;
    kineticConfig.blockSize[0] = c_kineticThreadsPerBlock;
    kineticConfig.blockSize[1] = 1;
    kineticConfig.blockSize[2] = 1;
    kineticConfig.gridSize[0] = (updateNumAtoms_ + c_kineticThreadsPerBlock - 1) / c_kineticThreadsPerBlock;
    kineticConfig.gridSize[1] = 1;
    kineticConfig.gridSize[2] = 1;
    kineticConfig.sharedMemorySize =
            (c_kineticThreadsPerBlock / 32) * (6 + updateNumTemperatureGroups_) * sizeof(float);

    auto       kineticKernel = accumulateHalfStepKineticKernel;
    const auto kineticArgs   = prepareGpuKernelArguments(kineticKernel,
                                                       kineticConfig,
                                                       &updateNumAtoms_,
                                                       &updateNumTemperatureGroups_,
                                                       &updateVelocities_,
                                                       &d_updateMasses_,
                                                       &d_updateTemperatureGroups_,
                                                       &d_updateCurrentHalfKinetic_);
    launchGpuKernel(kineticKernel,
                    kineticConfig,
                    *updateStream_,
                    nullptr,
                    "Accumulate GaMD GPU kinetic energy",
                    kineticArgs);

    KernelLaunchConfig finalizeConfig;
    finalizeConfig.blockSize[0]     = 1;
    finalizeConfig.blockSize[1]     = 1;
    finalizeConfig.blockSize[2]     = 1;
    finalizeConfig.gridSize[0]      = 1;
    finalizeConfig.gridSize[1]      = 1;
    finalizeConfig.gridSize[2]      = 1;
    finalizeConfig.sharedMemorySize = 0;

    auto       finalizeKernel = finalizeKineticHistoryKernel;
    const auto finalizeArgs   = prepareGpuKernelArguments(finalizeKernel,
                                                        finalizeConfig,
                                                        &updateNumTemperatureGroups_,
                                                        &recordSample,
                                                        &d_updateCurrentHalfKinetic_,
                                                        &d_updatePreviousHalfKinetic_,
                                                        &d_updateKineticHistoryCount_,
                                                        &sc_historyCapacity,
                                                        &d_updateKineticHistory_);
    launchGpuKernel(finalizeKernel,
                    finalizeConfig,
                    *updateStream_,
                    nullptr,
                    "Finalize GaMD GPU kinetic-energy history",
                    finalizeArgs);
}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::Impl::takePreviousHalfStepKineticEnergy()
{
    GMX_RELEASE_ASSERT(updateStream_ != nullptr && updateVelocities_ != nullptr
                               && d_updateMasses_ != nullptr && d_updateTemperatureGroups_ != nullptr,
                       "GaMD previous half-step kinetic state requires bound update buffers");
    clearDeviceBufferAsync(
            &d_updateGroupHalfKineticTensor_, 0, updateNumTemperatureGroups_ * DIM * DIM, *updateStream_);

    constexpr int      c_kineticThreadsPerBlock = 256;
    KernelLaunchConfig config;
    config.blockSize[0] = c_kineticThreadsPerBlock;
    config.blockSize[1] = 1;
    config.blockSize[2] = 1;
    config.gridSize[0] = (updateNumAtoms_ + c_kineticThreadsPerBlock - 1) / c_kineticThreadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = updateNumTemperatureGroups_ * DIM * DIM * sizeof(float);

    auto       kernel     = accumulateGroupHalfKineticTensorKernel;
    const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &updateNumAtoms_,
                                                      &updateNumTemperatureGroups_,
                                                      &updateVelocities_,
                                                      &d_updateMasses_,
                                                      &d_updateTemperatureGroups_,
                                                      &d_updateGroupHalfKineticTensor_);
    launchGpuKernel(kernel,
                    config,
                    *updateStream_,
                    nullptr,
                    "Stage GaMD temperature-group half-step kinetic energy",
                    kernelArgs);
    copyFromDeviceBuffer(h_updatePreviousHalfKinetic_.data(),
                         &d_updateGroupHalfKineticTensor_,
                         0,
                         updateNumTemperatureGroups_ * DIM * DIM,
                         *updateStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);

    std::vector<std::array<real, DIM * DIM>> tensors(updateNumTemperatureGroups_);
    for (int group = 0; group < updateNumTemperatureGroups_; ++group)
    {
        std::copy_n(h_updatePreviousHalfKinetic_.data() + group * DIM * DIM,
                    DIM * DIM,
                    tensors[group].begin());
    }
    return tensors;
}

void GaMDGpuExecutor::Impl::finishGamdDihedralBuffers()
{
    GMX_ASSERT(dihedralBufferEnabled(), "GaMD dihedral buffers requested while disabled");
    GMX_ASSERT(d_gamdDihedralForcesNbnxn_ != nullptr && d_gamdDihedralForcesState_ != nullptr
                       && d_nbnxnAtomOrder_ != nullptr,
               "GaMD dihedral buffers have not been initialized");

    wallcycle_start_nocount(wallCycleCounters_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wallCycleCounters_, WallCycleSubCounter::LaunchGpuBonded);

    // 关闭两种累加后，原生内核只按原子映射复制三个力分量。
    // 未使用的附加力也传入有效缓冲，避免原生入口对空指针计算偏移。
    launchForceReductionKernel(gamdStateForceSize_,
                               /* atomStart= */ 0,
                               /* addRvecForce= */ false,
                               /* accumulate= */ false,
                               d_gamdDihedralForcesNbnxn_,
                               d_gamdDihedralForcesState_,
                               d_gamdDihedralForcesState_,
                               d_nbnxnAtomOrder_,
                               deviceStream_,
                               nullptr,
                               0);

    launchGamdVirialReductionKernel(d_gamdDihedralForcesNbnxn_,
                                    d_gamdDihedralShiftForces_,
                                    &d_gamdDihedralVirial_,
                                    "Reduce GaMD dihedral virial");

    wallcycle_sub_stop(wallCycleCounters_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wallCycleCounters_, WallCycleCounter::LaunchGpuPp);
}

DeviceBuffer<float> GaMDGpuExecutor::Impl::gamdPmeEnergyStagingBuffer()
{
    GMX_ASSERT(energyEnabled(), "GaMD device energy requested while disabled");
    return d_gamdPmeEnergy_;
}

GpuEventSynchronizer* GaMDGpuExecutor::Impl::gamdPmeEnergyReadyEvent()
{
    GMX_ASSERT(gamdPmeEnergyReadyEvent_ != nullptr,
               "GaMD PME energy event requested while device energy is disabled");
    return gamdPmeEnergyReadyEvent_.get();
}

void GaMDGpuExecutor::Impl::launchGamdEnergyReduction(const GaMDBiasParameters& parameters,
                                                      const GaMDForceOverrides& forceOverrides,
                                                      bool                      recordEnergySample)
{
    GMX_ASSERT(energyEnabled(), "GaMD device energy requested while disabled");
    GMX_ASSERT(shortRangeLennardJonesEnergy_ != nullptr && shortRangeElectrostaticEnergy_ != nullptr
                       && listedEnergies_ != nullptr && d_gamdPmeEnergy_ != nullptr
                       && d_gamdState_ != nullptr,
               "GaMD device energy buffers have not been initialized");

    gamdPmeEnergyReadyEvent_->enqueueWaitEvent(deviceStream_);

    KernelLaunchConfig config;
    config.blockSize[0]     = 1;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = 1;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel     = reduceGamdRawEnergiesKernel;
    const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &shortRangeLennardJonesEnergy_,
                                                      &shortRangeElectrostaticEnergy_,
                                                      &d_gamdPmeEnergy_,
                                                      &listedEnergies_,
                                                      &d_gamdState_);
    launchGpuKernel(kernel, config, deviceStream_, nullptr, "Reduce GaMD raw energies", kernelArgs);
    launchGamdProductionScaleKernel(d_gamdState_, parameters, forceOverrides, deviceStream_);
    if (recordEnergySample)
    {
        GMX_RELEASE_ASSERT(useGpuResidentGaMD_ && d_gamdEnergyHistory_ != nullptr
                                   && d_gamdEnergyHistoryCount_ != nullptr,
                           "Deferred GaMD energy history requested while disabled");

        auto       historyKernel     = recordGamdEnergySampleKernel;
        const auto historyKernelArgs = prepareGpuKernelArguments(historyKernel,
                                                                 config,
                                                                 &shortRangeLennardJonesEnergy_,
                                                                 &shortRangeElectrostaticEnergy_,
                                                                 &d_gamdPmeEnergy_,
                                                                 &listedEnergies_,
                                                                 &d_gamdState_,
                                                                 &d_gamdEnergyHistoryCount_,
                                                                 &sc_historyCapacity,
                                                                 &d_gamdEnergyHistory_);
        launchGpuKernel(
                historyKernel, config, deviceStream_, nullptr, "Record deferred GaMD energy sample", historyKernelArgs);
    }
    if (gamdEnergyShadowDiagnosticsEnabled_)
    {
        copyFromDeviceBuffer(h_gamdDiagnosticState_.data(),
                             &d_gamdState_,
                             0,
                             c_gaMDStateSize,
                             deviceStream_,
                             GpuApiCallBehavior::Async,
                             nullptr);
    }
}

GaMDStepResult GaMDGpuExecutor::Impl::gamdDiagnosticState()
{
    GMX_ASSERT(gamdEnergyShadowDiagnosticsEnabled_,
               "GaMD device energy diagnostics requested while disabled");
    deviceStream_.synchronize();
    return {
        h_gamdDiagnosticState_[c_gaMDPotentialEnergy], h_gamdDiagnosticState_[c_gaMDDihedralEnergy],
        h_gamdDiagnosticState_[c_gaMDPotentialScale],  h_gamdDiagnosticState_[c_gaMDDihedralScale],
        h_gamdDiagnosticState_[c_gaMDPotentialBoost],  h_gamdDiagnosticState_[c_gaMDDihedralBoost]
    };
}

std::vector<std::array<real, F_NRE>> GaMDGpuExecutor::Impl::takeGamdEnergyHistory()
{
    GMX_RELEASE_ASSERT(useGpuResidentGaMD_, "Deferred GaMD energy history requested while disabled");
    copyFromDeviceBuffer(h_gamdEnergyHistoryCount_.data(),
                         &d_gamdEnergyHistoryCount_,
                         0,
                         1,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    const int sampleCount = h_gamdEnergyHistoryCount_[0];
    GMX_RELEASE_ASSERT(sampleCount <= sc_historyCapacity,
                       "Deferred GaMD energy history exceeded its fixed capacity");
    if (sampleCount == 0)
    {
        return {};
    }

    copyFromDeviceBuffer(h_gamdEnergyHistory_.data(),
                         &d_gamdEnergyHistory_,
                         0,
                         sampleCount * F_NRE,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);

    std::vector<std::array<real, F_NRE>> samples(sampleCount);
    for (int sample = 0; sample < sampleCount; ++sample)
    {
        std::copy_n(h_gamdEnergyHistory_.data() + sample * F_NRE, F_NRE, samples[sample].begin());
    }
    clearDeviceBufferAsync(&d_gamdEnergyHistoryCount_, 0, 1, deviceStream_);
    return samples;
}

void GaMDGpuExecutor::Impl::recordGamdForceVirialSample()
{
    GMX_RELEASE_ASSERT(useGpuResidentGaMD_ && d_gamdForceVirialHistory_ != nullptr,
                       "Deferred GaMD force-virial history requested while disabled");

    KernelLaunchConfig config;
    config.blockSize[0]     = 1;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = 1;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel     = recordGamdForceVirialSampleKernel;
    const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &d_gamdShortRangeVirial_,
                                                      &d_gamdDihedralVirial_,
                                                      &d_gamdPmeEnergy_,
                                                      &d_gamdState_,
                                                      &d_gamdEnergyHistoryCount_,
                                                      &sc_historyCapacity,
                                                      &d_gamdForceVirialHistory_);
    launchGpuKernel(kernel, config, deviceStream_, nullptr, "Record deferred GaMD force virial", kernelArgs);
}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::Impl::takeGamdForceVirialHistory(const int numSamples)
{
    GMX_RELEASE_ASSERT(useGpuResidentGaMD_,
                       "Deferred GaMD force-virial history requested while disabled");
    GMX_RELEASE_ASSERT(numSamples >= 0 && numSamples <= sc_historyCapacity,
                       "Deferred GaMD force-virial sample count is out of range");
    if (numSamples == 0)
    {
        return {};
    }

    copyFromDeviceBuffer(h_gamdForceVirialHistory_.data(),
                         &d_gamdForceVirialHistory_,
                         0,
                         numSamples * DIM * DIM,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    std::vector<std::array<real, DIM * DIM>> samples(numSamples);
    for (int sample = 0; sample < numSamples; ++sample)
    {
        std::copy_n(h_gamdForceVirialHistory_.data() + sample * DIM * DIM,
                    DIM * DIM,
                    samples[sample].begin());
    }
    return samples;
}

void GaMDGpuExecutor::Impl::launchGamdVirialReductionKernel(DeviceBuffer<Float3> forces,
                                                            DeviceBuffer<Float3> shiftForces,
                                                            DeviceBuffer<float>* virial,
                                                            const char*          kernelName)
{
    GMX_ASSERT(virial != nullptr, "GaMD virial reduction requires an output buffer");
    clearDeviceBufferAsync(virial, 0, DIM * DIM, deviceStream_);

    constexpr int      c_virialThreadsPerBlock = 256;
    KernelLaunchConfig virialConfig;
    virialConfig.blockSize[0] = c_virialThreadsPerBlock;
    virialConfig.blockSize[1] = 1;
    virialConfig.blockSize[2] = 1;
    virialConfig.gridSize[0] = (gamdNbnxnForceSize_ + c_virialThreadsPerBlock - 1) / c_virialThreadsPerBlock;
    virialConfig.gridSize[1]      = 1;
    virialConfig.gridSize[2]      = 1;
    virialConfig.sharedMemorySize = (c_virialThreadsPerBlock / 32) * DIM * DIM * sizeof(float);
    auto       virialKernel       = reduceGamdDihedralVirialKernel;
    const auto virialKernelArgs   = prepareGpuKernelArguments(virialKernel,
                                                            virialConfig,
                                                            &gamdNbnxnForceSize_,
                                                            &coordinatesAndCharges_,
                                                            &forces,
                                                            &shiftForces,
                                                            &pbcAiuc_,
                                                            virial);
    launchGpuKernel(virialKernel, virialConfig, deviceStream_, nullptr, kernelName, virialKernelArgs);
}

void GaMDGpuExecutor::Impl::launchGamdShortRangeVirialReduction()
{
    GMX_ASSERT(useGpuResidentGaMD_, "GaMD short-range virial reduction requested while disabled");
    wallcycle_start_nocount(wallCycleCounters_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wallCycleCounters_, WallCycleSubCounter::LaunchGpuBonded);
    launchGamdVirialReductionKernel(
            forces_, shiftForces_, &d_gamdShortRangeVirial_, "Reduce GaMD short-range virial");
    wallcycle_sub_stop(wallCycleCounters_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wallCycleCounters_, WallCycleCounter::LaunchGpuPp);
}

void GaMDGpuExecutor::Impl::launchGamdForceCorrectionKernel(DeviceBuffer<Float3> forces,
                                                            real                 totalScale,
                                                            real dihedralCorrectionScale,
                                                            GpuEventSynchronizer* rawForcesReady)
{
    GMX_ASSERT(useGpuResidentGaMD_, "GaMD GPU force correction requested while disabled");
    GMX_ASSERT(!useGpuResidentGaMD_ || d_gamdState_ != nullptr,
               "GaMD device scale requested without device state storage");
    rawForcesReady->enqueueWaitEvent(deviceStream_);

    constexpr int      c_threadsPerBlock = 256;
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (gamdStateForceSize_ + c_threadsPerBlock - 1) / c_threadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel     = gamdForceCorrectionKernel;
    const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &gamdStateForceSize_,
                                                      &totalScale,
                                                      &dihedralCorrectionScale,
                                                      &d_gamdState_,
                                                      &useGpuResidentGaMD_,
                                                      &d_gamdDihedralForcesState_,
                                                      &forces);
    launchGpuKernel(kernel, config, deviceStream_, nullptr, "GaMD in-place force correction", kernelArgs);
    gamdForcesReadyEvent_->markEvent(deviceStream_);
}

void GaMDGpuExecutor::Impl::launchGamdForceCorrectionShadowKernel(real totalScale, real dihedralCorrectionScale)
{
    GMX_ASSERT(gamdDihedralShadowEnabled_,
               "GaMD force-correction shadow kernel requested while disabled");
    GMX_ASSERT(d_gamdCorrectedForcesState_ != nullptr && d_gamdDihedralForcesState_ != nullptr,
               "GaMD force-correction shadow buffers have not been initialized");

    constexpr int      c_threadsPerBlock = 256;
    KernelLaunchConfig config;
    config.blockSize[0]     = c_threadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (gamdStateForceSize_ + c_threadsPerBlock - 1) / c_threadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = 0;

    auto       kernel         = gamdForceCorrectionKernel;
    const bool useDeviceScale = false;
    const auto kernelArgs     = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &gamdStateForceSize_,
                                                      &totalScale,
                                                      &dihedralCorrectionScale,
                                                      &d_gamdState_,
                                                      &useDeviceScale,
                                                      &d_gamdDihedralForcesState_,
                                                      &d_gamdCorrectedForcesState_);
    launchGpuKernel(kernel, config, deviceStream_, nullptr, "GaMD force-correction shadow", kernelArgs);
}

} // namespace gmx
//! \endcond
