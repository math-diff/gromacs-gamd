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
 *
 * \brief Implements the CUDA GaMD device executor.
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
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/typecasts_cuda_hip.h"
#include "gromacs/gpu_utils/vectype_ops_cuda.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

__global__ void mapNbnxnToStateForceKernel(const int     numAtoms,
                                           const int*    gm_nbnxnAtomOrder,
                                           const float3* gm_nbnxnForce,
                                           float3*       gm_stateForce)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom < numAtoms)
    {
        gm_stateForce[atom] = gm_nbnxnForce[gm_nbnxnAtomOrder[atom]];
    }
}

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
        gm_gamdEnergies[0] = totalEnergy;
        gm_gamdEnergies[1] = dihedralEnergy;
    }
}

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
        sample[F_EPOT] = gm_gamdState[0] + gm_gamdState[4] + gm_gamdState[5];
    }
}

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
        const float scaleP                  = gm_gamdState[2];
        const float dihedralCorrectionScale = scaleP * (gm_gamdState[3] - 1.0F);
        float*      sample                  = gm_forceVirialHistory + sampleIndex * DIM * DIM;
        for (int component = 0; component < DIM * DIM; ++component)
        {
            sample[component] =
                    scaleP * (gm_shortRangeVirial[component] + gm_pmeEnergyAndVirial[1 + component])
                    + dihedralCorrectionScale * gm_dihedralVirial[component];
        }
    }
}

__global__ void gamdForceCorrectionShadowKernel(const int     numAtoms,
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
        const float appliedTotalScale = useDeviceScale ? gm_gamdState[2] : totalScale;
        const float appliedDihedralCorrectionScale =
                useDeviceScale ? gm_gamdState[2] * (gm_gamdState[3] - 1.0F) : dihedralCorrectionScale;
        const float3 rawForce      = gm_force[atom];
        const float3 dihedralForce = gm_dihedralForce[atom];
        gm_force[atom]             = make_float3(
                appliedTotalScale * rawForce.x + appliedDihedralCorrectionScale * dihedralForce.x,
                appliedTotalScale * rawForce.y + appliedDihedralCorrectionScale * dihedralForce.y,
                appliedTotalScale * rawForce.z + appliedDihedralCorrectionScale * dihedralForce.z);
    }
}

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

void GaMDGpuExecutor::Impl::finishGamdDihedralBuffers()
{
    GMX_ASSERT(gamdDihedralBufferEnabled_, "GaMD dihedral buffers requested while disabled");
    GMX_ASSERT(d_gamdDihedralForcesNbnxn_ != nullptr && d_gamdDihedralForcesState_ != nullptr
                       && d_nbnxnAtomOrder_ != nullptr,
               "GaMD dihedral buffers have not been initialized");

    wallcycle_start_nocount(wallCycleCounters_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wallCycleCounters_, WallCycleSubCounter::LaunchGpuBonded);

    constexpr int      c_mappingThreadsPerBlock = 256;
    KernelLaunchConfig mappingConfig;
    mappingConfig.blockSize[0] = c_mappingThreadsPerBlock;
    mappingConfig.blockSize[1] = 1;
    mappingConfig.blockSize[2] = 1;
    mappingConfig.gridSize[0] =
            (gamdStateForceSize_ + c_mappingThreadsPerBlock - 1) / c_mappingThreadsPerBlock;
    mappingConfig.gridSize[1]      = 1;
    mappingConfig.gridSize[2]      = 1;
    mappingConfig.sharedMemorySize = 0;

    auto       mappingKernel     = mapNbnxnToStateForceKernel;
    const auto mappingKernelArgs = prepareGpuKernelArguments(mappingKernel,
                                                             mappingConfig,
                                                             &gamdStateForceSize_,
                                                             &d_nbnxnAtomOrder_,
                                                             &d_gamdDihedralForcesNbnxn_,
                                                             &d_gamdDihedralForcesState_);
    launchGpuKernel(mappingKernel,
                    mappingConfig,
                    deviceStream_,
                    nullptr,
                    "Map GaMD dihedral forces to state order",
                    mappingKernelArgs);

    launchGamdVirialReductionKernel(d_gamdDihedralForcesNbnxn_,
                                    d_gamdDihedralShiftForces_,
                                    &d_gamdDihedralVirial_,
                                    "Reduce GaMD dihedral virial");

    wallcycle_sub_stop(wallCycleCounters_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wallCycleCounters_, WallCycleCounter::LaunchGpuPp);
}

DeviceBuffer<float> GaMDGpuExecutor::Impl::gamdPmeEnergyStagingBuffer()
{
    GMX_ASSERT(gamdEnergyShadowEnabled_, "GaMD device energy shadow requested while disabled");
    return d_gamdPmeEnergy_;
}

GpuEventSynchronizer* GaMDGpuExecutor::Impl::gamdPmeEnergyReadyEvent()
{
    GMX_ASSERT(gamdPmeEnergyReadyEvent_ != nullptr,
               "GaMD PME energy event requested while device energy shadow is disabled");
    return gamdPmeEnergyReadyEvent_.get();
}

void GaMDGpuExecutor::Impl::launchGamdEnergyShadowReduction(int    igamd,
                                                            int    stage,
                                                            double thresholdP,
                                                            double kP,
                                                            double thresholdD,
                                                            double kD,
                                                            bool   recordEnergySample)
{
    GMX_ASSERT(gamdEnergyShadowEnabled_, "GaMD device energy shadow requested while disabled");
    GMX_ASSERT(shortRangeLennardJonesEnergy_ != nullptr && shortRangeElectrostaticEnergy_ != nullptr
                       && listedEnergies_ != nullptr && d_gamdPmeEnergy_ != nullptr
                       && d_gamdEnergyShadow_ != nullptr,
               "GaMD device energy shadow buffers have not been initialized");

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
                                                      &d_gamdEnergyShadow_);
    launchGpuKernel(kernel, config, deviceStream_, nullptr, "Reduce GaMD raw energies", kernelArgs);
    launchGamdProductionScaleKernel(
            d_gamdEnergyShadow_, igamd, stage, thresholdP, kP, thresholdD, kD, deviceStream_);
    if (recordEnergySample)
    {
        GMX_RELEASE_ASSERT(gamdEnergyHistoryEnabled_ && d_gamdEnergyHistory_ != nullptr
                                   && d_gamdEnergyHistoryCount_ != nullptr,
                           "Deferred GaMD energy history requested while disabled");

        auto       historyKernel     = recordGamdEnergySampleKernel;
        const auto historyKernelArgs = prepareGpuKernelArguments(historyKernel,
                                                                 config,
                                                                 &shortRangeLennardJonesEnergy_,
                                                                 &shortRangeElectrostaticEnergy_,
                                                                 &d_gamdPmeEnergy_,
                                                                 &listedEnergies_,
                                                                 &d_gamdEnergyShadow_,
                                                                 &d_gamdEnergyHistoryCount_,
                                                                 &c_gamdEnergyHistoryCapacity_,
                                                                 &d_gamdEnergyHistory_);
        launchGpuKernel(
                historyKernel, config, deviceStream_, nullptr, "Record deferred GaMD energy sample", historyKernelArgs);
    }
    if (gamdEnergyShadowDiagnosticsEnabled_)
    {
        copyFromDeviceBuffer(h_gamdEnergyShadow_.data(),
                             &d_gamdEnergyShadow_,
                             0,
                             6,
                             deviceStream_,
                             GpuApiCallBehavior::Async,
                             nullptr);
    }
}

std::array<double, 6> GaMDGpuExecutor::Impl::gamdEnergyShadowValues()
{
    GMX_ASSERT(gamdEnergyShadowDiagnosticsEnabled_,
               "GaMD device energy diagnostics requested while disabled");
    deviceStream_.synchronize();
    return { h_gamdEnergyShadow_[0], h_gamdEnergyShadow_[1], h_gamdEnergyShadow_[2],
             h_gamdEnergyShadow_[3], h_gamdEnergyShadow_[4], h_gamdEnergyShadow_[5] };
}

std::vector<std::array<real, F_NRE>> GaMDGpuExecutor::Impl::takeGamdEnergyHistory()
{
    GMX_RELEASE_ASSERT(gamdEnergyHistoryEnabled_,
                       "Deferred GaMD energy history requested while disabled");
    copyFromDeviceBuffer(h_gamdEnergyHistoryCount_.data(),
                         &d_gamdEnergyHistoryCount_,
                         0,
                         1,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    const int sampleCount = h_gamdEnergyHistoryCount_[0];
    GMX_RELEASE_ASSERT(sampleCount <= c_gamdEnergyHistoryCapacity_,
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
    GMX_RELEASE_ASSERT(gamdEnergyHistoryEnabled_ && d_gamdForceVirialHistory_ != nullptr,
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
                                                      &d_gamdEnergyShadow_,
                                                      &d_gamdEnergyHistoryCount_,
                                                      &c_gamdEnergyHistoryCapacity_,
                                                      &d_gamdForceVirialHistory_);
    launchGpuKernel(kernel, config, deviceStream_, nullptr, "Record deferred GaMD force virial", kernelArgs);
}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::Impl::takeGamdForceVirialHistory(const int numSamples)
{
    GMX_RELEASE_ASSERT(gamdEnergyHistoryEnabled_,
                       "Deferred GaMD force-virial history requested while disabled");
    GMX_RELEASE_ASSERT(numSamples >= 0 && numSamples <= c_gamdEnergyHistoryCapacity_,
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
    GMX_ASSERT(gamdForceCorrectionEnabled_,
               "GaMD short-range virial reduction requested while disabled");
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
    GMX_ASSERT(gamdForceCorrectionEnabled_, "GaMD GPU force correction requested while disabled");
    GMX_ASSERT(!gamdScaleFromDeviceEnabled_ || d_gamdEnergyShadow_ != nullptr,
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

    auto       kernel     = gamdForceCorrectionShadowKernel;
    const auto kernelArgs = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &gamdStateForceSize_,
                                                      &totalScale,
                                                      &dihedralCorrectionScale,
                                                      &d_gamdEnergyShadow_,
                                                      &gamdScaleFromDeviceEnabled_,
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

    auto       kernel         = gamdForceCorrectionShadowKernel;
    const bool useDeviceScale = false;
    const auto kernelArgs     = prepareGpuKernelArguments(kernel,
                                                      config,
                                                      &gamdStateForceSize_,
                                                      &totalScale,
                                                      &dihedralCorrectionScale,
                                                      &d_gamdEnergyShadow_,
                                                      &useDeviceScale,
                                                      &d_gamdDihedralForcesState_,
                                                      &d_gamdCorrectedForcesState_);
    launchGpuKernel(kernel, config, deviceStream_, nullptr, "GaMD force-correction shadow", kernelArgs);
}

} // namespace gmx
