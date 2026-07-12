/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
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
 * \brief Implements update and constraints class.
 *
 * The class combines Leap-Frog integrator with LINCS and SETTLE constraints.
 *
 * \todo The computational procedures in members should be integrated to improve
 *       computational performance.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "update_constrain_gpu_impl.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/gpu_utils/gputraits.h"
#if GMX_GPU_CUDA
#    include "gromacs/gpu_utils/cudautils.cuh"
#endif
#include "gromacs/mdlib/leapfrog_gpu.h"
#include "gromacs/mdlib/update_constrain_gpu.h"
#include "gromacs/mdlib/update_constrain_gpu_internal.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/mtop_util.h"

static constexpr bool sc_haveGpuConstraintSupport = (GMX_GPU_CUDA != 0) || (GMX_GPU_SYCL != 0);

namespace gmx
{

#if GMX_GPU_CUDA
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
            compact[component] = ((haveLincs ? gm_lincsVirial[component] : 0.0F)
                                  + (haveSettles ? gm_settleVirial[component] : 0.0F))
                                 * scaleFactor;
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
#endif

void UpdateConstrainGpu::Impl::integrate(GpuEventSynchronizer*             fReadyOnDevice,
                                         const real                        dt,
                                         const bool                        updateVelocities,
                                         const bool                        computeVirial,
                                         const bool                        deferVirialToHost,
                                         tensor                            virial,
                                         const bool                        doTemperatureScaling,
                                         gmx::ArrayRef<const t_grp_tcstat> tcstat,
                                         const bool                        doParrinelloRahman,
                                         const float                       dtPressureCouple,
                                         const Matrix3x3&                  prVelocityScalingMatrix)
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);

    if (!computeVirial)
    {
        clear_mat(virial);
    }

    // Make sure that the forces are ready on device before proceeding with the update.
    fReadyOnDevice->enqueueWaitEvent(deviceStream_);

    if (numAtoms_ != 0)
    {
        // A copy of the current coordinates is saved into d_x0_ by integrate(), and
        // d_x_ is updated by integration and constraints.
        integrator_->integrate(
                d_x_, d_x0_, d_v_, d_f_, dt, doTemperatureScaling, tcstat, doParrinelloRahman, dtPressureCouple, prVelocityScalingMatrix);
        if (sc_haveGpuConstraintSupport)
        {
            lincsGpu_->apply(d_x0_, d_x_, updateVelocities, d_v_, 1.0 / dt, computeVirial, pbcAiuc_);
            settleGpu_->apply(d_x0_, d_x_, updateVelocities, d_v_, 1.0 / dt, computeVirial, pbcAiuc_);
        }

        if (computeVirial && deferVirialToHost)
        {
            clear_mat(virial);
#if GMX_GPU_CUDA
            GMX_RELEASE_ASSERT(constraintVirialHistoryEnabled_,
                               "Deferred constraint virial requested while history is disabled");
            KernelLaunchConfig config;
            config.blockSize[0]      = 1;
            config.blockSize[1]      = 1;
            config.blockSize[2]      = 1;
            config.gridSize[0]       = 1;
            config.gridSize[1]       = 1;
            config.gridSize[2]       = 1;
            config.sharedMemorySize  = 0;
            const bool  haveLincs    = lincsGpu_->hasConstraints();
            const bool  haveSettles  = settleGpu_->hasSettles();
            const float scaleFactor  = 0.5F / (dt * dt);
            auto        lincsVirial  = lincsGpu_->virialDeviceBuffer();
            auto        settleVirial = settleGpu_->virialDeviceBuffer();
            auto        kernel       = recordConstraintVirialKernel;
            const auto  kernelArgs   = prepareGpuKernelArguments(kernel,
                                                              config,
                                                              &lincsVirial,
                                                              &haveLincs,
                                                              &settleVirial,
                                                              &haveSettles,
                                                              &scaleFactor,
                                                              &d_constraintVirialHistoryCount_,
                                                              &c_constraintVirialHistoryCapacity_,
                                                              &d_constraintVirialHistory_);
            launchGpuKernel(
                    kernel, config, deviceStream_, nullptr, "Record deferred constraint virial", kernelArgs);
#else
            GMX_RELEASE_ASSERT(false, "Deferred constraint virial requires CUDA");
#endif
        }
        else if (computeVirial)
        {
            copyConstraintVirialToHost(dt, virial);
        }

#if GMX_GPU_CUDA
        if (deviceGlobalHistoryEnabled_)
        {
            clearDeviceBufferAsync(&d_currentHalfKinetic_, 0, 6 + numTemperatureGroups_, deviceStream_);
            constexpr int      c_kineticThreadsPerBlock = 256;
            KernelLaunchConfig kineticConfig;
            kineticConfig.blockSize[0] = c_kineticThreadsPerBlock;
            kineticConfig.blockSize[1] = 1;
            kineticConfig.blockSize[2] = 1;
            kineticConfig.gridSize[0] =
                    (numAtoms_ + c_kineticThreadsPerBlock - 1) / c_kineticThreadsPerBlock;
            kineticConfig.gridSize[1] = 1;
            kineticConfig.gridSize[2] = 1;
            kineticConfig.sharedMemorySize =
                    (c_kineticThreadsPerBlock / 32) * (6 + numTemperatureGroups_) * sizeof(float);
            auto       kineticKernel = accumulateHalfStepKineticKernel;
            const auto kineticArgs   = prepareGpuKernelArguments(kineticKernel,
                                                               kineticConfig,
                                                               &numAtoms_,
                                                               &numTemperatureGroups_,
                                                               &d_v_,
                                                               &d_masses_,
                                                               &d_temperatureGroups_,
                                                               &d_currentHalfKinetic_);
            launchGpuKernel(kineticKernel,
                            kineticConfig,
                            deviceStream_,
                            nullptr,
                            "Accumulate GPU kinetic energy",
                            kineticArgs);

            KernelLaunchConfig finalizeConfig;
            finalizeConfig.blockSize[0]     = 1;
            finalizeConfig.blockSize[1]     = 1;
            finalizeConfig.blockSize[2]     = 1;
            finalizeConfig.gridSize[0]      = 1;
            finalizeConfig.gridSize[1]      = 1;
            finalizeConfig.gridSize[2]      = 1;
            finalizeConfig.sharedMemorySize = 0;
            const bool recordKineticSample  = deferVirialToHost;
            auto       finalizeKernel       = finalizeKineticHistoryKernel;
            const auto finalizeArgs         = prepareGpuKernelArguments(finalizeKernel,
                                                                finalizeConfig,
                                                                &numTemperatureGroups_,
                                                                &recordKineticSample,
                                                                &d_currentHalfKinetic_,
                                                                &d_previousHalfKinetic_,
                                                                &d_kineticHistoryCount_,
                                                                &c_constraintVirialHistoryCapacity_,
                                                                &d_kineticHistory_);
            launchGpuKernel(finalizeKernel,
                            finalizeConfig,
                            deviceStream_,
                            nullptr,
                            "Finalize GPU kinetic-energy history",
                            finalizeArgs);
        }
#endif
    }
    else
    {
        clear_mat(virial);
    }

    xUpdatedOnDeviceEvent_.markEvent(deviceStream_);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpuPp);
}

void UpdateConstrainGpu::Impl::copyConstraintVirialToHost(const real dt, tensor virial)
{
    clear_mat(virial);
    lincsGpu_->copyVirialToHost(virial);
    settleGpu_->copyVirialToHost(virial);

    // scaledVirial -> virial (methods above return scaled values)
    const float scaleFactor = 0.5F / (dt * dt);
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            virial[i][j] *= scaleFactor;
        }
    }
}

std::vector<std::array<real, DIM * DIM>> UpdateConstrainGpu::Impl::takeConstraintVirialHistory()
{
    GMX_RELEASE_ASSERT(constraintVirialHistoryEnabled_,
                       "Deferred constraint virial history requested while disabled");
    copyFromDeviceBuffer(h_constraintVirialHistoryCount_.data(),
                         &d_constraintVirialHistoryCount_,
                         0,
                         1,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    const int sampleCount = h_constraintVirialHistoryCount_[0];
    GMX_RELEASE_ASSERT(sampleCount <= c_constraintVirialHistoryCapacity_,
                       "Deferred constraint virial history exceeded its fixed capacity");
    if (sampleCount == 0)
    {
        return {};
    }

    copyFromDeviceBuffer(h_constraintVirialHistory_.data(),
                         &d_constraintVirialHistory_,
                         0,
                         sampleCount * DIM * DIM,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    std::vector<std::array<real, DIM * DIM>> samples(sampleCount);
    for (int sample = 0; sample < sampleCount; ++sample)
    {
        std::copy_n(h_constraintVirialHistory_.data() + sample * DIM * DIM,
                    DIM * DIM,
                    samples[sample].begin());
    }
    clearDeviceBufferAsync(&d_constraintVirialHistoryCount_, 0, 1, deviceStream_);
    return samples;
}

GpuKineticEnergyHistory UpdateConstrainGpu::Impl::takeKineticEnergyHistory()
{
    GMX_RELEASE_ASSERT(constraintVirialHistoryEnabled_,
                       "Deferred kinetic history requested while disabled");
    copyFromDeviceBuffer(h_kineticHistoryCount_.data(),
                         &d_kineticHistoryCount_,
                         0,
                         1,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    const int sampleCount = h_kineticHistoryCount_[0];
    GMX_RELEASE_ASSERT(sampleCount <= c_constraintVirialHistoryCapacity_,
                       "Deferred kinetic history exceeded its fixed capacity");
    GpuKineticEnergyHistory history;
    history.numGroups = numTemperatureGroups_;
    if (sampleCount == 0)
    {
        return history;
    }

    const int historyStride = DIM * DIM + numTemperatureGroups_;
    const int valueCount    = sampleCount * historyStride;
    copyFromDeviceBuffer(h_kineticHistory_.data(),
                         &d_kineticHistory_,
                         0,
                         valueCount,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    history.totalTensors.resize(sampleCount);
    history.groupKineticEnergies.resize(sampleCount * numTemperatureGroups_);
    for (int sample = 0; sample < sampleCount; ++sample)
    {
        const float* source = h_kineticHistory_.data() + sample * historyStride;
        std::copy_n(source, DIM * DIM, history.totalTensors[sample].begin());
        std::copy_n(source + DIM * DIM,
                    numTemperatureGroups_,
                    history.groupKineticEnergies.begin() + sample * numTemperatureGroups_);
    }
    clearDeviceBufferAsync(&d_kineticHistoryCount_, 0, 1, deviceStream_);
    return history;
}

std::vector<std::array<real, DIM * DIM>> UpdateConstrainGpu::Impl::previousHalfStepKineticEnergy()
{
    GMX_RELEASE_ASSERT(constraintVirialHistoryEnabled_,
                       "Previous GPU kinetic state requested while disabled");
    clearDeviceBufferAsync(&d_groupHalfKineticTensor_, 0, numTemperatureGroups_ * DIM * DIM, deviceStream_);
    constexpr int      c_kineticThreadsPerBlock = 256;
    KernelLaunchConfig config;
    config.blockSize[0]     = c_kineticThreadsPerBlock;
    config.blockSize[1]     = 1;
    config.blockSize[2]     = 1;
    config.gridSize[0]      = (numAtoms_ + c_kineticThreadsPerBlock - 1) / c_kineticThreadsPerBlock;
    config.gridSize[1]      = 1;
    config.gridSize[2]      = 1;
    config.sharedMemorySize = numTemperatureGroups_ * DIM * DIM * sizeof(float);
    auto       kernel       = accumulateGroupHalfKineticTensorKernel;
    const auto kernelArgs   = prepareGpuKernelArguments(
            kernel, config, &numAtoms_, &numTemperatureGroups_, &d_v_, &d_masses_, &d_temperatureGroups_, &d_groupHalfKineticTensor_);
    launchGpuKernel(kernel,
                    config,
                    deviceStream_,
                    nullptr,
                    "Stage temperature-group half-step kinetic energy",
                    kernelArgs);
    copyFromDeviceBuffer(h_previousHalfKinetic_.data(),
                         &d_groupHalfKineticTensor_,
                         0,
                         numTemperatureGroups_ * DIM * DIM,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    std::vector<std::array<real, DIM * DIM>> tensors(numTemperatureGroups_);
    for (int group = 0; group < numTemperatureGroups_; ++group)
    {
        std::copy_n(h_previousHalfKinetic_.data() + group * DIM * DIM, DIM * DIM, tensors[group].begin());
    }
    return tensors;
}

void UpdateConstrainGpu::Impl::scaleCoordinates(const Matrix3x3& scalingMatrix)
{
    if (numAtoms_ == 0)
    {
        return;
    }

    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);

    ScalingMatrix mu(scalingMatrix);

    launchScaleCoordinatesKernel(numAtoms_, d_x_, mu, deviceStream_);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpuPp);
}

void UpdateConstrainGpu::Impl::scaleVelocities(const Matrix3x3& scalingMatrix)
{
    if (numAtoms_ == 0)
    {
        return;
    }

    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);

    ScalingMatrix mu(scalingMatrix);

    launchScaleCoordinatesKernel(numAtoms_, d_v_, mu, deviceStream_);

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuUpdateConstrain);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpuPp);
}

UpdateConstrainGpu::Impl::Impl(const t_inputrec&    ir,
                               const gmx_mtop_t&    mtop,
                               const int            numTempScaleValues,
                               const DeviceContext& deviceContext,
                               const DeviceStream&  deviceStream,
                               gmx_wallcycle*       wcycle) :
    deviceContext_(deviceContext), deviceStream_(deviceStream), wcycle_(wcycle)
{
    numTemperatureGroups_ = numTempScaleValues;
    integrator_ = std::make_unique<LeapFrogGpu>(deviceContext_, deviceStream_, numTempScaleValues);
    if (sc_haveGpuConstraintSupport)
    {
        lincsGpu_ = std::make_unique<LincsGpu>(ir.nLincsIter, ir.nProjOrder, deviceContext_, deviceStream_);
        settleGpu_ = std::make_unique<SettleGpu>(mtop, deviceContext_, deviceStream_);
    }
#if GMX_GPU_CUDA
    const char* residentEnergyRequest = std::getenv("GMX_GAMD_GPU_RESIDENT_ENERGY");
    constraintVirialHistoryEnabled_ =
            residentEnergyRequest != nullptr && std::strcmp(residentEnergyRequest, "1") == 0;
    const char* deviceGlobalsRequest = std::getenv("GMX_GAMD_GPU_DEVICE_GLOBALS");
    deviceGlobalHistoryEnabled_ = constraintVirialHistoryEnabled_ && deviceGlobalsRequest != nullptr
                                  && std::strcmp(deviceGlobalsRequest, "1") == 0;
    if (constraintVirialHistoryEnabled_)
    {
        allocateDeviceBuffer(&d_constraintVirialHistory_,
                             c_constraintVirialHistoryCapacity_ * DIM * DIM,
                             deviceContext_);
        allocateDeviceBuffer(&d_constraintVirialHistoryCount_, 1, deviceContext_);
        clearDeviceBufferAsync(&d_constraintVirialHistoryCount_, 0, 1, deviceStream_);
        h_constraintVirialHistory_.resize(c_constraintVirialHistoryCapacity_ * DIM * DIM);
        h_constraintVirialHistoryCount_.resize(1);
        if (deviceGlobalHistoryEnabled_)
        {
            GMX_RELEASE_ASSERT(numTemperatureGroups_ > 0,
                               "Deferred kinetic history requires temperature groups");
            GMX_RELEASE_ASSERT(
                    numTemperatureGroups_ <= 2,
                    "Deferred kinetic history currently supports at most two temperature groups");
            allocateDeviceBuffer(&d_currentHalfKinetic_, 6 + numTemperatureGroups_, deviceContext_);
            allocateDeviceBuffer(&d_previousHalfKinetic_, 6 + numTemperatureGroups_, deviceContext_);
            allocateDeviceBuffer(
                    &d_groupHalfKineticTensor_, numTemperatureGroups_ * DIM * DIM, deviceContext_);
            allocateDeviceBuffer(&d_kineticHistory_,
                                 c_constraintVirialHistoryCapacity_ * (DIM * DIM + numTemperatureGroups_),
                                 deviceContext_);
            allocateDeviceBuffer(&d_kineticHistoryCount_, 1, deviceContext_);
            clearDeviceBufferAsync(&d_previousHalfKinetic_, 0, 6 + numTemperatureGroups_, deviceStream_);
            clearDeviceBufferAsync(&d_kineticHistoryCount_, 0, 1, deviceStream_);
            h_kineticHistory_.resize(c_constraintVirialHistoryCapacity_
                                     * (DIM * DIM + numTemperatureGroups_));
            h_previousHalfKinetic_.resize(numTemperatureGroups_ * DIM * DIM);
            h_kineticHistoryCount_.resize(1);
        }
    }
#endif
}

UpdateConstrainGpu::Impl::~Impl()
{
    freeDeviceBuffer(&d_x0_);
    freeDeviceBuffer(&d_inverseMasses_);
    if (d_constraintVirialHistory_ != nullptr)
    {
        freeDeviceBuffer(&d_constraintVirialHistory_);
    }
    if (d_constraintVirialHistoryCount_ != nullptr)
    {
        freeDeviceBuffer(&d_constraintVirialHistoryCount_);
    }
    if (d_masses_ != nullptr)
    {
        freeDeviceBuffer(&d_masses_);
    }
    if (d_temperatureGroups_ != nullptr)
    {
        freeDeviceBuffer(&d_temperatureGroups_);
    }
    if (d_currentHalfKinetic_ != nullptr)
    {
        freeDeviceBuffer(&d_currentHalfKinetic_);
    }
    if (d_previousHalfKinetic_ != nullptr)
    {
        freeDeviceBuffer(&d_previousHalfKinetic_);
    }
    if (d_groupHalfKineticTensor_ != nullptr)
    {
        freeDeviceBuffer(&d_groupHalfKineticTensor_);
    }
    if (d_kineticHistory_ != nullptr)
    {
        freeDeviceBuffer(&d_kineticHistory_);
    }
    if (d_kineticHistoryCount_ != nullptr)
    {
        freeDeviceBuffer(&d_kineticHistoryCount_);
    }
}

void UpdateConstrainGpu::Impl::set(DeviceBuffer<Float3>          d_x,
                                   DeviceBuffer<Float3>          d_v,
                                   const DeviceBuffer<Float3>    d_f,
                                   const InteractionDefinitions& idef,
                                   const t_mdatoms&              md)
{
    wallcycle_start(wcycle_, WallCycleCounter::GpuSetConstr);

    GMX_ASSERT(d_x, "Coordinates device buffer should not be null.");
    GMX_ASSERT(d_v, "Velocities device buffer should not be null.");
    GMX_ASSERT(d_f, "Forces device buffer should not be null.");

    d_x_ = d_x;
    d_v_ = d_v;
    d_f_ = d_f;

    numAtoms_ = md.homenr;

    reallocateDeviceBuffer(&d_x0_, numAtoms_, &numXp_, &numXpAlloc_, deviceContext_);

    reallocateDeviceBuffer(
            &d_inverseMasses_, numAtoms_, &numInverseMasses_, &numInverseMassesAlloc_, deviceContext_);

    if (deviceGlobalHistoryEnabled_)
    {
        reallocateDeviceBuffer(
                &d_masses_, numAtoms_, &numKineticAtoms_, &numKineticAtomsAlloc_, deviceContext_);
        reallocateDeviceBuffer(&d_temperatureGroups_,
                               numAtoms_,
                               &numTemperatureGroupAtoms_,
                               &numTemperatureGroupAtomsAlloc_,
                               deviceContext_);
        copyToDeviceBuffer(
                &d_masses_, md.massT.data(), 0, numAtoms_, deviceStream_, GpuApiCallBehavior::Sync, nullptr);
        if (md.cTC.empty())
        {
            std::vector<unsigned short> temperatureGroups(numAtoms_, 0);
            copyToDeviceBuffer(&d_temperatureGroups_,
                               temperatureGroups.data(),
                               0,
                               numAtoms_,
                               deviceStream_,
                               GpuApiCallBehavior::Sync,
                               nullptr);
        }
        else
        {
            copyToDeviceBuffer(
                    &d_temperatureGroups_, md.cTC.data(), 0, numAtoms_, deviceStream_, GpuApiCallBehavior::Sync, nullptr);
        }
    }

    // Integrator should also update something, but it does not even have a method yet
    integrator_->set(numAtoms_, md.invmass, md.cTC);
    if (sc_haveGpuConstraintSupport)
    {
        wallcycle_sub_start(wcycle_, WallCycleSubCounter::GpuSetLincs);
        lincsGpu_->set(idef, numAtoms_, md.invmass);
        wallcycle_sub_stop(wcycle_, WallCycleSubCounter::GpuSetLincs);
        wallcycle_sub_start(wcycle_, WallCycleSubCounter::GpuSetSettle);
        settleGpu_->set(idef);
        wallcycle_sub_stop(wcycle_, WallCycleSubCounter::GpuSetSettle);
    }
    else
    {
        GMX_ASSERT(idef.il[F_SETTLE].empty(), "SETTLE not supported");
        GMX_ASSERT(idef.il[F_CONSTR].empty(), "LINCS not supported");
    }

    wallcycle_stop(wcycle_, WallCycleCounter::GpuSetConstr);
}

void UpdateConstrainGpu::Impl::setPbc(const PbcType pbcType, const matrix box)
{
    // TODO wallcycle
    setPbcAiuc(numPbcDimensions(pbcType), box, &pbcAiuc_);
}

GpuEventSynchronizer* UpdateConstrainGpu::Impl::xUpdatedOnDeviceEvent()
{
    return &xUpdatedOnDeviceEvent_;
}

UpdateConstrainGpu::UpdateConstrainGpu(const t_inputrec&    ir,
                                       const gmx_mtop_t&    mtop,
                                       const int            numTempScaleValues,
                                       const DeviceContext& deviceContext,
                                       const DeviceStream&  deviceStream,
                                       gmx_wallcycle*       wcycle) :
    impl_(new Impl(ir, mtop, numTempScaleValues, deviceContext, deviceStream, wcycle))
{
}

UpdateConstrainGpu::~UpdateConstrainGpu() = default;

void UpdateConstrainGpu::integrate(GpuEventSynchronizer*             fReadyOnDevice,
                                   const real                        dt,
                                   const bool                        updateVelocities,
                                   const bool                        computeVirial,
                                   const bool                        deferVirialToHost,
                                   tensor                            virialScaled,
                                   const bool                        doTemperatureScaling,
                                   gmx::ArrayRef<const t_grp_tcstat> tcstat,
                                   const bool                        doParrinelloRahman,
                                   const float                       dtPressureCouple,
                                   const gmx::Matrix3x3&             prVelocityScalingMatrix)
{
    impl_->integrate(fReadyOnDevice,
                     dt,
                     updateVelocities,
                     computeVirial,
                     deferVirialToHost,
                     virialScaled,
                     doTemperatureScaling,
                     tcstat,
                     doParrinelloRahman,
                     dtPressureCouple,
                     prVelocityScalingMatrix);
}

void UpdateConstrainGpu::copyConstraintVirialToHost(const real dt, tensor virial)
{
    impl_->copyConstraintVirialToHost(dt, virial);
}

std::vector<std::array<real, DIM * DIM>> UpdateConstrainGpu::takeConstraintVirialHistory()
{
    return impl_->takeConstraintVirialHistory();
}

GpuKineticEnergyHistory UpdateConstrainGpu::takeKineticEnergyHistory()
{
    return impl_->takeKineticEnergyHistory();
}

std::vector<std::array<real, DIM * DIM>> UpdateConstrainGpu::previousHalfStepKineticEnergy()
{
    return impl_->previousHalfStepKineticEnergy();
}

void UpdateConstrainGpu::scaleCoordinates(const gmx::Matrix3x3& scalingMatrix)
{
    impl_->scaleCoordinates(scalingMatrix);
}

void UpdateConstrainGpu::scaleVelocities(const gmx::Matrix3x3& scalingMatrix)
{
    impl_->scaleVelocities(scalingMatrix);
}

void UpdateConstrainGpu::set(DeviceBuffer<Float3>          d_x,
                             DeviceBuffer<Float3>          d_v,
                             const DeviceBuffer<Float3>    d_f,
                             const InteractionDefinitions& idef,
                             const t_mdatoms&              md)
{
    impl_->set(d_x, d_v, d_f, idef, md);
}

void UpdateConstrainGpu::setPbc(const PbcType pbcType, const matrix box)
{
    impl_->setPbc(pbcType, box);
}

GpuEventSynchronizer* UpdateConstrainGpu::xUpdatedOnDeviceEvent()
{
    return impl_->xUpdatedOnDeviceEvent();
}

bool UpdateConstrainGpu::isNumCoupledConstraintsSupported(const gmx_mtop_t& mtop)
{
    return LincsGpu::isNumCoupledConstraintsSupported(mtop);
}

bool UpdateConstrainGpu::areConstraintsSupported()
{
    return sc_haveGpuConstraintSupport;
}

} // namespace gmx
