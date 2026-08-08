/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 */

#include "gmxpre.h"

#include <cstdlib>

#include <algorithm>

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#include "gamdgpu_impl.h"

namespace gmx
{

namespace
{

bool hasPerturbedEntries(const InteractionDefinitions& interactionDefinitions, int functionType)
{
    GMX_ASSERT(interactionDefinitions.ilsort == ilsortNO_FE || interactionDefinitions.ilsort == ilsortFE_SORTED,
               "Perturbed interactions should be sorted here");
    const InteractionList& interactionList = interactionDefinitions.il[functionType];
    return interactionDefinitions.ilsort != ilsortNO_FE
           && interactionDefinitions.numNonperturbedInteractions[functionType] != interactionList.size();
}

} // namespace

GaMDGpuExecutor::Impl::Impl(const bool           useGpuResidentGaMD,
                            const DeviceContext& deviceContext,
                            const DeviceStream&  deviceStream,
                            gmx_wallcycle*       wallCycleCounters) :
    deviceContext_(deviceContext), deviceStream_(deviceStream), wallCycleCounters_(wallCycleCounters)
{
    GMX_RELEASE_ASSERT(deviceStream_.isValid(),
                       "GaMD GPU execution requires a valid bonded-force stream");

    matrix box = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    setPbcAiuc(0, box, &pbcAiuc_);

    gamdDihedralShadowEnabled_          = std::getenv("GMX_GAMD_GPU_DIH_SHADOW") != nullptr;
    gamdEnergyShadowDiagnosticsEnabled_ = std::getenv("GMX_GAMD_GPU_ENERGY_SHADOW") != nullptr;
    gamdForceCorrectionEnabled_         = useGpuResidentGaMD;
    gamdScaleFromDeviceEnabled_         = useGpuResidentGaMD;
    gamdEnergyHistoryEnabled_           = useGpuResidentGaMD;
    gamdEnergyShadowEnabled_   = gamdEnergyShadowDiagnosticsEnabled_ || gamdScaleFromDeviceEnabled_;
    gamdDihedralBufferEnabled_ = gamdDihedralShadowEnabled_ || gamdForceCorrectionEnabled_;

    if (gamdForceCorrectionEnabled_)
    {
        if (gamdDihedralShadowEnabled_)
        {
            gamdForcesReadyEvent_ = std::make_unique<GpuEventSynchronizer>(2, 2);
        }
        else
        {
            gamdForcesReadyEvent_ = std::make_unique<GpuEventSynchronizer>();
        }
    }

    if (gamdDihedralBufferEnabled_)
    {
        allocateDeviceBuffer(&d_gamdDihedralShiftForces_, c_numShiftVectors, deviceContext_);
        allocateDeviceBuffer(&d_gamdDihedralVirial_, DIM * DIM, deviceContext_);
        allocateDeviceBuffer(&d_gamdShortRangeVirial_, DIM * DIM, deviceContext_);
    }

    if (gamdEnergyShadowEnabled_)
    {
        // 第一个元素保存 reciprocal energy，后面九个元素保存完整 virial 张量。
        allocateDeviceBuffer(&d_gamdPmeEnergy_, 1 + DIM * DIM, deviceContext_);
        allocateDeviceBuffer(&d_gamdEnergyShadow_, 6, deviceContext_);
        if (gamdEnergyHistoryEnabled_)
        {
            allocateDeviceBuffer(&d_gamdEnergyHistory_, c_historyCapacity_ * F_NRE, deviceContext_);
            allocateDeviceBuffer(&d_gamdForceVirialHistory_, c_historyCapacity_ * DIM * DIM, deviceContext_);
            allocateDeviceBuffer(&d_gamdEnergyHistoryCount_, 1, deviceContext_);
            clearDeviceBufferAsync(&d_gamdEnergyHistoryCount_, 0, 1, deviceStream_);
            h_gamdEnergyHistory_.resize(c_historyCapacity_ * F_NRE);
            h_gamdForceVirialHistory_.resize(c_historyCapacity_ * DIM * DIM);
            h_gamdEnergyHistoryCount_.resize(1);
        }
        if (gamdEnergyShadowDiagnosticsEnabled_)
        {
            h_gamdEnergyShadow_.resize(6);
        }
        gamdPmeEnergyReadyEvent_ = std::make_unique<GpuEventSynchronizer>();
    }
}

GaMDGpuExecutor::Impl::~Impl()
{
    deviceStream_.synchronize();
    if (updateStream_ != nullptr && updateStream_ != &deviceStream_)
    {
        // 两条流都可能仍在使用模块缓冲，释放前必须分别完成排队工作。
        updateStream_->synchronize();
    }

    if (d_gamdDihedralForcesNbnxn_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralForcesNbnxn_);
    }
    if (d_gamdDihedralForcesState_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralForcesState_);
    }
    if (d_gamdDihedralShiftForces_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralShiftForces_);
    }
    if (d_gamdDihedralVirial_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralVirial_);
    }
    if (d_gamdShortRangeVirial_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdShortRangeVirial_);
    }
    if (d_gamdCorrectedForcesState_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdCorrectedForcesState_);
    }
    if (d_nbnxnAtomOrder_ != nullptr)
    {
        freeDeviceBuffer(&d_nbnxnAtomOrder_);
    }
    if (d_gamdEnergyShadow_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdEnergyShadow_);
    }
    if (d_gamdPmeEnergy_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdPmeEnergy_);
    }
    if (d_gamdEnergyHistory_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdEnergyHistory_);
    }
    if (d_gamdForceVirialHistory_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdForceVirialHistory_);
    }
    if (d_gamdEnergyHistoryCount_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdEnergyHistoryCount_);
    }
    if (d_updateMasses_ != nullptr)
    {
        freeDeviceBuffer(&d_updateMasses_);
    }
    if (d_updateTemperatureGroups_ != nullptr)
    {
        freeDeviceBuffer(&d_updateTemperatureGroups_);
    }
    if (d_updateConstraintVirialHistory_ != nullptr)
    {
        freeDeviceBuffer(&d_updateConstraintVirialHistory_);
    }
    if (d_updateConstraintVirialHistoryCount_ != nullptr)
    {
        freeDeviceBuffer(&d_updateConstraintVirialHistoryCount_);
    }
    if (d_updateCurrentHalfKinetic_ != nullptr)
    {
        freeDeviceBuffer(&d_updateCurrentHalfKinetic_);
    }
    if (d_updatePreviousHalfKinetic_ != nullptr)
    {
        freeDeviceBuffer(&d_updatePreviousHalfKinetic_);
    }
    if (d_updateGroupHalfKineticTensor_ != nullptr)
    {
        freeDeviceBuffer(&d_updateGroupHalfKineticTensor_);
    }
    if (d_updateKineticHistory_ != nullptr)
    {
        freeDeviceBuffer(&d_updateKineticHistory_);
    }
    if (d_updateKineticHistoryCount_ != nullptr)
    {
        freeDeviceBuffer(&d_updateKineticHistoryCount_);
    }
}

void GaMDGpuExecutor::Impl::initializeUpdateHistory(const int           numTemperatureGroups,
                                                    const DeviceStream& updateStream)
{
    GMX_RELEASE_ASSERT(gamdForceCorrectionEnabled_,
                       "GaMD update history requires the GPU-resident execution mode");
    GMX_RELEASE_ASSERT(updateStream_ == nullptr,
                       "GaMD update history must be initialized exactly once");
    GMX_RELEASE_ASSERT(updateStream.isValid(), "GaMD update history requires a valid update stream");
    GMX_RELEASE_ASSERT(numTemperatureGroups > 0,
                       "GaMD deferred kinetic history requires temperature groups");
    GMX_RELEASE_ASSERT(
            numTemperatureGroups <= 2,
            "GaMD deferred kinetic history currently supports at most two temperature groups");

    updateStream_               = &updateStream;
    updateNumTemperatureGroups_ = numTemperatureGroups;

    // 固定容量覆盖两次 host staging 之间的生产步，避免普通步重新分配和同步。
    allocateDeviceBuffer(&d_updateConstraintVirialHistory_, c_historyCapacity_ * DIM * DIM, deviceContext_);
    allocateDeviceBuffer(&d_updateConstraintVirialHistoryCount_, 1, deviceContext_);
    allocateDeviceBuffer(&d_updateCurrentHalfKinetic_, 6 + updateNumTemperatureGroups_, deviceContext_);
    allocateDeviceBuffer(&d_updatePreviousHalfKinetic_, 6 + updateNumTemperatureGroups_, deviceContext_);
    allocateDeviceBuffer(
            &d_updateGroupHalfKineticTensor_, updateNumTemperatureGroups_ * DIM * DIM, deviceContext_);
    allocateDeviceBuffer(&d_updateKineticHistory_,
                         c_historyCapacity_ * (DIM * DIM + updateNumTemperatureGroups_),
                         deviceContext_);
    allocateDeviceBuffer(&d_updateKineticHistoryCount_, 1, deviceContext_);

    clearDeviceBufferAsync(&d_updateConstraintVirialHistoryCount_, 0, 1, *updateStream_);
    clearDeviceBufferAsync(&d_updatePreviousHalfKinetic_, 0, 6 + updateNumTemperatureGroups_, *updateStream_);
    clearDeviceBufferAsync(&d_updateKineticHistoryCount_, 0, 1, *updateStream_);

    h_updateConstraintVirialHistory_.resize(c_historyCapacity_ * DIM * DIM);
    h_updateConstraintVirialHistoryCount_.resize(1);
    h_updateKineticHistory_.resize(c_historyCapacity_ * (DIM * DIM + updateNumTemperatureGroups_));
    h_updatePreviousHalfKinetic_.resize(updateNumTemperatureGroups_ * DIM * DIM);
    h_updateKineticHistoryCount_.resize(1);
}

void GaMDGpuExecutor::Impl::bindUpdateBuffers(DeviceBuffer<Float3>           velocities,
                                              ArrayRef<const real>           masses,
                                              ArrayRef<const unsigned short> temperatureGroups)
{
    GMX_RELEASE_ASSERT(updateStream_ != nullptr,
                       "GaMD update history must be initialized before binding update buffers");
    GMX_RELEASE_ASSERT(velocities != nullptr && !masses.empty(),
                       "GaMD update history requires velocity and mass buffers");
    GMX_RELEASE_ASSERT(temperatureGroups.empty() || temperatureGroups.size() == masses.size(),
                       "GaMD temperature-group data must match the local atom count");

    updateVelocities_ = velocities;
    updateNumAtoms_   = masses.ssize();
    // 质量和温度组由模块保存设备副本，update 层只保留通用积分数据。
    reallocateDeviceBuffer(
            &d_updateMasses_, updateNumAtoms_, &updateMassCount_, &updateMassCapacity_, deviceContext_);
    reallocateDeviceBuffer(&d_updateTemperatureGroups_,
                           updateNumAtoms_,
                           &updateTemperatureGroupCount_,
                           &updateTemperatureGroupCapacity_,
                           deviceContext_);
    copyToDeviceBuffer(
            &d_updateMasses_, masses.data(), 0, updateNumAtoms_, *updateStream_, GpuApiCallBehavior::Sync, nullptr);

    if (temperatureGroups.empty())
    {
        std::vector<unsigned short> defaultTemperatureGroups(updateNumAtoms_, 0);
        copyToDeviceBuffer(&d_updateTemperatureGroups_,
                           defaultTemperatureGroups.data(),
                           0,
                           updateNumAtoms_,
                           *updateStream_,
                           GpuApiCallBehavior::Sync,
                           nullptr);
    }
    else
    {
        copyToDeviceBuffer(&d_updateTemperatureGroups_,
                           temperatureGroups.data(),
                           0,
                           updateNumAtoms_,
                           *updateStream_,
                           GpuApiCallBehavior::Sync,
                           nullptr);
    }
}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::Impl::takeUpdateConstraintVirialHistory()
{
    GMX_RELEASE_ASSERT(updateStream_ != nullptr,
                       "GaMD deferred constraint-virial history requested before initialization");
    copyFromDeviceBuffer(h_updateConstraintVirialHistoryCount_.data(),
                         &d_updateConstraintVirialHistoryCount_,
                         0,
                         1,
                         *updateStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    const int sampleCount = h_updateConstraintVirialHistoryCount_[0];
    GMX_RELEASE_ASSERT(sampleCount <= c_historyCapacity_,
                       "GaMD deferred constraint-virial history exceeded its fixed capacity");
    if (sampleCount == 0)
    {
        return {};
    }

    copyFromDeviceBuffer(h_updateConstraintVirialHistory_.data(),
                         &d_updateConstraintVirialHistory_,
                         0,
                         sampleCount * DIM * DIM,
                         *updateStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    std::vector<std::array<real, DIM * DIM>> samples(sampleCount);
    for (int sample = 0; sample < sampleCount; ++sample)
    {
        std::copy_n(h_updateConstraintVirialHistory_.data() + sample * DIM * DIM,
                    DIM * DIM,
                    samples[sample].begin());
    }
    clearDeviceBufferAsync(&d_updateConstraintVirialHistoryCount_, 0, 1, *updateStream_);
    return samples;
}

GaMDKineticEnergyHistory GaMDGpuExecutor::Impl::takeUpdateKineticEnergyHistory()
{
    GMX_RELEASE_ASSERT(updateStream_ != nullptr,
                       "GaMD deferred kinetic history requested before initialization");
    copyFromDeviceBuffer(h_updateKineticHistoryCount_.data(),
                         &d_updateKineticHistoryCount_,
                         0,
                         1,
                         *updateStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    const int sampleCount = h_updateKineticHistoryCount_[0];
    GMX_RELEASE_ASSERT(sampleCount <= c_historyCapacity_,
                       "GaMD deferred kinetic history exceeded its fixed capacity");
    GaMDKineticEnergyHistory history;
    history.numGroups = updateNumTemperatureGroups_;
    if (sampleCount == 0)
    {
        return history;
    }

    const int historyStride = DIM * DIM + updateNumTemperatureGroups_;
    copyFromDeviceBuffer(h_updateKineticHistory_.data(),
                         &d_updateKineticHistory_,
                         0,
                         sampleCount * historyStride,
                         *updateStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
    history.totalTensors.resize(sampleCount);
    history.groupKineticEnergies.resize(sampleCount * updateNumTemperatureGroups_);
    for (int sample = 0; sample < sampleCount; ++sample)
    {
        const float* source = h_updateKineticHistory_.data() + sample * historyStride;
        std::copy_n(source, DIM * DIM, history.totalTensors[sample].begin());
        std::copy_n(source + DIM * DIM,
                    updateNumTemperatureGroups_,
                    history.groupKineticEnergies.begin() + sample * updateNumTemperatureGroups_);
    }
    clearDeviceBufferAsync(&d_updateKineticHistoryCount_, 0, 1, *updateStream_);
    return history;
}

bool GaMDGpuExecutor::Impl::dihedralShadowEnabled() const
{
    return gamdDihedralShadowEnabled_;
}

bool GaMDGpuExecutor::Impl::dihedralBufferEnabled() const
{
    return gamdDihedralBufferEnabled_;
}

bool GaMDGpuExecutor::Impl::energyEnabled() const
{
    return gamdEnergyShadowEnabled_;
}

bool GaMDGpuExecutor::Impl::energyDiagnosticsEnabled() const
{
    return gamdEnergyShadowDiagnosticsEnabled_;
}

bool GaMDGpuExecutor::Impl::scaleFromDeviceEnabled() const
{
    return gamdScaleFromDeviceEnabled_;
}

void GaMDGpuExecutor::Impl::bindListedForceBuffers(ArrayRef<const int> nbnxnAtomOrder,
                                                   const InteractionDefinitions& interactionDefinitions,
                                                   DeviceBuffer<Float4> coordinatesAndCharges,
                                                   DeviceBuffer<Float3> forces,
                                                   DeviceBuffer<Float3> shiftForces,
                                                   DeviceBuffer<float> shortRangeLennardJonesEnergy,
                                                   DeviceBuffer<float> shortRangeElectrostaticEnergy,
                                                   DeviceBuffer<float> listedEnergies)
{
    if (gamdDihedralBufferEnabled_)
    {
        if (!interactionDefinitions.il[F_FOURDIHS].empty())
        {
            gmx_fatal(FARGS,
                      "The GaMD GPU force path does not support FOURDIHS. Use -update cpu or "
                      "add CUDA FOURDIHS support before using this topology.");
        }
        for (const int functionType : { F_PDIHS, F_RBDIHS, F_CMAP })
        {
            if (!interactionDefinitions.il[functionType].empty()
                && hasPerturbedEntries(interactionDefinitions, functionType))
            {
                gmx_fatal(FARGS,
                          "The GaMD GPU force path does not support perturbed %s interactions. "
                          "Use -update cpu for this topology.",
                          interaction_function[functionType].longname);
            }
        }

        GMX_RELEASE_ASSERT(!nbnxnAtomOrder.empty(),
                           "GaMD GPU dihedral forces require an atom-order mapping");
        const int nbnxnForceSize = *std::max_element(nbnxnAtomOrder.begin(), nbnxnAtomOrder.end()) + 1;
        reallocateDeviceBuffer(&d_gamdDihedralForcesNbnxn_,
                               nbnxnForceSize,
                               &gamdNbnxnForceSize_,
                               &gamdNbnxnForceCapacity_,
                               deviceContext_);
        reallocateDeviceBuffer(&d_gamdDihedralForcesState_,
                               nbnxnAtomOrder.ssize(),
                               &gamdStateForceSize_,
                               &gamdStateForceCapacity_,
                               deviceContext_);
        reallocateDeviceBuffer(&d_gamdCorrectedForcesState_,
                               nbnxnAtomOrder.ssize(),
                               &gamdCorrectedForceSize_,
                               &gamdCorrectedForceCapacity_,
                               deviceContext_);
        reallocateDeviceBuffer(&d_nbnxnAtomOrder_,
                               nbnxnAtomOrder.ssize(),
                               &nbnxnAtomOrderSize_,
                               &nbnxnAtomOrderCapacity_,
                               deviceContext_);
        copyToDeviceBuffer(&d_nbnxnAtomOrder_,
                           nbnxnAtomOrder.data(),
                           0,
                           nbnxnAtomOrder.ssize(),
                           deviceStream_,
                           GpuApiCallBehavior::Async,
                           nullptr);
    }

    coordinatesAndCharges_         = coordinatesAndCharges;
    forces_                        = forces;
    shiftForces_                   = shiftForces;
    shortRangeLennardJonesEnergy_  = shortRangeLennardJonesEnergy;
    shortRangeElectrostaticEnergy_ = shortRangeElectrostaticEnergy;
    listedEnergies_                = listedEnergies;
}

void GaMDGpuExecutor::Impl::setPbc(const PbcAiuc& pbcAiuc)
{
    pbcAiuc_ = pbcAiuc;
}

GaMDBondedDeviceBuffers GaMDGpuExecutor::Impl::prepareBondedForceAccumulation()
{
    GaMDBondedDeviceBuffers buffers;
    buffers.enabled = gamdDihedralBufferEnabled_;
    if (!buffers.enabled)
    {
        return buffers;
    }

    GMX_RELEASE_ASSERT(d_gamdDihedralForcesNbnxn_ != nullptr && d_gamdDihedralShiftForces_ != nullptr,
                       "GaMD listed-force buffers must be bound before launching GPU bondeds");
    clearDeviceBufferAsync(&d_gamdDihedralForcesNbnxn_, 0, gamdNbnxnForceSize_, deviceStream_);
    clearDeviceBufferAsync(&d_gamdDihedralShiftForces_, 0, c_numShiftVectors, deviceStream_);
    buffers.dihedralForces = d_gamdDihedralForcesNbnxn_;
    buffers.shiftForces    = d_gamdDihedralShiftForces_;
    return buffers;
}

void GaMDGpuExecutor::Impl::copyGamdDihedralShadowForces(ArrayRef<RVec> forces)
{
    GMX_RELEASE_ASSERT(gamdDihedralShadowEnabled_,
                       "GaMD GPU dihedral shadow-force copy requested while disabled");
    GMX_RELEASE_ASSERT(forces.ssize() == gamdStateForceSize_,
                       "GaMD GPU dihedral shadow-force host buffer has the wrong size");
    copyFromDeviceBuffer(reinterpret_cast<Float3*>(forces.data()),
                         &d_gamdDihedralForcesState_,
                         0,
                         gamdStateForceSize_,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
}

void GaMDGpuExecutor::Impl::copyGamdDihedralVirial(std::array<real, DIM * DIM>* virial)
{
    GMX_RELEASE_ASSERT(gamdDihedralBufferEnabled_,
                       "GaMD GPU dihedral virial copy requested while disabled");
    GMX_RELEASE_ASSERT(virial != nullptr, "GaMD GPU dihedral virial requires host storage");
    copyFromDeviceBuffer(
            virial->data(), &d_gamdDihedralVirial_, 0, virial->size(), deviceStream_, GpuApiCallBehavior::Sync, nullptr);
}

void GaMDGpuExecutor::Impl::copyGamdShortRangeVirial(std::array<real, DIM * DIM>* virial)
{
    GMX_RELEASE_ASSERT(gamdForceCorrectionEnabled_,
                       "GaMD GPU short-range virial copy requested while disabled");
    GMX_RELEASE_ASSERT(virial != nullptr, "GaMD GPU short-range virial requires host storage");
    copyFromDeviceBuffer(virial->data(),
                         &d_gamdShortRangeVirial_,
                         0,
                         virial->size(),
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
}

void GaMDGpuExecutor::Impl::applyGamdForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                                           real                 totalScale,
                                                           real           dihedralCorrectionScale,
                                                           ArrayRef<RVec> correctedForces)
{
    GMX_RELEASE_ASSERT(gamdDihedralShadowEnabled_,
                       "GaMD GPU force-correction shadow requested while disabled");
    GMX_RELEASE_ASSERT(rawForces.ssize() == gamdStateForceSize_ && correctedForces.ssize() == gamdStateForceSize_,
                       "GaMD GPU force-correction shadow buffers have the wrong size");
    copyToDeviceBuffer(&d_gamdCorrectedForcesState_,
                       reinterpret_cast<const Float3*>(rawForces.data()),
                       0,
                       gamdStateForceSize_,
                       deviceStream_,
                       GpuApiCallBehavior::Async,
                       nullptr);
    launchGamdForceCorrectionShadowKernel(totalScale, dihedralCorrectionScale);
    copyFromDeviceBuffer(reinterpret_cast<Float3*>(correctedForces.data()),
                         &d_gamdCorrectedForcesState_,
                         0,
                         gamdStateForceSize_,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
}

void GaMDGpuExecutor::Impl::applyGamdForceCorrection(DeviceBuffer<RVec>    forces,
                                                     real                  totalScale,
                                                     real                  dihedralCorrectionScale,
                                                     GpuEventSynchronizer* rawForcesReady)
{
    GMX_RELEASE_ASSERT(gamdForceCorrectionEnabled_,
                       "GaMD GPU force correction requested while disabled");
    GMX_RELEASE_ASSERT(forces != nullptr && rawForcesReady != nullptr,
                       "GaMD GPU force correction requires forces and a dependency event");
    launchGamdForceCorrectionKernel(forces, totalScale, dihedralCorrectionScale, rawForcesReady);
}

GpuEventSynchronizer* GaMDGpuExecutor::Impl::gamdForcesReadyEvent()
{
    GMX_RELEASE_ASSERT(gamdForcesReadyEvent_ != nullptr,
                       "GaMD corrected-force event requested while GPU correction is disabled");
    return gamdForcesReadyEvent_.get();
}

GaMDGpuExecutor::GaMDGpuExecutor(const bool           useGpuResidentGaMD,
                                 const DeviceContext& deviceContext,
                                 const DeviceStream&  deviceStream,
                                 gmx_wallcycle*       wallCycleCounters) :
    impl_(std::make_unique<Impl>(useGpuResidentGaMD, deviceContext, deviceStream, wallCycleCounters))
{
}

GaMDGpuExecutor::~GaMDGpuExecutor() = default;

bool GaMDGpuExecutor::dihedralShadowEnabled() const
{
    return impl_->dihedralShadowEnabled();
}

bool GaMDGpuExecutor::dihedralBufferEnabled() const
{
    return impl_->dihedralBufferEnabled();
}

bool GaMDGpuExecutor::energyEnabled() const
{
    return impl_->energyEnabled();
}

bool GaMDGpuExecutor::energyDiagnosticsEnabled() const
{
    return impl_->energyDiagnosticsEnabled();
}

bool GaMDGpuExecutor::scaleFromDeviceEnabled() const
{
    return impl_->scaleFromDeviceEnabled();
}

void GaMDGpuExecutor::bindListedForceBuffers(ArrayRef<const int>           nbnxnAtomOrder,
                                             const InteractionDefinitions& interactionDefinitions,
                                             DeviceBuffer<Float4>          coordinatesAndCharges,
                                             DeviceBuffer<Float3>          forces,
                                             DeviceBuffer<Float3>          shiftForces,
                                             DeviceBuffer<float> shortRangeLennardJonesEnergy,
                                             DeviceBuffer<float> shortRangeElectrostaticEnergy,
                                             DeviceBuffer<float> listedEnergies)
{
    impl_->bindListedForceBuffers(nbnxnAtomOrder,
                                  interactionDefinitions,
                                  coordinatesAndCharges,
                                  forces,
                                  shiftForces,
                                  shortRangeLennardJonesEnergy,
                                  shortRangeElectrostaticEnergy,
                                  listedEnergies);
}

void GaMDGpuExecutor::setPbc(const PbcAiuc& pbcAiuc)
{
    impl_->setPbc(pbcAiuc);
}

GaMDBondedDeviceBuffers GaMDGpuExecutor::prepareBondedForceAccumulation()
{
    return impl_->prepareBondedForceAccumulation();
}

void GaMDGpuExecutor::finishBondedForceAccumulation()
{
    impl_->finishGamdDihedralBuffers();
}

DeviceBuffer<float> GaMDGpuExecutor::pmeEnergyStagingBuffer()
{
    return impl_->gamdPmeEnergyStagingBuffer();
}

GpuEventSynchronizer* GaMDGpuExecutor::pmeEnergyReadyEvent()
{
    return impl_->gamdPmeEnergyReadyEvent();
}

void GaMDGpuExecutor::launchEnergyReduction(const int    boostType,
                                            const int    stage,
                                            const double thresholdP,
                                            const double forceConstantP,
                                            const double thresholdD,
                                            const double forceConstantD,
                                            const bool   recordEnergySample)
{
    impl_->launchGamdEnergyShadowReduction(
            boostType, stage, thresholdP, forceConstantP, thresholdD, forceConstantD, recordEnergySample);
}

std::array<double, 6> GaMDGpuExecutor::energyDiagnostics()
{
    return impl_->gamdEnergyShadowValues();
}

std::vector<std::array<real, F_NRE>> GaMDGpuExecutor::takeEnergyHistory()
{
    return impl_->takeGamdEnergyHistory();
}

void GaMDGpuExecutor::recordForceVirialSample()
{
    impl_->recordGamdForceVirialSample();
}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::takeForceVirialHistory(const int numSamples)
{
    return impl_->takeGamdForceVirialHistory(numSamples);
}

void GaMDGpuExecutor::initializeUpdateHistory(const int numTemperatureGroups, const DeviceStream& updateStream)
{
    impl_->initializeUpdateHistory(numTemperatureGroups, updateStream);
}

void GaMDGpuExecutor::bindUpdateBuffers(DeviceBuffer<Float3>           velocities,
                                        ArrayRef<const real>           masses,
                                        ArrayRef<const unsigned short> temperatureGroups)
{
    impl_->bindUpdateBuffers(velocities, masses, temperatureGroups);
}

void GaMDGpuExecutor::recordConstraintVirial(DeviceBuffer<float> lincsVirial,
                                             const bool          haveLincs,
                                             DeviceBuffer<float> settleVirial,
                                             const bool          haveSettles,
                                             const real          scaleFactor)
{
    impl_->recordUpdateConstraintVirial(lincsVirial, haveLincs, settleVirial, haveSettles, scaleFactor);
}

void GaMDGpuExecutor::recordKineticEnergy(const bool recordSample)
{
    impl_->recordUpdateKineticEnergy(recordSample);
}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::takeConstraintVirialHistory()
{
    return impl_->takeUpdateConstraintVirialHistory();
}

GaMDKineticEnergyHistory GaMDGpuExecutor::takeKineticEnergyHistory()
{
    return impl_->takeUpdateKineticEnergyHistory();
}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::previousHalfStepKineticEnergy()
{
    return impl_->takePreviousHalfStepKineticEnergy();
}

void GaMDGpuExecutor::copyDihedralShadowForces(ArrayRef<RVec> forces)
{
    impl_->copyGamdDihedralShadowForces(forces);
}

void GaMDGpuExecutor::copyDihedralVirial(std::array<real, DIM * DIM>* virial)
{
    impl_->copyGamdDihedralVirial(virial);
}

void GaMDGpuExecutor::launchShortRangeVirialReduction()
{
    impl_->launchGamdShortRangeVirialReduction();
}

void GaMDGpuExecutor::copyShortRangeVirial(std::array<real, DIM * DIM>* virial)
{
    impl_->copyGamdShortRangeVirial(virial);
}

void GaMDGpuExecutor::applyForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                                 const real           totalScale,
                                                 const real           dihedralCorrectionScale,
                                                 ArrayRef<RVec>       correctedForces)
{
    impl_->applyGamdForceCorrectionShadow(rawForces, totalScale, dihedralCorrectionScale, correctedForces);
}

void GaMDGpuExecutor::applyForceCorrection(DeviceBuffer<RVec>    forces,
                                           const real            totalScale,
                                           const real            dihedralCorrectionScale,
                                           GpuEventSynchronizer* rawForcesReady)
{
    impl_->applyGamdForceCorrection(forces, totalScale, dihedralCorrectionScale, rawForcesReady);
}

GpuEventSynchronizer* GaMDGpuExecutor::forcesReadyEvent()
{
    return impl_->gamdForcesReadyEvent();
}

} // namespace gmx
