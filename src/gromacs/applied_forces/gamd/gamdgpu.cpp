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

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/ifunc.h"
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
            allocateDeviceBuffer(&d_gamdEnergyHistory_, c_gamdEnergyHistoryCapacity_ * F_NRE, deviceContext_);
            allocateDeviceBuffer(
                    &d_gamdForceVirialHistory_, c_gamdEnergyHistoryCapacity_ * DIM * DIM, deviceContext_);
            allocateDeviceBuffer(&d_gamdEnergyHistoryCount_, 1, deviceContext_);
            clearDeviceBufferAsync(&d_gamdEnergyHistoryCount_, 0, 1, deviceStream_);
            h_gamdEnergyHistory_.resize(c_gamdEnergyHistoryCapacity_ * F_NRE);
            h_gamdForceVirialHistory_.resize(c_gamdEnergyHistoryCapacity_ * DIM * DIM);
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
