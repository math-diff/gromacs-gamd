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
 * \brief Owns CUDA buffers, bindings and deferred GaMD sample batches.
 *
 * \ingroup module_applied_forces
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
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "gamdgpu_impl.h"

//! \cond internal
namespace gmx
{

namespace
{

/*! \internal \brief Checks whether the listed interaction type contains unsupported perturbations. */
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
    useGpuResidentGaMD_(useGpuResidentGaMD),
    deviceContext_(deviceContext),
    deviceStream_(deviceStream),
    wallCycleCounters_(wallCycleCounters)
{
    GMX_RELEASE_ASSERT(deviceStream_.isValid(),
                       "GaMD GPU execution requires a valid bonded-force stream");

    matrix box = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    setPbcAiuc(0, box, &pbcAiuc_);

    // 输出最终 GPU 力时复用影子检查的回传，确保读取已经修正的设备数据。
    gamdDihedralShadowEnabled_ = std::getenv("GMX_GAMD_GPU_DIH_SHADOW") != nullptr
                                 || std::getenv("GMX_GAMD_FORCE_DEBUG_FILE") != nullptr;
    gamdEnergyShadowDiagnosticsEnabled_ = std::getenv("GMX_GAMD_GPU_ENERGY_SHADOW") != nullptr;


    if (useGpuResidentGaMD_)
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

    if (dihedralBufferEnabled())
    {
        allocateDeviceBuffer(&d_gamdDihedralShiftForces_, c_numShiftVectors, deviceContext_);
        allocateDeviceBuffer(&d_gamdDihedralVirial_, DIM * DIM, deviceContext_);
        allocateDeviceBuffer(&d_gamdShortRangeVirial_, DIM * DIM, deviceContext_);
    }

    if (energyEnabled())
    {
        // 第一个元素保存 reciprocal energy，后面九个元素保存完整 virial 张量。
        allocateDeviceBuffer(&d_gamdPmeEnergy_, 1 + DIM * DIM, deviceContext_);
        allocateDeviceBuffer(&d_gamdState_, c_gaMDStateSize, deviceContext_);
        if (useGpuResidentGaMD_)
        {
            allocateDeviceBuffer(&d_gamdEnergyHistory_, sc_historyCapacity * F_NRE, deviceContext_);
            allocateDeviceBuffer(&d_gamdForceVirialHistory_, sc_historyCapacity * DIM * DIM, deviceContext_);
            allocateDeviceBuffer(&d_gamdEnergyHistoryCount_, 1, deviceContext_);
            clearDeviceBufferAsync(&d_gamdEnergyHistoryCount_, 0, 1, deviceStream_);
            h_gamdEnergyHistory_.resize(sc_historyCapacity * F_NRE);
            h_gamdForceVirialHistory_.resize(sc_historyCapacity * DIM * DIM);
            h_gamdEnergyHistoryCount_.resize(1);
        }
        if (gamdEnergyShadowDiagnosticsEnabled_)
        {
            h_gamdDiagnosticState_.resize(c_gaMDStateSize);
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

    freeDeviceBuffer(&d_gamdDihedralForcesNbnxn_);
    freeDeviceBuffer(&d_gamdDihedralForcesState_);
    freeDeviceBuffer(&d_gamdDihedralShiftForces_);
    freeDeviceBuffer(&d_gamdDihedralVirial_);
    freeDeviceBuffer(&d_gamdShortRangeVirial_);
    freeDeviceBuffer(&d_gamdCorrectedForcesState_);
    freeDeviceBuffer(&d_nbnxnAtomOrder_);
    freeDeviceBuffer(&d_gamdState_);
    freeDeviceBuffer(&d_gamdPmeEnergy_);
    freeDeviceBuffer(&d_gamdEnergyHistory_);
    freeDeviceBuffer(&d_gamdForceVirialHistory_);
    freeDeviceBuffer(&d_gamdEnergyHistoryCount_);
    freeDeviceBuffer(&d_updateMasses_);
    freeDeviceBuffer(&d_updateTemperatureGroups_);
    freeDeviceBuffer(&d_updateConstraintVirialHistory_);
    freeDeviceBuffer(&d_updateConstraintVirialHistoryCount_);
    freeDeviceBuffer(&d_updateCurrentHalfKinetic_);
    freeDeviceBuffer(&d_updatePreviousHalfKinetic_);
    freeDeviceBuffer(&d_updateGroupHalfKineticTensor_);
    freeDeviceBuffer(&d_updateKineticHistory_);
    freeDeviceBuffer(&d_updateKineticHistoryCount_);
}

void GaMDGpuExecutor::Impl::initializeUpdateHistory(const int           numTemperatureGroups,
                                                    const DeviceStream& updateStream)
{
    GMX_RELEASE_ASSERT(useGpuResidentGaMD_,
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
    allocateDeviceBuffer(&d_updateConstraintVirialHistory_, sc_historyCapacity * DIM * DIM, deviceContext_);
    allocateDeviceBuffer(&d_updateConstraintVirialHistoryCount_, 1, deviceContext_);
    allocateDeviceBuffer(&d_updateCurrentHalfKinetic_, 6 + updateNumTemperatureGroups_, deviceContext_);
    allocateDeviceBuffer(&d_updatePreviousHalfKinetic_, 6 + updateNumTemperatureGroups_, deviceContext_);
    allocateDeviceBuffer(
            &d_updateGroupHalfKineticTensor_, updateNumTemperatureGroups_ * DIM * DIM, deviceContext_);
    allocateDeviceBuffer(&d_updateKineticHistory_,
                         sc_historyCapacity * (DIM * DIM + updateNumTemperatureGroups_),
                         deviceContext_);
    allocateDeviceBuffer(&d_updateKineticHistoryCount_, 1, deviceContext_);

    clearDeviceBufferAsync(&d_updateConstraintVirialHistoryCount_, 0, 1, *updateStream_);
    clearDeviceBufferAsync(&d_updatePreviousHalfKinetic_, 0, 6 + updateNumTemperatureGroups_, *updateStream_);
    clearDeviceBufferAsync(&d_updateKineticHistoryCount_, 0, 1, *updateStream_);

    h_updateConstraintVirialHistory_.resize(sc_historyCapacity * DIM * DIM);
    h_updateConstraintVirialHistoryCount_.resize(1);
    h_updateKineticHistory_.resize(sc_historyCapacity * (DIM * DIM + updateNumTemperatureGroups_));
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
    GMX_RELEASE_ASSERT(sampleCount <= sc_historyCapacity,
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
    GMX_RELEASE_ASSERT(sampleCount <= sc_historyCapacity,
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
    return gamdDihedralShadowEnabled_ || useGpuResidentGaMD_;
}

bool GaMDGpuExecutor::Impl::energyEnabled() const
{
    return gamdEnergyShadowDiagnosticsEnabled_ || useGpuResidentGaMD_;
}

bool GaMDGpuExecutor::Impl::energyDiagnosticsEnabled() const
{
    return gamdEnergyShadowDiagnosticsEnabled_;
}

bool GaMDGpuExecutor::Impl::scaleFromDeviceEnabled() const
{
    return useGpuResidentGaMD_;
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
    if (dihedralBufferEnabled())
    {
        if (!interactionDefinitions.il[F_FOURDIHS].empty())
        {
            GMX_THROW(InconsistentInputError(
                    "The GaMD GPU force path does not support FOURDIHS. Use -update cpu or "
                    "add CUDA FOURDIHS support before using this topology."));
        }
        for (const int functionType : { F_PDIHS, F_RBDIHS, F_CMAP })
        {
            if (!interactionDefinitions.il[functionType].empty()
                && hasPerturbedEntries(interactionDefinitions, functionType))
            {
                GMX_THROW(InconsistentInputError(formatString(
                        "The GaMD GPU force path does not support perturbed %s interactions. "
                        "Use -update cpu for this topology.",
                        interaction_function[functionType].longname)));
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
    buffers.enabled = dihedralBufferEnabled();
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

void GaMDGpuExecutor::Impl::copyGamdDihedralForces(ArrayRef<RVec> forces)
{
    GMX_RELEASE_ASSERT(dihedralBufferEnabled(),
                       "GaMD GPU dihedral-force copy requested without a force buffer");
    GMX_RELEASE_ASSERT(forces.ssize() == gamdStateForceSize_,
                       "GaMD GPU dihedral-force host buffer has the wrong size");
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
    GMX_RELEASE_ASSERT(dihedralBufferEnabled(),
                       "GaMD GPU dihedral virial copy requested while disabled");
    GMX_RELEASE_ASSERT(virial != nullptr, "GaMD GPU dihedral virial requires host storage");
    copyFromDeviceBuffer(
            virial->data(), &d_gamdDihedralVirial_, 0, virial->size(), deviceStream_, GpuApiCallBehavior::Sync, nullptr);
}

void GaMDGpuExecutor::Impl::copyGamdShortRangeVirial(std::array<real, DIM * DIM>* virial)
{
    GMX_RELEASE_ASSERT(useGpuResidentGaMD_,
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
    GMX_RELEASE_ASSERT(useGpuResidentGaMD_, "GaMD GPU force correction requested while disabled");
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
    // 调用方已发起异步上传；拓扑错误必须在这里收口，不能展开其调用栈。
    try
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
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
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

void GaMDGpuExecutor::launchEnergyReduction(const GaMDBiasParameters& parameters,
                                            const GaMDForceOverrides& forceOverrides,
                                            const bool                recordEnergySample)
{
    impl_->launchGamdEnergyReduction(parameters, forceOverrides, recordEnergySample);
}

GaMDStepResult GaMDGpuExecutor::energyDiagnostics()
{
    return impl_->gamdDiagnosticState();
}

GaMDHistoryBatch GaMDGpuExecutor::Impl::takeHistory()
{
    // 保持原来的两条流的读取顺序，在返回之前统一检查四组记录。
    GaMDHistoryBatch history;
    history.energySamples     = takeGamdEnergyHistory();
    history.forceVirials      = takeGamdForceVirialHistory(history.energySamples.size());
    history.constraintVirials = takeUpdateConstraintVirialHistory();
    history.kineticEnergy     = takeUpdateKineticEnergyHistory();
    const Index sampleCount   = ssize(history.energySamples);
    GMX_RELEASE_ASSERT(ssize(history.forceVirials) == sampleCount && ssize(history.constraintVirials) == sampleCount
                               && ssize(history.kineticEnergy.totalTensors) == sampleCount,
                       "Deferred GaMD energy, virial, and kinetic sample counts must match");
    GMX_RELEASE_ASSERT(history.kineticEnergy.groupKineticEnergies.size()
                               == sampleCount * history.kineticEnergy.numGroups,
                       "Deferred GaMD kinetic samples must include every temperature group");
    return history;
}

GaMDHistoryBatch GaMDGpuExecutor::takeHistory()
{
    return impl_->takeHistory();
}

void GaMDGpuExecutor::recordForceVirialSample()
{
    impl_->recordGamdForceVirialSample();
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

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::previousHalfStepKineticEnergy()
{
    return impl_->takePreviousHalfStepKineticEnergy();
}

void GaMDGpuExecutor::copyDihedralForces(ArrayRef<RVec> forces)
{
    impl_->copyGamdDihedralForces(forces);
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
//! \endcond
