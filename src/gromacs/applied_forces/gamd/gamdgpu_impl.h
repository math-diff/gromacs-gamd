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

#ifndef GMX_APPLIED_FORCES_GAMD_GAMDGPU_IMPL_H
#define GMX_APPLIED_FORCES_GAMD_GAMDGPU_IMPL_H

#include "gromacs/applied_forces/gamd/gamdgpu.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/gpu_utils/hostallocator.h"

namespace gmx
{

class GaMDGpuExecutor::Impl
{
public:
    Impl(bool                 useGpuResidentGaMD,
         const DeviceContext& deviceContext,
         const DeviceStream&  deviceStream,
         gmx_wallcycle*       wallCycleCounters);
    ~Impl();

    bool dihedralShadowEnabled() const;
    bool dihedralBufferEnabled() const;
    bool energyEnabled() const;
    bool energyDiagnosticsEnabled() const;
    bool scaleFromDeviceEnabled() const;

    void bindListedForceBuffers(ArrayRef<const int>           nbnxnAtomOrder,
                                const InteractionDefinitions& interactionDefinitions,
                                DeviceBuffer<Float4>          coordinatesAndCharges,
                                DeviceBuffer<Float3>          forces,
                                DeviceBuffer<Float3>          shiftForces,
                                DeviceBuffer<float>           shortRangeLennardJonesEnergy,
                                DeviceBuffer<float>           shortRangeElectrostaticEnergy,
                                DeviceBuffer<float>           listedEnergies);
    void setPbc(const PbcAiuc& pbcAiuc);

    GaMDBondedDeviceBuffers prepareBondedForceAccumulation();
    void                    finishGamdDihedralBuffers();

    DeviceBuffer<float>   gamdPmeEnergyStagingBuffer();
    GpuEventSynchronizer* gamdPmeEnergyReadyEvent();
    void                  launchGamdEnergyReduction(const GaMDBiasParameters& parameters,
                                                    const GaMDForceOverrides& forceOverrides,
                                                    bool                      recordEnergySample);
    GaMDStepResult        gamdDiagnosticState();
    GaMDHistoryBatch      takeHistory();
    void                  recordGamdForceVirialSample();

    void initializeUpdateHistory(int numTemperatureGroups, const DeviceStream& updateStream);
    void bindUpdateBuffers(DeviceBuffer<Float3>           velocities,
                           ArrayRef<const real>           masses,
                           ArrayRef<const unsigned short> temperatureGroups);
    void recordUpdateConstraintVirial(DeviceBuffer<float> lincsVirial,
                                      bool                haveLincs,
                                      DeviceBuffer<float> settleVirial,
                                      bool                haveSettles,
                                      real                scaleFactor);
    void recordUpdateKineticEnergy(bool recordSample);
    std::vector<std::array<real, DIM * DIM>> takePreviousHalfStepKineticEnergy();

    void copyGamdDihedralForces(ArrayRef<RVec> forces);
    void copyGamdDihedralVirial(std::array<real, DIM * DIM>* virial);
    void launchGamdShortRangeVirialReduction();
    void copyGamdShortRangeVirial(std::array<real, DIM * DIM>* virial);

    void                  applyGamdForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                                         real                 totalScale,
                                                         real                 dihedralCorrectionScale,
                                                         ArrayRef<RVec>       correctedForces);
    void                  applyGamdForceCorrection(DeviceBuffer<RVec>    forces,
                                                   real                  totalScale,
                                                   real                  dihedralCorrectionScale,
                                                   GpuEventSynchronizer* rawForcesReady);
    GpuEventSynchronizer* gamdForcesReadyEvent();

    void launchGamdForceCorrectionShadowKernel(real totalScale, real dihedralCorrectionScale);
    void launchGamdVirialReductionKernel(DeviceBuffer<Float3> forces,
                                         DeviceBuffer<Float3> shiftForces,
                                         DeviceBuffer<float>* virial,
                                         const char*          kernelName);
    void launchGamdForceCorrectionKernel(DeviceBuffer<Float3>  forces,
                                         real                  totalScale,
                                         real                  dihedralCorrectionScale,
                                         GpuEventSynchronizer* rawForcesReady);

private:
    std::vector<std::array<real, F_NRE>>     takeGamdEnergyHistory();
    std::vector<std::array<real, DIM * DIM>> takeGamdForceVirialHistory(int numSamples);
    std::vector<std::array<real, DIM * DIM>> takeUpdateConstraintVirialHistory();
    GaMDKineticEnergyHistory                 takeUpdateKineticEnergyHistory();

    static constexpr int c_historyCapacity_ = 4096;

    const bool           useGpuResidentGaMD_;
    const DeviceContext& deviceContext_;
    const DeviceStream&  deviceStream_;
    const DeviceStream*  updateStream_      = nullptr;
    gmx_wallcycle*       wallCycleCounters_ = nullptr;

    PbcAiuc pbcAiuc_;

    DeviceBuffer<Float4> coordinatesAndCharges_         = nullptr;
    DeviceBuffer<Float3> forces_                        = nullptr;
    DeviceBuffer<Float3> shiftForces_                   = nullptr;
    DeviceBuffer<float>  shortRangeLennardJonesEnergy_  = nullptr;
    DeviceBuffer<float>  shortRangeElectrostaticEnergy_ = nullptr;
    DeviceBuffer<float>  listedEnergies_                = nullptr;

    DeviceBuffer<Float3> d_gamdDihedralForcesNbnxn_  = nullptr;
    DeviceBuffer<Float3> d_gamdDihedralForcesState_  = nullptr;
    DeviceBuffer<Float3> d_gamdDihedralShiftForces_  = nullptr;
    DeviceBuffer<float>  d_gamdDihedralVirial_       = nullptr;
    DeviceBuffer<float>  d_gamdShortRangeVirial_     = nullptr;
    DeviceBuffer<Float3> d_gamdCorrectedForcesState_ = nullptr;
    DeviceBuffer<int>    d_nbnxnAtomOrder_           = nullptr;
    int                  gamdNbnxnForceSize_         = 0;
    int                  gamdNbnxnForceCapacity_     = 0;
    int                  gamdStateForceSize_         = 0;
    int                  gamdStateForceCapacity_     = 0;
    int                  gamdCorrectedForceSize_     = 0;
    int                  gamdCorrectedForceCapacity_ = 0;
    int                  nbnxnAtomOrderSize_         = 0;
    int                  nbnxnAtomOrderCapacity_     = 0;

    bool gamdDihedralShadowEnabled_          = false;
    bool gamdEnergyShadowDiagnosticsEnabled_ = false;

    DeviceBuffer<float> d_gamdPmeEnergy_          = nullptr;
    DeviceBuffer<float> d_gamdState_              = nullptr;
    DeviceBuffer<float> d_gamdEnergyHistory_      = nullptr;
    DeviceBuffer<float> d_gamdForceVirialHistory_ = nullptr;
    DeviceBuffer<int>   d_gamdEnergyHistoryCount_ = nullptr;

    HostVector<float> h_gamdDiagnosticState_ = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<float> h_gamdEnergyHistory_ = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<float> h_gamdForceVirialHistory_ = { {},
                                                    gmx::HostAllocationPolicy(
                                                            gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<int>   h_gamdEnergyHistoryCount_ = { {},
                                                    gmx::HostAllocationPolicy(
                                                          gmx::PinningPolicy::PinnedIfSupported) };

    int                          updateNumTemperatureGroups_           = 0;
    int                          updateNumAtoms_                       = 0;
    int                          updateMassCount_                      = -1;
    int                          updateMassCapacity_                   = -1;
    int                          updateTemperatureGroupCount_          = -1;
    int                          updateTemperatureGroupCapacity_       = -1;
    DeviceBuffer<Float3>         updateVelocities_                     = nullptr;
    DeviceBuffer<real>           d_updateMasses_                       = nullptr;
    DeviceBuffer<unsigned short> d_updateTemperatureGroups_            = nullptr;
    DeviceBuffer<float>          d_updateConstraintVirialHistory_      = nullptr;
    DeviceBuffer<int>            d_updateConstraintVirialHistoryCount_ = nullptr;
    DeviceBuffer<float>          d_updateCurrentHalfKinetic_           = nullptr;
    DeviceBuffer<float>          d_updatePreviousHalfKinetic_          = nullptr;
    DeviceBuffer<float>          d_updateGroupHalfKineticTensor_       = nullptr;
    DeviceBuffer<float>          d_updateKineticHistory_               = nullptr;
    DeviceBuffer<int>            d_updateKineticHistoryCount_          = nullptr;
    HostVector<float>            h_updateConstraintVirialHistory_      = {
        {},
        gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported)
    };
    HostVector<int> h_updateConstraintVirialHistoryCount_ = {
        {},
        gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported)
    };
    HostVector<float> h_updateKineticHistory_      = { {},
                                                       gmx::HostAllocationPolicy(
                                                          gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<float> h_updatePreviousHalfKinetic_ = { {},
                                                       gmx::HostAllocationPolicy(
                                                               gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<int>   h_updateKineticHistoryCount_ = { {},
                                                       gmx::HostAllocationPolicy(
                                                             gmx::PinningPolicy::PinnedIfSupported) };

    std::unique_ptr<GpuEventSynchronizer> gamdPmeEnergyReadyEvent_;
    std::unique_ptr<GpuEventSynchronizer> gamdForcesReadyEvent_;
};

} // namespace gmx

#endif
