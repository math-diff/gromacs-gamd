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

    DeviceBuffer<float>                      gamdPmeEnergyStagingBuffer();
    GpuEventSynchronizer*                    gamdPmeEnergyReadyEvent();
    void                                     launchGamdEnergyShadowReduction(int    boostType,
                                                                             int    stage,
                                                                             double thresholdP,
                                                                             double forceConstantP,
                                                                             double thresholdD,
                                                                             double forceConstantD,
                                                                             bool   recordEnergySample);
    std::array<double, 6>                    gamdEnergyShadowValues();
    std::vector<std::array<real, F_NRE>>     takeGamdEnergyHistory();
    void                                     recordGamdForceVirialSample();
    std::vector<std::array<real, DIM * DIM>> takeGamdForceVirialHistory(int numSamples);

    void copyGamdDihedralShadowForces(ArrayRef<RVec> forces);
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
    static constexpr int c_gamdEnergyHistoryCapacity_ = 4096;

    const DeviceContext& deviceContext_;
    const DeviceStream&  deviceStream_;
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
    bool gamdDihedralBufferEnabled_          = false;
    bool gamdForceCorrectionEnabled_         = false;
    bool gamdEnergyShadowEnabled_            = false;
    bool gamdEnergyShadowDiagnosticsEnabled_ = false;
    bool gamdScaleFromDeviceEnabled_         = false;
    bool gamdEnergyHistoryEnabled_           = false;

    DeviceBuffer<float> d_gamdPmeEnergy_          = nullptr;
    DeviceBuffer<float> d_gamdEnergyShadow_       = nullptr;
    DeviceBuffer<float> d_gamdEnergyHistory_      = nullptr;
    DeviceBuffer<float> d_gamdForceVirialHistory_ = nullptr;
    DeviceBuffer<int>   d_gamdEnergyHistoryCount_ = nullptr;

    HostVector<float> h_gamdEnergyShadow_ = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<float> h_gamdEnergyHistory_ = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<float> h_gamdForceVirialHistory_ = { {},
                                                    gmx::HostAllocationPolicy(
                                                            gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<int>   h_gamdEnergyHistoryCount_ = { {},
                                                    gmx::HostAllocationPolicy(
                                                          gmx::PinningPolicy::PinnedIfSupported) };

    std::unique_ptr<GpuEventSynchronizer> gamdPmeEnergyReadyEvent_;
    std::unique_ptr<GpuEventSynchronizer> gamdForcesReadyEvent_;
};

} // namespace gmx

#endif
