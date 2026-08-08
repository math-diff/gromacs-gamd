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

#ifndef GMX_APPLIED_FORCES_GAMD_GAMDGPU_H
#define GMX_APPLIED_FORCES_GAMD_GAMDGPU_H

#include <array>
#include <memory>
#include <vector>

#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/pbcutil/pbc_aiuc.h"
#include "gromacs/topology/idef.h"

class DeviceContext;
class DeviceStream;
class GpuEventSynchronizer;
struct gmx_wallcycle;

namespace gmx
{

template<typename>
class ArrayRef;

struct GaMDBondedDeviceBuffers
{
    bool                 enabled        = false;
    DeviceBuffer<Float3> dihedralForces = nullptr;
    DeviceBuffer<Float3> shiftForces    = nullptr;
};

struct GaMDKineticEnergyHistory
{
    int                                      numGroups = 0;
    std::vector<std::array<real, DIM * DIM>> totalTensors;
    std::vector<real>                        groupKineticEnergies;

    int numSamples() const { return static_cast<int>(totalTensors.size()); }
};

class GaMDGpuExecutor final
{
public:
    GaMDGpuExecutor(bool                 useGpuResidentGaMD,
                    const DeviceContext& deviceContext,
                    const DeviceStream&  deviceStream,
                    gmx_wallcycle*       wallCycleCounters);
    ~GaMDGpuExecutor();

    GaMDGpuExecutor(const GaMDGpuExecutor&)            = delete;
    GaMDGpuExecutor& operator=(const GaMDGpuExecutor&) = delete;

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
    void                    finishBondedForceAccumulation();

    DeviceBuffer<float>   pmeEnergyStagingBuffer();
    GpuEventSynchronizer* pmeEnergyReadyEvent();

    void                                 launchEnergyReduction(int    boostType,
                                                               int    stage,
                                                               double thresholdP,
                                                               double forceConstantP,
                                                               double thresholdD,
                                                               double forceConstantD,
                                                               bool   recordEnergySample);
    std::array<double, 6>                energyDiagnostics();
    std::vector<std::array<real, F_NRE>> takeEnergyHistory();

    void                                     recordForceVirialSample();
    std::vector<std::array<real, DIM * DIM>> takeForceVirialHistory(int numSamples);

    /*! \brief Bind the update stream and allocate GaMD-owned deferred global histories. */
    void initializeUpdateHistory(int numTemperatureGroups, const DeviceStream& updateStream);
    /*! \brief Bind device velocity data and copy immutable mass/group metadata. */
    void bindUpdateBuffers(DeviceBuffer<Float3>           velocities,
                           ArrayRef<const real>           masses,
                           ArrayRef<const unsigned short> temperatureGroups);
    /*! \brief Record one constraint-virial sample on the bound update stream. */
    void recordConstraintVirial(DeviceBuffer<float> lincsVirial,
                                bool                haveLincs,
                                DeviceBuffer<float> settleVirial,
                                bool                haveSettles,
                                real                scaleFactor);
    /*! \brief Advance half-step kinetic state and optionally record a full-step sample. */
    void                                     recordKineticEnergy(bool recordSample);
    std::vector<std::array<real, DIM * DIM>> takeConstraintVirialHistory();
    GaMDKineticEnergyHistory                 takeKineticEnergyHistory();
    std::vector<std::array<real, DIM * DIM>> previousHalfStepKineticEnergy();

    void copyDihedralShadowForces(ArrayRef<RVec> forces);
    void copyDihedralVirial(std::array<real, DIM * DIM>* virial);
    void launchShortRangeVirialReduction();
    void copyShortRangeVirial(std::array<real, DIM * DIM>* virial);

    void                  applyForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                                     real                 totalScale,
                                                     real                 dihedralCorrectionScale,
                                                     ArrayRef<RVec>       correctedForces);
    void                  applyForceCorrection(DeviceBuffer<RVec>    forces,
                                               real                  totalScale,
                                               real                  dihedralCorrectionScale,
                                               GpuEventSynchronizer* rawForcesReady);
    GpuEventSynchronizer* forcesReadyEvent();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
