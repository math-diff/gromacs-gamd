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

#ifndef GMX_APPLIED_FORCES_GAMD_GAMDGPU_H
#define GMX_APPLIED_FORCES_GAMD_GAMDGPU_H

#include <array>
#include <memory>
#include <vector>

#include "gromacs/applied_forces/gamd/gamdtypes.h"
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

/*! \brief Complete, count-checked records for the same deferred simulation steps. */
struct GaMDHistoryBatch
{
    std::vector<std::array<real, F_NRE>>     energySamples;
    std::vector<std::array<real, DIM * DIM>> forceVirials;
    std::vector<std::array<real, DIM * DIM>> constraintVirials;
    GaMDKineticEnergyHistory                 kineticEnergy;
};

class GaMDGpuExecutor final
{
public:
    //! 主机调度和所有设备历史缓冲共用此容量，满后必须先回传整批记录。
    static constexpr int c_historyCapacity = 4096;

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

    void           launchEnergyReduction(const GaMDBiasParameters& parameters,
                                         const GaMDForceOverrides& forceOverrides,
                                         bool                      recordEnergySample);
    GaMDStepResult energyDiagnostics();
    //! Read all deferred records together and clear their counters in stream order.
    GaMDHistoryBatch takeHistory();

    void recordForceVirialSample();

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
    std::vector<std::array<real, DIM * DIM>> previousHalfStepKineticEnergy();

    void copyDihedralForces(ArrayRef<RVec> forces);
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
