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
/*! \libinternal \file
 * \brief Declares the GaMD device executor and complete history batches.
 *
 * \inlibraryapi
 * \ingroup module_applied_forces
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

//! \cond libapi
//! \addtogroup module_applied_forces
//! \{

template<typename>
class ArrayRef;

/*! \libinternal \brief Device views supplied to listed-force accumulation. */
struct GaMDBondedDeviceBuffers
{
    //! Whether these views enable dihedral accumulation.
    bool enabled = false;
    //! Borrowed device view for raw dihedral-force accumulation.
    DeviceBuffer<Float3> dihedralForces = nullptr;
    //! Borrowed device view for periodic shift-force accumulation.
    DeviceBuffer<Float3> shiftForces = nullptr;
};

/*! \libinternal \brief Kinetic records for the ordered steps in one deferred batch. */
struct GaMDKineticEnergyHistory
{
    //! Number of temperature groups represented by each sample.
    int numGroups = 0;
    //! One full-step kinetic tensor per deferred sample.
    std::vector<std::array<real, DIM * DIM>> totalTensors;
    //! Sample-major kinetic energy values for every temperature group.
    std::vector<real> groupKineticEnergies;

    //! Returns the number of full-step kinetic samples.
    int numSamples() const { return static_cast<int>(totalTensors.size()); }
};

/*! \brief Complete, count-checked records for the same deferred simulation steps. */
struct GaMDHistoryBatch
{
    //! Raw energy components and corrected total energy, in step order.
    std::vector<std::array<real, F_NRE>> energySamples;
    //! Corrected force virial for each deferred step.
    std::vector<std::array<real, DIM * DIM>> forceVirials;
    //! Constraint virial for the same deferred steps.
    std::vector<std::array<real, DIM * DIM>> constraintVirials;
    //! Matching kinetic tensors and temperature-group energy samples.
    GaMDKineticEnergyHistory kineticEnergy;
};

/*! \libinternal \brief Owns GaMD device resources and their stream-ordering contracts. */
class GaMDGpuExecutor final
{
public:
    //! 主机调度和所有设备历史缓冲共用此容量，满后必须先回传整批记录。
    static constexpr int sc_historyCapacity = 4096;

    //! Constructs the CUDA executor, or reports an unavailable backend.
    GaMDGpuExecutor(bool                 useGpuResidentGaMD,
                    const DeviceContext& deviceContext,
                    const DeviceStream&  deviceStream,
                    gmx_wallcycle*       wallCycleCounters);
    //! Releases the executor's device buffers and host staging storage.
    ~GaMDGpuExecutor();

    //! Disables copying of device buffers and stream-bound state.
    GaMDGpuExecutor(const GaMDGpuExecutor&) = delete;
    //! Disables assignment of device buffers and stream-bound state.
    GaMDGpuExecutor& operator=(const GaMDGpuExecutor&) = delete;

    //! Reports whether CPU/GPU dihedral and force comparisons are enabled.
    bool dihedralShadowEnabled() const;
    //! Reports whether production or diagnostics require dihedral forces.
    bool dihedralBufferEnabled() const;
    //! Reports whether device energy reduction is available.
    bool energyEnabled() const;
    //! Reports whether the host/device energy comparison is enabled.
    bool energyDiagnosticsEnabled() const;
    //! Reports whether force correction consumes the device bias state.
    bool scaleFromDeviceEnabled() const;

    /*! \brief Binds borrowed views and updates the atom-order mapping.
     * Contains errors at this legacy boundary because the caller may have
     * already started asynchronous device transfers.
     */
    void bindListedForceBuffers(ArrayRef<const int>           nbnxnAtomOrder,
                                const InteractionDefinitions& interactionDefinitions,
                                DeviceBuffer<Float4>          coordinatesAndCharges,
                                DeviceBuffer<Float3>          forces,
                                DeviceBuffer<Float3>          shiftForces,
                                DeviceBuffer<float>           shortRangeLennardJonesEnergy,
                                DeviceBuffer<float>           shortRangeElectrostaticEnergy,
                                DeviceBuffer<float>           listedEnergies);
    //! Updates the periodic geometry used by device virial reduction.
    void setPbc(const PbcAiuc& pbcAiuc);

    //! Clears and exposes module-owned dihedral accumulation buffers.
    GaMDBondedDeviceBuffers prepareBondedForceAccumulation();
    //! Maps completed dihedral forces to state order and computes their virial.
    void finishBondedForceAccumulation();

    //! Returns reciprocal energy followed by the nine PME virial components.
    DeviceBuffer<float> pmeEnergyStagingBuffer();
    //! Returns the event connecting PME production to GaMD reduction.
    GpuEventSynchronizer* pmeEnergyReadyEvent();

    //! Reduces raw energies and evaluates device bias, optionally recording a sample.
    void launchEnergyReduction(const GaMDBiasParameters& parameters,
                               const GaMDForceOverrides& forceOverrides,
                               bool                      recordEnergySample);
    //! Waits for and returns the current device bias state for comparison.
    GaMDStepResult energyDiagnostics();
    //! Read all deferred records together and clear their counters in stream order.
    GaMDHistoryBatch takeHistory();

    //! Records corrected force virial in the current deferred sample slot.
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
    void recordKineticEnergy(bool recordSample);
    //! Returns previous half-step tensors for resuming host global work.
    std::vector<std::array<real, DIM * DIM>> previousHalfStepKineticEnergy();

    //! Synchronously copies raw dihedral forces in state atom order.
    void copyDihedralForces(ArrayRef<RVec> forces);
    //! Synchronously copies the completed dihedral virial.
    void copyDihedralVirial(std::array<real, DIM * DIM>* virial);
    //! Queues short-range virial reduction on the force stream.
    void launchShortRangeVirialReduction();
    //! Synchronously copies the short-range virial.
    void copyShortRangeVirial(std::array<real, DIM * DIM>* virial);

    //! Evaluates a GPU correction of CPU reference forces for diagnostics.
    void applyForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                    real                 totalScale,
                                    real                 dihedralCorrectionScale,
                                    ArrayRef<RVec>       correctedForces);
    //! Waits for raw forces, corrects the live device buffer and records readiness.
    void applyForceCorrection(DeviceBuffer<RVec>    forces,
                              real                  totalScale,
                              real                  dihedralCorrectionScale,
                              GpuEventSynchronizer* rawForcesReady);
    //! Returns the post-correction event consumed by GPU update.
    GpuEventSynchronizer* forcesReadyEvent();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

//! \}
//! \endcond

} // namespace gmx

#endif
