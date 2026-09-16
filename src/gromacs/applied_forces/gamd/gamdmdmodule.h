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
/*! \libinternal \file
 * \brief Declares the built-in GaMD module and its legacy MD entry points.
 *
 * \inlibraryapi
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDMDMODULE_H
#define GMX_APPLIED_FORCES_GAMD_GAMDMDMODULE_H

#include <array>
#include <memory>

#include "gromacs/applied_forces/gamd/gamdtypes.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

class DeviceContext;
class DeviceStream;
struct gmx_wallcycle;

namespace gmx
{

//! \cond libapi
//! \addtogroup module_applied_forces
//! \{

class GaMDGpuExecutor;
class GaMDDiagnostics;
class GaMDRuntime;

/*! \libinternal
 * \brief Owns the GaMD runtime for one simulation.
 *
 * The module is constructed by MDModules and remains alive until the simulator
 * has completed. Performance-critical force and update paths receive a
 * non-owning pointer through SimulatorModules instead of discovering GaMD
 * state through process-wide globals.
 *
 * Runtime failures are translated at these legacy MD entry points with
 * GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR. The internal GaMDRuntime interface
 * reports exceptions; an instance must be reinitialized after failed input
 * preparation. Simulation-setup notifications supply the MDLogger targets.
 */
class GaMDMDModule final : public IMDModule
{
public:
    //! Constructs the host runtime and optional diagnostic outputs.
    GaMDMDModule();
    //! Releases the GPU executor, runtime and diagnostics owned by this simulation.
    ~GaMDMDModule() override;

    //! Returns no MDP provider; GaMD parameters use the legacy input file.
    IMdpOptionProvider* mdpOptionProvider() override;
    //! Returns no separate output provider; the MD loop owns output timing.
    IMDOutputProvider* outputProvider() override;
    //! Leaves force-provider registration empty; correction uses the MD bridge.
    void initForceProviders(ForceProviders* forceProviders) override;
    //! Subscribes to the simulation logger notification.
    void subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifiers) override;
    //! Requires no preprocessing callbacks for the legacy input.
    void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifiers) override;

    //! Prepares the immutable work description for one step.
    const GaMDStepPlan& prepareStep(int64_t step, int rank, bool pressureCouplingThisStep = false);
    //! Returns the current immutable step work description.
    const GaMDStepPlan& stepPlan() const;
    //! Returns the host result for the current step.
    const GaMDStepResult& stepResult() const;
    //! Returns the current bias parameters shared by the CPU and GPU evaluators.
    GaMDBiasParameters biasParameters() const;
    //! Creates the optional executor that owns the GaMD device state.
    void initializeGpuExecutor(bool                 useGpuResidentGaMD,
                               const DeviceContext& deviceContext,
                               const DeviceStream&  deviceStream,
                               gmx_wallcycle*       wallCycleCounters);
    //! Returns the module-owned device executor, or nullptr when unavailable.
    GaMDGpuExecutor* gpuExecutor();
    //! Returns the optional executor without allowing modification.
    const GaMDGpuExecutor* gpuExecutor() const;
    //! Returns diagnostics and host scratch state owned by this simulation module.
    GaMDDiagnostics& diagnostics();
    //! Returns the per-instance diagnostics without allowing modification.
    const GaMDDiagnostics& diagnostics() const;
    //! Updates host statistics and the host step result from current raw energies.
    void finalizeCurrentStep(int64_t        step,
                             int            rank,
                             double         totalPotentialEnergy,
                             double         dihedralEnergy,
                             gmx_wallcycle* wallCycle = nullptr);
    //! Reports whether the current step needs host energy staging.
    bool requiresHostEnergyThisStep(int64_t step) const;
    //! Configures whether production text output is reconstructed in batches.
    void setBufferedProductionOutputEnabled(bool enabled);
    //! Reports whether buffered production text output is active.
    bool bufferedProductionOutputEnabled() const;
    //! Writes deferred production records in their original step order.
    void writeBufferedProductionOutput(ArrayRef<const int64_t> deferredSteps,
                                       ArrayRef<const std::array<real, F_NRE>> deferredEnergySamples,
                                       int64_t        currentStep,
                                       int            rank,
                                       gmx_wallcycle* wallCycle = nullptr);
    //! Mirrors the GROMACS checkpoint decision for this step.
    void setCheckpointingThisStep(bool checkpointingThisStep);
    //! Writes restart state before standard energy buffers are reset.
    void writeRestartState(int64_t step);
    //! Warns once when the requested run cannot reach production.
    void warnIfRunTooShort(int64_t initStep, int64_t numSteps, int rank);
    //! Resets this instance for an independent unit-test scenario.
    void resetStateForTesting();

private:
    std::unique_ptr<GaMDDiagnostics> diagnostics_;
    std::unique_ptr<GaMDRuntime>     runtime_;
    std::unique_ptr<GaMDGpuExecutor> gpuExecutor_;
};

//! Constructs the built-in GaMD module.
std::unique_ptr<GaMDMDModule> createGaMDModule();

//! \}
//! \endcond

} // namespace gmx

#endif
