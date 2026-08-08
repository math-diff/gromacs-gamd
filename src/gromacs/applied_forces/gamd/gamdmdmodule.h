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
/*! \internal \file
 * \brief Declares the built-in GaMD simulation module.
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMMDMODULE_H
#define GMX_APPLIED_FORCES_GAMD_GAMMDMODULE_H

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

class GaMDGpuExecutor;
class GaMDDiagnostics;
class GaMDRuntime;

/*! \internal
 * \brief Owns the GaMD runtime for one simulation.
 *
 * The module is constructed by MDModules and remains alive until the simulator
 * has completed. Performance-critical force and update paths receive a
 * non-owning pointer through SimulatorModules instead of discovering GaMD
 * state through process-wide globals.
 */
class GaMDMDModule final : public IMDModule
{
public:
    GaMDMDModule();
    ~GaMDMDModule() override;

    IMdpOptionProvider* mdpOptionProvider() override;
    IMDOutputProvider*  outputProvider() override;
    void                initForceProviders(ForceProviders* forceProviders) override;
    void subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifiers) override;
    void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifiers) override;

    //! Prepares the immutable work description for one step.
    const GaMDStepPlan& prepareStep(int64_t step, int rank, bool pressureCouplingThisStep = false);
    //! Returns the current immutable step work description.
    const GaMDStepPlan& stepPlan() const;
    //! Returns the host result for the current step.
    const GaMDStepResult& stepResult() const;
    //! Returns parameters consumed by the device-resident production path.
    GaMDGpuProductionParameters gpuProductionParameters() const;
    //! Creates the optional executor that owns the GaMD device state.
    void initializeGpuExecutor(bool                 useGpuResidentGaMD,
                               const DeviceContext& deviceContext,
                               const DeviceStream&  deviceStream,
                               gmx_wallcycle*       wallCycleCounters);
    //! Returns the module-owned device executor, or nullptr when unavailable.
    GaMDGpuExecutor*       gpuExecutor();
    const GaMDGpuExecutor* gpuExecutor() const;
    //! Returns diagnostics and host scratch state owned by this simulation module.
    GaMDDiagnostics&       diagnostics();
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
    std::unique_ptr<GaMDRuntime>     runtime_;
    std::unique_ptr<GaMDDiagnostics> diagnostics_;
    std::unique_ptr<GaMDGpuExecutor> gpuExecutor_;
};

//! Constructs the built-in GaMD module.
std::unique_ptr<GaMDMDModule> createGaMDModule();

} // namespace gmx

#endif
