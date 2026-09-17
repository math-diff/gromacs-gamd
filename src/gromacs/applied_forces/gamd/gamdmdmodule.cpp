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
 * \brief Connects the GaMD runtime to the legacy MD loop and setup notifications.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gamdmdmodule.h"

#include <functional>

#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#include "gamddiagnostics.h"
#include "gamdgpu.h"
#include "gamdruntime.h"

//! \cond internal
namespace gmx
{

GaMDMDModule::GaMDMDModule() :
    diagnostics_(std::make_unique<GaMDDiagnostics>()),
    runtime_(std::make_unique<GaMDRuntime>(*diagnostics_))
{
}

GaMDMDModule::~GaMDMDModule() = default;

IMdpOptionProvider* GaMDMDModule::mdpOptionProvider()
{
    return nullptr;
}

IMDOutputProvider* GaMDMDModule::outputProvider()
{
    return nullptr;
}

void GaMDMDModule::initForceProviders(ForceProviders* /* forceProviders */) {}

void GaMDMDModule::subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifiers)
{
    // 复用引擎的日志目标，不再绕过它直接写 stderr。
    std::function<void(const MDLogger&)> setLogger =
            std::bind(&GaMDDiagnostics::setLogger, diagnostics_.get(), std::placeholders::_1);
    notifiers->simulationSetupNotifier_.subscribe(setLogger);
}

void GaMDMDModule::subscribeToPreProcessingNotifications(MDModulesNotifiers* /* notifiers */) {}

const GaMDStepPlan& GaMDMDModule::prepareStep(int64_t step, int rank, bool pressureCouplingThisStep)
{
    try
    {
        return runtime_->prepareStep(step, rank, pressureCouplingThisStep);
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

const GaMDStepResult& GaMDMDModule::stepResult() const
{
    return runtime_->stepResult();
}

GaMDBiasParameters GaMDMDModule::biasParameters() const
{
    return runtime_->biasParameters();
}

void GaMDMDModule::initializeGpuExecutor(const bool           useGpuResidentGaMD,
                                         const DeviceContext& deviceContext,
                                         const DeviceStream&  deviceStream,
                                         gmx_wallcycle*       wallCycleCounters)
{
    GMX_RELEASE_ASSERT(gpuExecutor_ == nullptr,
                       "The GaMD GPU executor can only be initialized once");
    gpuExecutor_ = std::make_unique<GaMDGpuExecutor>(
            useGpuResidentGaMD, deviceContext, deviceStream, wallCycleCounters);
}

GaMDGpuExecutor* GaMDMDModule::gpuExecutor()
{
    return gpuExecutor_.get();
}

const GaMDGpuExecutor* GaMDMDModule::gpuExecutor() const
{
    return gpuExecutor_.get();
}

GaMDDiagnostics& GaMDMDModule::diagnostics()
{
    return *diagnostics_;
}

const GaMDDiagnostics& GaMDMDModule::diagnostics() const
{
    return *diagnostics_;
}

void GaMDMDModule::finalizeCurrentStep(int64_t        step,
                                       int            rank,
                                       double         totalPotentialEnergy,
                                       double         dihedralEnergy,
                                       gmx_wallcycle* wallCycle)
{
    try
    {
        runtime_->finalizeCurrentStep(step, rank, totalPotentialEnergy, dihedralEnergy, wallCycle);
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

bool GaMDMDModule::requiresHostEnergyThisStep(int64_t step) const
{
    return runtime_->requiresHostEnergyThisStep(step);
}

void GaMDMDModule::setBufferedProductionOutputEnabled(bool enabled)
{
    runtime_->setBufferedProductionOutputEnabled(enabled);
}

bool GaMDMDModule::bufferedProductionOutputEnabled() const
{
    return runtime_->bufferedProductionOutputEnabled();
}

void GaMDMDModule::writeBufferedProductionOutput(ArrayRef<const int64_t> deferredSteps,
                                                 ArrayRef<const std::array<real, F_NRE>> deferredEnergySamples,
                                                 int64_t        currentStep,
                                                 int            rank,
                                                 gmx_wallcycle* wallCycle)
{
    try
    {
        runtime_->writeBufferedProductionOutput(
                deferredSteps, deferredEnergySamples, currentStep, rank, wallCycle);
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

void GaMDMDModule::setCheckpointingThisStep(bool checkpointingThisStep)
{
    runtime_->setCheckpointingThisStep(checkpointingThisStep);
}

void GaMDMDModule::writeRestartState(int64_t step)
{
    try
    {
        runtime_->writeRestartState(step);
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

void GaMDMDModule::warnIfRunTooShort(int64_t initStep, int64_t numSteps, int rank)
{
    try
    {
        runtime_->warnIfRunTooShort(initStep, numSteps, rank);
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

void GaMDMDModule::resetStateForTesting()
{
    runtime_.reset();
    diagnostics_ = std::make_unique<GaMDDiagnostics>();
    runtime_     = std::make_unique<GaMDRuntime>(*diagnostics_);
}

std::unique_ptr<GaMDMDModule> createGaMDModule()
{
    return std::make_unique<GaMDMDModule>();
}

} // namespace gmx
//! \endcond
