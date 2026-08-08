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

#include "gamdmdmodule.h"

#include "gamdruntime.h"

namespace gmx
{

GaMDMDModule::GaMDMDModule() : runtime_(std::make_unique<GaMDRuntime>()) {}

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

void GaMDMDModule::subscribeToSimulationSetupNotifications(MDModulesNotifiers* /* notifiers */) {}

void GaMDMDModule::subscribeToPreProcessingNotifications(MDModulesNotifiers* /* notifiers */) {}

const GaMDStepPlan& GaMDMDModule::prepareStep(int64_t step, int rank, bool pressureCouplingThisStep)
{
    return runtime_->prepareStep(step, rank, pressureCouplingThisStep);
}

const GaMDStepPlan& GaMDMDModule::stepPlan() const
{
    return runtime_->stepPlan();
}

const GaMDStepResult& GaMDMDModule::stepResult() const
{
    return runtime_->stepResult();
}

GaMDGpuProductionParameters GaMDMDModule::gpuProductionParameters() const
{
    return runtime_->gpuProductionParameters();
}

void GaMDMDModule::finalizeCurrentStep(int64_t        step,
                                       int            rank,
                                       double         totalPotentialEnergy,
                                       double         dihedralEnergy,
                                       gmx_wallcycle* wallCycle)
{
    runtime_->finalizeCurrentStep(step, rank, totalPotentialEnergy, dihedralEnergy, wallCycle);
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
    runtime_->writeBufferedProductionOutput(
            deferredSteps, deferredEnergySamples, currentStep, rank, wallCycle);
}

void GaMDMDModule::setCheckpointingThisStep(bool checkpointingThisStep)
{
    runtime_->setCheckpointingThisStep(checkpointingThisStep);
}

void GaMDMDModule::writeRestartState(int64_t step)
{
    runtime_->writeRestartState(step);
}

void GaMDMDModule::warnIfRunTooShort(int64_t initStep, int64_t numSteps, int rank)
{
    runtime_->warnIfRunTooShort(initStep, numSteps, rank);
}

void GaMDMDModule::resetStateForTesting()
{
    runtime_ = std::make_unique<GaMDRuntime>();
}

std::unique_ptr<GaMDMDModule> createGaMDModule()
{
    return std::make_unique<GaMDMDModule>();
}

} // namespace gmx
