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
 * \brief GaMD step scheduling and accounting.
 */
#include "gmxpre.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <limits>

#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#include "gamddiagnostics.h"
#include "gamdruntime.h"

namespace gmx
{

void GaMDRuntime::setCheckpointingThisStep(bool checkpointingThisStep)
{
    checkpointingThisStep_  = checkpointingThisStep;
    stepPlan_.checkpointing = checkpointingThisStep;
    if (checkpointingThisStep && stepPlan_.enabled)
    {
        stepPlan_.requiresHostEnergy = true;
        stepPlan_.gpuGraphEligible   = false;
    }
}

GaMDStage GaMDRuntime::determineStage(int64_t step) const
{
    if (forceProductionRestart_)
    {
        return GaMDStage::Production;
    }

    if (step <= parameters_.conventionalPrepEnd)
    {
        return GaMDStage::ConventionalPreparation;
    }
    if (step <= parameters_.conventionalEnd)
    {
        return GaMDStage::ConventionalStatistics;
    }
    if (step <= parameters_.conventionalEnd + parameters_.boostPrepLength)
    {
        return GaMDStage::BoostPreparation;
    }
    if (step <= parameters_.conventionalEnd + parameters_.boostLength)
    {
        return GaMDStage::BoostStatistics;
    }

    return GaMDStage::Production;
}

const GaMDStepPlan& GaMDRuntime::prepareStep(int64_t step, int rank, bool pressureCouplingThisStep)
{
    if (lastPreparedStep_ == step)
    {
        return stepPlan_;
    }
    lastPreparedStep_ = step;

    currentRank_ = rank;

    loadParameters(rank);
    loadRestartStateIfNeeded(step, rank);

    // Default to a neutral state. Current-step GaMD applies the final force
    // correction later in do_force(), after raw energies and raw listed forces
    // for the current coordinates are available.
    stepResult_         = {};
    stepPlan_           = {};
    stepPlan_.step      = step;
    stepPlan_.boostType = parameters_.boostType;
    stepPlan_.debug     = debug_;

    if (parameters_.boostType == 0)
    {
        return stepPlan_;
    }

    stepPlan_.enabled       = true;
    stepPlan_.stage         = determineStage(step);
    stepPlan_.thresholdP    = potentialStatistics_.threshold;
    stepPlan_.kP            = potentialStatistics_.forceConstant;
    stepPlan_.thresholdD    = dihedralStatistics_.threshold;
    stepPlan_.kD            = dihedralStatistics_.forceConstant;
    stepPlan_.checkpointing = checkpointingThisStep_;
    stepPlan_.writesParameters =
            parameters_.parameterInterval > 0 && step % parameters_.parameterInterval == 0;
    stepPlan_.writesReweight =
            step > 0 && parameters_.reweightInterval > 0 && step % parameters_.reweightInterval == 0;

    const bool forcedScale            = diagnostics_.forceOverrides().enabled;
    const bool writesDihedralForces   = diagnostics_.dihedralForceOutputRequested(step);
    const bool writesUnbufferedOutput = (stepPlan_.writesParameters || stepPlan_.writesReweight)
                                        && !bufferedProductionOutputEnabled();
    stepPlan_.requiresHostEnergy = stepPlan_.stage != GaMDStage::Production
                                   || checkpointingThisStep_ || pressureCouplingThisStep
                                   || forcedScale || writesUnbufferedOutput || writesDihedralForces;
    stepPlan_.gpuGraphEligible = stepPlan_.stage == GaMDStage::Production && !checkpointingThisStep_
                                 && !pressureCouplingThisStep && !writesDihedralForces;
    return stepPlan_;
}

GaMDBiasParameters GaMDRuntime::biasParameters() const
{
    return { stepPlan_.boostType, stepPlan_.stage,      stepPlan_.thresholdP,
             stepPlan_.kP,        stepPlan_.thresholdD, stepPlan_.kD };
}

bool GaMDRuntime::requiresHostEnergyThisStep(int64_t step) const
{
    GMX_RELEASE_ASSERT(stepPlan_.step == step, "GaMD host-staging query must use the prepared step");
    return stepPlan_.requiresHostEnergy;
}

bool GaMDRuntime::bufferedProductionOutputEnabled() const
{
    // 强制诊断已经即时写出当前步，不能再由缓冲输出重复写一遍。
    return bufferedProductionOutputEnabled_ && !diagnostics_.forceOverrides().enabled;
}

void GaMDRuntime::setBufferedProductionOutputEnabled(bool enabled)
{
    bufferedProductionOutputEnabled_ = enabled;
}

void GaMDRuntime::warnIfRunTooShort(int64_t initStep, int64_t nsteps, int rank)
{
    if (warnedAboutRunLength_)
    {
        return;
    }
    warnedAboutRunLength_ = true;

    if (!parametersLoaded_ || parameters_.boostType == 0 || rank != 0 || nsteps < 0
        || parameters_.restartMode != 0 || forceProductionRestart_)
    {
        return;
    }

    // 阶段长度已在输入时检查；模拟总步数也要在相加之前排除溢出。
    if (initStep > std::numeric_limits<int64_t>::max() - nsteps)
    {
        gmx_fatal(FARGS, "GaMD init_step + nsteps exceeds the signed 64-bit step range.");
    }
    const int64_t adaptivePhaseEnd = parameters_.conventionalEnd + parameters_.boostLength;
    const int64_t plannedFinalStep = initStep + nsteps;
    if (adaptivePhaseEnd >= plannedFinalStep)
    {
        std::fprintf(
                stderr,
                "\n[GaMD WARNING] ntcmd + nteb = %" PRId64 " >= planned final step = %" PRId64
                ".\n"
                "GaMD requires ntcmd + nteb < init_step + nsteps to reach the production stage.\n"
                "Increase nsteps or reduce ntcmd/nteb if production sampling is intended.\n\n",
                adaptivePhaseEnd,
                plannedFinalStep);
    }
}

double GaMDRuntime::validateAndClampScale(const char*                 label,
                                          double                      rawScale,
                                          double                      energy,
                                          const GaMDEnergyStatistics& stat,
                                          int64_t                     step,
                                          int                         rank,
                                          bool*                       warnedAtFloor)
{
    if (!std::isfinite(rawScale) || !std::isfinite(energy) || !std::isfinite(stat.threshold)
        || !std::isfinite(stat.forceConstant))
    {
        gmx_fatal(FARGS,
                  "Non-finite GaMD %s boost quantity at step %" PRId64
                  ": raw_scale=%g energy=%g "
                  "threshold_E=%g k=%g.",
                  label,
                  step,
                  rawScale,
                  energy,
                  stat.threshold,
                  stat.forceConstant);
    }

    if (rawScale <= c_gaMDScaleFloor && !*warnedAtFloor)
    {
        if (rank == 0)
        {
            std::fprintf(stderr,
                         "\n[GaMD WARNING] %s raw force weight reached the 0.01 safety floor "
                         "at step %" PRId64
                         ": raw_weight=%.17g, energy=%.17g kJ/mol, E=%.17g "
                         "kJ/mol, k=%.17g (kJ/mol)^-1. The applied weight is clamped to "
                         "0.01. Inspect equilibration and GaMD parameters before using this "
                         "trajectory for production analysis.\n\n",
                         label,
                         step,
                         rawScale,
                         energy,
                         stat.threshold,
                         stat.forceConstant);
        }
        *warnedAtFloor = true;
    }
    return std::max(c_gaMDScaleFloor, rawScale);
}

void GaMDRuntime::finalizeCurrentStep(int64_t        step,
                                      int            rank,
                                      double         totalPotentialEnergy,
                                      double         dihedralEnergy,
                                      gmx_wallcycle* wallCycle)
{
    if (parameters_.boostType == 0 || lastAccountedStep_ == step)
    {
        return;
    }
    wallcycle_start(wallCycle, WallCycleCounter::GaMDFinalize);
    lastAccountedStep_ = step;

    currentRank_ = rank;
    GMX_RELEASE_ASSERT(stepPlan_.step == step, "GaMD finalization must use the prepared step");
    stepResult_.potentialEnergy = totalPotentialEnergy;
    stepResult_.dihedralEnergy  = dihedralEnergy;

    const GaMDForceOverrides& forceOverrides = diagnostics_.forceOverrides();
    if (forceOverrides.enabled)
    {
        const GaMDEnergyStatistics overrideStats{};
        stepResult_.scaleP = validateAndClampScale("total-potential",
                                                   forceOverrides.totalScale,
                                                   totalPotentialEnergy,
                                                   overrideStats,
                                                   step,
                                                   rank,
                                                   &warnedPotentialScaleFloor_);
        stepResult_.scaleD = validateAndClampScale(
                "dihedral", forceOverrides.dihedralScale, dihedralEnergy, overrideStats, step, rank, &warnedDihedralScaleFloor_);
        stepResult_.boostP = forceOverrides.potentialBoost;
        stepResult_.boostD = forceOverrides.dihedralBoost;
        wallcycle_stop(wallCycle, WallCycleCounter::GaMDFinalize);
        wallcycle_start(wallCycle, WallCycleCounter::GaMDOutput);
        writeParameterLine(step, rank, true);
        writeReweightLine(step, rank, true);
        wallcycle_stop(wallCycle, WallCycleCounter::GaMDOutput);
        return;
    }

    const bool updateStat = (stepPlan_.stage == GaMDStage::ConventionalStatistics
                             || stepPlan_.stage == GaMDStage::BoostStatistics);
    stepResult_ = evaluateGaMDBias(biasParameters(), totalPotentialEnergy, dihedralEnergy);

    if (stepPlan_.stage >= GaMDStage::BoostPreparation)
    {
        double totalEnergyForPotentialBoost = totalPotentialEnergy;
        if (parameters_.boostType == 3)
        {
            totalEnergyForPotentialBoost += stepResult_.boostD;
        }
        if (!std::isfinite(stepResult_.boostP) || !std::isfinite(stepResult_.boostD))
        {
            gmx_fatal(FARGS,
                      "Non-finite GaMD boost energy at step %" PRId64 ": boostP=%g boostD=%g.",
                      step,
                      stepResult_.boostP,
                      stepResult_.boostD);
        }
        stepResult_.scaleP = validateAndClampScale("total-potential",
                                                   stepResult_.scaleP,
                                                   totalEnergyForPotentialBoost,
                                                   potentialStatistics_,
                                                   step,
                                                   rank,
                                                   &warnedPotentialScaleFloor_);
        stepResult_.scaleD = validateAndClampScale(
                "dihedral", stepResult_.scaleD, dihedralEnergy, dihedralStatistics_, step, rank, &warnedDihedralScaleFloor_);
    }

    if (updateStat)
    {
        const bool updateAverageWindow    = (step % parameters_.averagingWindow == 0);
        double     totalPotentialForStats = totalPotentialEnergy;
        if (stepPlan_.stage == GaMDStage::BoostStatistics)
        {
            // 保留原来的相加顺序，避免改变统计量的浮点舍入结果。
            totalPotentialForStats = totalPotentialEnergy + stepResult_.boostP + stepResult_.boostD;
        }

        if (parameters_.boostType == 1 || parameters_.boostType == 3)
        {
            updateGaMDStatistics(totalPotentialForStats, potentialStatistics_, potentialWindowStatistics_);
            if (updateAverageWindow)
            {
                finalizeWindowStatistics(potentialWindowStatistics_, potentialStatistics_);
            }
        }

        if (parameters_.boostType == 2 || parameters_.boostType == 3)
        {
            updateGaMDStatistics(dihedralEnergy, dihedralStatistics_, dihedralWindowStatistics_);
            if (updateAverageWindow)
            {
                finalizeWindowStatistics(dihedralWindowStatistics_, dihedralStatistics_);
            }
        }

        const bool updateBiasParameters = (stepPlan_.stage == GaMDStage::ConventionalStatistics
                                           && step == parameters_.conventionalEnd)
                                          || stepPlan_.stage == GaMDStage::BoostStatistics;
        if (updateBiasParameters)
        {
            if (parameters_.boostType == 1 || parameters_.boostType == 3)
            {
                calculateThresholdAndForceConstant(
                        parameters_.potentialMode, parameters_.potentialSigmaLimit, potentialStatistics_);
            }
            if (parameters_.boostType == 2 || parameters_.boostType == 3)
            {
                calculateThresholdAndForceConstant(
                        parameters_.dihedralMode, parameters_.dihedralSigmaLimit, dihedralStatistics_);
            }
        }
    }

    wallcycle_stop(wallCycle, WallCycleCounter::GaMDFinalize);
    wallcycle_start(wallCycle, WallCycleCounter::GaMDOutput);
    if (!bufferedProductionOutputEnabled() || stepPlan_.stage != GaMDStage::Production)
    {
        writeParameterLine(step, rank, true);
        writeReweightLine(step, rank, true);
    }
    wallcycle_stop(wallCycle, WallCycleCounter::GaMDOutput);
}

void GaMDRuntime::writeBufferedProductionOutput(ArrayRef<const int64_t> deferredSteps,
                                                ArrayRef<const std::array<real, F_NRE>> deferredEnergySamples,
                                                int64_t        currentStep,
                                                int            rank,
                                                gmx_wallcycle* wallCycle)
{
    if (!bufferedProductionOutputEnabled())
    {
        return;
    }
    GMX_RELEASE_ASSERT(deferredSteps.size() == deferredEnergySamples.size(),
                       "Buffered GaMD output steps and energy samples must match");
    if (stepPlan_.stage != GaMDStage::Production)
    {
        GMX_RELEASE_ASSERT(deferredSteps.empty(),
                           "Buffered GaMD output may defer samples only in production stage 5");
        return;
    }

    wallcycle_start(wallCycle, WallCycleCounter::GaMDOutput);
    for (std::size_t sampleIndex = 0; sampleIndex < deferredEnergySamples.size(); ++sampleIndex)
    {
        const auto& sample          = deferredEnergySamples[sampleIndex];
        float       potentialEnergy = 0;
        for (int interactionType = 0; interactionType < F_EPOT; ++interactionType)
        {
            if (interactionType != F_DISRESVIOL && interactionType != F_ORIRESDEV)
            {
                potentialEnergy += sample[interactionType];
            }
        }
        const float dihedralEnergy =
                sample[F_PDIHS] + sample[F_RBDIHS] + sample[F_FOURDIHS] + sample[F_CMAP];

        GaMDStepResult record = evaluateGaMDBias(biasParameters(), potentialEnergy, dihedralEnergy);
        double         totalEnergyForP = potentialEnergy;
        if (parameters_.boostType == 3)
        {
            totalEnergyForP += record.boostD;
        }
        record.scaleP = validateAndClampScale("total-potential",
                                              record.scaleP,
                                              totalEnergyForP,
                                              potentialStatistics_,
                                              deferredSteps[sampleIndex],
                                              rank,
                                              &warnedPotentialScaleFloor_);
        record.scaleD = validateAndClampScale("dihedral",
                                              record.scaleD,
                                              dihedralEnergy,
                                              dihedralStatistics_,
                                              deferredSteps[sampleIndex],
                                              rank,
                                              &warnedDihedralScaleFloor_);

        writeParameterLine(deferredSteps[sampleIndex], rank, false);
        writeReweightRecord(deferredSteps[sampleIndex], rank, record, false);
    }

    writeParameterLine(currentStep, rank, false);
    writeReweightLine(currentStep, rank, false);
    if ((parameterFile_ != nullptr && std::fflush(parameterFile_) != 0)
        || (reweightFile_ != nullptr && std::fflush(reweightFile_) != 0))
    {
        gmx_fatal(FARGS, "Failed to flush buffered GaMD output at step %" PRId64 ".", currentStep);
    }
    wallcycle_stop(wallCycle, WallCycleCounter::GaMDOutput);
}

} // namespace gmx
