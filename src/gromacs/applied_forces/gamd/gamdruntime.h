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
 * \brief Declares instance-owned GaMD host runtime state.
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDRUNTIME_H
#define GMX_APPLIED_FORCES_GAMD_GAMDRUNTIME_H

#include <cstdint>
#include <cstdio>

#include <array>

#include "gromacs/applied_forces/gamd/gamdtypes.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

struct gmx_wallcycle;

namespace gmx
{

struct GaMDParameters
{
    int     boostType           = 0;
    int     restartMode         = 0;
    int     thresholdMode       = 1;
    int     potentialMode       = 1;
    int     dihedralMode        = 1;
    int64_t conventionalPrepEnd = 200000;
    int64_t conventionalEnd     = 1000000;
    int64_t boostPrepLength     = 200000;
    int64_t boostLength         = 1000000;
    int64_t averagingWindow     = 50000;
    int64_t reweightInterval    = 1;
    int64_t parameterInterval   = 0;
    double  potentialSigmaLimit = 6.0 * 4.184;
    double  dihedralSigmaLimit  = 6.0 * 4.184;
};

struct GaMDEnergyStatistics
{
    double  maximum       = -1e99;
    double  minimum       = 1e99;
    double  average       = 0.0;
    double  deviation     = 0.001;
    double  m2            = 0.0;
    int64_t count         = 0;
    double  threshold     = 0.0;
    double  k0            = 0.0;
    double  forceConstant = 0.0;
};

struct GaMDWindowStatistics
{
    int64_t count = 0;
    double  mean  = 0.0;
    double  m2    = 0.0;
};

struct GaMDProductionOutputRecord
{
    double potentialEnergy = 0.0;
    double dihedralEnergy  = 0.0;
    double scaleP          = 1.0;
    double scaleD          = 1.0;
    double boostP          = 0.0;
    double boostD          = 0.0;
};

/*! \internal
 * \brief Owns all mutable GaMD host state for one simulation.
 */
class GaMDRuntime
{
public:
    GaMDRuntime() = default;
    ~GaMDRuntime();

    const GaMDStepPlan&         prepareStep(int64_t step, int rank, bool pressureCouplingThisStep);
    const GaMDStepPlan&         stepPlan() const { return stepPlan_; }
    const GaMDStepResult&       stepResult() const { return stepResult_; }
    GaMDGpuProductionParameters gpuProductionParameters() const;
    void                        finalizeCurrentStep(int64_t        step,
                                                    int            rank,
                                                    double         totalPotentialEnergy,
                                                    double         dihedralEnergy,
                                                    gmx_wallcycle* wallCycle);
    bool                        requiresHostEnergyThisStep(int64_t step) const;
    void                        setBufferedProductionOutputEnabled(bool enabled);
    bool                        bufferedProductionOutputEnabled() const;
    void                        writeBufferedProductionOutput(ArrayRef<const int64_t> deferredSteps,
                                                              ArrayRef<const std::array<real, F_NRE>> deferredEnergySamples,
                                                              int64_t        currentStep,
                                                              int            rank,
                                                              gmx_wallcycle* wallCycle);
    void                        setCheckpointingThisStep(bool checkpointingThisStep);
    void                        writeRestartState(int64_t step);
    void                        warnIfRunTooShort(int64_t initStep, int64_t numSteps, int rank);

private:
    void loadParameters(int rank);
    void loadRestartStateIfNeeded(int64_t step, int rank);
    void saveRestartState(int64_t step, int rank);
    int  determineStage(int64_t step) const;

    FILE* reweightFile(int rank);
    FILE* parameterFile(int rank);
    void  writeParameterLine(int64_t step, int rank, bool flushOutput);
    void writeReweightRecord(int64_t step, int rank, const GaMDProductionOutputRecord& record, bool flushOutput);
    void writeReweightLine(int64_t step, int rank, bool flushOutput);
    bool suppressTextOutputForStep(int64_t step) const;
    void closeOutputFiles();

    double validateAndClampScale(const char*                 label,
                                 double                      rawScale,
                                 double                      energy,
                                 const GaMDEnergyStatistics& statistics,
                                 int64_t                     step,
                                 int                         rank,
                                 bool*                       warnedAtFloor);

    GaMDParameters       parameters_;
    bool                 parametersLoaded_ = false;
    GaMDEnergyStatistics potentialStatistics_;
    GaMDEnergyStatistics dihedralStatistics_;
    GaMDWindowStatistics potentialWindowStatistics_;
    GaMDWindowStatistics dihedralWindowStatistics_;

    GaMDStepPlan   stepPlan_;
    GaMDStepResult stepResult_;
    int64_t        lastPreparedStep_   = -1;
    int64_t        lastAccountedStep_  = -1;
    bool           stepReadyForOutput_ = false;

    bool    restartLoadAttempted_            = false;
    bool    forceProductionRestart_          = false;
    int     currentRank_                     = 0;
    bool    checkpointingThisStep_           = false;
    bool    bufferedProductionOutputEnabled_ = false;
    int64_t suppressTextOutputStep_          = -1;
    bool    warnedPotentialScaleFloor_       = false;
    bool    warnedDihedralScaleFloor_        = false;
    bool    warnedAboutRunLength_            = false;
    int     debug_                           = 0;

    FILE* reweightFile_  = nullptr;
    FILE* parameterFile_ = nullptr;
};

} // namespace gmx

#endif
