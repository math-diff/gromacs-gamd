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
 * \brief Declares instance-owned GaMD diagnostic state and host scratch buffers.
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDDIAGNOSTICS_H
#define GMX_APPLIED_FORCES_GAMD_GAMDDIAGNOSTICS_H

#include <cstdint>
#include <cstdio>

#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{

/*! \internal
 * \brief Reusable host buffers whose lifetime is tied to one GaMD simulation.
 */
struct GaMDHostWorkspace
{
    std::vector<RVec> gpuModeDihedralForces;
    std::vector<RVec> gpuModeDihedralShiftForces;
    std::vector<RVec> gpuShadowDihedralForces;
    std::vector<RVec> referenceCorrectedForces;
    std::vector<RVec> liveGpuCorrectedForces;
    std::vector<RVec> rawReducedForcesForShadow;

    std::vector<RVec> rawForcesForCorrectionShadow;
    std::vector<RVec> correctedForcesForShadow;
    bool              correctionShadowCompared = false;

    std::vector<RVec> cpuDihedralForces;
    std::vector<RVec> cpuDihedralShiftForces;
    std::vector<RVec> gpuDihedralForcesForShadow;
    bool              dihedralShadowCompared = false;
};

/*! \internal
 * \brief Owns optional GaMD diagnostics for one simulation instance.
 */
class GaMDDiagnostics final
{
public:
    GaMDDiagnostics();
    ~GaMDDiagnostics();

    GaMDHostWorkspace& hostWorkspace();

    bool forcedTotalScale(double* scale) const;
    bool forcedDihedralRatio(double* ratio) const;

    void prepareDihedralStep(int64_t step, int numAtoms);
    void addBoostedDihedralForces(int        atomIndexI,
                                  const rvec forceI,
                                  int        atomIndexJ,
                                  const rvec forceJ,
                                  int        atomIndexK,
                                  const rvec forceK,
                                  int        atomIndexL,
                                  const rvec forceL);
    void addRawDihedralForces(int        atomIndexI,
                              const rvec forceI,
                              int        atomIndexJ,
                              const rvec forceJ,
                              int        atomIndexK,
                              const rvec forceK,
                              int        atomIndexL,
                              const rvec forceL);
    void flushDihedralForces(int64_t step);
    void dumpFinalForces(int64_t step, ArrayRef<const RVec> forces, bool isMainRank);

    bool shouldPrintNstcalcenergyMessage();
    bool shouldPrintResidentEnergyMessage();
    bool shouldPrintGpuWorkloadDiagnostics();

private:
    void addBoostedDihedralForce(int atomIndex, const rvec force);
    void addRawDihedralForce(int atomIndex, const rvec force);

    GaMDHostWorkspace hostWorkspace_;

    bool   totalScaleOverrideEnabled_    = false;
    double totalScaleOverride_           = 1.0;
    bool   dihedralRatioOverrideEnabled_ = false;
    double dihedralRatioOverride_        = 1.0;

    FILE*   rawDihedralFile_        = nullptr;
    int64_t rawDihedralCurrentStep_ = -1;

    FILE*             boostedDihedralFile_         = nullptr;
    int64_t           boostedDihedralStep_         = -1;
    int64_t           boostedDihedralCurrentStep_  = -1;
    int64_t           boostedDihedralPreparedStep_ = -1;
    int64_t           boostedDihedralFlushedStep_  = -1;
    int               boostedDihedralBegin_        = 0;
    int               boostedDihedralCount_        = -1;
    std::vector<RVec> boostedDihedralForces_;

    FILE*   finalForceFile_  = nullptr;
    int64_t finalForceStep_  = -1;
    int     finalForceBegin_ = 0;
    int     finalForceCount_ = -1;

    bool nstcalcenergyMessagePrinted_   = false;
    bool residentEnergyMessagePrinted_  = false;
    bool gpuWorkloadDiagnosticsEnabled_ = false;
    bool gpuWorkloadDiagnosticsPrinted_ = false;
};

} // namespace gmx

#endif
