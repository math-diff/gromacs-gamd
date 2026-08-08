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

#include "gamddiagnostics.h"

#include <cmath>
#include <cstdlib>

#include <algorithm>

#include "gromacs/math/vec.h"

namespace gmx
{

GaMDDiagnostics::GaMDDiagnostics()
{
    // 诊断变量只在模块构造时读取一次，生产热路径不反复查询进程环境。
    if (const char* totalScale = std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP"))
    {
        totalScaleOverride_        = std::max(0.01, std::atof(totalScale));
        totalScaleOverrideEnabled_ = true;

        double dihedralRatio = std::atof(totalScale);
        if (const char* configuredRatio = std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALED"))
        {
            dihedralRatio = std::atof(configuredRatio);
        }
        dihedralRatioOverride_        = std::max(0.01, dihedralRatio);
        dihedralRatioOverrideEnabled_ = true;
    }

    if (const char* filename = std::getenv("GMX_GAMD_DIH_RAW_DEBUG_FILE"))
    {
        rawDihedralFile_ = std::fopen(filename, "w");
        if (rawDihedralFile_ != nullptr)
        {
            std::fprintf(
                    rawDihedralFile_,
                    "# step atom fx fy fz   (raw applied dihedral-like force contributions)\n");
            std::fflush(rawDihedralFile_);
        }
    }

    if (const char* configuredStep = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP"))
    {
        boostedDihedralStep_ = std::atol(configuredStep);
    }
    if (const char* configuredBegin = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_BEGIN"))
    {
        boostedDihedralBegin_ = std::atoi(configuredBegin);
    }
    if (const char* configuredCount = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_COUNT"))
    {
        boostedDihedralCount_ = std::atoi(configuredCount);
    }
    if (const char* filename = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_FILE"))
    {
        boostedDihedralFile_ = std::fopen(filename, "w");
        if (boostedDihedralFile_ != nullptr)
        {
            std::fprintf(boostedDihedralFile_,
                         "# step atom fx fy fz |f|   (boosted dihedral-like force only)\n");
            std::fflush(boostedDihedralFile_);
        }
    }

    if (const char* configuredStep = std::getenv("GMX_GAMD_FORCE_DEBUG_STEP"))
    {
        finalForceStep_ = std::atol(configuredStep);
    }
    if (const char* configuredBegin = std::getenv("GMX_GAMD_FORCE_DEBUG_BEGIN"))
    {
        finalForceBegin_ = std::atoi(configuredBegin);
    }
    if (const char* configuredCount = std::getenv("GMX_GAMD_FORCE_DEBUG_COUNT"))
    {
        finalForceCount_ = std::atoi(configuredCount);
    }
    if (const char* filename = std::getenv("GMX_GAMD_FORCE_DEBUG_FILE"))
    {
        finalForceFile_ = std::fopen(filename, "w");
        if (finalForceFile_ != nullptr)
        {
            std::fprintf(finalForceFile_, "# step atom fx fy fz |f|\n");
            std::fflush(finalForceFile_);
        }
    }

    gpuWorkloadDiagnosticsEnabled_ = std::getenv("GMX_GAMD_GPU_DIAGNOSTICS") != nullptr;
}

GaMDDiagnostics::~GaMDDiagnostics()
{
    if (rawDihedralFile_ != nullptr)
    {
        std::fclose(rawDihedralFile_);
    }
    if (boostedDihedralFile_ != nullptr)
    {
        std::fclose(boostedDihedralFile_);
    }
    if (finalForceFile_ != nullptr)
    {
        std::fclose(finalForceFile_);
    }
}

GaMDHostWorkspace& GaMDDiagnostics::hostWorkspace()
{
    return hostWorkspace_;
}

bool GaMDDiagnostics::forcedTotalScale(double* scale) const
{
    if (totalScaleOverrideEnabled_ && scale != nullptr)
    {
        *scale = totalScaleOverride_;
    }
    return totalScaleOverrideEnabled_;
}

bool GaMDDiagnostics::forcedDihedralRatio(double* ratio) const
{
    if (dihedralRatioOverrideEnabled_ && ratio != nullptr)
    {
        *ratio = dihedralRatioOverride_;
    }
    return dihedralRatioOverrideEnabled_;
}

void GaMDDiagnostics::prepareDihedralStep(const int64_t step, const int numAtoms)
{
    rawDihedralCurrentStep_     = step;
    boostedDihedralCurrentStep_ = step;
    if (boostedDihedralFile_ == nullptr || boostedDihedralStep_ < 0
        || boostedDihedralCurrentStep_ != boostedDihedralStep_)
    {
        return;
    }

    const int begin = std::max(0, std::min(boostedDihedralBegin_, numAtoms));
    int       count = boostedDihedralCount_;
    if (count < 0)
    {
        count = numAtoms - begin;
    }
    count                 = std::max(0, std::min(count, numAtoms - begin));
    boostedDihedralBegin_ = begin;

    if (boostedDihedralPreparedStep_ != boostedDihedralCurrentStep_
        || static_cast<int>(boostedDihedralForces_.size()) != count)
    {
        // 每个目标步只清零一次，随后各 bonded kernel 把贡献累加到同一缓冲。
        boostedDihedralForces_.resize(count);
        for (RVec& force : boostedDihedralForces_)
        {
            clear_rvec(force);
        }
        boostedDihedralPreparedStep_ = boostedDihedralCurrentStep_;
    }
}

void GaMDDiagnostics::addBoostedDihedralForce(const int atomIndex, const rvec force)
{
    if (boostedDihedralFile_ == nullptr || boostedDihedralStep_ < 0
        || boostedDihedralCurrentStep_ != boostedDihedralStep_ || boostedDihedralCount_ == 0)
    {
        return;
    }

    const int end = boostedDihedralBegin_ + static_cast<int>(boostedDihedralForces_.size());
    if (atomIndex < boostedDihedralBegin_ || atomIndex >= end)
    {
        return;
    }
    rvec_inc(boostedDihedralForces_[atomIndex - boostedDihedralBegin_], force);
}

void GaMDDiagnostics::addBoostedDihedralForces(const int  atomIndexI,
                                               const rvec forceI,
                                               const int  atomIndexJ,
                                               const rvec forceJ,
                                               const int  atomIndexK,
                                               const rvec forceK,
                                               const int  atomIndexL,
                                               const rvec forceL)
{
    addBoostedDihedralForce(atomIndexI, forceI);
    addBoostedDihedralForce(atomIndexJ, forceJ);
    addBoostedDihedralForce(atomIndexK, forceK);
    addBoostedDihedralForce(atomIndexL, forceL);
}

void GaMDDiagnostics::addRawDihedralForce(const int atomIndex, const rvec force)
{
    if (rawDihedralFile_ == nullptr)
    {
        return;
    }
    std::fprintf(rawDihedralFile_,
                 "%lld %d %.9e %.9e %.9e\n",
                 static_cast<long long>(rawDihedralCurrentStep_),
                 atomIndex,
                 static_cast<double>(force[XX]),
                 static_cast<double>(force[YY]),
                 static_cast<double>(force[ZZ]));
}

void GaMDDiagnostics::addRawDihedralForces(const int  atomIndexI,
                                           const rvec forceI,
                                           const int  atomIndexJ,
                                           const rvec forceJ,
                                           const int  atomIndexK,
                                           const rvec forceK,
                                           const int  atomIndexL,
                                           const rvec forceL)
{
    addRawDihedralForce(atomIndexI, forceI);
    addRawDihedralForce(atomIndexJ, forceJ);
    addRawDihedralForce(atomIndexK, forceK);
    addRawDihedralForce(atomIndexL, forceL);
}

void GaMDDiagnostics::flushDihedralForces(const int64_t step)
{
    if (boostedDihedralFile_ == nullptr || step != boostedDihedralStep_
        || boostedDihedralFlushedStep_ == step || boostedDihedralPreparedStep_ != step)
    {
        return;
    }

    const int count = static_cast<int>(boostedDihedralForces_.size());
    for (int forceIndex = 0; forceIndex < count; ++forceIndex)
    {
        const int    atomIndex = boostedDihedralBegin_ + forceIndex;
        const double forceX    = boostedDihedralForces_[forceIndex][XX];
        const double forceY    = boostedDihedralForces_[forceIndex][YY];
        const double forceZ    = boostedDihedralForces_[forceIndex][ZZ];
        const double forceNorm = std::sqrt(forceX * forceX + forceY * forceY + forceZ * forceZ);
        std::fprintf(boostedDihedralFile_,
                     "%lld %d %.9e %.9e %.9e %.9e\n",
                     static_cast<long long>(step),
                     atomIndex,
                     forceX,
                     forceY,
                     forceZ,
                     forceNorm);
    }
    std::fflush(boostedDihedralFile_);
    boostedDihedralFlushedStep_ = step;
}

void GaMDDiagnostics::dumpFinalForces(const int64_t step, const ArrayRef<const RVec> forces, const bool isMainRank)
{
    if (finalForceFile_ == nullptr || !isMainRank || step != finalForceStep_)
    {
        return;
    }

    const int numAtoms = forces.ssize();
    const int begin    = std::max(0, std::min(finalForceBegin_, numAtoms));
    int       end      = numAtoms;
    if (finalForceCount_ >= 0)
    {
        end = std::max(begin, std::min(begin + finalForceCount_, numAtoms));
    }

    std::fprintf(finalForceFile_,
                 "# dump step %lld nAtoms %d begin %d end %d\n",
                 static_cast<long long>(step),
                 numAtoms,
                 begin,
                 end);
    for (int atomIndex = begin; atomIndex < end; ++atomIndex)
    {
        const double forceX    = forces[atomIndex][XX];
        const double forceY    = forces[atomIndex][YY];
        const double forceZ    = forces[atomIndex][ZZ];
        const double forceNorm = std::sqrt(forceX * forceX + forceY * forceY + forceZ * forceZ);
        std::fprintf(finalForceFile_,
                     "%lld %d %.9e %.9e %.9e %.9e\n",
                     static_cast<long long>(step),
                     atomIndex,
                     forceX,
                     forceY,
                     forceZ,
                     forceNorm);
    }
    std::fflush(finalForceFile_);
}

bool GaMDDiagnostics::shouldPrintNstcalcenergyMessage()
{
    if (nstcalcenergyMessagePrinted_)
    {
        return false;
    }
    nstcalcenergyMessagePrinted_ = true;
    return true;
}

bool GaMDDiagnostics::shouldPrintResidentEnergyMessage()
{
    if (residentEnergyMessagePrinted_)
    {
        return false;
    }
    residentEnergyMessagePrinted_ = true;
    return true;
}

bool GaMDDiagnostics::shouldPrintGpuWorkloadDiagnostics()
{
    if (!gpuWorkloadDiagnosticsEnabled_ || gpuWorkloadDiagnosticsPrinted_)
    {
        return false;
    }
    gpuWorkloadDiagnosticsPrinted_ = true;
    return true;
}

} // namespace gmx
