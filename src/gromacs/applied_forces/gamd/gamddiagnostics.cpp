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
#include "gmxpre.h"

#include "gamddiagnostics.h"

#include <cmath>
#include <cstdlib>

#include <algorithm>

#include "gromacs/math/vec.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/strconvert.h"

#include "gamdmath.h"

namespace gmx
{
namespace
{

struct GaMDForceComparison
{
    double     maxAbsoluteError = 0.0;
    double     rmsError         = 0.0;
    gmx::Index worstAtom        = 0;
    int        worstComponent   = 0;
};

static GaMDForceComparison compareGaMDForceBuffers(ArrayRef<const RVec> reference,
                                                   ArrayRef<const RVec> candidate)
{
    GMX_RELEASE_ASSERT(reference.size() == candidate.size() && !reference.empty(),
                       "GaMD GPU shadow-force comparison requires equally sized non-empty buffers");

    GaMDForceComparison comparison;
    double              sumSquaredError = 0.0;
    for (gmx::Index atom = 0; atom < reference.ssize(); ++atom)
    {
        for (int component = 0; component < DIM; ++component)
        {
            const double error = static_cast<double>(candidate[atom][component])
                                 - static_cast<double>(reference[atom][component]);
            const double absoluteError = std::abs(error);
            sumSquaredError += error * error;
            if (absoluteError > comparison.maxAbsoluteError)
            {
                comparison.maxAbsoluteError = absoluteError;
                comparison.worstAtom        = atom;
                comparison.worstComponent   = component;
            }
        }
    }
    comparison.rmsError = std::sqrt(sumSquaredError / (DIM * reference.size()));
    return comparison;
}

} // namespace

void validateGaMDGpuShadowForces(const char*          label,
                                 int64_t              step,
                                 ArrayRef<const RVec> reference,
                                 ArrayRef<const RVec> candidate,
                                 double               maxAbsoluteTolerance,
                                 double               rmsTolerance)
{
    const GaMDForceComparison comparison = compareGaMDForceBuffers(reference, candidate);
    std::fprintf(stderr,
                 "[GaMD GPU shadow] %s step=%lld max_abs=%.9g rms=%.9g "
                 "worst_atom=%td component=%d cpu=%.9g gpu=%.9g\n",
                 label,
                 static_cast<long long>(step),
                 comparison.maxAbsoluteError,
                 comparison.rmsError,
                 comparison.worstAtom,
                 comparison.worstComponent,
                 static_cast<double>(reference[comparison.worstAtom][comparison.worstComponent]),
                 static_cast<double>(candidate[comparison.worstAtom][comparison.worstComponent]));

    if (!std::isfinite(comparison.maxAbsoluteError) || !std::isfinite(comparison.rmsError)
        || comparison.maxAbsoluteError > maxAbsoluteTolerance || comparison.rmsError > rmsTolerance)
    {
        gmx_fatal(FARGS,
                  "GaMD GPU %s shadow forces exceed provisional mixed-precision tolerances "
                  "(max %.9g > %.9g or RMS %.9g > %.9g).",
                  label,
                  comparison.maxAbsoluteError,
                  maxAbsoluteTolerance,
                  comparison.rmsError,
                  rmsTolerance);
    }
}

double readGaMDDiagnosticReal(const char* variableName, double defaultValue)
{
    const char* configuredValue = std::getenv(variableName);
    if (configuredValue == nullptr)
    {
        return defaultValue;
    }

    double value = 0.0;
    bool   valid = false;
    try
    {
        value = doubleFromString(configuredValue);
        valid = std::isfinite(value);
    }
    catch (const InvalidInputError&)
    {
        // 下面统一报告变量名，不能把解析失败变成可用的零值。
    }
    if (!valid)
    {
        gmx_fatal(FARGS,
                  "GaMD diagnostic variable %s must be a finite number without trailing text.",
                  variableName);
    }
    return value;
}


GaMDDiagnostics::GaMDDiagnostics()
{
    // 诊断变量只在模块构造时读取一次，生产热路径不反复查询进程环境。
    if (std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP") != nullptr)
    {
        const double totalScale    = readGaMDDiagnosticReal("GMX_GAMD_FORCE_OVERRIDE_SCALEP", 1.0);
        totalScaleOverride_        = std::max(c_gaMDScaleFloor, totalScale);
        totalScaleOverrideEnabled_ = true;
        const double dihedralRatio = readGaMDDiagnosticReal("GMX_GAMD_FORCE_OVERRIDE_SCALED", totalScale);
        dihedralRatioOverride_        = std::max(c_gaMDScaleFloor, dihedralRatio);
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
        boostedDihedralStep_ = int64FromString(configuredStep);
    }
    if (const char* configuredBegin = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_BEGIN"))
    {
        boostedDihedralBegin_ = intFromString(configuredBegin);
    }
    if (const char* configuredCount = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_COUNT"))
    {
        boostedDihedralCount_ = intFromString(configuredCount);
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
        finalForceStep_ = int64FromString(configuredStep);
    }
    if (const char* configuredBegin = std::getenv("GMX_GAMD_FORCE_DEBUG_BEGIN"))
    {
        finalForceBegin_ = intFromString(configuredBegin);
    }
    if (const char* configuredCount = std::getenv("GMX_GAMD_FORCE_DEBUG_COUNT"))
    {
        finalForceCount_ = intFromString(configuredCount);
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
