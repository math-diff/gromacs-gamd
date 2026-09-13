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

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <algorithm>

#include "gromacs/math/vec.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/strconvert.h"

namespace gmx
{
namespace
{

void flushGaMDDiagnosticFile(FILE* output, const std::string& filename)
{
    if (std::fflush(output) != 0 || std::ferror(output) != 0)
    {
        gmx_fatal(FARGS, "Failed to write GaMD diagnostic file '%s': %s.", filename.c_str(), std::strerror(errno));
    }
}

FILE* openGaMDDiagnosticFile(const char* variableName, const char* header, std::string* filename)
{
    const char* configuredName = std::getenv(variableName);
    if (configuredName == nullptr)
    {
        return nullptr;
    }
    *filename    = configuredName;
    FILE* output = std::fopen(filename->c_str(), "w");
    if (output == nullptr)
    {
        gmx_fatal(FARGS,
                  "Cannot open GaMD diagnostic file '%s' requested by %s: %s.",
                  filename->c_str(),
                  variableName,
                  std::strerror(errno));
    }
    // 用户明确请求了文件，连表头都写不进去时就立即报告。
    if (std::fprintf(output, "%s", header) < 0)
    {
        gmx_fatal(FARGS,
                  "Failed to write GaMD diagnostic file '%s': %s.",
                  filename->c_str(),
                  std::strerror(errno));
    }
    flushGaMDDiagnosticFile(output, *filename);
    return output;
}

void closeGaMDDiagnosticFile(FILE* output, const std::string& filename)
{
    if (output != nullptr && std::fclose(output) != 0)
    {
        gmx_fatal(FARGS, "Failed to close GaMD diagnostic file '%s': %s.", filename.c_str(), std::strerror(errno));
    }
}

void writeGaMDForceSnapshot(FILE*                output,
                            const std::string&   filename,
                            int64_t              step,
                            ArrayRef<const RVec> forces,
                            int                  requestedBegin,
                            int                  requestedCount,
                            double               forceWeight,
                            ArrayRef<const int>  globalAtomIndices)
{
    const int numAtoms = forces.ssize();
    const int begin    = std::clamp(requestedBegin, 0, numAtoms);
    int       end      = numAtoms;
    if (requestedCount >= 0)
    {
        end = begin + std::min(requestedCount, numAtoms - begin);
    }
    GMX_RELEASE_ASSERT(globalAtomIndices.empty() || globalAtomIndices.size() == forces.size(),
                       "GaMD diagnostic atom mapping must match the force buffer");

    // 只为选中的输出范围建立编号映射，按拓扑顺序写出 CPU 重排后的数据。
    std::vector<Index> localIndices;
    if (!globalAtomIndices.empty())
    {
        localIndices.resize(end - begin, -1);
        for (Index localIndex = 0; localIndex < forces.ssize(); ++localIndex)
        {
            const int globalIndex = globalAtomIndices[localIndex];
            if (globalIndex >= begin && globalIndex < end)
            {
                localIndices[globalIndex - begin] = localIndex;
            }
        }
    }
    if (std::fprintf(
                output, "# dump step %lld nAtoms %d begin %d end %d\n", static_cast<long long>(step), numAtoms, begin, end)
        < 0)
    {
        gmx_fatal(FARGS, "Failed to write GaMD diagnostic file '%s': %s.", filename.c_str(), std::strerror(errno));
    }
    for (int atomIndex = begin; atomIndex < end; ++atomIndex)
    {
        Index localIndex = atomIndex;
        if (!globalAtomIndices.empty())
        {
            localIndex = localIndices[atomIndex - begin];
            GMX_RELEASE_ASSERT(localIndex >= 0,
                               "GaMD diagnostic output requires a complete topology atom mapping");
        }
        const double forceX    = forceWeight * forces[localIndex][XX];
        const double forceY    = forceWeight * forces[localIndex][YY];
        const double forceZ    = forceWeight * forces[localIndex][ZZ];
        const double forceNorm = std::sqrt(forceX * forceX + forceY * forceY + forceZ * forceZ);
        if (std::fprintf(
                    output, "%lld %d %.9e %.9e %.9e %.9e\n", static_cast<long long>(step), atomIndex, forceX, forceY, forceZ, forceNorm)
            < 0)
        {
            gmx_fatal(FARGS,
                      "Failed to write GaMD diagnostic file '%s': %s.",
                      filename.c_str(),
                      std::strerror(errno));
        }
    }
    flushGaMDDiagnosticFile(output, filename);
}

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
    for (gmx::Index atomIndex = 0; atomIndex < reference.ssize(); ++atomIndex)
    {
        for (int componentIndex = 0; componentIndex < DIM; ++componentIndex)
        {
            const double referenceValue = reference[atomIndex][componentIndex];
            const double candidateValue = candidate[atomIndex][componentIndex];
            const double error          = candidateValue - referenceValue;
            // 不能依赖最大值聚合发现 NaN；它可能让后面的大小比较全部失效。
            if (!std::isfinite(referenceValue) || !std::isfinite(candidateValue) || !std::isfinite(error))
            {
                gmx_fatal(FARGS,
                          "Non-finite GaMD shadow force at atom %td component %d: cpu=%g gpu=%g.",
                          atomIndex,
                          componentIndex,
                          referenceValue,
                          candidateValue);
            }
            const double absoluteError = std::abs(error);
            sumSquaredError += error * error;
            if (absoluteError > comparison.maxAbsoluteError)
            {
                comparison.maxAbsoluteError = absoluteError;
                comparison.worstAtom        = atomIndex;
                comparison.worstComponent   = componentIndex;
            }
        }
    }
    comparison.rmsError = std::sqrt(sumSquaredError / (DIM * reference.size()));
    return comparison;
}

} // namespace

void validateGaMDGpuShadowEnergy(int64_t step, const GaMDStepResult& reference, const GaMDStepResult& candidate)
{
    const std::array<double, 5> referenceValues = { reference.potentialEnergy,
                                                    reference.dihedralEnergy,
                                                    reference.scaleP,
                                                    reference.scaleD,
                                                    reference.totalBoost() };
    const std::array<double, 5> candidateValues = { candidate.potentialEnergy,
                                                    candidate.dihedralEnergy,
                                                    candidate.scaleP,
                                                    candidate.scaleD,
                                                    candidate.totalBoost() };
    // 保留已有混合精度阈值；TC5b 总势能的 0.25 kJ/mol 对应四个浮点间隔。
    constexpr std::array<double, 5> tolerances  = { 0.25, 0.001, 2e-5, 2e-5, 0.1 };
    std::array<double, 5>           differences = {};
    bool                            mismatch    = false;
    for (std::size_t quantityIndex = 0; quantityIndex < differences.size(); ++quantityIndex)
    {
        differences[quantityIndex] = candidateValues[quantityIndex] - referenceValues[quantityIndex];
        if (!std::isfinite(referenceValues[quantityIndex]) || !std::isfinite(candidateValues[quantityIndex])
            || !std::isfinite(differences[quantityIndex]))
        {
            gmx_fatal(FARGS,
                      "Non-finite GaMD GPU energy/state shadow at step %lld, quantity %zu: "
                      "cpu=%g gpu=%g.",
                      static_cast<long long>(step),
                      quantityIndex,
                      referenceValues[quantityIndex],
                      candidateValues[quantityIndex]);
        }
        if (std::abs(differences[quantityIndex]) > tolerances[quantityIndex])
        {
            mismatch = true;
        }
    }
    std::fprintf(stderr,
                 "[GaMD GPU energy shadow] step=%lld total_diff=%.9g dihedral_diff=%.9g "
                 "scaleP_diff=%.9g scaleD_diff=%.9g total_boost_diff=%.9g\n",
                 static_cast<long long>(step),
                 differences[0],
                 differences[1],
                 differences[2],
                 differences[3],
                 differences[4]);
    if (mismatch)
    {
        gmx_fatal(FARGS,
                  "GaMD GPU energy/state shadow mismatch at step %lld: total %.9g, "
                  "dihedral %.9g, scaleP %.9g, scaleD %.9g, total boost %.9g.",
                  static_cast<long long>(step),
                  differences[0],
                  differences[1],
                  differences[2],
                  differences[3],
                  differences[4]);
    }
}

void validateGaMDGpuShadowVirial(int64_t step, const tensor reference, const std::array<real, DIM * DIM>& candidate)
{
    double maxAbsoluteError = 0.0;
    for (int rowIndex = 0; rowIndex < DIM; ++rowIndex)
    {
        for (int columnIndex = 0; columnIndex < DIM; ++columnIndex)
        {
            const double referenceValue = reference[rowIndex][columnIndex];
            const double candidateValue = candidate[rowIndex * DIM + columnIndex];
            const double difference     = candidateValue - referenceValue;
            // 每个分量都检查，避免 std::max 把 NaN 隐藏在有限的最大误差后面。
            if (!std::isfinite(referenceValue) || !std::isfinite(candidateValue)
                || !std::isfinite(difference))
            {
                gmx_fatal(FARGS,
                          "Non-finite GaMD GPU virial shadow at step %lld, component %d,%d: "
                          "cpu=%g gpu=%g.",
                          static_cast<long long>(step),
                          rowIndex,
                          columnIndex,
                          referenceValue,
                          candidateValue);
            }
            maxAbsoluteError = std::max(maxAbsoluteError, std::abs(difference));
        }
    }
    std::fprintf(stderr,
                 "[GaMD GPU shadow] live-dihedral-virial step=%lld max_abs=%.9g\n",
                 static_cast<long long>(step),
                 maxAbsoluteError);
    if (maxAbsoluteError > 0.01)
    {
        gmx_fatal(FARGS,
                  "GaMD GPU dihedral virial exceeds provisional mixed-precision "
                  "tolerance (max %.9g > 0.01).",
                  maxAbsoluteError);
    }
}

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

static double readGaMDDiagnosticReal(const char* variableName, double defaultValue)
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
    // 先解析数字，再打开文件，避免参数写错时先截断已有诊断文件。
    if (const char* configuredStep = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP"))
    {
        dihedralForceStep_ = int64FromString(configuredStep);
    }
    if (const char* configuredBegin = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_BEGIN"))
    {
        dihedralForceBegin_ = intFromString(configuredBegin);
    }
    if (const char* configuredCount = std::getenv("GMX_GAMD_DIH_FORCE_DEBUG_COUNT"))
    {
        dihedralForceCount_ = intFromString(configuredCount);
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
    rawDihedralFile_   = openGaMDDiagnosticFile("GMX_GAMD_DIH_RAW_DEBUG_FILE",
                                              "# step atom fx fy fz (raw CPU dihedral evaluation "
                                                "contributions; recomputations included)\n",
                                              &rawDihedralFilename_);
    dihedralForceFile_ = openGaMDDiagnosticFile("GMX_GAMD_DIH_FORCE_DEBUG_FILE",
                                                "# step atom fx fy fz |f| (scaleP * scaleD * raw "
                                                "dihedral force; zero-based topology atom index)\n",
                                                &dihedralForceFilename_);
    finalForceFile_    = openGaMDDiagnosticFile(
            "GMX_GAMD_FORCE_DEBUG_FILE", "# step atom fx fy fz |f|\n", &finalForceFilename_);
    gpuWorkloadDiagnosticsEnabled_ = std::getenv("GMX_GAMD_GPU_DIAGNOSTICS") != nullptr;
}

GaMDDiagnostics::~GaMDDiagnostics()
{
    closeGaMDDiagnosticFile(rawDihedralFile_, rawDihedralFilename_);
    closeGaMDDiagnosticFile(dihedralForceFile_, dihedralForceFilename_);
    closeGaMDDiagnosticFile(finalForceFile_, finalForceFilename_);
}

GaMDHostWorkspace& GaMDDiagnostics::hostWorkspace()
{
    return hostWorkspace_;
}

const GaMDForceOverrides& GaMDDiagnostics::forceOverrides()
{
    if (!forceOverridesRead_)
    {
        // 只有 GaMD 真正使用时才读取，普通模拟不解析这些诊断设置。
        if (std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP") != nullptr)
        {
            forceOverrides_.totalScale = readGaMDDiagnosticReal("GMX_GAMD_FORCE_OVERRIDE_SCALEP", 1.0);
            forceOverrides_.dihedralScale = readGaMDDiagnosticReal("GMX_GAMD_FORCE_OVERRIDE_SCALED",
                                                                   forceOverrides_.totalScale);
            forceOverrides_.potentialBoost =
                    readGaMDDiagnosticReal("GMX_GAMD_FORCE_OVERRIDE_BOOSTP", 0.0);
            forceOverrides_.dihedralBoost = readGaMDDiagnosticReal("GMX_GAMD_FORCE_OVERRIDE_BOOSTD", 0.0);
            forceOverrides_.enabled = true;
        }
        forceOverridesRead_ = true;
    }
    return forceOverrides_;
}

void GaMDDiagnostics::prepareDihedralStep(const int64_t step)
{
    rawDihedralCurrentStep_ = step;
}

bool GaMDDiagnostics::dihedralForceOutputRequested(const int64_t step) const
{
    return dihedralForceFile_ != nullptr && dihedralForceStep_ >= 0 && step == dihedralForceStep_
           && step != dihedralForceWrittenStep_;
}

void GaMDDiagnostics::dumpDihedralForces(const int64_t        step,
                                         ArrayRef<const RVec> rawForces,
                                         const double         forceWeight,
                                         const bool           isMainRank,
                                         ArrayRef<const int>  globalAtomIndices)
{
    if (!isMainRank || !dihedralForceOutputRequested(step))
    {
        return;
    }
    // 使用已经完成的力缓冲，不再按 bonded 内核的调用次数累加。
    writeGaMDForceSnapshot(dihedralForceFile_,
                           dihedralForceFilename_,
                           step,
                           rawForces,
                           dihedralForceBegin_,
                           dihedralForceCount_,
                           forceWeight,
                           globalAtomIndices);
    dihedralForceWrittenStep_ = step;
}

void GaMDDiagnostics::addRawDihedralForce(const int atomIndex, const rvec force)
{
    if (rawDihedralFile_ == nullptr)
    {
        return;
    }
    if (std::fprintf(rawDihedralFile_,
                     "%lld %d %.9e %.9e %.9e\n",
                     static_cast<long long>(rawDihedralCurrentStep_),
                     atomIndex,
                     static_cast<double>(force[XX]),
                     static_cast<double>(force[YY]),
                     static_cast<double>(force[ZZ]))
        < 0)
    {
        gmx_fatal(FARGS,
                  "Failed to write GaMD diagnostic file '%s': %s.",
                  rawDihedralFilename_.c_str(),
                  std::strerror(errno));
    }
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

void GaMDDiagnostics::finishDihedralStep(const int64_t step, const bool isMainRank)
{
    if (rawDihedralFile_ != nullptr)
    {
        flushGaMDDiagnosticFile(rawDihedralFile_, rawDihedralFilename_);
    }
    // 到达了请求的步号却没有样本时，不能把空文件当成有效诊断结果。
    if (isMainRank && dihedralForceOutputRequested(step))
    {
        gmx_fatal(FARGS,
                  "No GaMD dihedral force snapshot was produced at step %lld for '%s'.",
                  static_cast<long long>(step),
                  dihedralForceFilename_.c_str());
    }
}

void GaMDDiagnostics::dumpFinalForces(const int64_t step, const ArrayRef<const RVec> forces, const bool isMainRank)
{
    if (finalForceFile_ == nullptr || !isMainRank || step != finalForceStep_)
    {
        return;
    }
    writeGaMDForceSnapshot(
            finalForceFile_, finalForceFilename_, step, forces, finalForceBegin_, finalForceCount_, 1.0, {});
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
