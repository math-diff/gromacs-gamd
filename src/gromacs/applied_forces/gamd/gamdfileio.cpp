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
 * \brief Legacy GaMD restart recovery and text output.
 */
#include "gmxpre.h"

#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"

#include "gamdruntime.h"

namespace gmx
{

static bool fileExists(const char* filename)
{
    std::ifstream input(filename);
    return input.good();
}

static void flushAndSyncFileOrFatal(FILE* fp, const char* filename)
{
    if (fp == nullptr)
    {
        return;
    }
    if (std::fflush(fp) != 0 || gmx_fsync(fp) != 0)
    {
        gmx_fatal(FARGS, "Failed to flush and synchronize GaMD file '%s'.", filename);
    }
}

static bool parseGaMDTextRow(const std::string& line, int expectedColumns, int64_t* step)
{
    std::istringstream       input(line);
    std::vector<std::string> tokens;
    std::string              token;
    while (input >> token)
    {
        tokens.push_back(token);
    }
    if (static_cast<int>(tokens.size()) != expectedColumns)
    {
        return false;
    }

    for (const std::string& value : tokens)
    {
        char* end = nullptr;
        errno     = 0;
        std::strtod(value.c_str(), &end);
        if (end == value.c_str() || *end != '\0' || errno == ERANGE)
        {
            return false;
        }
    }

    char* stepEnd            = nullptr;
    errno                    = 0;
    const int64_t parsedStep = std::strtoll(tokens[1].c_str(), &stepEnd, 10);
    if (stepEnd == tokens[1].c_str() || *stepEnd != '\0' || errno == ERANGE)
    {
        return false;
    }
    *step = parsedStep;
    return true;
}

/* The GROMACS checkpoint and these human-readable files cannot be committed as
 * one filesystem transaction.  Make continuation deterministic instead: the
 * restart file is the commit record, and on load each text file is atomically
 * rebuilt to contain exactly one complete row per step through savedStep.
 * Re-running this repair after another interruption is safe and idempotent. */
static void reconcileGaMDTextFile(const char* filename, int expectedColumns, int64_t savedStep, int rank)
{
    if (rank != 0 || !fileExists(filename))
    {
        return;
    }

    std::ifstream input(filename);
    if (!input.is_open())
    {
        gmx_fatal(FARGS, "Failed to open GaMD text file '%s' for continuation repair.", filename);
    }

    std::vector<std::string>       headers;
    std::set<std::string>          seenHeaders;
    std::map<int64_t, std::string> rowsByStep;
    int64_t                        malformedRows = 0;
    int64_t                        futureRows    = 0;
    int64_t                        duplicateRows = 0;
    std::string                    line;
    while (std::getline(input, line))
    {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos)
        {
            continue;
        }
        if (line[first] == '#' || line[first] == ';')
        {
            if (seenHeaders.insert(line).second)
            {
                headers.push_back(line);
            }
            continue;
        }

        int64_t rowStep = -1;
        if (!parseGaMDTextRow(line, expectedColumns, &rowStep))
        {
            malformedRows++;
            continue;
        }
        if (rowStep > savedStep)
        {
            futureRows++;
            continue;
        }
        if (!rowsByStep.emplace(rowStep, line).second)
        {
            duplicateRows++;
        }
    }
    if (input.bad())
    {
        gmx_fatal(FARGS, "Failed while reading GaMD text file '%s'.", filename);
    }
    input.close();

    const std::string tmpName = std::string(filename) + ".reconcile.tmp";
    FILE*             fp      = std::fopen(tmpName.c_str(), "w");
    if (fp == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open '%s' for continuation repair.", tmpName.c_str());
    }
    for (const std::string& header : headers)
    {
        std::fprintf(fp, "%s\n", header.c_str());
    }
    for (const auto& [rowStep, row] : rowsByStep)
    {
        (void)rowStep;
        std::fprintf(fp, "%s\n", row.c_str());
    }
    flushAndSyncFileOrFatal(fp, tmpName.c_str());
    if (std::fclose(fp) != 0)
    {
        gmx_fatal(FARGS, "Failed to close repaired GaMD text file '%s'.", tmpName.c_str());
    }
    if (std::rename(tmpName.c_str(), filename) != 0)
    {
        gmx_fatal(FARGS, "Failed to atomically replace repaired GaMD text file '%s'.", filename);
    }

    if (malformedRows > 0 || futureRows > 0 || duplicateRows > 0)
    {
        std::fprintf(stderr,
                     "[GaMD WARNING] Repaired %s at checkpoint step %" PRId64 ": removed %" PRId64
                     " "
                     "malformed/partial, %" PRId64 " post-checkpoint, and %" PRId64
                     " duplicate rows.\n",
                     filename,
                     savedStep,
                     malformedRows,
                     futureRows,
                     duplicateRows);
    }
}

static void writeStatsLine(FILE* fp, const char* name, const GaMDEnergyStatistics& s)
{
    std::fprintf(fp,
                 "%s %.17g %.17g %.17g %.17g %.17g %" PRId64 " %.17g %.17g %.17g\n",
                 name,
                 s.maximum,
                 s.minimum,
                 s.average,
                 s.deviation,
                 s.m2,
                 s.count,
                 s.threshold,
                 s.k0,
                 s.forceConstant);
}

static bool readStatsLine(std::istringstream& iss, GaMDEnergyStatistics* s)
{
    return static_cast<bool>(iss >> s->maximum >> s->minimum >> s->average >> s->deviation >> s->m2
                             >> s->count >> s->threshold >> s->k0 >> s->forceConstant);
}

static void writeWindowStatsLine(FILE* fp, const char* name, const GaMDWindowStatistics& s)
{
    std::fprintf(fp, "%s %" PRId64 " %.17g %.17g\n", name, s.count, s.mean, s.m2);
}

static bool readWindowStatsLine(std::istringstream& iss, GaMDWindowStatistics* s)
{
    return static_cast<bool>(iss >> s->count >> s->mean >> s->m2);
}

void GaMDRuntime::saveRestartState(int64_t step, int rank)
{
    if (rank != 0)
    {
        return;
    }

    if (!parametersLoaded_ || parameters_.boostType == 0)
    {
        return;
    }

    if (step < 0)
    {
        return;
    }

    const GaMDStage currentStage          = determineStage(step);
    const bool      saveAtCmdStatBoundary = (step == parameters_.conventionalEnd);
    const bool saveAtBoostStatBoundary = (step == parameters_.conventionalEnd + parameters_.boostLength);
    const bool inBoostStatPhase     = (currentStage == GaMDStage::BoostStatistics);
    const bool inProductionPhase    = (currentStage == GaMDStage::Production);
    const bool saveDuringBoostStat  = (inBoostStatPhase && checkpointingThisStep_);
    const bool saveDuringProduction = (inProductionPhase && checkpointingThisStep_);

    if (!saveAtCmdStatBoundary && !saveAtBoostStatBoundary && !saveDuringBoostStat && !saveDuringProduction)
    {
        return;
    }

    /* During the pure cMD preparation window (step <= ntcmdprep), GaMD has not
     * started accumulating statistics yet. Writing a restart file there only
     * produces an almost empty state that is not useful for resume logic.
     *
     * Save once when the cMD statistics stage ends (step == ntcmd), save during
     * the boost-stat stage on the same steps where .cpt checkpointing happens,
     * and force one final save at step == ntcmd + nteb when the adaptive
     * parameter updates finish.
     */
    if (step <= parameters_.conventionalPrepEnd && !inProductionPhase)
    {
        return;
    }

    const char* tmpName      = "gamd-restart.dat.tmp";
    const char* finalName    = "gamd-restart.dat";
    const char* previousName = "gamd-restart-prev.dat";

    // Rows already emitted before this checkpoint must reach durable storage
    // before the restart state is used as their commit record.
    flushAndSyncFileOrFatal(reweightFile_, "gamd-reweight.dat");
    flushAndSyncFileOrFatal(parameterFile_, "gamd-para.dat");

    FILE* fp = std::fopen(tmpName, "w");
    if (fp == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open %s for writing GaMD restart state.", tmpName);
    }

    std::fprintf(fp, "# GaMD restart state\n");
    std::fprintf(fp, "version 3\n");
    std::fprintf(fp, "saved_step %" PRId64 "\n", step);
    std::fprintf(fp, "production_restart %d\n", static_cast<int>(forceProductionRestart_));

    // 参数也写出，续跑时做一致性检查
    std::fprintf(fp,
                 "params %d %d %d %d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64
                 " %" PRId64 " %.17g %.17g\n",
                 parameters_.boostType,
                 parameters_.thresholdMode,
                 parameters_.potentialMode,
                 parameters_.dihedralMode,
                 parameters_.conventionalPrepEnd,
                 parameters_.conventionalEnd,
                 parameters_.boostPrepLength,
                 parameters_.boostLength,
                 parameters_.averagingWindow,
                 parameters_.reweightInterval,
                 parameters_.potentialSigmaLimit,
                 parameters_.dihedralSigmaLimit);

    writeWindowStatsLine(fp, "windowP", potentialWindowStatistics_);
    writeWindowStatsLine(fp, "windowD", dihedralWindowStatistics_);
    writeStatsLine(fp, "statP", potentialStatistics_);
    writeStatsLine(fp, "statD", dihedralStatistics_);

    flushAndSyncFileOrFatal(fp, tmpName);
    if (std::fclose(fp) != 0)
    {
        gmx_fatal(FARGS, "Failed to close %s after writing GaMD restart state.", tmpName);
    }

    // Keep the previous commit record. If the process dies after this update
    // but before the matching .cpt is committed, continuation can select the
    // previous restart whose saved_step matches the checkpoint.
    if (fileExists(finalName))
    {
        std::remove(previousName);
        if (std::rename(finalName, previousName) != 0)
        {
            gmx_fatal(FARGS, "Failed to rotate previous GaMD restart %s -> %s.", finalName, previousName);
        }
    }
    if (std::rename(tmpName, finalName) != 0)
    {
        gmx_fatal(FARGS, "Failed to atomically replace GaMD restart %s -> %s.", tmpName, finalName);
    }
}

// 重启数据必须先通过完整性检查，再交给算法状态。
static bool validStats(const GaMDEnergyStatistics& statistics)
{
    return std::isfinite(statistics.maximum) && std::isfinite(statistics.minimum)
           && std::isfinite(statistics.average) && std::isfinite(statistics.deviation)
           && std::isfinite(statistics.m2) && std::isfinite(statistics.threshold)
           && std::isfinite(statistics.k0) && std::isfinite(statistics.forceConstant)
           && statistics.count >= 0 && statistics.deviation >= 0.0 && statistics.m2 >= 0.0
           && statistics.k0 >= 0.0 && statistics.k0 <= 1.0 && statistics.forceConstant >= 0.0
           && (statistics.count == 0 || statistics.maximum >= statistics.minimum);
}

static bool validWindowStats(const GaMDWindowStatistics& statistics)
{
    return statistics.count >= 0 && std::isfinite(statistics.mean) && std::isfinite(statistics.m2)
           && statistics.m2 >= 0.0;
}

// 先形成完整、已验证的快照，再选择文件和更新运行状态。
struct GaMDRestartState
{
    int                  version              = 0;
    int64_t              savedStep            = -1;
    bool                 productionRestart    = false;
    bool                 hasProductionRestart = false;
    GaMDParameters       parameters;
    GaMDEnergyStatistics potentialStatistics;
    GaMDEnergyStatistics dihedralStatistics;
    GaMDWindowStatistics potentialWindow;
    GaMDWindowStatistics dihedralWindow;
};

static std::optional<GaMDRestartState> readGaMDRestartState(const char* filename, std::string* error)
{
    error->clear();
    std::ifstream input(filename);
    if (!input.is_open())
    {
        return std::nullopt;
    }

    GaMDRestartState      state;
    std::set<std::string> fields;
    std::string           line;
    int64_t               lineNumber = 0;
    while (std::getline(input, line))
    {
        lineNumber++;
        std::istringstream columns(line);
        std::string        field;
        if (!(columns >> field) || field[0] == '#' || field[0] == ';')
        {
            continue;
        }
        const std::string location = std::string(filename) + ":" + std::to_string(lineNumber) + ": ";
        if (!fields.insert(field).second)
        {
            *error = location + "Duplicate GaMD restart field '" + field + "'.";
            return std::nullopt;
        }

        bool parsed = false;
        if (field == "version")
        {
            parsed = static_cast<bool>(columns >> state.version);
        }
        else if (field == "saved_step")
        {
            parsed = static_cast<bool>(columns >> state.savedStep);
        }
        else if (field == "production_restart")
        {
            int productionRestart = 0;
            parsed                = static_cast<bool>(columns >> productionRestart);
            if (parsed && productionRestart != 0 && productionRestart != 1)
            {
                *error = location + "production_restart must be 0 or 1.";
                return std::nullopt;
            }
            state.productionRestart    = (productionRestart == 1);
            state.hasProductionRestart = true;
        }
        else if (field == "params")
        {
            GaMDParameters& parameters = state.parameters;
            parsed = static_cast<bool>(columns >> parameters.boostType >> parameters.thresholdMode
                                       >> parameters.potentialMode >> parameters.dihedralMode
                                       >> parameters.conventionalPrepEnd >> parameters.conventionalEnd
                                       >> parameters.boostPrepLength >> parameters.boostLength
                                       >> parameters.averagingWindow >> parameters.reweightInterval
                                       >> parameters.potentialSigmaLimit >> parameters.dihedralSigmaLimit);
        }
        else if (field == "statP")
        {
            parsed = readStatsLine(columns, &state.potentialStatistics);
        }
        else if (field == "statD")
        {
            parsed = readStatsLine(columns, &state.dihedralStatistics);
        }
        else if (field == "windowP")
        {
            parsed = readWindowStatsLine(columns, &state.potentialWindow);
        }
        else if (field == "windowD")
        {
            parsed = readWindowStatsLine(columns, &state.dihedralWindow);
        }
        else
        {
            *error = location + "Unknown GaMD restart field '" + field + "'.";
            return std::nullopt;
        }
        if (!parsed)
        {
            *error = location + "Invalid or incomplete GaMD restart field '" + field + "'.";
            return std::nullopt;
        }
        std::string trailing;
        if ((columns >> trailing) && trailing[0] != '#' && trailing[0] != ';')
        {
            *error = location + "Unexpected trailing token in GaMD restart field '" + field + "'.";
            return std::nullopt;
        }
    }
    if (input.bad())
    {
        *error = "Failed while reading GaMD restart data.";
        return std::nullopt;
    }

    const char* requiredFields[] = { "version", "saved_step", "params", "statP", "statD" };
    for (int fieldIndex = 0; fieldIndex < 5; fieldIndex++)
    {
        if (fields.count(requiredFields[fieldIndex]) == 0)
        {
            *error = std::string("Missing GaMD restart field '") + requiredFields[fieldIndex] + "'.";
            return std::nullopt;
        }
    }
    if (state.version < 1 || state.version > 3)
    {
        *error = "Unsupported GaMD restart version " + std::to_string(state.version) + ".";
        return std::nullopt;
    }
    if (state.version >= 2 && (fields.count("windowP") == 0 || fields.count("windowD") == 0))
    {
        *error = "GaMD restart is missing window statistics.";
        return std::nullopt;
    }
    if (state.version == 3 && !state.hasProductionRestart)
    {
        *error = "GaMD restart version 3 requires production_restart.";
        return std::nullopt;
    }
    // 连备份也先校验完整阶段参数，不能把坏计数恢复到正在运行的模拟。
    *error = gaMDStageParameterError(state.parameters);
    if (!error->empty())
    {
        return std::nullopt;
    }
    if (state.savedStep < 0 || !validStats(state.potentialStatistics) || !validStats(state.dihedralStatistics)
        || (state.version >= 2
            && (!validWindowStats(state.potentialWindow) || !validWindowStats(state.dihedralWindow)))
        || !std::isfinite(state.parameters.potentialSigmaLimit)
        || !std::isfinite(state.parameters.dihedralSigmaLimit))
    {
        *error = "GaMD restart contains invalid or non-finite saved state.";
        return std::nullopt;
    }
    return state;
}

void GaMDRuntime::loadRestartStateIfNeeded(int64_t step, int rank)
{
    if (restartLoadAttempted_)
    {
        return;
    }

    const bool explicitProductionRestart = (parameters_.restartMode != 0);

    // 只尝试一次：
    // - fresh run 若从 step 0 开始，不要后面误加载旧文件
    // - continue run 第一调用通常就是 step > 0，会在这里加载
    // - irest_gamd 模式允许从新模拟的首步直接加载
    restartLoadAttempted_ = true;

    if (step <= 0 && !explicitProductionRestart)
    {
        if (fileExists("gamd-reweight.dat") || fileExists("gamd-para.dat")
            || fileExists("gamd-restart.dat") || fileExists("gamd-restart-prev.dat"))
        {
            gmx_fatal(FARGS,
                      "A fresh GaMD run found existing GaMD output/restart files in the working "
                      "directory. Refusing to append unrelated data; use a clean directory or "
                      "continue with the matching checkpoint.");
        }
        return;
    }
    std::string currentError;
    std::optional<GaMDRestartState> currentState = readGaMDRestartState("gamd-restart.dat", &currentError);
    const bool currentMatches =
            currentState && (currentState->savedStep == step || currentState->savedStep == step - 1);
    std::string                     previousError;
    std::optional<GaMDRestartState> previousState;
    if (!currentMatches && step > 0)
    {
        previousState = readGaMDRestartState("gamd-restart-prev.dat", &previousError);
    }
    const bool previousMatches =
            previousState && (previousState->savedStep == step || previousState->savedStep == step - 1);
    const bool checkpointContinuation = step > 0 && (currentMatches || previousMatches);

    const char*             restartFilename = "gamd-restart.dat";
    const GaMDRestartState* selectedState   = nullptr;
    if (explicitProductionRestart && !checkpointContinuation)
    {
        if (fileExists("gamd-reweight.dat") || fileExists("gamd-para.dat"))
        {
            gmx_fatal(FARGS,
                      "irest_gamd found existing output but no GaMD restart matches continued step "
                      "%" PRId64 ".",
                      step);
        }
        if (!currentState)
        {
            gmx_fatal(FARGS,
                      "GaMD restart '%s' is missing or invalid: %s",
                      restartFilename,
                      currentError.c_str());
        }
        selectedState = &currentState.value();
    }
    else if (currentMatches)
    {
        selectedState = &currentState.value();
    }
    else if (previousMatches)
    {
        restartFilename = "gamd-restart-prev.dat";
        selectedState   = &previousState.value();
        if (rank == 0)
        {
            std::fprintf(stderr,
                         "[GaMD WARNING] Current restart does not match checkpoint step %" PRId64
                         "; using validated %s saved_step=%" PRId64 ". %s\n",
                         step,
                         restartFilename,
                         selectedState->savedStep,
                         currentError.c_str());
        }
    }
    else
    {
        const bool haveArtifacts = fileExists("gamd-reweight.dat") || fileExists("gamd-para.dat")
                                   || fileExists("gamd-restart.dat")
                                   || fileExists("gamd-restart-prev.dat");
        if (haveArtifacts || !currentError.empty() || !previousError.empty())
        {
            gmx_fatal(FARGS,
                      "No valid GaMD restart matches first continued step %" PRId64 ". %s %s",
                      step,
                      currentError.c_str(),
                      previousError.c_str());
        }
        return;
    }

    const GaMDRestartState& savedState      = *selectedState;
    const GaMDParameters&   savedParameters = savedState.parameters;
    const bool              physicsParamsMismatch =
            (savedParameters.boostType != parameters_.boostType
             || savedParameters.thresholdMode != parameters_.thresholdMode
             || savedParameters.potentialMode != parameters_.potentialMode
             || savedParameters.dihedralMode != parameters_.dihedralMode
             || std::abs(savedParameters.potentialSigmaLimit - parameters_.potentialSigmaLimit) > 1e-9
             || std::abs(savedParameters.dihedralSigmaLimit - parameters_.dihedralSigmaLimit) > 1e-9);
    const bool stageParamsMismatch =
            (savedParameters.conventionalPrepEnd != parameters_.conventionalPrepEnd
             || savedParameters.conventionalEnd != parameters_.conventionalEnd
             || savedParameters.boostPrepLength != parameters_.boostPrepLength
             || savedParameters.boostLength != parameters_.boostLength
             || savedParameters.averagingWindow != parameters_.averagingWindow);

    // 只检查会影响物理轨迹/boost 的参数。
    // reweight_nst 只影响输出格式，不影响动力学，因此不作为拒绝条件。
    // irest_gamd 模式下，ntcmd/nteb/ntave 等分段控制参数允许在 gamd.in 中
    // 置零；生产阶段直接从 restart 状态起步时，它们不再控制动力学流程。
    if (physicsParamsMismatch || (!explicitProductionRestart && stageParamsMismatch))
    {
        gmx_fatal(FARGS,
                  "Current gamd.in does not match GaMD restart '%s'; refusing an "
                  "inconsistent continuation (irest_gamd=%d).",
                  restartFilename,
                  parameters_.restartMode);
    }

    potentialStatistics_ = savedState.potentialStatistics;
    dihedralStatistics_  = savedState.dihedralStatistics;
    if (savedState.version >= 2)
    {
        potentialWindowStatistics_ = savedState.potentialWindow;
        dihedralWindowStatistics_  = savedState.dihedralWindow;
    }
    else
    {
        potentialWindowStatistics_ = {};
        dihedralWindowStatistics_  = {};
    }

    if (explicitProductionRestart)
    {
        const int64_t restartNtcmd = savedParameters.conventionalEnd;
        const int64_t restartNteb  = savedParameters.boostLength;
        const bool    inferredProductionRestart =
                (!savedState.hasProductionRestart && savedState.savedStep < restartNtcmd);
        const bool restartFileIsProductionRestart =
                (savedState.productionRestart || inferredProductionRestart);

        if (savedState.savedStep < restartNtcmd && !restartFileIsProductionRestart)
        {
            gmx_fatal(FARGS,
                      "irest_gamd = %d requires GaMD parameters that were already initialized, "
                      "but gamd-restart.dat was saved at step %" PRId64 " before ntcmd = %" PRId64
                      ".",
                      parameters_.restartMode,
                      savedState.savedStep,
                      restartNtcmd);
        }

        // In irest_gamd mode the stage-control counters can be omitted from
        // gamd.in. Keep the values from the restart file for provenance/output.
        parameters_.conventionalPrepEnd = savedParameters.conventionalPrepEnd;
        parameters_.conventionalEnd     = savedParameters.conventionalEnd;
        parameters_.boostPrepLength     = savedParameters.boostPrepLength;
        parameters_.boostLength         = savedParameters.boostLength;
        parameters_.averagingWindow     = savedParameters.averagingWindow;

        forceProductionRestart_ = true;

        if (checkpointContinuation)
        {
            reconcileGaMDTextFile("gamd-reweight.dat", 9, savedState.savedStep, rank);
            reconcileGaMDTextFile("gamd-para.dat", 17, savedState.savedStep, rank);
            if (savedState.savedStep == step)
            {
                suppressTextOutputStep_ = savedState.savedStep;
            }
        }

        if (rank == 0)
        {
            if (inferredProductionRestart)
            {
                std::fprintf(stderr,
                             "[GaMD INFO] Inferred production_restart=1 from gamd-restart.dat "
                             "because saved_step=%" PRId64 " is smaller than ntcmd=%" PRId64
                             ". "
                             "This is expected for restart files written by a prior irest_gamd "
                             "production run.\n",
                             savedState.savedStep,
                             restartNtcmd);
            }

            std::fprintf(stderr,
                         "[GaMD INFO] irest_gamd=%d ignores ntcmdprep/ntcmd/ntebprep/nteb "
                         "from gamd.in and uses the values stored in gamd-restart.dat.\n",
                         parameters_.restartMode);

            if (!checkpointContinuation && savedState.savedStep < restartNtcmd + restartNteb)
            {
                std::fprintf(stderr,
                             "[GaMD WARNING] irest_gamd=%d loaded saved_step=%" PRId64
                             " before the "
                             "adaptive boost-stat stage ended at step=%" PRId64
                             ". "
                             "The new simulation will enter production immediately using the "
                             "latest saved GaMD parameters.\n",
                             parameters_.restartMode,
                             savedState.savedStep,
                             restartNtcmd + restartNteb);
            }

            if (checkpointContinuation)
            {
                std::fprintf(stderr,
                             "[GaMD INFO] Continued irest_gamd production from %s "
                             "saved_step=%" PRId64 " at checkpoint step=%" PRId64
                             "; text outputs are "
                             "reconciled to the checkpoint.\n",
                             restartFilename,
                             savedState.savedStep,
                             step);
            }
            else
            {
                std::fprintf(stderr,
                             "[GaMD INFO] Loaded GaMD restart state from saved_step=%" PRId64
                             " for a "
                             "new simulation with irest_gamd=%d; forcing GaMD production stage "
                             "from step=%" PRId64 ".\n",
                             savedState.savedStep,
                             parameters_.restartMode,
                             step);
            }
        }
    }
    else
    {
        reconcileGaMDTextFile("gamd-reweight.dat", 9, savedState.savedStep, rank);
        reconcileGaMDTextFile("gamd-para.dat", 17, savedState.savedStep, rank);

        if (savedState.savedStep == step)
        {
            suppressTextOutputStep_ = savedState.savedStep;
        }

        if (rank == 0)
        {
            std::fprintf(stderr,
                         "[GaMD INFO] Loaded GaMD restart state from %s saved_step=%" PRId64
                         " for "
                         "continued step=%" PRId64
                         "; text outputs are reconciled to the checkpoint.\n",
                         restartFilename,
                         savedState.savedStep,
                         step);
        }
    }
}

void GaMDRuntime::writeRestartState(int64_t step)
{
    saveRestartState(step, currentRank_);
}

GaMDRuntime::~GaMDRuntime()
{
    closeOutputFiles();
}

void GaMDRuntime::closeOutputFiles()
{
    if (reweightFile_ != nullptr)
    {
        std::fclose(reweightFile_);
        reweightFile_ = nullptr;
    }
    if (parameterFile_ != nullptr)
    {
        std::fclose(parameterFile_);
        parameterFile_ = nullptr;
    }
}

FILE* GaMDRuntime::reweightFile(int rank)
{
    if (rank != 0)
    {
        return nullptr;
    }

    if (reweightFile_ != nullptr)
    {
        return reweightFile_;
    }

    const bool outputFileExists = fileExists("gamd-reweight.dat");

    reweightFile_ = std::fopen("gamd-reweight.dat", "a");
    if (reweightFile_ == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open gamd-reweight.dat for writing.");
    }

    if (!outputFileExists)
    {
        std::fprintf(reweightFile_, "# GaMD reweighting file\n");
        std::fprintf(reweightFile_, "# All energy terms stored in units of kcal/mol\n");
        std::fprintf(reweightFile_,
                     "# reweight_nst,total_nstep,Unboosted-Potential-Energy,"
                     "Unboosted-Dihedral-Energy,Total-Force-Weight,Dihedral-Force-Weight,"
                     "Effective-Dihedral-Force-Weight,Boost-Energy-Potential,"
                     "Boost-Energy-Dihedral\n");
        flushAndSyncFileOrFatal(reweightFile_, "gamd-reweight.dat");
    }

    return reweightFile_;
}

FILE* GaMDRuntime::parameterFile(int rank)
{
    if (rank != 0)
    {
        return nullptr;
    }

    if (parameterFile_ != nullptr)
    {
        return parameterFile_;
    }

    const bool outputFileExists = fileExists("gamd-para.dat");

    parameterFile_ = std::fopen("gamd-para.dat", "a");
    if (parameterFile_ == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open gamd-para.dat for writing.");
    }

    if (!outputFileExists)
    {
        std::fprintf(parameterFile_, "# GaMD parameter history\n");
        std::fprintf(parameterFile_,
                     "# Energies are in kJ/mol; k0 is dimensionless; k is in (kJ/mol)^-1\n");
        std::fprintf(parameterFile_,
                     "# para_nst,total_nstep,stage,"
                     "VmaxP,VminP,VavgP,SigmaVP,EP,k0P,kP,"
                     "VmaxD,VminD,VavgD,SigmaVD,ED,k0D,kD\n");
        flushAndSyncFileOrFatal(parameterFile_, "gamd-para.dat");
    }

    return parameterFile_;
}

bool GaMDRuntime::suppressTextOutputForStep(int64_t step) const
{
    return (suppressTextOutputStep_ == step);
}

static void writeGaMDEnergyStatisticsColumns(FILE* file, bool enabled, const GaMDEnergyStatistics& statistics)
{
    const char* columnFormat = " %18.9g %18.9g %18.9g %18.9g %18.9g %18.9g %18.9g";
    if (enabled)
    {
        std::fprintf(file,
                     columnFormat,
                     statistics.maximum,
                     statistics.minimum,
                     statistics.average,
                     statistics.deviation,
                     statistics.threshold,
                     statistics.k0,
                     statistics.forceConstant);
    }
    else
    {
        const double unavailable = std::numeric_limits<double>::quiet_NaN();
        std::fprintf(
                file, columnFormat, unavailable, unavailable, unavailable, unavailable, unavailable, unavailable, unavailable);
    }
}

void GaMDRuntime::writeParameterLine(int64_t step, int rank, bool flushOutput)
{
    if (rank != 0)
    {
        return;
    }

    if (parameters_.boostType == 0 || parameters_.parameterInterval <= 0)
    {
        return;
    }

    if (stepPlan_.stage < GaMDStage::ConventionalStatistics || step == 0)
    {
        return;
    }

    if (step % parameters_.parameterInterval != 0)
    {
        return;
    }

    if (suppressTextOutputForStep(step))
    {
        return;
    }

    FILE* fp = parameterFile(rank);
    if (fp == nullptr)
    {
        return;
    }

    const bool hasP = (parameters_.boostType == 1 || parameters_.boostType == 3);
    const bool hasD = (parameters_.boostType == 2 || parameters_.boostType == 3);

    if (std::fprintf(fp,
                     "%10" PRId64 " %12" PRId64 " %6d",
                     parameters_.parameterInterval,
                     step,
                     static_cast<int>(stepPlan_.stage))
        < 0)
    {
        gmx_fatal(FARGS, "Failed to write gamd-para.dat at step %" PRId64 ".", step);
    }
    writeGaMDEnergyStatisticsColumns(fp, hasP, potentialStatistics_);
    writeGaMDEnergyStatisticsColumns(fp, hasD, dihedralStatistics_);
    if (std::fputc('\n', fp) == EOF || (flushOutput && std::fflush(fp) != 0))
    {
        gmx_fatal(FARGS, "Failed to flush gamd-para.dat at step %" PRId64 ".", step);
    }
}

void GaMDRuntime::writeReweightRecord(int64_t step, int rank, const GaMDStepResult& record, bool flushOutput)
{
    if (rank != 0)
    {
        return;
    }

    if (parameters_.boostType == 0)
    {
        return;
    }

    if (stepPlan_.stage < GaMDStage::BoostPreparation)
    {
        return;
    }

    if (step == 0)
    {
        return;
    }

    if (parameters_.reweightInterval <= 0)
    {
        return;
    }

    if (step % parameters_.reweightInterval != 0)
    {
        return;
    }

    if (suppressTextOutputForStep(step))
    {
        return;
    }

    FILE* fp = reweightFile(rank);
    if (fp == nullptr)
    {
        return;
    }

    constexpr double kJ_to_kcal = 1.0 / 4.184;

    const double VP_kcal           = record.potentialEnergy * kJ_to_kcal;
    const double VD_kcal           = record.dihedralEnergy * kJ_to_kcal;
    const double boostP_kcal       = record.boostP * kJ_to_kcal;
    const double boostD_kcal       = record.boostD * kJ_to_kcal;
    const double effectiveDihScale = record.scaleP * record.scaleD;

    if (std::fprintf(fp,
                     "%10" PRId64 " %12" PRId64
                     " %20.9f %20.9f %14.9f %14.9f %14.9f %20.9f %20.9f\n",
                     parameters_.reweightInterval,
                     step,
                     VP_kcal,
                     VD_kcal,
                     record.scaleP,
                     record.scaleD,
                     effectiveDihScale,
                     boostP_kcal,
                     boostD_kcal)
                < 0
        || (flushOutput && std::fflush(fp) != 0))
    {
        gmx_fatal(FARGS, "Failed to write or flush gamd-reweight.dat at step %" PRId64 ".", step);
    }
}

void GaMDRuntime::writeReweightLine(int64_t step, int rank, bool flushOutput)
{
    writeReweightRecord(step, rank, stepResult_, flushOutput);
}

} // namespace gmx
