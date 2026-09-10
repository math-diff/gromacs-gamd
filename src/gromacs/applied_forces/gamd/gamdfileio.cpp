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

static bool readRestartSavedStep(const char* filename, int64_t* savedStep)
{
    std::ifstream input(filename);
    if (!input.is_open())
    {
        return false;
    }

    std::string line;
    std::string key;
    while (std::getline(input, line))
    {
        std::istringstream lineStream(line);
        if ((lineStream >> key) && key == "saved_step" && (lineStream >> *savedStep))
        {
            return true;
        }
    }
    return false;
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

    const int  currentStage          = determineStage(step);
    const bool saveAtCmdStatBoundary = (step == parameters_.conventionalEnd);
    const bool saveAtBoostStatBoundary = (step == parameters_.conventionalEnd + parameters_.boostLength);
    const bool inBoostStatPhase     = (currentStage == 4);
    const bool inProductionPhase    = (currentStage == 5);
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
    const char* restartFilename      = "gamd-restart.dat";
    int64_t     currentSavedStep     = -1;
    int64_t     previousSavedStep    = -1;
    const bool  haveCurrentSavedStep = readRestartSavedStep("gamd-restart.dat", &currentSavedStep);
    const bool havePreviousSavedStep = readRestartSavedStep("gamd-restart-prev.dat", &previousSavedStep);

    const bool currentMatches =
            haveCurrentSavedStep && (currentSavedStep == step || currentSavedStep == step - 1);
    const bool previousMatches =
            havePreviousSavedStep && (previousSavedStep == step || previousSavedStep == step - 1);
    const bool checkpointContinuation = (step > 0 && (currentMatches || previousMatches));

    if (explicitProductionRestart && !checkpointContinuation
        && (fileExists("gamd-reweight.dat") || fileExists("gamd-para.dat")))
    {
        gmx_fatal(FARGS,
                  "irest_gamd starts a new production simulation but existing gamd-reweight.dat "
                  "or gamd-para.dat was found and no GaMD restart matches continued step %" PRId64
                  ". "
                  "Use a clean output directory for a new production segment, or provide its "
                  "matching checkpoint for continuation.",
                  step);
    }

    if (!explicitProductionRestart || checkpointContinuation)
    {
        if (!currentMatches && previousMatches)
        {
            restartFilename = "gamd-restart-prev.dat";
            if (rank == 0)
            {
                std::fprintf(stderr,
                             "[GaMD WARNING] gamd-restart.dat saved_step=%" PRId64
                             " does not match "
                             "continued step=%" PRId64
                             "; using gamd-restart-prev.dat saved_step=%" PRId64
                             ". "
                             "This indicates interruption between GaMD and GROMACS checkpoint "
                             "commits.\n",
                             currentSavedStep,
                             step,
                             previousSavedStep);
            }
        }
        else if (!currentMatches)
        {
            // A unit or deliberately offset fresh run can begin above step 0,
            // but once any GaMD output exists, silently resetting statistics is
            // unsafe and must be rejected.
            const bool haveGaMDArtifacts =
                    fileExists("gamd-reweight.dat") || fileExists("gamd-para.dat")
                    || fileExists("gamd-restart.dat") || fileExists("gamd-restart-prev.dat");
            if (haveGaMDArtifacts)
            {
                gmx_fatal(FARGS,
                          "No GaMD restart matches first continued step %" PRId64
                          " "
                          "(gamd-restart.dat saved_step=%" PRId64 ", previous saved_step=%" PRId64
                          "). "
                          "Refusing to reset GaMD statistics or append inconsistent output.",
                          step,
                          currentSavedStep,
                          previousSavedStep);
            }
            return;
        }
    }

    // 注意：所有 rank 都读同一个文件，这样每个进程都有相同 GaMD 状态
    std::ifstream infile(restartFilename);
    if (!infile.is_open())
    {
        gmx_fatal(FARGS, "GaMD restart mode requires '%s', but the file was not found.", restartFilename);
    }

    int     version   = 0;
    int64_t savedStep = -1;

    int     file_igamd        = 0;
    int     file_iE           = 1;
    int     file_iEP          = 1;
    int     file_iED          = 1;
    int64_t file_ntcmdprep    = 0;
    int64_t file_ntcmd        = 0;
    int64_t file_ntebprep     = 0;
    int64_t file_nteb         = 0;
    int64_t file_ntave        = 1;
    int64_t file_reweight_nst = 1;
    double  file_sigma0P      = 0.0;
    double  file_sigma0D      = 0.0;

    GaMDEnergyStatistics statP;
    GaMDEnergyStatistics statD;
    GaMDWindowStatistics windowStatP;
    GaMDWindowStatistics windowStatD;

    bool foundVersion           = false;
    bool foundSavedStep         = false;
    bool foundParams            = false;
    bool foundProductionRestart = false;
    bool foundStatP             = false;
    bool foundStatD             = false;
    bool foundWindowP           = false;
    bool foundWindowD           = false;
    bool fileProductionRestart  = false;

    std::string line;
    std::string key;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        if (!(iss >> key))
        {
            continue;
        }
        if (key[0] == '#' || key[0] == ';')
        {
            continue;
        }

        if (key == "version")
        {
            if (iss >> version)
            {
                foundVersion = true;
            }
        }
        else if (key == "saved_step")
        {
            if (iss >> savedStep)
            {
                foundSavedStep = true;
            }
        }
        else if (key == "production_restart")
        {
            int productionRestart = 0;
            if (iss >> productionRestart)
            {
                fileProductionRestart  = (productionRestart != 0);
                foundProductionRestart = true;
            }
        }
        else if (key == "params")
        {
            if (iss >> file_igamd >> file_iE >> file_iEP >> file_iED >> file_ntcmdprep >> file_ntcmd >> file_ntebprep
                >> file_nteb >> file_ntave >> file_reweight_nst >> file_sigma0P >> file_sigma0D)
            {
                foundParams = true;
            }
        }
        else if (key == "statP")
        {
            foundStatP = readStatsLine(iss, &statP);
        }
        else if (key == "statD")
        {
            foundStatD = readStatsLine(iss, &statD);
        }
        else if (key == "windowP")
        {
            foundWindowP = readWindowStatsLine(iss, &windowStatP);
        }
        else if (key == "windowD")
        {
            foundWindowD = readWindowStatsLine(iss, &windowStatD);
        }
    }

    if (!foundVersion || !foundSavedStep || !foundParams || !foundStatP || !foundStatD)
    {
        gmx_fatal(FARGS, "GaMD restart '%s' is incomplete or corrupted.", restartFilename);
    }

    if (version != 1 && version != 2 && version != 3)
    {
        gmx_fatal(FARGS, "Unsupported GaMD restart version %d in '%s'.", version, restartFilename);
    }

    if (version >= 2 && (!foundWindowP || !foundWindowD))
    {
        gmx_fatal(FARGS, "GaMD restart '%s' is missing window statistics.", restartFilename);
    }

    if (savedStep < 0 || !validStats(statP) || !validStats(statD)
        || (version >= 2 && (!validWindowStats(windowStatP) || !validWindowStats(windowStatD))))
    {
        gmx_fatal(FARGS, "GaMD restart '%s' contains invalid or non-finite saved state.", restartFilename);
    }

    // 最关键保护：restart 文件必须对应上一完成步
    // 接受两种 saved_step 语义：
    // legacy: saved_step = step - 1
    // current: saved_step = step
    if ((!explicitProductionRestart || checkpointContinuation) && savedStep != step && savedStep != step - 1)
    {
        gmx_fatal(FARGS,
                  "GaMD restart '%s' saved_step=%" PRId64
                  " does not match first continued step=%" PRId64 ".",
                  restartFilename,
                  savedStep,
                  step);
    }

    const bool physicsParamsMismatch =
            (file_igamd != parameters_.boostType || file_iE != parameters_.thresholdMode
             || file_iEP != parameters_.potentialMode || file_iED != parameters_.dihedralMode
             || std::abs(file_sigma0P - parameters_.potentialSigmaLimit) > 1e-9
             || std::abs(file_sigma0D - parameters_.dihedralSigmaLimit) > 1e-9);
    const bool stageParamsMismatch =
            (file_ntcmdprep != parameters_.conventionalPrepEnd
             || file_ntcmd != parameters_.conventionalEnd || file_ntebprep != parameters_.boostPrepLength
             || file_nteb != parameters_.boostLength || file_ntave != parameters_.averagingWindow);

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

    potentialStatistics_ = statP;
    dihedralStatistics_  = statD;
    if (version >= 2)
    {
        potentialWindowStatistics_ = windowStatP;
        dihedralWindowStatistics_  = windowStatD;
    }
    else
    {
        potentialWindowStatistics_ = {};
        dihedralWindowStatistics_  = {};
    }

    if (explicitProductionRestart)
    {
        const int64_t restartNtcmd = file_ntcmd;
        const int64_t restartNteb  = file_nteb;
        const bool inferredProductionRestart = (!foundProductionRestart && savedStep < restartNtcmd);
        const bool restartFileIsProductionRestart = (fileProductionRestart || inferredProductionRestart);

        if (savedStep < restartNtcmd && !restartFileIsProductionRestart)
        {
            gmx_fatal(FARGS,
                      "irest_gamd = %d requires GaMD parameters that were already initialized, "
                      "but gamd-restart.dat was saved at step %" PRId64 " before ntcmd = %" PRId64
                      ".",
                      parameters_.restartMode,
                      savedStep,
                      restartNtcmd);
        }

        // In irest_gamd mode the stage-control counters can be omitted from
        // gamd.in. Keep the values from the restart file for provenance/output.
        parameters_.conventionalPrepEnd = file_ntcmdprep;
        parameters_.conventionalEnd     = file_ntcmd;
        parameters_.boostPrepLength     = file_ntebprep;
        parameters_.boostLength         = file_nteb;
        parameters_.averagingWindow     = std::max<int64_t>(1, file_ntave);

        forceProductionRestart_ = true;

        if (checkpointContinuation)
        {
            reconcileGaMDTextFile("gamd-reweight.dat", 9, savedStep, rank);
            reconcileGaMDTextFile("gamd-para.dat", 17, savedStep, rank);
            if (savedStep == step)
            {
                suppressTextOutputStep_ = savedStep;
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
                             savedStep,
                             restartNtcmd);
            }

            std::fprintf(stderr,
                         "[GaMD INFO] irest_gamd=%d ignores ntcmdprep/ntcmd/ntebprep/nteb "
                         "from gamd.in and uses the values stored in gamd-restart.dat.\n",
                         parameters_.restartMode);

            if (!checkpointContinuation && savedStep < restartNtcmd + restartNteb)
            {
                std::fprintf(stderr,
                             "[GaMD WARNING] irest_gamd=%d loaded saved_step=%" PRId64
                             " before the "
                             "adaptive boost-stat stage ended at step=%" PRId64
                             ". "
                             "The new simulation will enter production immediately using the "
                             "latest saved GaMD parameters.\n",
                             parameters_.restartMode,
                             savedStep,
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
                             savedStep,
                             step);
            }
            else
            {
                std::fprintf(stderr,
                             "[GaMD INFO] Loaded GaMD restart state from saved_step=%" PRId64
                             " for a "
                             "new simulation with irest_gamd=%d; forcing GaMD production stage "
                             "from step=%" PRId64 ".\n",
                             savedStep,
                             parameters_.restartMode,
                             step);
            }
        }
    }
    else
    {
        reconcileGaMDTextFile("gamd-reweight.dat", 9, savedStep, rank);
        reconcileGaMDTextFile("gamd-para.dat", 17, savedStep, rank);

        if (savedStep == step)
        {
            suppressTextOutputStep_ = savedStep;
        }

        if (rank == 0)
        {
            std::fprintf(stderr,
                         "[GaMD INFO] Loaded GaMD restart state from %s saved_step=%" PRId64
                         " for "
                         "continued step=%" PRId64
                         "; text outputs are reconciled to the checkpoint.\n",
                         restartFilename,
                         savedStep,
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

    bool fileExists = false;
    {
        std::ifstream test("gamd-reweight.dat");
        fileExists = test.good();
    }

    reweightFile_ = std::fopen("gamd-reweight.dat", "a");
    if (reweightFile_ == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open gamd-reweight.dat for writing.");
    }

    if (!fileExists)
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

    bool fileExists = false;
    {
        std::ifstream test("gamd-para.dat");
        fileExists = test.good();
    }

    parameterFile_ = std::fopen("gamd-para.dat", "a");
    if (parameterFile_ == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open gamd-para.dat for writing.");
    }

    if (!fileExists)
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

    if (stepPlan_.stage < 2 || step == 0)
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
                     stepPlan_.stage)
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

    if (!stepReadyForOutput_ || stepPlan_.stage < 3)
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
