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

#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"

#include "gamdruntime.h"

namespace gmx
{

// ============================================================================
// Restart helpers
// ============================================================================

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

static bool readRestartSavedStep(const char* filename, long* savedStep)
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

static bool parseGaMDTextRow(const std::string& line, int expectedColumns, long* step)
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

    char* stepEnd         = nullptr;
    errno                 = 0;
    const long parsedStep = std::strtol(tokens[1].c_str(), &stepEnd, 10);
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
static void reconcileGaMDTextFile(const char* filename, int expectedColumns, long savedStep, int nodeid)
{
    if (nodeid != 0 || !fileExists(filename))
    {
        return;
    }

    std::ifstream input(filename);
    if (!input.is_open())
    {
        gmx_fatal(FARGS, "Failed to open GaMD text file '%s' for continuation repair.", filename);
    }

    std::vector<std::string>    headers;
    std::set<std::string>       seenHeaders;
    std::map<long, std::string> rowsByStep;
    long                        malformedRows = 0;
    long                        futureRows    = 0;
    long                        duplicateRows = 0;
    std::string                 line;
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

        long rowStep = -1;
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
                     "[GaMD WARNING] Repaired %s at checkpoint step %ld: removed %ld "
                     "malformed/partial, %ld post-checkpoint, and %ld duplicate rows.\n",
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
                 "%s %.17g %.17g %.17g %.17g %.17g %ld %.17g %.17g %.17g\n",
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
    std::fprintf(fp, "%s %ld %.17g %.17g\n", name, s.count, s.mean, s.m2);
}

static bool readWindowStatsLine(std::istringstream& iss, GaMDWindowStatistics* s)
{
    return static_cast<bool>(iss >> s->count >> s->mean >> s->m2);
}

void GaMDRuntime::saveRestartState(int64_t step, int nodeid)
{
    if (nodeid != 0)
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
    std::fprintf(fp, "saved_step %ld\n", step);
    std::fprintf(fp, "production_restart %d\n", forceProductionRestart_ ? 1 : 0);

    // 参数也写出，续跑时做一致性检查
    std::fprintf(fp,
                 "params %d %d %d %d %ld %ld %ld %ld %ld %ld %.17g %.17g\n",
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

void GaMDRuntime::loadRestartStateIfNeeded(int64_t step, int nodeid)
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
    long        currentSavedStep     = -1;
    long        previousSavedStep    = -1;
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
                  "or gamd-para.dat was found and no GaMD restart matches continued step %ld. "
                  "Use a clean output directory for a new production segment, or provide its "
                  "matching checkpoint for continuation.",
                  step);
    }

    if (!explicitProductionRestart || checkpointContinuation)
    {
        if (!currentMatches && previousMatches)
        {
            restartFilename = "gamd-restart-prev.dat";
            if (nodeid == 0)
            {
                std::fprintf(stderr,
                             "[GaMD WARNING] gamd-restart.dat saved_step=%ld does not match "
                             "continued step=%ld; using gamd-restart-prev.dat saved_step=%ld. "
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
                          "No GaMD restart matches first continued step %ld "
                          "(gamd-restart.dat saved_step=%ld, previous saved_step=%ld). "
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

    int  version   = 0;
    long savedStep = -1;

    int    file_igamd        = 0;
    int    file_iE           = 1;
    int    file_iEP          = 1;
    int    file_iED          = 1;
    long   file_ntcmdprep    = 0;
    long   file_ntcmd        = 0;
    long   file_ntebprep     = 0;
    long   file_nteb         = 0;
    long   file_ntave        = 1;
    long   file_reweight_nst = 1;
    double file_sigma0P      = 0.0;
    double file_sigma0D      = 0.0;

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

    auto validStats = [](const GaMDEnergyStatistics& stat)
    {
        return std::isfinite(stat.maximum) && std::isfinite(stat.minimum)
               && std::isfinite(stat.average) && std::isfinite(stat.deviation)
               && std::isfinite(stat.m2) && std::isfinite(stat.threshold) && std::isfinite(stat.k0)
               && std::isfinite(stat.forceConstant) && stat.count >= 0 && stat.deviation >= 0.0
               && stat.m2 >= 0.0 && stat.k0 >= 0.0 && stat.k0 <= 1.0 && stat.forceConstant >= 0.0
               && (stat.count == 0 || stat.maximum >= stat.minimum);
    };
    auto validWindowStats = [](const GaMDWindowStatistics& stat) {
        return stat.count >= 0 && std::isfinite(stat.mean) && std::isfinite(stat.m2) && stat.m2 >= 0.0;
    };
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
                  "GaMD restart '%s' saved_step=%ld does not match first continued step=%ld.",
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
        const long restartNtcmd = file_ntcmd;
        const long restartNteb  = file_nteb;
        const bool inferredProductionRestart = (!foundProductionRestart && savedStep < restartNtcmd);
        const bool restartFileIsProductionRestart = (fileProductionRestart || inferredProductionRestart);

        if (savedStep < restartNtcmd && !restartFileIsProductionRestart)
        {
            gmx_fatal(FARGS,
                      "irest_gamd = %d requires GaMD parameters that were already initialized, "
                      "but gamd-restart.dat was saved at step %ld before ntcmd = %ld.",
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
        parameters_.averagingWindow     = std::max<long>(1, file_ntave);

        forceProductionRestart_ = true;

        if (checkpointContinuation)
        {
            reconcileGaMDTextFile("gamd-reweight.dat", 9, savedStep, nodeid);
            reconcileGaMDTextFile("gamd-para.dat", 17, savedStep, nodeid);
            if (savedStep == step)
            {
                suppressTextOutputStep_ = savedStep;
            }
        }

        if (nodeid == 0)
        {
            if (inferredProductionRestart)
            {
                std::fprintf(stderr,
                             "[GaMD INFO] Inferred production_restart=1 from gamd-restart.dat "
                             "because saved_step=%ld is smaller than ntcmd=%ld. "
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
                             "[GaMD WARNING] irest_gamd=%d loaded saved_step=%ld before the "
                             "adaptive boost-stat stage ended at step=%ld. "
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
                             "saved_step=%ld at checkpoint step=%ld; text outputs are "
                             "reconciled to the checkpoint.\n",
                             restartFilename,
                             savedStep,
                             step);
            }
            else
            {
                std::fprintf(stderr,
                             "[GaMD INFO] Loaded GaMD restart state from saved_step=%ld for a "
                             "new simulation with irest_gamd=%d; forcing GaMD production stage "
                             "from step=%ld.\n",
                             savedStep,
                             parameters_.restartMode,
                             step);
            }
        }
    }
    else
    {
        reconcileGaMDTextFile("gamd-reweight.dat", 9, savedStep, nodeid);
        reconcileGaMDTextFile("gamd-para.dat", 17, savedStep, nodeid);

        if (savedStep == step)
        {
            suppressTextOutputStep_ = savedStep;
        }

        if (nodeid == 0)
        {
            std::fprintf(stderr,
                         "[GaMD INFO] Loaded GaMD restart state from %s saved_step=%ld for "
                         "continued step=%ld; text outputs are reconciled to the checkpoint.\n",
                         restartFilename,
                         savedStep,
                         step);
        }
    }
}

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

// ============================================================================
// Parameter loading
// ============================================================================

void GaMDRuntime::loadParameters(int nodeid)
{
    if (parametersLoaded_)
    {
        return;
    }

    std::ifstream infile("gamd.in");
    if (!infile.is_open())
    {
        gmx_fatal(FARGS, "GaMD is enabled but required parameter file 'gamd.in' was not found.");
    }

    std::set<std::string> seenKeys;
    std::string           line;
    std::string           key;
    long                  lineNumber = 0;

    while (std::getline(infile, line))
    {
        lineNumber++;
        std::istringstream iss(line);
        if (!(iss >> key) || key[0] == '#' || key[0] == ';')
        {
            continue;
        }

        auto markUnique = [&]()
        {
            if (!seenKeys.insert(key).second)
            {
                gmx_fatal(FARGS, "Duplicate GaMD parameter '%s' at gamd.in line %ld.", key.c_str(), lineNumber);
            }
        };
        auto readValue = [&](auto* value)
        {
            markUnique();
            if (!(iss >> *value))
            {
                gmx_fatal(FARGS,
                          "Missing or invalid value for GaMD parameter '%s' at gamd.in line %ld.",
                          key.c_str(),
                          lineNumber);
            }
            std::string trailing;
            if ((iss >> trailing) && trailing[0] != '#' && trailing[0] != ';')
            {
                gmx_fatal(FARGS,
                          "Unexpected trailing token '%s' for GaMD parameter '%s' at "
                          "gamd.in line %ld.",
                          trailing.c_str(),
                          key.c_str(),
                          lineNumber);
            }
        };

        if (key == "igamd")
        {
            readValue(&parameters_.boostType);
        }
        else if (key == "irest_gamd")
        {
            readValue(&parameters_.restartMode);
        }
        else if (key == "iE")
        {
            readValue(&parameters_.thresholdMode);
        }
        else if (key == "iEP")
        {
            readValue(&parameters_.potentialMode);
        }
        else if (key == "iED")
        {
            readValue(&parameters_.dihedralMode);
        }
        else if (key == "ntcmdprep")
        {
            readValue(&parameters_.conventionalPrepEnd);
        }
        else if (key == "ntcmd")
        {
            readValue(&parameters_.conventionalEnd);
        }
        else if (key == "ntebprep")
        {
            readValue(&parameters_.boostPrepLength);
        }
        else if (key == "nteb")
        {
            readValue(&parameters_.boostLength);
        }
        else if (key == "ntave")
        {
            readValue(&parameters_.averagingWindow);
        }
        else if (key == "reweight_nst")
        {
            readValue(&parameters_.reweightInterval);
        }
        else if (key == "para_nst")
        {
            readValue(&parameters_.parameterInterval);
        }
        else if (key == "sigma0P")
        {
            double sigmaKcal = 0.0;
            readValue(&sigmaKcal);
            parameters_.potentialSigmaLimit = sigmaKcal * 4.184;
        }
        else if (key == "sigma0D")
        {
            double sigmaKcal = 0.0;
            readValue(&sigmaKcal);
            parameters_.dihedralSigmaLimit = sigmaKcal * 4.184;
        }
        else if (key == "debug")
        {
            readValue(&debug_);
        }
        else
        {
            gmx_fatal(FARGS, "Unknown GaMD parameter '%s' at gamd.in line %ld.", key.c_str(), lineNumber);
        }
    }
    if (infile.bad())
    {
        gmx_fatal(FARGS, "Failed while reading required parameter file 'gamd.in'.");
    }

    auto requireKey = [&](const char* requiredKey)
    {
        if (seenKeys.count(requiredKey) == 0)
        {
            gmx_fatal(FARGS, "Required GaMD parameter '%s' is missing from gamd.in.", requiredKey);
        }
    };

    requireKey("igamd");
    requireKey("iE");
    requireKey("reweight_nst");

    if (parameters_.boostType < 1 || parameters_.boostType > 3)
    {
        gmx_fatal(FARGS, "GaMD parameter igamd must be 1, 2, or 3; got %d.", parameters_.boostType);
    }
    if (parameters_.restartMode != 0 && parameters_.restartMode != 1)
    {
        gmx_fatal(FARGS, "GaMD parameter irest_gamd must be 0 or 1; got %d.", parameters_.restartMode);
    }
    if (parameters_.thresholdMode < 1 || parameters_.thresholdMode > 2 || parameters_.potentialMode < 1
        || parameters_.potentialMode > 2 || parameters_.dihedralMode < 1 || parameters_.dihedralMode > 2)
    {
        gmx_fatal(FARGS, "GaMD parameters iE, iEP, and iED must each be 1 or 2.");
    }
    if (parameters_.reweightInterval <= 0)
    {
        gmx_fatal(FARGS, "GaMD parameter reweight_nst must be positive; got %ld.", parameters_.reweightInterval);
    }
    if (parameters_.parameterInterval < 0)
    {
        gmx_fatal(FARGS, "GaMD parameter para_nst must be non-negative; got %ld.", parameters_.parameterInterval);
    }
    if (debug_ != 0 && debug_ != 1)
    {
        gmx_fatal(FARGS, "GaMD parameter debug must be 0 or 1; got %d.", debug_);
    }

    const bool usesPotentialBoost = (parameters_.boostType == 1 || parameters_.boostType == 3);
    const bool usesDihedralBoost  = (parameters_.boostType == 2 || parameters_.boostType == 3);
    if (usesPotentialBoost)
    {
        requireKey("iEP");
        requireKey("sigma0P");
        if (!std::isfinite(parameters_.potentialSigmaLimit) || parameters_.potentialSigmaLimit <= 0.0)
        {
            gmx_fatal(FARGS, "GaMD parameter sigma0P must be finite and positive.");
        }
    }
    if (usesDihedralBoost)
    {
        requireKey("iED");
        requireKey("sigma0D");
        if (!std::isfinite(parameters_.dihedralSigmaLimit) || parameters_.dihedralSigmaLimit <= 0.0)
        {
            gmx_fatal(FARGS, "GaMD parameter sigma0D must be finite and positive.");
        }
    }

    if (parameters_.restartMode == 0)
    {
        requireKey("ntcmdprep");
        requireKey("ntcmd");
        requireKey("ntebprep");
        requireKey("nteb");
        requireKey("ntave");

        if (parameters_.averagingWindow <= 0 || parameters_.conventionalPrepEnd < 0
            || parameters_.conventionalEnd <= parameters_.conventionalPrepEnd
            || parameters_.boostPrepLength < 0 || parameters_.boostLength <= parameters_.boostPrepLength)
        {
            gmx_fatal(FARGS,
                      "Invalid GaMD stage parameters: require ntave>0, 0<=ntcmdprep<ntcmd, "
                      "and 0<=ntebprep<nteb; got %ld, %ld, %ld, %ld, %ld.",
                      parameters_.averagingWindow,
                      parameters_.conventionalPrepEnd,
                      parameters_.conventionalEnd,
                      parameters_.boostPrepLength,
                      parameters_.boostLength);
        }

        if (parameters_.conventionalPrepEnd % parameters_.averagingWindow != 0
            || parameters_.conventionalEnd % parameters_.averagingWindow != 0
            || parameters_.boostPrepLength % parameters_.averagingWindow != 0
            || parameters_.boostLength % parameters_.averagingWindow != 0)
        {
            gmx_fatal(FARGS,
                      "GaMD requires ntcmdprep (%ld), ntcmd (%ld), ntebprep (%ld), and nteb "
                      "(%ld) to be multiples of ntave (%ld) for aligned statistics windows.",
                      parameters_.conventionalPrepEnd,
                      parameters_.conventionalEnd,
                      parameters_.boostPrepLength,
                      parameters_.boostLength,
                      parameters_.averagingWindow);
        }
    }

    parametersLoaded_ = true;

    if (nodeid == 0)
    {
        std::fprintf(stderr, "\n=== GaMD Parameters Loaded from gamd.in ===\n");
        std::fprintf(stderr, "igamd        = %d\n", parameters_.boostType);
        std::fprintf(stderr, "irest_gamd   = %d\n", parameters_.restartMode);
        std::fprintf(stderr, "iE           = %d\n", parameters_.thresholdMode);
        std::fprintf(stderr, "iEP          = %d\n", parameters_.potentialMode);
        std::fprintf(stderr, "iED          = %d\n", parameters_.dihedralMode);
        std::fprintf(stderr, "ntcmdprep    = %ld\n", parameters_.conventionalPrepEnd);
        std::fprintf(stderr, "ntcmd        = %ld\n", parameters_.conventionalEnd);
        std::fprintf(stderr, "ntebprep     = %ld\n", parameters_.boostPrepLength);
        std::fprintf(stderr, "nteb         = %ld\n", parameters_.boostLength);
        std::fprintf(stderr, "ntave        = %ld\n", parameters_.averagingWindow);
        std::fprintf(stderr, "reweight_nst = %ld\n", parameters_.reweightInterval);
        std::fprintf(stderr, "para_nst     = %ld\n", parameters_.parameterInterval);
        std::fprintf(stderr, "sigma0P      = %.4f kJ/mol\n", parameters_.potentialSigmaLimit);
        std::fprintf(stderr, "sigma0D      = %.4f kJ/mol\n", parameters_.dihedralSigmaLimit);
        std::fprintf(stderr, "DEBUG MODE   = %d\n", debug_);
        std::fprintf(stderr, "===========================================\n\n");
    }
}

// ============================================================================
// Reweight file helpers
// ============================================================================

FILE* GaMDRuntime::reweightFile(int nodeid)
{
    if (nodeid != 0)
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

FILE* GaMDRuntime::parameterFile(int nodeid)
{
    if (nodeid != 0)
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

static void writeGaMDEnergyStatisticsColumns(FILE* fp, bool enabled, const GaMDEnergyStatistics& stat)
{
    const double nanValue = std::numeric_limits<double>::quiet_NaN();
    const double Vmax     = enabled ? stat.maximum : nanValue;
    const double Vmin     = enabled ? stat.minimum : nanValue;
    const double Vavg     = enabled ? stat.average : nanValue;
    const double sigmaV   = enabled ? stat.deviation : nanValue;
    const double E        = enabled ? stat.threshold : nanValue;
    const double k0       = enabled ? stat.k0 : nanValue;
    const double k        = enabled ? stat.forceConstant : nanValue;

    std::fprintf(fp, " %18.9g %18.9g %18.9g %18.9g %18.9g %18.9g %18.9g", Vmax, Vmin, Vavg, sigmaV, E, k0, k);
}

void GaMDRuntime::writeParameterLine(int64_t step, int nodeid, bool flushOutput)
{
    if (nodeid != 0)
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

    FILE* fp = parameterFile(nodeid);
    if (fp == nullptr)
    {
        return;
    }

    const bool hasP = (parameters_.boostType == 1 || parameters_.boostType == 3);
    const bool hasD = (parameters_.boostType == 2 || parameters_.boostType == 3);

    if (std::fprintf(fp, "%10ld %12ld %6d", parameters_.parameterInterval, step, stepPlan_.stage) < 0)
    {
        gmx_fatal(FARGS, "Failed to write gamd-para.dat at step %ld.", step);
    }
    writeGaMDEnergyStatisticsColumns(fp, hasP, potentialStatistics_);
    writeGaMDEnergyStatisticsColumns(fp, hasD, dihedralStatistics_);
    if (std::fputc('\n', fp) == EOF || (flushOutput && std::fflush(fp) != 0))
    {
        gmx_fatal(FARGS, "Failed to flush gamd-para.dat at step %ld.", step);
    }
}

void GaMDRuntime::writeReweightRecord(int64_t step, int nodeid, const GaMDProductionOutputRecord& record, bool flushOutput)
{
    if (nodeid != 0)
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

    FILE* fp = reweightFile(nodeid);
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
                     "%10ld %12ld %20.9f %20.9f %14.9f %14.9f %14.9f %20.9f %20.9f\n",
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
        gmx_fatal(FARGS, "Failed to write or flush gamd-reweight.dat at step %ld.", step);
    }
}

void GaMDRuntime::writeReweightLine(int64_t step, int nodeid, bool flushOutput)
{
    writeReweightRecord(step,
                        nodeid,
                        { stepResult_.potentialEnergy,
                          stepResult_.dihedralEnergy,
                          stepResult_.scaleP,
                          stepResult_.scaleD,
                          stepResult_.boostP,
                          stepResult_.boostD },
                        flushOutput);
}

// ============================================================================
// Core math helpers
// ============================================================================

static void calculateThresholdAndForceConstant(int iE, double sigma0, GaMDEnergyStatistics& stat)
{
    if (stat.maximum <= stat.minimum || stat.deviation < 1e-6 || stat.maximum == stat.average)
    {
        return;
    }

    if (iE == 1)
    {
        stat.threshold        = stat.maximum;
        const double k0_prime = (sigma0 / stat.deviation) * (stat.maximum - stat.minimum)
                                / (stat.maximum - stat.average);
        stat.k0 = std::min(1.0, k0_prime);
    }
    else if (iE == 2)
    {
        const double k0_dprime = (1.0 - sigma0 / stat.deviation) * (stat.maximum - stat.minimum)
                                 / std::max(1e-6, (stat.average - stat.minimum));

        if (k0_dprime > 0.0 && k0_dprime <= 1.0)
        {
            stat.k0        = k0_dprime;
            stat.threshold = stat.minimum + (stat.maximum - stat.minimum) / stat.k0;
        }
        else
        {
            stat.threshold        = stat.maximum;
            const double k0_prime = (sigma0 / stat.deviation) * (stat.maximum - stat.minimum)
                                    / (stat.maximum - stat.average);
            stat.k0 = std::min(1.0, k0_prime);
        }
    }

    stat.forceConstant = stat.k0 / std::max(1e-6, (stat.maximum - stat.minimum));
}

static void updateStatisticsMinMax(double energy, GaMDEnergyStatistics& statistics)
{
    if (energy > statistics.maximum)
    {
        statistics.maximum = energy;
    }
    if (energy < statistics.minimum)
    {
        statistics.minimum = energy;
    }

    statistics.count++;
}

static void updateWindowStatistics(double energy, GaMDWindowStatistics& statistics)
{
    statistics.count++;
    const double delta = energy - statistics.mean;
    statistics.mean += delta / statistics.count;
    const double delta2 = energy - statistics.mean;
    statistics.m2 += delta * delta2;
}

static void finalizeWindowStatistics(GaMDWindowStatistics& windowStatistics, GaMDEnergyStatistics& statistics)
{
    if (windowStatistics.count <= 0)
    {
        return;
    }

    statistics.average = windowStatistics.mean;
    if (windowStatistics.count > 1)
    {
        statistics.deviation = std::sqrt(windowStatistics.m2 / windowStatistics.count);
    }
    else
    {
        statistics.deviation = 0.0;
    }

    windowStatistics = {};
}

int GaMDRuntime::determineStage(int64_t step) const
{
    if (forceProductionRestart_)
    {
        return 5;
    }

    if (step <= parameters_.conventionalPrepEnd)
    {
        return 1;
    }
    if (step <= parameters_.conventionalEnd)
    {
        return 2;
    }
    if (step <= parameters_.conventionalEnd + parameters_.boostPrepLength)
    {
        return 3;
    }
    if (step <= parameters_.conventionalEnd + parameters_.boostLength)
    {
        return 4;
    }

    return 5;
}

static bool stageUsesBoost(const int stage)
{
    return (stage >= 3);
}

static bool stageUpdatesStats(const int stage)
{
    return (stage == 2 || stage == 4);
}

const GaMDStepPlan& GaMDRuntime::prepareStep(int64_t step, int nodeid, bool pressureCouplingThisStep)
{
    if (lastPreparedStep_ == step)
    {
        return stepPlan_;
    }
    lastPreparedStep_ = step;

    currentRank_ = nodeid;

    loadParameters(nodeid);
    loadRestartStateIfNeeded(step, nodeid);

    // Default to a neutral state. Current-step GaMD applies the final force
    // correction later in do_force(), after raw energies and raw listed forces
    // for the current coordinates are available.
    stepResult_         = {};
    stepReadyForOutput_ = false;
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
    stepReadyForOutput_ = (stepPlan_.stage >= 3);

    const bool forcedScale            = std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP") != nullptr;
    const bool writesUnbufferedOutput = (stepPlan_.writesParameters || stepPlan_.writesReweight)
                                        && !bufferedProductionOutputEnabled_;
    stepPlan_.requiresHostEnergy = stepPlan_.stage != 5 || checkpointingThisStep_
                                   || pressureCouplingThisStep || forcedScale || writesUnbufferedOutput;
    stepPlan_.gpuGraphEligible =
            stepPlan_.stage == 5 && !checkpointingThisStep_ && !pressureCouplingThisStep;
    return stepPlan_;
}

GaMDGpuProductionParameters GaMDRuntime::gpuProductionParameters() const
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
    return bufferedProductionOutputEnabled_;
}

void GaMDRuntime::setBufferedProductionOutputEnabled(bool enabled)
{
    bufferedProductionOutputEnabled_ = enabled;
}

void GaMDRuntime::warnIfRunTooShort(int64_t initStep, int64_t nsteps, int nodeid)
{
    if (warnedAboutRunLength_)
    {
        return;
    }
    warnedAboutRunLength_ = true;

    if (!parametersLoaded_ || parameters_.boostType == 0 || nodeid != 0 || nsteps < 0
        || parameters_.restartMode != 0 || forceProductionRestart_)
    {
        return;
    }

    const int64_t adaptivePhaseEnd = static_cast<int64_t>(parameters_.conventionalEnd)
                                     + static_cast<int64_t>(parameters_.boostLength);
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
                                          int                         nodeid,
                                          bool*                       warnedAtFloor)
{
    if (!std::isfinite(rawScale) || !std::isfinite(energy) || !std::isfinite(stat.threshold)
        || !std::isfinite(stat.forceConstant))
    {
        gmx_fatal(FARGS,
                  "Non-finite GaMD %s boost quantity at step %ld: raw_scale=%g energy=%g "
                  "threshold_E=%g k=%g.",
                  label,
                  step,
                  rawScale,
                  energy,
                  stat.threshold,
                  stat.forceConstant);
    }

    constexpr double c_scaleFloor = 0.01;
    if (rawScale <= c_scaleFloor && !*warnedAtFloor)
    {
        if (nodeid == 0)
        {
            std::fprintf(stderr,
                         "\n[GaMD WARNING] %s raw force weight reached the 0.01 safety floor "
                         "at step %ld: raw_weight=%.17g, energy=%.17g kJ/mol, E=%.17g "
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
    return std::max(c_scaleFloor, rawScale);
}

void GaMDRuntime::finalizeCurrentStep(int64_t        step,
                                      int            nodeid,
                                      double         totalPotentialEnergy,
                                      double         dihedralEnergy,
                                      gmx_wallcycle* wcycle)
{
    if (parameters_.boostType == 0 || lastAccountedStep_ == step)
    {
        return;
    }
    wallcycle_start(wcycle, WallCycleCounter::GaMDFinalize);
    lastAccountedStep_ = step;

    currentRank_ = nodeid;
    GMX_RELEASE_ASSERT(stepPlan_.step == step, "GaMD finalization must use the prepared step");
    stepReadyForOutput_         = (stepPlan_.stage >= 3);
    stepResult_.potentialEnergy = totalPotentialEnergy;
    stepResult_.dihedralEnergy  = dihedralEnergy;

    if (const char* sp = std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP"))
    {
        const double scaleP = std::atof(sp);

        double scaleD = scaleP;
        if (const char* sd = std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALED"))
        {
            scaleD = std::atof(sd);
        }

        double boostP = 0.0;
        double boostD = 0.0;
        if (const char* bp = std::getenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTP"))
        {
            boostP = std::atof(bp);
        }
        if (const char* bd = std::getenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTD"))
        {
            boostD = std::atof(bd);
        }

        const GaMDEnergyStatistics overrideStats{};
        stepResult_.scaleP = validateAndClampScale(
                "total-potential", scaleP, totalPotentialEnergy, overrideStats, step, nodeid, &warnedPotentialScaleFloor_);
        stepResult_.scaleD = validateAndClampScale(
                "dihedral", scaleD, dihedralEnergy, overrideStats, step, nodeid, &warnedDihedralScaleFloor_);
        if (!std::isfinite(boostP) || !std::isfinite(boostD))
        {
            gmx_fatal(FARGS, "Non-finite forced GaMD boost at step %ld: boostP=%g boostD=%g.", step, boostP, boostD);
        }
        stepResult_.boostP = boostP;
        stepResult_.boostD = boostD;
        wallcycle_stop(wcycle, WallCycleCounter::GaMDFinalize);
        wallcycle_start(wcycle, WallCycleCounter::GaMDOutput);
        writeParameterLine(step, nodeid, true);
        writeReweightLine(step, nodeid, true);
        wallcycle_stop(wcycle, WallCycleCounter::GaMDOutput);
        return;
    }

    const bool updateStat = stageUpdatesStats(stepPlan_.stage);
    const bool applyBoost = stageUsesBoost(stepPlan_.stage);

    stepResult_.scaleP = 1.0;
    stepResult_.scaleD = 1.0;
    stepResult_.boostP = 0.0;
    stepResult_.boostD = 0.0;

    if (applyBoost)
    {
        double boostP          = 0.0;
        double boostD          = 0.0;
        double scaleP          = 1.0;
        double scaleD          = 1.0;
        double totalEnergyForP = totalPotentialEnergy;

        if ((parameters_.boostType == 2 || parameters_.boostType == 3)
            && dihedralEnergy < dihedralStatistics_.threshold)
        {
            boostD = 0.5 * dihedralStatistics_.forceConstant
                     * (dihedralStatistics_.threshold - dihedralEnergy)
                     * (dihedralStatistics_.threshold - dihedralEnergy);
            scaleD = 1.0 - dihedralStatistics_.forceConstant * (dihedralStatistics_.threshold - dihedralEnergy);
        }

        if (parameters_.boostType == 3)
        {
            totalEnergyForP += boostD;
        }

        if ((parameters_.boostType == 1 || parameters_.boostType == 3)
            && totalEnergyForP < potentialStatistics_.threshold)
        {
            boostP = 0.5 * potentialStatistics_.forceConstant
                     * (potentialStatistics_.threshold - totalEnergyForP)
                     * (potentialStatistics_.threshold - totalEnergyForP);
            scaleP = 1.0 - potentialStatistics_.forceConstant * (potentialStatistics_.threshold - totalEnergyForP);
        }

        if (!std::isfinite(boostP) || !std::isfinite(boostD))
        {
            gmx_fatal(FARGS, "Non-finite GaMD boost energy at step %ld: boostP=%g boostD=%g.", step, boostP, boostD);
        }
        stepResult_.scaleP = validateAndClampScale(
                "total-potential", scaleP, totalEnergyForP, potentialStatistics_, step, nodeid, &warnedPotentialScaleFloor_);
        stepResult_.scaleD = validateAndClampScale(
                "dihedral", scaleD, dihedralEnergy, dihedralStatistics_, step, nodeid, &warnedDihedralScaleFloor_);
        stepResult_.boostP = boostP;
        stepResult_.boostD = boostD;
    }

    if (updateStat)
    {
        const bool   updateAverageWindow = (step % parameters_.averagingWindow == 0);
        const double totalPotentialForStats =
                (stepPlan_.stage == 4 ? totalPotentialEnergy + stepResult_.boostP + stepResult_.boostD
                                      : totalPotentialEnergy);

        if (parameters_.boostType == 1 || parameters_.boostType == 3)
        {
            updateStatisticsMinMax(totalPotentialForStats, potentialStatistics_);
            updateWindowStatistics(totalPotentialForStats, potentialWindowStatistics_);
            if (updateAverageWindow)
            {
                finalizeWindowStatistics(potentialWindowStatistics_, potentialStatistics_);
            }
        }

        if (parameters_.boostType == 2 || parameters_.boostType == 3)
        {
            updateStatisticsMinMax(dihedralEnergy, dihedralStatistics_);
            updateWindowStatistics(dihedralEnergy, dihedralWindowStatistics_);
            if (updateAverageWindow)
            {
                finalizeWindowStatistics(dihedralWindowStatistics_, dihedralStatistics_);
            }
        }

        if (stepPlan_.stage == 2 && step == parameters_.conventionalEnd)
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
        else if (stepPlan_.stage == 4)
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

    wallcycle_stop(wcycle, WallCycleCounter::GaMDFinalize);
    wallcycle_start(wcycle, WallCycleCounter::GaMDOutput);
    if (!bufferedProductionOutputEnabled() || stepPlan_.stage != 5)
    {
        writeParameterLine(step, nodeid, true);
        writeReweightLine(step, nodeid, true);
    }
    wallcycle_stop(wcycle, WallCycleCounter::GaMDOutput);
}

void GaMDRuntime::writeBufferedProductionOutput(ArrayRef<const int64_t> deferredSteps,
                                                ArrayRef<const std::array<real, F_NRE>> deferredEnergySamples,
                                                int64_t        currentStep,
                                                int            nodeid,
                                                gmx_wallcycle* wcycle)
{
    if (!bufferedProductionOutputEnabled())
    {
        return;
    }
    GMX_RELEASE_ASSERT(deferredSteps.size() == deferredEnergySamples.size(),
                       "Buffered GaMD output steps and energy samples must match");
    if (stepPlan_.stage != 5)
    {
        GMX_RELEASE_ASSERT(deferredSteps.empty(),
                           "Buffered GaMD output may defer samples only in production stage 5");
        return;
    }

    wallcycle_start(wcycle, WallCycleCounter::GaMDOutput);
    for (std::size_t sampleIndex = 0; sampleIndex < deferredEnergySamples.size(); ++sampleIndex)
    {
        const auto& sample          = deferredEnergySamples[sampleIndex];
        float       potentialEnergy = 0;
        for (int fType = 0; fType < F_EPOT; ++fType)
        {
            if (fType != F_DISRESVIOL && fType != F_ORIRESDEV)
            {
                potentialEnergy += sample[fType];
            }
        }
        const float dihedralEnergy =
                sample[F_PDIHS] + sample[F_RBDIHS] + sample[F_FOURDIHS] + sample[F_CMAP];

        GaMDProductionOutputRecord record;
        record.potentialEnergy = potentialEnergy;
        record.dihedralEnergy  = dihedralEnergy;
        double totalEnergyForP = potentialEnergy;
        if ((parameters_.boostType == 2 || parameters_.boostType == 3)
            && dihedralEnergy < dihedralStatistics_.threshold)
        {
            const double deltaD = dihedralStatistics_.threshold - dihedralEnergy;
            record.boostD       = 0.5 * dihedralStatistics_.forceConstant * deltaD * deltaD;
            record.scaleD       = 1.0 - dihedralStatistics_.forceConstant * deltaD;
        }
        if (parameters_.boostType == 3)
        {
            totalEnergyForP += record.boostD;
        }
        if ((parameters_.boostType == 1 || parameters_.boostType == 3)
            && totalEnergyForP < potentialStatistics_.threshold)
        {
            const double deltaP = potentialStatistics_.threshold - totalEnergyForP;
            record.boostP       = 0.5 * potentialStatistics_.forceConstant * deltaP * deltaP;
            record.scaleP       = 1.0 - potentialStatistics_.forceConstant * deltaP;
        }
        record.scaleP = validateAndClampScale("total-potential",
                                              record.scaleP,
                                              totalEnergyForP,
                                              potentialStatistics_,
                                              deferredSteps[sampleIndex],
                                              nodeid,
                                              &warnedPotentialScaleFloor_);
        record.scaleD = validateAndClampScale("dihedral",
                                              record.scaleD,
                                              dihedralEnergy,
                                              dihedralStatistics_,
                                              deferredSteps[sampleIndex],
                                              nodeid,
                                              &warnedDihedralScaleFloor_);

        writeParameterLine(deferredSteps[sampleIndex], nodeid, false);
        writeReweightRecord(deferredSteps[sampleIndex], nodeid, record, false);
    }

    writeParameterLine(currentStep, nodeid, false);
    writeReweightLine(currentStep, nodeid, false);
    if ((parameterFile_ != nullptr && std::fflush(parameterFile_) != 0)
        || (reweightFile_ != nullptr && std::fflush(reweightFile_) != 0))
    {
        gmx_fatal(FARGS, "Failed to flush buffered GaMD output at step %ld.", currentStep);
    }
    wallcycle_stop(wcycle, WallCycleCounter::GaMDOutput);
}

} // namespace gmx
