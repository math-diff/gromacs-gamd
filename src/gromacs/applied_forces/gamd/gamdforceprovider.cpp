#include "gamdforceprovider.h"

#include "config.h"

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

// ============================================================================
// Globals shared with other translation units
// ============================================================================

// 导出给 bonded.cpp 的二面角补偿系数
double g_gamd_dih_ratio = 1.0;

// 导出给 md.cpp 的全局力缩放系数
double g_gamd_scale_P = 1.0;

int g_gamd_debug = 0;

// Cache the total boost added to F_EPOT for bookkeeping/debug
double g_gamd_last_total_boost = 0.0;

// 可由 md.cpp 更新的当前 step 调试变量
long g_gamd_debug_current_step = -1;

// ============================================================================
// File-local runtime state
// ============================================================================

static double g_gamd_boostP_current  = 0.0;
static double g_gamd_boostD_current  = 0.0;
static double g_gamd_scale_D_current = 1.0;
static double g_gamd_VP_used         = 0.0;
static double g_gamd_VD_used         = 0.0;

static long g_gamd_lastPreparedStep  = -1;
static long g_gamd_lastAccountedStep = -1;

// 仅当当前 step 可用于 reweight 输出时为 true
static bool g_gamd_step_ready_for_output = false;
static int  g_gamd_currentStage          = 0; // 1..5

// restart load control
static bool g_gamd_restart_load_attempted   = false;
static bool g_gamd_force_production_restart = false;

// 当前 rank id，供写 restart 时使用
static int g_gamd_current_nodeid = 0;

// Mirror the .cpt decision from md.cpp so GaMD restart state is written on the
// same steps as checkpointing.
static bool g_gamd_checkpointing_this_step = false;

static long g_gamd_suppressTextOutputStep = -1;

static bool g_gamd_warnedScalePFloor = false;
static bool g_gamd_warnedScaleDFloor = false;

static FILE* g_gamd_reweight_fp = nullptr;
static FILE* g_gamd_para_fp     = nullptr;

namespace gmx
{

struct GaMDRealParams
{
    int    igamd        = 0;
    int    irest_gamd   = 0;
    int    iE           = 1;
    int    iEP          = 1;
    int    iED          = 1;
    long   ntcmdprep    = 200000;
    long   ntcmd        = 1000000;
    long   ntebprep     = 200000;
    long   nteb         = 1000000;
    long   ntave        = 50000;
    long   reweight_nst = 1;
    long   para_nst     = 0;
    double sigma0P      = 6.0 * 4.184; // kcal/mol -> kJ/mol
    double sigma0D      = 6.0 * 4.184;
};

struct GaMDStats
{
    double Vmax   = -1e99;
    double Vmin   = 1e99;
    double Vavg   = 0.0;
    double sigmaV = 0.001;
    double M2     = 0.0;
    long   count  = 0;
    double E      = 0.0;
    double k0     = 0.0;
    double k      = 0.0;
};

struct GaMDWindowStats
{
    long   count = 0;
    double mean  = 0.0;
    double M2    = 0.0;
};

// 全局参数
static GaMDRealParams g_params;
static bool           g_params_loaded = false;

// 为了 restart/continue 可恢复，这里不要用 thread_local
static GaMDStats       g_statP;
static GaMDStats       g_statD;
static GaMDWindowStats g_windowStatP;
static GaMDWindowStats g_windowStatD;

static int determineGaMDStage(long step);

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

static void writeStatsLine(FILE* fp, const char* name, const GaMDStats& s)
{
    std::fprintf(fp,
                 "%s %.17g %.17g %.17g %.17g %.17g %ld %.17g %.17g %.17g\n",
                 name,
                 s.Vmax,
                 s.Vmin,
                 s.Vavg,
                 s.sigmaV,
                 s.M2,
                 s.count,
                 s.E,
                 s.k0,
                 s.k);
}

static bool readStatsLine(std::istringstream& iss, GaMDStats* s)
{
    return static_cast<bool>(iss >> s->Vmax >> s->Vmin >> s->Vavg >> s->sigmaV >> s->M2 >> s->count
                             >> s->E >> s->k0 >> s->k);
}

static void writeWindowStatsLine(FILE* fp, const char* name, const GaMDWindowStats& s)
{
    std::fprintf(fp, "%s %ld %.17g %.17g\n", name, s.count, s.mean, s.M2);
}

static bool readWindowStatsLine(std::istringstream& iss, GaMDWindowStats* s)
{
    return static_cast<bool>(iss >> s->count >> s->mean >> s->M2);
}

static void saveGaMDRestartState(long step, int nodeid)
{
    if (nodeid != 0)
    {
        return;
    }

    if (!g_params_loaded || g_params.igamd == 0)
    {
        return;
    }

    if (step < 0)
    {
        return;
    }

    const int  currentStage            = determineGaMDStage(step);
    const bool saveAtCmdStatBoundary   = (step == g_params.ntcmd);
    const bool saveAtBoostStatBoundary = (step == g_params.ntcmd + g_params.nteb);
    const bool inBoostStatPhase        = (currentStage == 4);
    const bool inProductionPhase       = (currentStage == 5);
    const bool saveDuringBoostStat     = (inBoostStatPhase && g_gamd_checkpointing_this_step);
    const bool saveDuringProduction    = (inProductionPhase && g_gamd_checkpointing_this_step);

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
    if (step <= g_params.ntcmdprep && !inProductionPhase)
    {
        return;
    }

    const char* tmpName      = "gamd-restart.dat.tmp";
    const char* finalName    = "gamd-restart.dat";
    const char* previousName = "gamd-restart-prev.dat";

    // Rows already emitted before this checkpoint must reach durable storage
    // before the restart state is used as their commit record.
    flushAndSyncFileOrFatal(g_gamd_reweight_fp, "gamd-reweight.dat");
    flushAndSyncFileOrFatal(g_gamd_para_fp, "gamd-para.dat");

    FILE* fp = std::fopen(tmpName, "w");
    if (fp == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open %s for writing GaMD restart state.", tmpName);
    }

    std::fprintf(fp, "# GaMD restart state\n");
    std::fprintf(fp, "version 3\n");
    std::fprintf(fp, "saved_step %ld\n", step);
    std::fprintf(fp, "production_restart %d\n", g_gamd_force_production_restart ? 1 : 0);

    // 参数也写出，续跑时做一致性检查
    std::fprintf(fp,
                 "params %d %d %d %d %ld %ld %ld %ld %ld %ld %.17g %.17g\n",
                 g_params.igamd,
                 g_params.iE,
                 g_params.iEP,
                 g_params.iED,
                 g_params.ntcmdprep,
                 g_params.ntcmd,
                 g_params.ntebprep,
                 g_params.nteb,
                 g_params.ntave,
                 g_params.reweight_nst,
                 g_params.sigma0P,
                 g_params.sigma0D);

    writeWindowStatsLine(fp, "windowP", g_windowStatP);
    writeWindowStatsLine(fp, "windowD", g_windowStatD);
    writeStatsLine(fp, "statP", g_statP);
    writeStatsLine(fp, "statD", g_statD);

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

static void loadGaMDRestartStateIfNeeded(long step, int nodeid)
{
    if (g_gamd_restart_load_attempted)
    {
        return;
    }

    const bool explicitProductionRestart = (g_params.irest_gamd != 0);

    // 只尝试一次：
    // - fresh run 若从 step 0 开始，不要后面误加载旧文件
    // - continue run 第一调用通常就是 step > 0，会在这里加载
    // - irest_gamd 模式允许从新模拟的首步直接加载
    g_gamd_restart_load_attempted = true;

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

    GaMDStats       statP;
    GaMDStats       statD;
    GaMDWindowStats windowStatP;
    GaMDWindowStats windowStatD;

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

    auto validStats = [](const GaMDStats& stat)
    {
        return std::isfinite(stat.Vmax) && std::isfinite(stat.Vmin) && std::isfinite(stat.Vavg)
               && std::isfinite(stat.sigmaV) && std::isfinite(stat.M2) && std::isfinite(stat.E)
               && std::isfinite(stat.k0) && std::isfinite(stat.k) && stat.count >= 0
               && stat.sigmaV >= 0.0 && stat.M2 >= 0.0 && stat.k0 >= 0.0 && stat.k0 <= 1.0
               && stat.k >= 0.0 && (stat.count == 0 || stat.Vmax >= stat.Vmin);
    };
    auto validWindowStats = [](const GaMDWindowStats& stat) {
        return stat.count >= 0 && std::isfinite(stat.mean) && std::isfinite(stat.M2) && stat.M2 >= 0.0;
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
            (file_igamd != g_params.igamd || file_iE != g_params.iE || file_iEP != g_params.iEP
             || file_iED != g_params.iED || std::abs(file_sigma0P - g_params.sigma0P) > 1e-9
             || std::abs(file_sigma0D - g_params.sigma0D) > 1e-9);
    const bool stageParamsMismatch = (file_ntcmdprep != g_params.ntcmdprep
                                      || file_ntcmd != g_params.ntcmd || file_ntebprep != g_params.ntebprep
                                      || file_nteb != g_params.nteb || file_ntave != g_params.ntave);

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
                  g_params.irest_gamd);
    }

    g_statP = statP;
    g_statD = statD;
    if (version >= 2)
    {
        g_windowStatP = windowStatP;
        g_windowStatD = windowStatD;
    }
    else
    {
        g_windowStatP = {};
        g_windowStatD = {};
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
                      g_params.irest_gamd,
                      savedStep,
                      restartNtcmd);
        }

        // In irest_gamd mode the stage-control counters can be omitted from
        // gamd.in. Keep the values from the restart file for provenance/output.
        g_params.ntcmdprep = file_ntcmdprep;
        g_params.ntcmd     = file_ntcmd;
        g_params.ntebprep  = file_ntebprep;
        g_params.nteb      = file_nteb;
        g_params.ntave     = std::max<long>(1, file_ntave);

        g_gamd_force_production_restart = true;

        if (checkpointContinuation)
        {
            reconcileGaMDTextFile("gamd-reweight.dat", 9, savedStep, nodeid);
            reconcileGaMDTextFile("gamd-para.dat", 17, savedStep, nodeid);
            if (savedStep == step)
            {
                g_gamd_suppressTextOutputStep = savedStep;
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
                         g_params.irest_gamd);

            if (!checkpointContinuation && savedStep < restartNtcmd + restartNteb)
            {
                std::fprintf(stderr,
                             "[GaMD WARNING] irest_gamd=%d loaded saved_step=%ld before the "
                             "adaptive boost-stat stage ended at step=%ld. "
                             "The new simulation will enter production immediately using the "
                             "latest saved GaMD parameters.\n",
                             g_params.irest_gamd,
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
                             g_params.irest_gamd,
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
            g_gamd_suppressTextOutputStep = savedStep;
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

void gamdSetCheckpointingThisStep(bool checkpointingThisStep)
{
    g_gamd_checkpointing_this_step = checkpointingThisStep;
}

// 提供给 enerdata_utils.cpp：在 reset 前写出当前 GaMD restart 状态
void gamdWriteRestartState(long step)
{
    saveGaMDRestartState(step, g_gamd_current_nodeid);
}

void gamdResetStateForTesting()
{
    if (g_gamd_reweight_fp != nullptr)
    {
        std::fclose(g_gamd_reweight_fp);
        g_gamd_reweight_fp = nullptr;
    }
    if (g_gamd_para_fp != nullptr)
    {
        std::fclose(g_gamd_para_fp);
        g_gamd_para_fp = nullptr;
    }

    g_gamd_dih_ratio          = 1.0;
    g_gamd_scale_P            = 1.0;
    g_gamd_debug              = 0;
    g_gamd_last_total_boost   = 0.0;
    g_gamd_debug_current_step = -1;

    g_gamd_boostP_current  = 0.0;
    g_gamd_boostD_current  = 0.0;
    g_gamd_scale_D_current = 1.0;
    g_gamd_VP_used         = 0.0;
    g_gamd_VD_used         = 0.0;

    g_gamd_lastPreparedStep  = -1;
    g_gamd_lastAccountedStep = -1;

    g_gamd_step_ready_for_output = false;
    g_gamd_currentStage          = 0;

    g_gamd_restart_load_attempted   = false;
    g_gamd_force_production_restart = false;
    g_gamd_current_nodeid           = 0;
    g_gamd_checkpointing_this_step  = false;
    g_gamd_suppressTextOutputStep   = -1;
    g_gamd_warnedScalePFloor        = false;
    g_gamd_warnedScaleDFloor        = false;

    g_params        = {};
    g_params_loaded = false;
    g_statP         = {};
    g_statD         = {};
    g_windowStatP   = {};
    g_windowStatD   = {};
}

const char* currentStepGaMDGpuIncompatibilityReason(bool /*useGpuUpdate*/, bool /*haveGpuBondedWork*/)
{
    return nullptr;
}

const char* gaMDExecutionModeName(GaMDExecutionMode mode)
{
    switch (mode)
    {
        case GaMDExecutionMode::CpuReference: return "CPU reference";
        case GaMDExecutionMode::GpuScalarSynchronized: return "GPU scalar-synchronized";
        case GaMDExecutionMode::GpuResident: return "GPU resident";
    }
    return "unknown";
}

GaMDExecutionMode selectGaMDExecutionMode(bool useGaMD,
                                          bool useGpuNonbonded,
                                          bool useGpuPme,
                                          bool useGpuBonded,
                                          bool useGpuUpdate,
                                          bool havePpDomainDecomposition,
                                          bool haveSeparatePmeRank,
                                          bool useMts,
                                          bool havePressureCoupling,
                                          int  nstfout)
{
    if (!useGaMD)
    {
        return GaMDExecutionMode::CpuReference;
    }

    const char* request = std::getenv("GMX_GAMD_GPU");
    if (request == nullptr || std::string(request) == "0")
    {
        return GaMDExecutionMode::CpuReference;
    }
    if (std::string(request) != "1")
    {
        gmx_fatal(FARGS, "GMX_GAMD_GPU must be exactly 0 or 1; got '%s'.", request);
    }

    if (!GMX_GPU_CUDA)
    {
        gmx_fatal(FARGS, "GMX_GAMD_GPU=1 currently requires a CUDA build.");
    }
    if (!useGpuNonbonded)
    {
        gmx_fatal(FARGS, "GMX_GAMD_GPU=1 requires -nb gpu.");
    }
    if (!useGpuPme)
    {
        gmx_fatal(FARGS, "GMX_GAMD_GPU=1 requires -pme gpu for the current production target.");
    }
    if (!useGpuBonded)
    {
        gmx_fatal(FARGS, "GMX_GAMD_GPU=1 requires -bonded gpu.");
    }
    if (!useGpuUpdate)
    {
        gmx_fatal(FARGS, "GMX_GAMD_GPU=1 requires -update gpu.");
    }
    if (havePpDomainDecomposition || haveSeparatePmeRank)
    {
        gmx_fatal(FARGS,
                  "GMX_GAMD_GPU=1 currently supports one PP rank without domain decomposition "
                  "or a separate PME rank.");
    }
    if (useMts)
    {
        gmx_fatal(FARGS, "GMX_GAMD_GPU=1 does not support MTS.");
    }
    if (havePressureCoupling)
    {
        gmx_fatal(FARGS,
                  "GMX_GAMD_GPU=1 does not yet support pressure coupling because device virial "
                  "correction is not implemented.");
    }
    if (nstfout != 0)
    {
        gmx_fatal(FARGS,
                  "GMX_GAMD_GPU=1 currently requires nstfout=0 because corrected-force output "
                  "staging is not implemented.");
    }

    return GaMDExecutionMode::GpuScalarSynchronized;
}

// ============================================================================
// Parameter loading
// ============================================================================

void loadGaMDParams(int nodeid)
{
    if (g_params_loaded)
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
            readValue(&g_params.igamd);
        }
        else if (key == "irest_gamd")
        {
            readValue(&g_params.irest_gamd);
        }
        else if (key == "iE")
        {
            readValue(&g_params.iE);
        }
        else if (key == "iEP")
        {
            readValue(&g_params.iEP);
        }
        else if (key == "iED")
        {
            readValue(&g_params.iED);
        }
        else if (key == "ntcmdprep")
        {
            readValue(&g_params.ntcmdprep);
        }
        else if (key == "ntcmd")
        {
            readValue(&g_params.ntcmd);
        }
        else if (key == "ntebprep")
        {
            readValue(&g_params.ntebprep);
        }
        else if (key == "nteb")
        {
            readValue(&g_params.nteb);
        }
        else if (key == "ntave")
        {
            readValue(&g_params.ntave);
        }
        else if (key == "reweight_nst")
        {
            readValue(&g_params.reweight_nst);
        }
        else if (key == "para_nst")
        {
            readValue(&g_params.para_nst);
        }
        else if (key == "sigma0P")
        {
            double sigmaKcal = 0.0;
            readValue(&sigmaKcal);
            g_params.sigma0P = sigmaKcal * 4.184;
        }
        else if (key == "sigma0D")
        {
            double sigmaKcal = 0.0;
            readValue(&sigmaKcal);
            g_params.sigma0D = sigmaKcal * 4.184;
        }
        else if (key == "debug")
        {
            readValue(&g_gamd_debug);
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

    if (g_params.igamd < 1 || g_params.igamd > 3)
    {
        gmx_fatal(FARGS, "GaMD parameter igamd must be 1, 2, or 3; got %d.", g_params.igamd);
    }
    if (g_params.irest_gamd != 0 && g_params.irest_gamd != 1)
    {
        gmx_fatal(FARGS, "GaMD parameter irest_gamd must be 0 or 1; got %d.", g_params.irest_gamd);
    }
    if (g_params.iE < 1 || g_params.iE > 2 || g_params.iEP < 1 || g_params.iEP > 2
        || g_params.iED < 1 || g_params.iED > 2)
    {
        gmx_fatal(FARGS, "GaMD parameters iE, iEP, and iED must each be 1 or 2.");
    }
    if (g_params.reweight_nst <= 0)
    {
        gmx_fatal(FARGS, "GaMD parameter reweight_nst must be positive; got %ld.", g_params.reweight_nst);
    }
    if (g_params.para_nst < 0)
    {
        gmx_fatal(FARGS, "GaMD parameter para_nst must be non-negative; got %ld.", g_params.para_nst);
    }
    if (g_gamd_debug != 0 && g_gamd_debug != 1)
    {
        gmx_fatal(FARGS, "GaMD parameter debug must be 0 or 1; got %d.", g_gamd_debug);
    }

    const bool usesPotentialBoost = (g_params.igamd == 1 || g_params.igamd == 3);
    const bool usesDihedralBoost  = (g_params.igamd == 2 || g_params.igamd == 3);
    if (usesPotentialBoost)
    {
        requireKey("iEP");
        requireKey("sigma0P");
        if (!std::isfinite(g_params.sigma0P) || g_params.sigma0P <= 0.0)
        {
            gmx_fatal(FARGS, "GaMD parameter sigma0P must be finite and positive.");
        }
    }
    if (usesDihedralBoost)
    {
        requireKey("iED");
        requireKey("sigma0D");
        if (!std::isfinite(g_params.sigma0D) || g_params.sigma0D <= 0.0)
        {
            gmx_fatal(FARGS, "GaMD parameter sigma0D must be finite and positive.");
        }
    }

    if (g_params.irest_gamd == 0)
    {
        requireKey("ntcmdprep");
        requireKey("ntcmd");
        requireKey("ntebprep");
        requireKey("nteb");
        requireKey("ntave");

        if (g_params.ntave <= 0 || g_params.ntcmdprep < 0 || g_params.ntcmd <= g_params.ntcmdprep
            || g_params.ntebprep < 0 || g_params.nteb <= g_params.ntebprep)
        {
            gmx_fatal(FARGS,
                      "Invalid GaMD stage parameters: require ntave>0, 0<=ntcmdprep<ntcmd, "
                      "and 0<=ntebprep<nteb; got %ld, %ld, %ld, %ld, %ld.",
                      g_params.ntave,
                      g_params.ntcmdprep,
                      g_params.ntcmd,
                      g_params.ntebprep,
                      g_params.nteb);
        }

        if (g_params.ntcmdprep % g_params.ntave != 0 || g_params.ntcmd % g_params.ntave != 0
            || g_params.ntebprep % g_params.ntave != 0 || g_params.nteb % g_params.ntave != 0)
        {
            gmx_fatal(FARGS,
                      "GaMD requires ntcmdprep (%ld), ntcmd (%ld), ntebprep (%ld), and nteb "
                      "(%ld) to be multiples of ntave (%ld) for aligned statistics windows.",
                      g_params.ntcmdprep,
                      g_params.ntcmd,
                      g_params.ntebprep,
                      g_params.nteb,
                      g_params.ntave);
        }
    }

    g_params_loaded = true;

    if (nodeid == 0)
    {
        std::fprintf(stderr, "\n=== GaMD Parameters Loaded from gamd.in ===\n");
        std::fprintf(stderr, "igamd        = %d\n", g_params.igamd);
        std::fprintf(stderr, "irest_gamd   = %d\n", g_params.irest_gamd);
        std::fprintf(stderr, "iE           = %d\n", g_params.iE);
        std::fprintf(stderr, "iEP          = %d\n", g_params.iEP);
        std::fprintf(stderr, "iED          = %d\n", g_params.iED);
        std::fprintf(stderr, "ntcmdprep    = %ld\n", g_params.ntcmdprep);
        std::fprintf(stderr, "ntcmd        = %ld\n", g_params.ntcmd);
        std::fprintf(stderr, "ntebprep     = %ld\n", g_params.ntebprep);
        std::fprintf(stderr, "nteb         = %ld\n", g_params.nteb);
        std::fprintf(stderr, "ntave        = %ld\n", g_params.ntave);
        std::fprintf(stderr, "reweight_nst = %ld\n", g_params.reweight_nst);
        std::fprintf(stderr, "para_nst     = %ld\n", g_params.para_nst);
        std::fprintf(stderr, "sigma0P      = %.4f kJ/mol\n", g_params.sigma0P);
        std::fprintf(stderr, "sigma0D      = %.4f kJ/mol\n", g_params.sigma0D);
        std::fprintf(stderr, "DEBUG MODE   = %d\n", g_gamd_debug);
        std::fprintf(stderr, "===========================================\n\n");
    }
}

// ============================================================================
// Reweight file helpers
// ============================================================================

static FILE* getGaMDReweightFile(int nodeid)
{
    if (nodeid != 0)
    {
        return nullptr;
    }

    if (g_gamd_reweight_fp != nullptr)
    {
        return g_gamd_reweight_fp;
    }

    bool fileExists = false;
    {
        std::ifstream test("gamd-reweight.dat");
        fileExists = test.good();
    }

    g_gamd_reweight_fp = std::fopen("gamd-reweight.dat", "a");
    if (g_gamd_reweight_fp == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open gamd-reweight.dat for writing.");
    }

    if (!fileExists)
    {
        std::fprintf(g_gamd_reweight_fp, "# GaMD reweighting file\n");
        std::fprintf(g_gamd_reweight_fp, "# All energy terms stored in units of kcal/mol\n");
        std::fprintf(g_gamd_reweight_fp,
                     "# reweight_nst,total_nstep,Unboosted-Potential-Energy,"
                     "Unboosted-Dihedral-Energy,Total-Force-Weight,Dihedral-Force-Weight,"
                     "Effective-Dihedral-Force-Weight,Boost-Energy-Potential,"
                     "Boost-Energy-Dihedral\n");
        flushAndSyncFileOrFatal(g_gamd_reweight_fp, "gamd-reweight.dat");
    }

    return g_gamd_reweight_fp;
}

static FILE* getGaMDParaFile(int nodeid)
{
    if (nodeid != 0)
    {
        return nullptr;
    }

    if (g_gamd_para_fp != nullptr)
    {
        return g_gamd_para_fp;
    }

    bool fileExists = false;
    {
        std::ifstream test("gamd-para.dat");
        fileExists = test.good();
    }

    g_gamd_para_fp = std::fopen("gamd-para.dat", "a");
    if (g_gamd_para_fp == nullptr)
    {
        gmx_fatal(FARGS, "Failed to open gamd-para.dat for writing.");
    }

    if (!fileExists)
    {
        std::fprintf(g_gamd_para_fp, "# GaMD parameter history\n");
        std::fprintf(g_gamd_para_fp,
                     "# Energies are in kJ/mol; k0 is dimensionless; k is in (kJ/mol)^-1\n");
        std::fprintf(g_gamd_para_fp,
                     "# para_nst,total_nstep,stage,"
                     "VmaxP,VminP,VavgP,SigmaVP,EP,k0P,kP,"
                     "VmaxD,VminD,VavgD,SigmaVD,ED,k0D,kD\n");
        flushAndSyncFileOrFatal(g_gamd_para_fp, "gamd-para.dat");
    }

    return g_gamd_para_fp;
}

static bool suppressGaMDTextOutputForStep(long step)
{
    return (g_gamd_suppressTextOutputStep == step);
}

static void writeGaMDStatsColumns(FILE* fp, bool enabled, const GaMDStats& stat)
{
    const double nanValue = std::numeric_limits<double>::quiet_NaN();
    const double Vmax     = enabled ? stat.Vmax : nanValue;
    const double Vmin     = enabled ? stat.Vmin : nanValue;
    const double Vavg     = enabled ? stat.Vavg : nanValue;
    const double sigmaV   = enabled ? stat.sigmaV : nanValue;
    const double E        = enabled ? stat.E : nanValue;
    const double k0       = enabled ? stat.k0 : nanValue;
    const double k        = enabled ? stat.k : nanValue;

    std::fprintf(fp, " %18.9g %18.9g %18.9g %18.9g %18.9g %18.9g %18.9g", Vmax, Vmin, Vavg, sigmaV, E, k0, k);
}

struct GaMDProductionOutputRecord
{
    double potentialEnergy = 0;
    double dihedralEnergy  = 0;
    double scaleP          = 1;
    double scaleD          = 1;
    double boostP          = 0;
    double boostD          = 0;
};

static void writeGaMDParaLine(long step, int nodeid, bool flushOutput)
{
    if (nodeid != 0)
    {
        return;
    }

    if (g_params.igamd == 0 || g_params.para_nst <= 0)
    {
        return;
    }

    if (g_gamd_currentStage < 2 || step == 0)
    {
        return;
    }

    if (step % g_params.para_nst != 0)
    {
        return;
    }

    if (suppressGaMDTextOutputForStep(step))
    {
        return;
    }

    FILE* fp = getGaMDParaFile(nodeid);
    if (fp == nullptr)
    {
        return;
    }

    const bool hasP = (g_params.igamd == 1 || g_params.igamd == 3);
    const bool hasD = (g_params.igamd == 2 || g_params.igamd == 3);

    if (std::fprintf(fp, "%10ld %12ld %6d", g_params.para_nst, step, g_gamd_currentStage) < 0)
    {
        gmx_fatal(FARGS, "Failed to write gamd-para.dat at step %ld.", step);
    }
    writeGaMDStatsColumns(fp, hasP, g_statP);
    writeGaMDStatsColumns(fp, hasD, g_statD);
    if (std::fputc('\n', fp) == EOF || (flushOutput && std::fflush(fp) != 0))
    {
        gmx_fatal(FARGS, "Failed to flush gamd-para.dat at step %ld.", step);
    }
}

static void writeGaMDReweightRecord(long step, int nodeid, const GaMDProductionOutputRecord& record, bool flushOutput)
{
    if (nodeid != 0)
    {
        return;
    }

    if (g_params.igamd == 0)
    {
        return;
    }

    if (!g_gamd_step_ready_for_output || g_gamd_currentStage < 3)
    {
        return;
    }

    if (step == 0)
    {
        return;
    }

    if (g_params.reweight_nst <= 0)
    {
        return;
    }

    if (step % g_params.reweight_nst != 0)
    {
        return;
    }

    if (suppressGaMDTextOutputForStep(step))
    {
        return;
    }

    FILE* fp = getGaMDReweightFile(nodeid);
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
                     g_params.reweight_nst,
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

static void writeGaMDReweightLine(long step, int nodeid, bool flushOutput)
{
    writeGaMDReweightRecord(
            step,
            nodeid,
            { g_gamd_VP_used, g_gamd_VD_used, g_gamd_scale_P, g_gamd_scale_D_current, g_gamd_boostP_current, g_gamd_boostD_current },
            flushOutput);
}

// ============================================================================
// Core math helpers
// ============================================================================

void calc_E_k0(int iE, double sigma0, GaMDStats& stat)
{
    if (stat.Vmax <= stat.Vmin || stat.sigmaV < 1e-6 || stat.Vmax == stat.Vavg)
    {
        return;
    }

    if (iE == 1)
    {
        stat.E = stat.Vmax;
        const double k0_prime =
                (sigma0 / stat.sigmaV) * (stat.Vmax - stat.Vmin) / (stat.Vmax - stat.Vavg);
        stat.k0 = std::min(1.0, k0_prime);
    }
    else if (iE == 2)
    {
        const double k0_dprime = (1.0 - sigma0 / stat.sigmaV) * (stat.Vmax - stat.Vmin)
                                 / std::max(1e-6, (stat.Vavg - stat.Vmin));

        if (k0_dprime > 0.0 && k0_dprime <= 1.0)
        {
            stat.k0 = k0_dprime;
            stat.E  = stat.Vmin + (stat.Vmax - stat.Vmin) / stat.k0;
        }
        else
        {
            stat.E = stat.Vmax;
            const double k0_prime =
                    (sigma0 / stat.sigmaV) * (stat.Vmax - stat.Vmin) / (stat.Vmax - stat.Vavg);
            stat.k0 = std::min(1.0, k0_prime);
        }
    }

    stat.k = stat.k0 / std::max(1e-6, (stat.Vmax - stat.Vmin));
}

void update_stats_minmax(double V, GaMDStats& stat)
{
    if (V > stat.Vmax)
    {
        stat.Vmax = V;
    }
    if (V < stat.Vmin)
    {
        stat.Vmin = V;
    }

    stat.count++;
}

void update_stats_window(double V, GaMDWindowStats& stat)
{
    stat.count++;
    const double delta = V - stat.mean;
    stat.mean += delta / stat.count;
    const double delta2 = V - stat.mean;
    stat.M2 += delta * delta2;
}

void finalize_window_stats(GaMDWindowStats& windowStat, GaMDStats& stat)
{
    if (windowStat.count <= 0)
    {
        return;
    }

    stat.Vavg   = windowStat.mean;
    stat.sigmaV = (windowStat.count > 1 ? std::sqrt(windowStat.M2 / windowStat.count) : 0.0);

    windowStat = {};
}

static int determineGaMDStage(const long step)
{
    if (g_gamd_force_production_restart)
    {
        return 5;
    }

    if (step <= g_params.ntcmdprep)
    {
        return 1;
    }
    if (step <= g_params.ntcmd)
    {
        return 2;
    }
    if (step <= g_params.ntcmd + g_params.ntebprep)
    {
        return 3;
    }
    if (step <= g_params.ntcmd + g_params.nteb)
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

// ============================================================================
// Provider implementation
// ============================================================================

GaMDForceProvider::GaMDForceProvider(const GaMDParams& params, const t_inputrec& ir, const gmx_mtop_t& mtop) :
    params_(params)
{
    (void)ir;
    (void)mtop;
}

void gamdPrepareStep(long step, int nodeid)
{
    if (g_gamd_lastPreparedStep == step)
    {
        return;
    }
    g_gamd_lastPreparedStep = step;

    g_gamd_current_nodeid = nodeid;

    loadGaMDParams(nodeid);
    loadGaMDRestartStateIfNeeded(step, nodeid);

    // Default to a neutral state. Current-step GaMD applies the final force
    // correction later in do_force(), after raw energies and raw listed forces
    // for the current coordinates are available.
    g_gamd_scale_P               = 1.0;
    g_gamd_scale_D_current       = 1.0;
    g_gamd_dih_ratio             = 1.0;
    g_gamd_boostP_current        = 0.0;
    g_gamd_boostD_current        = 0.0;
    g_gamd_last_total_boost      = 0.0;
    g_gamd_VP_used               = 0.0;
    g_gamd_VD_used               = 0.0;
    g_gamd_step_ready_for_output = false;
    g_gamd_currentStage          = 0;

    if (g_params.igamd == 0)
    {
        return;
    }
    g_gamd_currentStage          = determineGaMDStage(step);
    g_gamd_step_ready_for_output = (g_gamd_currentStage >= 3);
}

GaMDGpuProductionParameters gamdGpuProductionParameters()
{
    return { g_params.igamd, g_gamd_currentStage, g_statP.E, g_statP.k, g_statD.E, g_statD.k };
}

bool gamdRequiresHostEnergyThisStep(long step)
{
    if (g_params.igamd == 0)
    {
        return false;
    }
    if (g_gamd_currentStage != 5 || g_gamd_checkpointing_this_step)
    {
        return true;
    }
    if (std::getenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP") != nullptr)
    {
        return true;
    }
    const bool writesParameters = g_params.para_nst > 0 && step % g_params.para_nst == 0;
    const bool writesReweight = step > 0 && g_params.reweight_nst > 0 && step % g_params.reweight_nst == 0;
    return (writesParameters || writesReweight) && !gamdBufferedProductionOutputEnabled();
}

bool gamdBufferedProductionOutputEnabled()
{
    const char* request = std::getenv("GMX_GAMD_GPU_BUFFERED_OUTPUT");
    return request != nullptr && request[0] == '1' && request[1] == '\0';
}

void gamdWarnIfRunTooShort(int64_t initStep, int64_t nsteps, int nodeid)
{
    static bool warnedAboutRunLength = false;
    if (warnedAboutRunLength)
    {
        return;
    }
    warnedAboutRunLength = true;

    if (!g_params_loaded || g_params.igamd == 0 || nodeid != 0 || nsteps < 0
        || g_params.irest_gamd != 0 || g_gamd_force_production_restart)
    {
        return;
    }

    const int64_t adaptivePhaseEnd =
            static_cast<int64_t>(g_params.ntcmd) + static_cast<int64_t>(g_params.nteb);
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

static double validateAndClampGaMDScale(const char*      label,
                                        double           rawScale,
                                        double           energy,
                                        const GaMDStats& stat,
                                        long             step,
                                        int              nodeid,
                                        bool*            warnedAtFloor)
{
    if (!std::isfinite(rawScale) || !std::isfinite(energy) || !std::isfinite(stat.E)
        || !std::isfinite(stat.k))
    {
        gmx_fatal(FARGS,
                  "Non-finite GaMD %s boost quantity at step %ld: raw_scale=%g energy=%g "
                  "threshold_E=%g k=%g.",
                  label,
                  step,
                  rawScale,
                  energy,
                  stat.E,
                  stat.k);
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
                         stat.E,
                         stat.k);
        }
        *warnedAtFloor = true;
    }
    return std::max(c_scaleFloor, rawScale);
}

void gamdFinalizeCurrentStep(long           step,
                             int            nodeid,
                             double         totalPotentialEnergy,
                             double         dihedralEnergy,
                             gmx_wallcycle* wcycle)
{
    if (g_params.igamd == 0 || g_gamd_lastAccountedStep == step)
    {
        return;
    }
    wallcycle_start(wcycle, WallCycleCounter::GaMDFinalize);
    g_gamd_lastAccountedStep = step;

    g_gamd_current_nodeid        = nodeid;
    g_gamd_currentStage          = determineGaMDStage(step);
    g_gamd_step_ready_for_output = (g_gamd_currentStage >= 3);
    g_gamd_VP_used               = totalPotentialEnergy;
    g_gamd_VD_used               = dihedralEnergy;

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

        const GaMDStats overrideStats{};
        g_gamd_scale_P = validateAndClampGaMDScale(
                "total-potential", scaleP, totalPotentialEnergy, overrideStats, step, nodeid, &g_gamd_warnedScalePFloor);
        g_gamd_scale_D_current = validateAndClampGaMDScale(
                "dihedral", scaleD, dihedralEnergy, overrideStats, step, nodeid, &g_gamd_warnedScaleDFloor);
        if (!std::isfinite(boostP) || !std::isfinite(boostD))
        {
            gmx_fatal(FARGS, "Non-finite forced GaMD boost at step %ld: boostP=%g boostD=%g.", step, boostP, boostD);
        }
        g_gamd_dih_ratio        = g_gamd_scale_D_current;
        g_gamd_boostP_current   = boostP;
        g_gamd_boostD_current   = boostD;
        g_gamd_last_total_boost = boostP + boostD;
        wallcycle_stop(wcycle, WallCycleCounter::GaMDFinalize);
        wallcycle_start(wcycle, WallCycleCounter::GaMDOutput);
        writeGaMDParaLine(step, nodeid, true);
        writeGaMDReweightLine(step, nodeid, true);
        wallcycle_stop(wcycle, WallCycleCounter::GaMDOutput);
        return;
    }

    const bool updateStat = stageUpdatesStats(g_gamd_currentStage);
    const bool applyBoost = stageUsesBoost(g_gamd_currentStage);

    g_gamd_scale_P         = 1.0;
    g_gamd_scale_D_current = 1.0;
    g_gamd_dih_ratio       = 1.0;
    g_gamd_boostP_current  = 0.0;
    g_gamd_boostD_current  = 0.0;

    if (applyBoost)
    {
        double boostP          = 0.0;
        double boostD          = 0.0;
        double scaleP          = 1.0;
        double scaleD          = 1.0;
        double totalEnergyForP = totalPotentialEnergy;

        if ((g_params.igamd == 2 || g_params.igamd == 3) && dihedralEnergy < g_statD.E)
        {
            boostD = 0.5 * g_statD.k * (g_statD.E - dihedralEnergy) * (g_statD.E - dihedralEnergy);
            scaleD = 1.0 - g_statD.k * (g_statD.E - dihedralEnergy);
        }

        if (g_params.igamd == 3)
        {
            totalEnergyForP += boostD;
        }

        if ((g_params.igamd == 1 || g_params.igamd == 3) && totalEnergyForP < g_statP.E)
        {
            boostP = 0.5 * g_statP.k * (g_statP.E - totalEnergyForP) * (g_statP.E - totalEnergyForP);
            scaleP = 1.0 - g_statP.k * (g_statP.E - totalEnergyForP);
        }

        if (!std::isfinite(boostP) || !std::isfinite(boostD))
        {
            gmx_fatal(FARGS, "Non-finite GaMD boost energy at step %ld: boostP=%g boostD=%g.", step, boostP, boostD);
        }
        g_gamd_scale_P = validateAndClampGaMDScale(
                "total-potential", scaleP, totalEnergyForP, g_statP, step, nodeid, &g_gamd_warnedScalePFloor);
        g_gamd_scale_D_current = validateAndClampGaMDScale(
                "dihedral", scaleD, dihedralEnergy, g_statD, step, nodeid, &g_gamd_warnedScaleDFloor);
        g_gamd_dih_ratio      = g_gamd_scale_D_current;
        g_gamd_boostP_current = boostP;
        g_gamd_boostD_current = boostD;
    }

    if (updateStat)
    {
        const bool   updateAverageWindow = (step % g_params.ntave == 0);
        const double totalPotentialForStats =
                (g_gamd_currentStage == 4 ? totalPotentialEnergy + g_gamd_boostP_current + g_gamd_boostD_current
                                          : totalPotentialEnergy);

        if (g_params.igamd == 1 || g_params.igamd == 3)
        {
            update_stats_minmax(totalPotentialForStats, g_statP);
            update_stats_window(totalPotentialForStats, g_windowStatP);
            if (updateAverageWindow)
            {
                finalize_window_stats(g_windowStatP, g_statP);
            }
        }

        if (g_params.igamd == 2 || g_params.igamd == 3)
        {
            update_stats_minmax(dihedralEnergy, g_statD);
            update_stats_window(dihedralEnergy, g_windowStatD);
            if (updateAverageWindow)
            {
                finalize_window_stats(g_windowStatD, g_statD);
            }
        }

        if (g_gamd_currentStage == 2 && step == g_params.ntcmd)
        {
            if (g_params.igamd == 1 || g_params.igamd == 3)
            {
                calc_E_k0(g_params.iEP, g_params.sigma0P, g_statP);
            }
            if (g_params.igamd == 2 || g_params.igamd == 3)
            {
                calc_E_k0(g_params.iED, g_params.sigma0D, g_statD);
            }
        }
        else if (g_gamd_currentStage == 4)
        {
            if (g_params.igamd == 1 || g_params.igamd == 3)
            {
                calc_E_k0(g_params.iEP, g_params.sigma0P, g_statP);
            }
            if (g_params.igamd == 2 || g_params.igamd == 3)
            {
                calc_E_k0(g_params.iED, g_params.sigma0D, g_statD);
            }
        }
    }

    g_gamd_last_total_boost = g_gamd_boostP_current + g_gamd_boostD_current;
    wallcycle_stop(wcycle, WallCycleCounter::GaMDFinalize);
    wallcycle_start(wcycle, WallCycleCounter::GaMDOutput);
    if (!gamdBufferedProductionOutputEnabled() || g_gamd_currentStage != 5)
    {
        writeGaMDParaLine(step, nodeid, true);
        writeGaMDReweightLine(step, nodeid, true);
    }
    wallcycle_stop(wcycle, WallCycleCounter::GaMDOutput);
}

void gamdWriteBufferedProductionOutput(ArrayRef<const int64_t> deferredSteps,
                                       ArrayRef<const std::array<real, F_NRE>> deferredEnergySamples,
                                       long           currentStep,
                                       int            nodeid,
                                       gmx_wallcycle* wcycle)
{
    if (!gamdBufferedProductionOutputEnabled())
    {
        return;
    }
    GMX_RELEASE_ASSERT(deferredSteps.size() == deferredEnergySamples.size(),
                       "Buffered GaMD output steps and energy samples must match");
    if (g_gamd_currentStage != 5)
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
        if ((g_params.igamd == 2 || g_params.igamd == 3) && dihedralEnergy < g_statD.E)
        {
            const double deltaD = g_statD.E - dihedralEnergy;
            record.boostD       = 0.5 * g_statD.k * deltaD * deltaD;
            record.scaleD       = 1.0 - g_statD.k * deltaD;
        }
        if (g_params.igamd == 3)
        {
            totalEnergyForP += record.boostD;
        }
        if ((g_params.igamd == 1 || g_params.igamd == 3) && totalEnergyForP < g_statP.E)
        {
            const double deltaP = g_statP.E - totalEnergyForP;
            record.boostP       = 0.5 * g_statP.k * deltaP * deltaP;
            record.scaleP       = 1.0 - g_statP.k * deltaP;
        }
        record.scaleP = validateAndClampGaMDScale("total-potential",
                                                  record.scaleP,
                                                  totalEnergyForP,
                                                  g_statP,
                                                  deferredSteps[sampleIndex],
                                                  nodeid,
                                                  &g_gamd_warnedScalePFloor);
        record.scaleD = validateAndClampGaMDScale(
                "dihedral", record.scaleD, dihedralEnergy, g_statD, deferredSteps[sampleIndex], nodeid, &g_gamd_warnedScaleDFloor);

        writeGaMDParaLine(deferredSteps[sampleIndex], nodeid, false);
        writeGaMDReweightRecord(deferredSteps[sampleIndex], nodeid, record, false);
    }

    writeGaMDParaLine(currentStep, nodeid, false);
    writeGaMDReweightLine(currentStep, nodeid, false);
    if ((g_gamd_para_fp != nullptr && std::fflush(g_gamd_para_fp) != 0)
        || (g_gamd_reweight_fp != nullptr && std::fflush(g_gamd_reweight_fp) != 0))
    {
        gmx_fatal(FARGS, "Failed to flush buffered GaMD output at step %ld.", currentStep);
    }
    wallcycle_stop(wcycle, WallCycleCounter::GaMDOutput);
}

void GaMDForceProvider::calculateForces(const ForceProviderInput& fpin, ForceProviderOutput* fpout)
{
    (void)fpout;

    const int  nodeid = fpin.cr_.nodeid;
    const long step   = fpin.step_;


    gamdPrepareStep(step, nodeid);
}

} // namespace gmx
