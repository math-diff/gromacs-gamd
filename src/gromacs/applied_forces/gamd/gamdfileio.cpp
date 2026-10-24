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
 * \brief Reads and writes the compatible GaMD restart and text protocols.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "gromacs/math/units.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textreader.h"

#include "gamddiagnostics.h"
#include "gamdruntime.h"

//! \cond internal
namespace gmx
{
namespace
{

/*! \internal \brief Tests accessibility using the legacy not-found policy. */
bool fileExists(const char* filename)
{
    return File::exists(filename, File::returnFalseOnError);
}

/*! \internal \brief Checks all stream errors before synchronizing a GaMD output file. */
void flushAndSyncGaMDFile(FILE* fp, const char* filename)
{
    if (fp == nullptr)
    {
        return;
    }
    // 刷新成功也不能抹去此前的写入失败，否则会提交不完整的重整文件。
    if (std::fflush(fp) != 0 || std::ferror(fp) != 0 || gmx_fsync(fp) != 0)
    {
        GMX_THROW(FileIOError(formatString("Failed to flush and synchronize GaMD file '%s'.", filename)));
    }
}

//! Header of a new gamd-reweight.dat.
const char* const c_reweightFileHeader =
        "# GaMD reweighting file\n"
        "# All energy terms stored in units of kcal/mol\n"
        "# reweight_nst,total_nstep,Unboosted-Potential-Energy,"
        "Unboosted-Dihedral-Energy,Total-Force-Weight,Dihedral-Force-Weight,"
        "Effective-Dihedral-Force-Weight,Boost-Energy-Potential,"
        "Boost-Energy-Dihedral\n";

//! Header of a new gamd-para.dat.
const char* const c_parameterFileHeader =
        "# GaMD parameter history\n"
        "# Energies are in kJ/mol; k0 is dimensionless; k is in (kJ/mol)^-1\n"
        "# para_nst,total_nstep,stage,"
        "VmaxP,VminP,VavgP,SigmaVP,EP,k0P,kP,"
        "VmaxD,VminD,VavgD,SigmaVD,ED,k0D,kD\n";

/*! \internal \brief Opens a legacy text output on first use and returns the borrowed handle.
 *
 * Continuations append to the existing file; a new file starts with its header.
 */
FILE* openGaMDTextOutput(FilePtr* file, const char* filename, const char* header)
{
    if (*file != nullptr)
    {
        return file->get();
    }

    const bool outputFileExists = fileExists(filename);
    file->reset(std::fopen(filename, "a"));
    if (*file == nullptr)
    {
        GMX_THROW(FileIOError(formatString("Failed to open %s for writing.", filename)));
    }
    if (!outputFileExists)
    {
        std::fprintf(file->get(), "%s", header);
        flushAndSyncGaMDFile(file->get(), filename);
    }
    return file->get();
}

/*! \internal \brief Validates a complete legacy output row, including NaN placeholders. */
bool parseGaMDTextRow(const std::string& line, int expectedColumns, int64_t* step)
{
    const std::vector<std::string> tokens = splitString(line);
    if (static_cast<int>(tokens.size()) != expectedColumns)
    {
        return false;
    }

    try
    {
        for (const std::string& value : tokens)
        {
            // NaN 占位列也符合旧输出协议，doubleFromString 会保留它们。
            doubleFromString(value.c_str());
        }
        *step = int64FromString(tokens[1].c_str());
    }
    catch (const InvalidInputError&)
    {
        return false;
    }

    return true;
}

/* The GROMACS checkpoint and these human-readable files cannot be committed as
 * one filesystem transaction.  Make continuation deterministic instead: the
 * restart file is the commit record, and on load each text file is atomically
 * rebuilt to contain exactly one complete row per step through savedStep.
 * Re-running this repair after another interruption is safe and idempotent. */
/*! \internal \brief Atomically retains complete unique output rows through the checkpoint step. */
void reconcileGaMDTextFile(const char* filename, int expectedColumns, int64_t savedStep, const MDLogger& logger)
{
    if (!fileExists(filename))
    {
        return;
    }

    TextReader input(filename);

    std::vector<std::string>       headers;
    std::set<std::string>          seenHeaders;
    std::map<int64_t, std::string> rowsByStep;
    int64_t                        malformedRows = 0;
    int64_t                        futureRows    = 0;
    int64_t                        duplicateRows = 0;
    std::string                    line;
    while (input.readLine(&line))
    {
        // 只去掉 getline 原来去掉的换行符，保留原记录的空格和 CR。
        if (!line.empty() && line.back() == '\n')
        {
            line.pop_back();
        }
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
    input.close();

    const std::string tmpName = std::string(filename) + ".reconcile.tmp";
    FilePtr           outputFile(std::fopen(tmpName.c_str(), "w"));
    FILE*             fp = outputFile.get();
    if (fp == nullptr)
    {
        GMX_THROW(FileIOError(
                formatString("Failed to open '%s' for continuation repair.", tmpName.c_str())));
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
    flushAndSyncGaMDFile(fp, tmpName.c_str());
    if (std::fclose(outputFile.release()) != 0)
    {
        GMX_THROW(FileIOError(
                formatString("Failed to close repaired GaMD text file '%s'.", tmpName.c_str())));
    }
    gmx_file_rename(tmpName, filename);

    if (malformedRows > 0 || futureRows > 0 || duplicateRows > 0)
    {
        GMX_LOG(logger.warning)
                .asParagraph()
                .appendTextFormatted("[GaMD WARNING] Repaired %s at checkpoint step %" PRId64
                                     ": removed %" PRId64
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

/*! \internal \brief Writes all persisted statistics in the legacy column order. */
void writeStatsLine(FILE* fp, const char* name, const GaMDEnergyStatistics& s)
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

/*! \internal \brief Converts the nine statistics fields of a size-checked restart row. */
GaMDEnergyStatistics readStatsLine(ArrayRef<const std::string> columns)
{
    // 列顺序与版本 1--3 的文本协议一致，转换失败由候选快照解析器统一报告。
    return { doubleFromString(columns[1].c_str()), doubleFromString(columns[2].c_str()),
             doubleFromString(columns[3].c_str()), doubleFromString(columns[4].c_str()),
             doubleFromString(columns[5].c_str()), int64FromString(columns[6].c_str()),
             doubleFromString(columns[7].c_str()), doubleFromString(columns[8].c_str()),
             doubleFromString(columns[9].c_str()) };
}

/*! \internal \brief Writes the active population-statistics window. */
void writeWindowStatsLine(FILE* fp, const char* name, const GaMDWindowStatistics& s)
{
    std::fprintf(fp, "%s %" PRId64 " %.17g %.17g\n", name, s.count, s.mean, s.m2);
}

/*! \internal \brief Converts the three window fields of a size-checked restart row. */
GaMDWindowStatistics readWindowStatsLine(ArrayRef<const std::string> columns)
{
    return { int64FromString(columns[1].c_str()),
             doubleFromString(columns[2].c_str()),
             doubleFromString(columns[3].c_str()) };
}

} // namespace

void GaMDRuntime::saveRestartState(int64_t step)
{
    if (!parametersLoaded_)
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
    flushAndSyncGaMDFile(reweightFile_.get(), "gamd-reweight.dat");
    flushAndSyncGaMDFile(parameterFile_.get(), "gamd-para.dat");

    FilePtr outputFile(std::fopen(tmpName, "w"));
    FILE*   fp = outputFile.get();
    if (fp == nullptr)
    {
        GMX_THROW(FileIOError(formatString("Failed to open %s for writing GaMD restart state.", tmpName)));
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

    flushAndSyncGaMDFile(fp, tmpName);
    if (std::fclose(outputFile.release()) != 0)
    {
        GMX_THROW(FileIOError(formatString("Failed to close %s after writing GaMD restart state.", tmpName)));
    }

    // Keep the previous commit record. If the process dies after this update
    // but before the matching .cpt is committed, continuation can select the
    // previous restart whose saved_step matches the checkpoint.
    if (fileExists(finalName))
    {
        std::remove(previousName);
        gmx_file_rename(finalName, previousName);
    }
    gmx_file_rename(tmpName, finalName);
}

namespace
{

// 重启数据必须先通过完整性检查，再交给算法状态。
/*! \internal \brief Checks finite values and invariants of persisted energy statistics. */
bool validStats(const GaMDEnergyStatistics& statistics)
{
    return std::isfinite(statistics.maximum) && std::isfinite(statistics.minimum)
           && std::isfinite(statistics.average) && std::isfinite(statistics.deviation)
           && std::isfinite(statistics.m2) && std::isfinite(statistics.threshold)
           && std::isfinite(statistics.k0) && std::isfinite(statistics.forceConstant)
           && statistics.count >= 0 && statistics.deviation >= 0.0 && statistics.m2 >= 0.0
           && statistics.k0 >= 0.0 && statistics.k0 <= 1.0 && statistics.forceConstant >= 0.0
           && (statistics.count == 0 || statistics.maximum >= statistics.minimum);
}

/*! \internal \brief Checks invariants of an incomplete Welford window. */
bool validWindowStats(const GaMDWindowStatistics& statistics)
{
    return statistics.count >= 0 && std::isfinite(statistics.mean) && std::isfinite(statistics.m2)
           && statistics.m2 >= 0.0;
}

// 先形成完整、已验证的快照，再选择文件和更新运行状态。
/*! \internal \brief Complete candidate state validated before restoring a simulation. */
struct GaMDRestartState
{
    //! Parsed legacy restart version.
    int version = 0;
    //! Step associated with this checkpoint commit record.
    int64_t savedStep = -1;
    //! Whether the saved simulation forced production mode.
    bool productionRestart = false;
    //! Whether the input explicitly supplied the production-mode field.
    bool hasProductionRestart = false;
    //! Persisted physical and stage parameters.
    GaMDParameters parameters;
    //! Persisted potential-energy statistics.
    GaMDEnergyStatistics potentialStatistics;
    //! Persisted dihedral-energy statistics.
    GaMDEnergyStatistics dihedralStatistics;
    //! Incomplete potential-energy statistics window.
    GaMDWindowStatistics potentialWindow;
    //! Incomplete dihedral-energy statistics window.
    GaMDWindowStatistics dihedralWindow;
};

/*! \internal \brief Parses and validates a candidate snapshot without changing live state. */
std::optional<GaMDRestartState> readGaMDRestartState(const char* filename, std::string* error)
{
    error->clear();
    if (!fileExists(filename))
    {
        return std::nullopt;
    }
    GaMDRestartState      state;
    std::set<std::string> fields;
    std::string           line;
    int64_t               lineNumber = 0;
    try
    {
        TextReader input(filename);
        while (input.readLine(&line))
        {
            lineNumber++;
            const std::vector<std::string>    fieldsInLine = splitGaMDInputLine(line);
            const ArrayRef<const std::string> columns(fieldsInLine);
            if (columns.empty())
            {
                continue;
            }
            const std::string& field = columns[0];
            const std::string  location =
                    std::string(filename) + ":" + std::to_string(lineNumber) + ": ";
            if (!fields.insert(field).second)
            {
                *error = location + "Duplicate GaMD restart field '" + field + "'.";
                return std::nullopt;
            }
            int expectedColumns = 0;
            if (field == "version" || field == "saved_step" || field == "production_restart")
            {
                expectedColumns = 2;
            }
            else if (field == "params")
            {
                expectedColumns = 13;
            }
            else if (field == "statP" || field == "statD")
            {
                expectedColumns = 10;
            }
            else if (field == "windowP" || field == "windowD")
            {
                expectedColumns = 4;
            }
            else
            {
                *error = location + "Unknown GaMD restart field '" + field + "'.";
                return std::nullopt;
            }
            if (columns.ssize() < expectedColumns)
            {
                *error = location + "Invalid or incomplete GaMD restart field '" + field + "'.";
                return std::nullopt;
            }
            if (columns.ssize() > expectedColumns)
            {
                *error = location + "Unexpected trailing token in GaMD restart field '" + field
                         + "'.";
                return std::nullopt;
            }
            try
            {
                if (field == "version")
                {
                    state.version = intFromString(columns[1].c_str());
                }
                else if (field == "saved_step")
                {
                    state.savedStep = int64FromString(columns[1].c_str());
                }
                else if (field == "production_restart")
                {
                    const int productionRestart = intFromString(columns[1].c_str());
                    if (productionRestart != 0 && productionRestart != 1)
                    {
                        *error = location + "production_restart must be 0 or 1.";
                        return std::nullopt;
                    }
                    state.productionRestart    = (productionRestart == 1);
                    state.hasProductionRestart = true;
                }
                else if (field == "params")
                {
                    GaMDParameters& parameters     = state.parameters;
                    parameters.boostType           = intFromString(columns[1].c_str());
                    parameters.thresholdMode       = intFromString(columns[2].c_str());
                    parameters.potentialMode       = intFromString(columns[3].c_str());
                    parameters.dihedralMode        = intFromString(columns[4].c_str());
                    parameters.conventionalPrepEnd = int64FromString(columns[5].c_str());
                    parameters.conventionalEnd     = int64FromString(columns[6].c_str());
                    parameters.boostPrepLength     = int64FromString(columns[7].c_str());
                    parameters.boostLength         = int64FromString(columns[8].c_str());
                    parameters.averagingWindow     = int64FromString(columns[9].c_str());
                    parameters.reweightInterval    = int64FromString(columns[10].c_str());
                    parameters.potentialSigmaLimit = doubleFromString(columns[11].c_str());
                    parameters.dihedralSigmaLimit  = doubleFromString(columns[12].c_str());
                }
                else if (field == "statP")
                {
                    state.potentialStatistics = readStatsLine(columns);
                }
                else if (field == "statD")
                {
                    state.dihedralStatistics = readStatsLine(columns);
                }
                else if (field == "windowP")
                {
                    state.potentialWindow = readWindowStatsLine(columns);
                }
                else if (field == "windowD")
                {
                    state.dihedralWindow = readWindowStatsLine(columns);
                }
            }
            catch (const InvalidInputError&)
            {
                *error = location + "Invalid or incomplete GaMD restart field '" + field + "'.";
                return std::nullopt;
            }
        }
        input.close();
    }
    catch (const FileIOError& exception)
    {
        *error = "Failed while reading GaMD restart data: " + std::string(exception.what());
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

} // namespace

void GaMDRuntime::loadRestartStateIfNeeded(int64_t step)
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
            GMX_THROW(InconsistentInputError(
                    "A fresh GaMD run found existing GaMD output/restart files in the working "
                    "directory. Refusing to append unrelated data; use a clean directory or "
                    "continue with the matching checkpoint."));
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
            GMX_THROW(InconsistentInputError(formatString(
                    "irest_gamd found existing output but no GaMD restart matches continued step "
                    "%" PRId64 ".",
                    step)));
        }
        if (!currentState)
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "GaMD restart '%s' is missing or invalid: %s", restartFilename, currentError.c_str())));
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
        GMX_LOG(diagnostics_.logger().warning)
                .asParagraph()
                .appendTextFormatted(
                        "[GaMD WARNING] Current restart does not match checkpoint step %" PRId64
                        "; using validated %s saved_step=%" PRId64 ". %s\n",
                        step,
                        restartFilename,
                        selectedState->savedStep,
                        currentError.c_str());
    }
    else
    {
        const bool haveArtifacts = fileExists("gamd-reweight.dat") || fileExists("gamd-para.dat")
                                   || fileExists("gamd-restart.dat")
                                   || fileExists("gamd-restart-prev.dat");
        if (haveArtifacts || !currentError.empty() || !previousError.empty())
        {
            GMX_THROW(InconsistentInputError(formatString(
                    "No valid GaMD restart matches first continued step %" PRId64 ". %s %s",
                    step,
                    currentError.c_str(),
                    previousError.c_str())));
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
        GMX_THROW(InconsistentInputError(
                formatString("Current gamd.in does not match GaMD restart '%s'; refusing an "
                             "inconsistent continuation (irest_gamd=%d).",
                             restartFilename,
                             parameters_.restartMode)));
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

    // 检查点续跑时把两个文本输出重整到快照步；不带 irest_gamd 的续跑一定走到这里。
    if (checkpointContinuation)
    {
        reconcileGaMDTextFile("gamd-reweight.dat", 9, savedState.savedStep, diagnostics_.logger());
        reconcileGaMDTextFile("gamd-para.dat", 17, savedState.savedStep, diagnostics_.logger());
        if (savedState.savedStep == step)
        {
            suppressTextOutputStep_ = savedState.savedStep;
        }
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
            GMX_THROW(InconsistentInputError(formatString(
                    "irest_gamd = %d requires GaMD parameters that were already initialized, "
                    "but gamd-restart.dat was saved at step %" PRId64 " before ntcmd = %" PRId64 ".",
                    parameters_.restartMode,
                    savedState.savedStep,
                    restartNtcmd)));
        }

        // In irest_gamd mode the stage-control counters can be omitted from
        // gamd.in. Keep the values from the restart file for provenance/output.
        parameters_.conventionalPrepEnd = savedParameters.conventionalPrepEnd;
        parameters_.conventionalEnd     = savedParameters.conventionalEnd;
        parameters_.boostPrepLength     = savedParameters.boostPrepLength;
        parameters_.boostLength         = savedParameters.boostLength;
        parameters_.averagingWindow     = savedParameters.averagingWindow;

        forceProductionRestart_ = true;

        if (inferredProductionRestart)
        {
            GMX_LOG(diagnostics_.logger().info)
                    .asParagraph()
                    .appendTextFormatted(
                            "[GaMD INFO] Inferred production_restart=1 from gamd-restart.dat "
                            "because saved_step=%" PRId64 " is smaller than ntcmd=%" PRId64
                            ". "
                            "This is expected for restart files written by a prior irest_gamd "
                            "production run.\n",
                            savedState.savedStep,
                            restartNtcmd);
        }

        GMX_LOG(diagnostics_.logger().info)
                .asParagraph()
                .appendTextFormatted(
                        "[GaMD INFO] irest_gamd=%d ignores ntcmdprep/ntcmd/ntebprep/nteb "
                        "from gamd.in and uses the values stored in gamd-restart.dat.\n",
                        parameters_.restartMode);

        if (!checkpointContinuation && savedState.savedStep < restartNtcmd + restartNteb)
        {
            GMX_LOG(diagnostics_.logger().warning)
                    .asParagraph()
                    .appendTextFormatted(
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
            GMX_LOG(diagnostics_.logger().info)
                    .asParagraph()
                    .appendTextFormatted(
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
            GMX_LOG(diagnostics_.logger().info)
                    .asParagraph()
                    .appendTextFormatted(
                            "[GaMD INFO] Loaded GaMD restart state from saved_step=%" PRId64
                            " for a "
                            "new simulation with irest_gamd=%d; forcing GaMD production stage "
                            "from step=%" PRId64 ".\n",
                            savedState.savedStep,
                            parameters_.restartMode,
                            step);
        }
    }
    else
    {
        GMX_LOG(diagnostics_.logger().info)
                .asParagraph()
                .appendTextFormatted(
                        "[GaMD INFO] Loaded GaMD restart state from %s saved_step=%" PRId64
                        " for "
                        "continued step=%" PRId64
                        "; text outputs are reconciled to the checkpoint.\n",
                        restartFilename,
                        savedState.savedStep,
                        step);
    }
}

void GaMDRuntime::writeRestartState(int64_t step)
{
    saveRestartState(step);
}

GaMDRuntime::~GaMDRuntime()
{
    try
    {
        closeOutputFiles();
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

void GaMDRuntime::closeOutputFiles()
{
    if (reweightFile_ != nullptr && std::fclose(reweightFile_.release()) != 0)
    {
        GMX_THROW(FileIOError("Failed to close gamd-reweight.dat."));
    }
    if (parameterFile_ != nullptr && std::fclose(parameterFile_.release()) != 0)
    {
        GMX_THROW(FileIOError("Failed to close gamd-para.dat."));
    }
}

bool GaMDRuntime::suppressTextOutputForStep(int64_t step) const
{
    return (suppressTextOutputStep_ == step);
}

namespace
{

/*! \internal \brief Writes enabled statistics or the compatible NaN placeholder columns. */
void writeGaMDEnergyStatisticsColumns(FILE* file, bool enabled, const GaMDEnergyStatistics& statistics)
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

} // namespace

void GaMDRuntime::writeParameterLine(int64_t step, bool flushOutput)
{
    if (parameters_.parameterInterval <= 0)
    {
        return;
    }

    if (stepPlan_.bias.stage < GaMDStage::ConventionalStatistics || step == 0)
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

    FILE* fp = openGaMDTextOutput(&parameterFile_, "gamd-para.dat", c_parameterFileHeader);

    const bool hasP = (parameters_.boostType == 1 || parameters_.boostType == 3);
    const bool hasD = (parameters_.boostType == 2 || parameters_.boostType == 3);

    if (std::fprintf(fp,
                     "%10" PRId64 " %12" PRId64 " %6d",
                     parameters_.parameterInterval,
                     step,
                     static_cast<int>(stepPlan_.bias.stage))
        < 0)
    {
        GMX_THROW(FileIOError(formatString("Failed to write gamd-para.dat at step %" PRId64 ".", step)));
    }
    writeGaMDEnergyStatisticsColumns(fp, hasP, potentialStatistics_);
    writeGaMDEnergyStatisticsColumns(fp, hasD, dihedralStatistics_);
    if (std::fputc('\n', fp) == EOF || (flushOutput && std::fflush(fp) != 0) || std::ferror(fp) != 0)
    {
        GMX_THROW(FileIOError(formatString("Failed to flush gamd-para.dat at step %" PRId64 ".", step)));
    }
}

void GaMDRuntime::writeReweightRecord(int64_t step, const GaMDStepResult& record, bool flushOutput)
{
    if (stepPlan_.bias.stage < GaMDStage::BoostPreparation)
    {
        return;
    }

    if (step == 0)
    {
        return;
    }

    // 输入校验已保证 reweight_nst 为正，取余前不必再检查。
    if (step % parameters_.reweightInterval != 0)
    {
        return;
    }

    if (suppressTextOutputForStep(step))
    {
        return;
    }

    FILE* fp = openGaMDTextOutput(&reweightFile_, "gamd-reweight.dat", c_reweightFileHeader);

    constexpr double kJ_to_kcal = 1.0 / c_cal2Joule;

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
        GMX_THROW(FileIOError(formatString(
                "Failed to write or flush gamd-reweight.dat at step %" PRId64 ".", step)));
    }
}

} // namespace gmx
//! \endcond
