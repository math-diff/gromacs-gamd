/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
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
 * \brief
 * Tests for GaMD behavior.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "config.h"

#include <csignal>
#include <cstdlib>

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#if defined(__linux__)
#    include <sys/resource.h>
#endif

#include "gromacs/applied_forces/gamd/gamddiagnostics.h"
#include "gromacs/applied_forces/gamd/gamdexecution.h"
#include "gromacs/applied_forces/gamd/gamdmdmodule.h"
#include "gromacs/applied_forces/gamd/gamdruntime.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/textreader.h"

#include "testutils/loggertest.h"
#include "testutils/setenv.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

class ScopedWorkingDirectory
{
public:
    explicit ScopedWorkingDirectory(const std::filesystem::path& path) :
        originalPath_(std::filesystem::current_path()), path_(path)
    {
        std::filesystem::create_directories(path_);
        std::filesystem::current_path(path_);
    }

    ~ScopedWorkingDirectory()
    {
        std::error_code errorCode;
        std::filesystem::current_path(originalPath_, errorCode);
        std::filesystem::remove_all(path_, errorCode);
    }

private:
    std::filesystem::path originalPath_;
    std::filesystem::path path_;
};

GaMDExecutionMode selectSupportedGaMDExecutionMode()
{
    return selectGaMDExecutionMode(
            true, true, true, true, true, false, false, false, PressureCoupling::No, PressureCouplingType::Isotropic, 0);
}

std::vector<std::vector<double>> readNumericRows(const std::filesystem::path& path)
{
    std::ifstream                    input(path);
    std::string                      line;
    std::vector<std::vector<double>> rows;

    while (std::getline(input, line))
    {
        std::istringstream lineStream(line);
        std::string        firstToken;
        if (!(lineStream >> firstToken) || firstToken[0] == '#')
        {
            continue;
        }

        std::vector<double> row;
        row.push_back(std::stod(firstToken));

        double value = 0.0;
        while (lineStream >> value)
        {
            row.push_back(value);
        }

        rows.push_back(std::move(row));
    }

    return rows;
}

std::vector<double> readLastNumericColumns(const std::filesystem::path& path)
{
    const auto rows = readNumericRows(path);
    if (rows.empty())
    {
        return {};
    }

    return rows.back();
}

Index countNumericRows(const std::filesystem::path& path)
{
    return ssize(readNumericRows(path));
}

class GaMDTest : public ::testing::Test
{
protected:
    GaMDMDModule gaMDModule_;
};

const std::string c_validGaMDRestart =
        "version 3\nsaved_step 10\nproduction_restart 1\n"
        "params 1 1 1 1 10 20 10 20 10 50 25.104 25.104\n"
        "windowP 0 0 0\nwindowD 0 0 0\n"
        "statP 10 0 5 2 0 2 10 1 0.1\n"
        "statD -1e99 1e99 0 0.001 0 0 0 0 0\n";

void writeGaMDProductionInput(const std::string& restartContents)
{
    std::ofstream input("gamd.in");
    input << "igamd 1\nirest_gamd 1\niE 1\niEP 1\nreweight_nst 50\nsigma0P 6.0\n";
    std::ofstream restart("gamd-restart.dat");
    restart << restartContents;
}

TEST(GaMDParameterTest, LegacyInputCommentsUseNativeTokenization)
{
    const std::vector<std::string> expected{ "ntcmd", "200" };
    EXPECT_EQ(splitGaMDInputLine("ntcmd 200#comment"), expected);
    EXPECT_EQ(splitGaMDInputLine("  ntcmd 200 ; comment\r\n"), expected);
    EXPECT_TRUE(splitGaMDInputLine("; comment only").empty());
    EXPECT_TRUE(splitGaMDInputLine("  # comment only").empty());
}

TEST(GaMDModuleOwnershipTest, LoggerComesFromSimulationSetup)
{
    LoggerTestHelper log;
    log.expectEntryMatchingRegex(MDLogger::LogLevel::Info, "GaMD GPU energy shadow");
    GaMDMDModule       module;
    MDModulesNotifiers notifiers;
    module.subscribeToSimulationSetupNotifications(&notifiers);
    notifiers.simulationSetupNotifier_.notify(log.logger());
    validateGaMDGpuShadowEnergy(module.diagnostics().logger(), 1, GaMDStepResult{}, GaMDStepResult{});
}

TEST_F(GaMDTest, NativeReaderPreservesAttachedRestartComments)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    std::string            restart = c_validGaMDRestart;
    restart.replace(restart.find("version 3\n"),
                    std::string("version 3\n").size(),
                    "version 3#version comment\r\n");
    restart.replace(restart.find("saved_step 10\n"),
                    std::string("saved_step 10\n").size(),
                    "saved_step 10;saved-step comment\n");
    writeGaMDProductionInput(restart);
    {
        std::ofstream input("gamd.in", std::ios::app);
        input << "debug 0#input comment\n";
    }
    const GaMDStepPlan& plan = gaMDModule_.prepareStep(0, 0);
    EXPECT_EQ(plan.stage, GaMDStage::Production);
    EXPECT_DOUBLE_EQ(plan.thresholdP, 10.0);
}

TEST_F(GaMDTest, RuntimeInputErrorsUseProjectExceptions)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    {
        std::ofstream input("gamd.in");
        input << "igamd invalid\n";
    }
    GaMDDiagnostics diagnostics;
    GaMDRuntime     runtime(diagnostics);
    EXPECT_THROW_GMX(runtime.prepareStep(0, 0, false), InvalidInputError);
}

TEST(GaMDDiagnosticsTest, RequestedOutputOpenErrorsAreReported)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory           workingDirectory(directory);
    const std::array<const char*, 3> fileVariables = { "GMX_GAMD_DIH_RAW_DEBUG_FILE",
                                                       "GMX_GAMD_DIH_FORCE_DEBUG_FILE",
                                                       "GMX_GAMD_FORCE_DEBUG_FILE" };
    for (Index fileIndex = 0; fileIndex < ssize(fileVariables); ++fileIndex)
    {
        gmxSetenv(fileVariables[fileIndex], "missing-directory/diagnostic.dat", 1);
        EXPECT_THROW_GMX(GaMDDiagnostics(), FileIOError);
        gmxUnsetenv(fileVariables[fileIndex]);
    }
}

TEST(GaMDDiagnosticsTest, RequestedOutputWriteErrorsAreReported)
{
#if GMX_NATIVE_WINDOWS
    GTEST_SKIP() << "The /dev/full write-error check requires a POSIX device.";
#else
    if (!std::filesystem::exists("/dev/full"))
    {
        GTEST_SKIP() << "The /dev/full device is unavailable.";
    }
    const std::array<const char*, 3> fileVariables = { "GMX_GAMD_DIH_RAW_DEBUG_FILE",
                                                       "GMX_GAMD_DIH_FORCE_DEBUG_FILE",
                                                       "GMX_GAMD_FORCE_DEBUG_FILE" };
    for (Index fileIndex = 0; fileIndex < ssize(fileVariables); ++fileIndex)
    {
        gmxSetenv(fileVariables[fileIndex], "/dev/full", 1);
        EXPECT_THROW_GMX(GaMDDiagnostics(), FileIOError);
        gmxUnsetenv(fileVariables[fileIndex]);
    }
#endif
}

TEST(GaMDDiagnosticsTest, DihedralSnapshotIsWeightedOrderedAndWrittenOnce)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_FILE", "dihedral.dat", 1);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP", "7", 1);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_BEGIN", "1", 1);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_COUNT", "2147483647", 1);
    gmxSetenv("GMX_GAMD_DIH_RAW_DEBUG_FILE", "raw-calls.dat", 1);
    {
        GaMDDiagnostics           diagnostics;
        const std::array<RVec, 3> rawForces = { RVec(4, 0, 0), RVec(0, 8, 0), RVec(0, 0, 12) };
        const std::array<int, 3>  topologyIndices = { 2, 0, 1 };
        diagnostics.prepareDihedralStep(7);
        // 原始调用可以出现多次，但不能再改变加权快照的数值。
        diagnostics.addRawDihedralForces(
                0, rawForces[0], 1, rawForces[1], 2, rawForces[2], 0, rawForces[0]);
        diagnostics.addRawDihedralForces(
                0, rawForces[0], 1, rawForces[1], 2, rawForces[2], 0, rawForces[0]);
        EXPECT_FALSE(diagnostics.dihedralForceOutputRequested(6));
        EXPECT_TRUE(diagnostics.dihedralForceOutputRequested(7));
        diagnostics.dumpDihedralForces(7, rawForces, 0.25, false, topologyIndices);
        EXPECT_TRUE(diagnostics.dihedralForceOutputRequested(7));
        diagnostics.dumpDihedralForces(6, rawForces, 0.25, true, topologyIndices);
        diagnostics.dumpDihedralForces(7, rawForces, 0.25, true, topologyIndices);
        EXPECT_FALSE(diagnostics.dihedralForceOutputRequested(7));
        diagnostics.dumpDihedralForces(7, rawForces, 0.5, true, topologyIndices);
        diagnostics.finishDihedralStep(7, true);
        EXPECT_EQ(rawForces[0][XX], 4);
        EXPECT_EQ(rawForces[2][ZZ], 12);
    }
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_FILE");
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP");
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_BEGIN");
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_COUNT");
    gmxUnsetenv("GMX_GAMD_DIH_RAW_DEBUG_FILE");
    const auto rows = readNumericRows("dihedral.dat");
    ASSERT_EQ(rows.size(), 2U);
    ASSERT_EQ(rows[0].size(), 6U);
    ASSERT_EQ(rows[1].size(), 6U);
    EXPECT_EQ(rows[0][0], 7);
    EXPECT_EQ(rows[0][1], 1);
    EXPECT_EQ(rows[0][4], 3);
    EXPECT_EQ(rows[0][5], 3);
    EXPECT_EQ(rows[1][1], 2);
    EXPECT_EQ(rows[1][2], 1);
    EXPECT_EQ(rows[1][5], 1);
    EXPECT_EQ(readNumericRows("raw-calls.dat").size(), 8U);
}

TEST(GaMDDiagnosticsTest, MissingRequestedDihedralSnapshotIsRejected)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_FILE", "dihedral.dat", 1);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP", "7", 1);
    {
        GaMDDiagnostics diagnostics;
        EXPECT_THROW_GMX(diagnostics.finishDihedralStep(7, true), InternalError);
    }
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_FILE");
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP");
}

TEST(GaMDDiagnosticsTest, DihedralOutputStagesOnlyTheRequestedStep)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    writeGaMDProductionInput(c_validGaMDRestart);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_FILE", "dihedral.dat", 1);
    gmxSetenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP", "7", 1);
    {
        GaMDMDModule module;
        module.setBufferedProductionOutputEnabled(true);
        module.prepareStep(6, 0);
        EXPECT_FALSE(module.requiresHostEnergyThisStep(6));
        module.prepareStep(7, 0);
        EXPECT_TRUE(module.requiresHostEnergyThisStep(7));
        const std::array<RVec, 1> forces = { RVec(1, 2, 3) };
        module.diagnostics().dumpDihedralForces(7, forces, 1.0, true);
        module.prepareStep(8, 0);
        EXPECT_FALSE(module.requiresHostEnergyThisStep(8));
    }
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_FILE");
    gmxUnsetenv("GMX_GAMD_DIH_FORCE_DEBUG_STEP");
}

TEST(GaMDDiagnosticsTest, FinalForceSnapshotKeepsValuesAndClampsCount)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    gmxSetenv("GMX_GAMD_FORCE_DEBUG_FILE", "final.dat", 1);
    gmxSetenv("GMX_GAMD_FORCE_DEBUG_STEP", "7", 1);
    gmxSetenv("GMX_GAMD_FORCE_DEBUG_BEGIN", "1", 1);
    gmxSetenv("GMX_GAMD_FORCE_DEBUG_COUNT", "2147483647", 1);
    {
        GaMDDiagnostics           diagnostics;
        const std::array<RVec, 3> forces = { RVec(4, 0, 0), RVec(0, 8, 0), RVec(0, 0, 12) };
        diagnostics.dumpFinalForces(7, forces, true);
    }
    gmxUnsetenv("GMX_GAMD_FORCE_DEBUG_FILE");
    gmxUnsetenv("GMX_GAMD_FORCE_DEBUG_STEP");
    gmxUnsetenv("GMX_GAMD_FORCE_DEBUG_BEGIN");
    gmxUnsetenv("GMX_GAMD_FORCE_DEBUG_COUNT");
    const auto rows = readNumericRows("final.dat");
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0][1], 1);
    EXPECT_EQ(rows[0][3], 8);
    EXPECT_EQ(rows[1][1], 2);
    EXPECT_EQ(rows[1][4], 12);
}

TEST(GaMDDiagnosticsTest, NonFiniteEnergyStateIsRejected)
{
    const GaMDStepResult        reference;
    const std::array<double, 3> invalidValues = { std::numeric_limits<double>::quiet_NaN(),
                                                  std::numeric_limits<double>::infinity(),
                                                  -std::numeric_limits<double>::infinity() };
    for (Index valueIndex = 0; valueIndex < ssize(invalidValues); ++valueIndex)
    {
        // 六种物理量分别破坏一次，主机和设备两侧都必须拒绝。
        std::array<GaMDStepResult, 6> invalidStates = {};
        invalidStates[0].potentialEnergy            = invalidValues[valueIndex];
        invalidStates[1].dihedralEnergy             = invalidValues[valueIndex];
        invalidStates[2].scaleP                     = invalidValues[valueIndex];
        invalidStates[3].scaleD                     = invalidValues[valueIndex];
        invalidStates[4].boostP                     = invalidValues[valueIndex];
        invalidStates[5].boostD                     = invalidValues[valueIndex];
        for (Index stateIndex = 0; stateIndex < ssize(invalidStates); ++stateIndex)
        {
            EXPECT_THROW_GMX(
                    validateGaMDGpuShadowEnergy(MDLogger{}, 1, reference, invalidStates[stateIndex]),
                    SimulationInstabilityError);
            EXPECT_THROW_GMX(validateGaMDGpuShadowEnergy(MDLogger{}, 1, invalidStates[stateIndex], reference),
                             SimulationInstabilityError);
        }
    }
}

TEST(GaMDDiagnosticsTest, NonFiniteVirialComponentsAreRejected)
{
    tensor                      reference     = {};
    std::array<real, DIM * DIM> candidate     = {};
    const std::array<real, 3>   invalidValues = { std::numeric_limits<real>::quiet_NaN(),
                                                  std::numeric_limits<real>::infinity(),
                                                  -std::numeric_limits<real>::infinity() };
    for (Index valueIndex = 0; valueIndex < ssize(invalidValues); ++valueIndex)
    {
        for (int componentIndex = 0; componentIndex < DIM * DIM; ++componentIndex)
        {
            candidate[componentIndex] = invalidValues[valueIndex];
            EXPECT_THROW_GMX(validateGaMDGpuShadowVirial(MDLogger{}, 1, reference, candidate),
                             SimulationInstabilityError);
            candidate[componentIndex]                             = 0;
            reference[componentIndex / DIM][componentIndex % DIM] = invalidValues[valueIndex];
            EXPECT_THROW_GMX(validateGaMDGpuShadowVirial(MDLogger{}, 1, reference, candidate),
                             SimulationInstabilityError);
            reference[componentIndex / DIM][componentIndex % DIM] = 0;
        }
    }
}

TEST(GaMDDiagnosticsTest, NonFiniteForcesAndExcessiveDifferencesAreRejected)
{
    const std::array<RVec, 1> referenceForces = { RVec(0, 0, 0) };
    std::array<RVec, 1>       candidateForces = { RVec(0, 0, 0) };
    candidateForces[0][ZZ]                    = std::numeric_limits<real>::quiet_NaN();
    EXPECT_THROW_GMX(validateGaMDGpuShadowForces(
                             MDLogger{}, "test", 1, referenceForces, candidateForces, 0.005, 0.0001),
                     SimulationInstabilityError);
    EXPECT_THROW_GMX(validateGaMDGpuShadowForces(
                             MDLogger{}, "test", 1, candidateForces, referenceForces, 0.005, 0.0001),
                     SimulationInstabilityError);
    candidateForces[0][ZZ] = 1;
    EXPECT_THROW_GMX(validateGaMDGpuShadowForces(
                             MDLogger{}, "test", 1, referenceForces, candidateForces, 0.005, 0.0001),
                     SimulationInstabilityError);

    GaMDStepResult candidateEnergy;
    candidateEnergy.potentialEnergy = 1;
    EXPECT_THROW_GMX(validateGaMDGpuShadowEnergy(MDLogger{}, 1, GaMDStepResult{}, candidateEnergy),
                     SimulationInstabilityError);
    tensor                      referenceVirial = {};
    std::array<real, DIM * DIM> candidateVirial = {};
    candidateVirial.back()                      = 1;
    EXPECT_THROW_GMX(validateGaMDGpuShadowVirial(MDLogger{}, 1, referenceVirial, candidateVirial),
                     SimulationInstabilityError);
}

TEST(GaMDDiagnosticsTest, FiniteShadowsWithinExistingTolerancesAreAccepted)
{
    GaMDStepResult candidateEnergy;
    candidateEnergy.potentialEnergy = 0.1;
    candidateEnergy.dihedralEnergy  = 0.0005;
    candidateEnergy.scaleP += 0.00001;
    candidateEnergy.scaleD -= 0.00001;
    candidateEnergy.boostP = 0.05;
    validateGaMDGpuShadowEnergy(MDLogger{}, 1, GaMDStepResult{}, candidateEnergy);
    tensor                      referenceVirial = {};
    std::array<real, DIM * DIM> candidateVirial = {};
    candidateVirial.back()                      = 0.005_real;
    validateGaMDGpuShadowVirial(MDLogger{}, 1, referenceVirial, candidateVirial);
    const std::array<RVec, 1> referenceForces = { RVec(0, 0, 0) };
    const std::array<RVec, 1> candidateForces = { RVec(0.0001_real, 0, 0) };
    validateGaMDGpuShadowForces(MDLogger{}, "test", 1, referenceForces, candidateForces, 0.005, 0.0001);
}

TEST(GaMDParameterTest, StageBoundariesAreCheckedBeforeAdding)
{
    GaMDParameters parameters;
    parameters.conventionalPrepEnd = 0;
    parameters.conventionalEnd     = std::numeric_limits<int64_t>::max() - 1;
    parameters.boostPrepLength     = 0;
    parameters.boostLength         = 1;
    parameters.averagingWindow     = 1;
    EXPECT_TRUE(gaMDStageParameterError(parameters).empty());
    parameters.boostLength = 2;
    EXPECT_NE(gaMDStageParameterError(parameters).find("64-bit"), std::string::npos);
    parameters.conventionalEnd = 20;
    parameters.boostLength     = 20;
    parameters.averagingWindow = 0;
    EXPECT_NE(gaMDStageParameterError(parameters).find("ntave>0"), std::string::npos);
    parameters.averagingWindow = 3;
    EXPECT_NE(gaMDStageParameterError(parameters).find("multiples of ntave"), std::string::npos);
}

TEST_F(GaMDTest, FreshInputRejectsStageAndRunLengthOverflow)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    {
        std::ofstream input("gamd.in");
        input << "igamd 1\niE 1\niEP 1\nreweight_nst 50\nsigma0P 6\n"
                 "ntcmdprep 0\nntcmd 9223372036854775806\n"
                 "ntebprep 0\nnteb 2\nntave 1\n";
    }
    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "64-bit");
    {
        std::ofstream input("gamd.in");
        input << "igamd 1\niE 1\niEP 1\nreweight_nst 50\nsigma0P 6\n"
                 "ntcmdprep 0\nntcmd 2\nntebprep 0\nnteb 2\nntave 1\n";
    }
    gaMDModule_.prepareStep(0, 0);
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            gaMDModule_.warnIfRunTooShort(std::numeric_limits<int64_t>::max() - 1, 2, 0), "64-bit");
}

TEST_F(GaMDTest, RestartRejectsInvalidStageCounters)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory           workingDirectory(directory);
    const std::array<std::string, 3> invalidParameters = {
        "params 1 1 1 1 10 20 10 20 0 50 25.104 25.104",
        "params 1 1 1 1 0 9223372036854775806 0 2 1 50 25.104 25.104",
        "params 1 1 1 1 10 20 10 20 3 50 25.104 25.104"
    };
    for (Index caseIndex = 0; caseIndex < ssize(invalidParameters); ++caseIndex)
    {
        std::string restart        = c_validGaMDRestart;
        const auto  parameterStart = restart.find("params ");
        const auto  parameterEnd   = restart.find('\n', parameterStart);
        restart.replace(parameterStart, parameterEnd - parameterStart, invalidParameters[caseIndex]);
        writeGaMDProductionInput(restart);
        GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "invalid");
    }
}

TEST_F(GaMDTest, ProductionRestartMayIgnoreZeroStageInput)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    writeGaMDProductionInput(c_validGaMDRestart);
    {
        std::ofstream input("gamd.in", std::ios::app);
        input << "ntcmdprep 0\nntcmd 0\nntebprep 0\nnteb 0\nntave 0\n";
    }
    const GaMDStepPlan& plan = gaMDModule_.prepareStep(0, 0);
    EXPECT_EQ(plan.stage, GaMDStage::Production);
    EXPECT_DOUBLE_EQ(plan.thresholdP, 10.0);
}

TEST_F(GaMDTest, StageTransitionsPreserveNumericOutput)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    {
        std::ofstream input("gamd.in");
        input << "igamd 1\niE 1\niEP 1\nreweight_nst 1\npara_nst 1\nsigma0P 6\n"
                 "ntcmdprep 2\nntcmd 4\nntebprep 2\nnteb 4\nntave 2\n";
    }
    // 固定旧协议的阶段数字，同时覆盖每个边界的前后两侧。
    const std::array<int, 10> expectedStages = { 1, 1, 1, 2, 2, 3, 3, 4, 4, 5 };
    for (Index stepIndex = 0; stepIndex < ssize(expectedStages); ++stepIndex)
    {
        const GaMDStepPlan& plan = gaMDModule_.prepareStep(stepIndex, 0);
        EXPECT_EQ(static_cast<int>(plan.stage), expectedStages[stepIndex]);
        gaMDModule_.finalizeCurrentStep(stepIndex, 0, stepIndex, 0);
    }
    const auto rows = readNumericRows("gamd-para.dat");
    ASSERT_EQ(rows.size(), 7U);
    for (Index rowIndex = 0; rowIndex < ssize(rows); ++rowIndex)
    {
        ASSERT_GE(rows[rowIndex].size(), 3U);
        const auto stepIndex = static_cast<Index>(rows[rowIndex][1]);
        ASSERT_LT(stepIndex, expectedStages.size());
        EXPECT_EQ(rows[rowIndex][2], expectedStages[stepIndex]);
    }
}

TEST(GaMDDiagnosticsTest, InvalidForceOverridesAreRejected)
{
    const std::array<const char*, 5> invalidValues{ "abc", "0.8junk", "nan", "inf", "1e999" };
    const std::array<const char*, 4> variableNames{ "GMX_GAMD_FORCE_OVERRIDE_SCALEP",
                                                    "GMX_GAMD_FORCE_OVERRIDE_SCALED",
                                                    "GMX_GAMD_FORCE_OVERRIDE_BOOSTP",
                                                    "GMX_GAMD_FORCE_OVERRIDE_BOOSTD" };
    for (Index variableIndex = 0; variableIndex < ssize(variableNames); variableIndex++)
    {
        gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP", "0.8", 1);
        for (Index valueIndex = 0; valueIndex < ssize(invalidValues); valueIndex++)
        {
            gmxSetenv(variableNames[variableIndex], invalidValues[valueIndex], 1);
            EXPECT_THROW_GMX(GaMDDiagnostics().forceOverrides(), InvalidInputError);
        }
        gmxUnsetenv(variableNames[variableIndex]);
    }
#if !GMX_NATIVE_WINDOWS
    // Windows 的环境变量接口把空值当作未设置，Unix 则应拒绝空字符串。
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP", "", 1);
    EXPECT_THROW_GMX(GaMDDiagnostics().forceOverrides(), InvalidInputError);
#endif
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
}

TEST(GaMDDiagnosticsTest, ValidOverridesAndIntegerValidationUseNativeConverters)
{
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP", "0.8", 1);
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALED", "0.6", 1);
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTP", "4.0", 1);
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTD", "2.0", 1);
    {
        GaMDDiagnostics           diagnostics;
        const GaMDForceOverrides& overrides = diagnostics.forceOverrides();
        EXPECT_TRUE(overrides.enabled);
        EXPECT_DOUBLE_EQ(overrides.totalScale, 0.8);
        EXPECT_DOUBLE_EQ(overrides.dihedralScale, 0.6);
        EXPECT_DOUBLE_EQ(overrides.potentialBoost, 4.0);
        EXPECT_DOUBLE_EQ(overrides.dihedralBoost, 2.0);

        // 第一次读取后配置固定；下一个模拟实例才采用新的环境值。
        gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP", "0.5", 1);
        EXPECT_DOUBLE_EQ(diagnostics.forceOverrides().totalScale, 0.8);
        EXPECT_DOUBLE_EQ(GaMDDiagnostics().forceOverrides().totalScale, 0.5);
    }
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALED");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTD");
    gmxSetenv("GMX_GAMD_FORCE_DEBUG_BEGIN", "1bad", 1);
    EXPECT_THROW(GaMDDiagnostics(), InvalidInputError);
    gmxUnsetenv("GMX_GAMD_FORCE_DEBUG_BEGIN");
}

TEST(GaMDDiagnosticsTest, UnusedOverridesDoNotAffectModuleConstruction)
{
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP", "invalid", 1);
    EXPECT_NO_THROW(GaMDMDModule());
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
}

TEST_F(GaMDTest, ForceOverrideSnapshotGovernsStagingAndHostResult)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    writeGaMDProductionInput(c_validGaMDRestart);
    gaMDModule_.setBufferedProductionOutputEnabled(true);
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP", "0.5", 1);
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_SCALED", "1.0", 1);
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTP", "4.0", 1);
    gmxSetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTD", "2.0", 1);
    EXPECT_TRUE(gaMDModule_.prepareStep(0, 0).requiresHostEnergy);
    EXPECT_FALSE(gaMDModule_.bufferedProductionOutputEnabled());

    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALED");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTD");
    gaMDModule_.finalizeCurrentStep(0, 0, 1000.0, 200.0);
    EXPECT_DOUBLE_EQ(gaMDModule_.stepResult().scaleP, 0.5);
    EXPECT_DOUBLE_EQ(gaMDModule_.stepResult().scaleD, 1.0);
    EXPECT_DOUBLE_EQ(gaMDModule_.stepResult().totalBoost(), 6.0);
    EXPECT_DOUBLE_EQ(gaMDModule_.stepResult().potentialEnergy, 1000.0);
    EXPECT_DOUBLE_EQ(gaMDModule_.stepResult().dihedralEnergy, 200.0);
    EXPECT_TRUE(gaMDModule_.prepareStep(1, 0).requiresHostEnergy);
}

TEST(GaMDModuleOwnershipTest, DiagnosticsAndHostWorkspaceAreInstanceOwned)
{
    LoggerTestHelper firstLog;
    LoggerTestHelper secondLog;
    firstLog.expectEntryMatchingRegex(MDLogger::LogLevel::Info, "nstcalcenergy = 1");
    firstLog.expectEntryMatchingRegex(MDLogger::LogLevel::Info, "GaMD output, log");
    secondLog.expectEntryMatchingRegex(MDLogger::LogLevel::Info, "nstcalcenergy = 1");
    secondLog.expectEntryMatchingRegex(MDLogger::LogLevel::Info, "standard energy/log");
    GaMDMDModule firstModule;
    GaMDMDModule secondModule;

    firstModule.diagnostics().setLogger(firstLog.logger());
    secondModule.diagnostics().setLogger(secondLog.logger());
    firstModule.diagnostics().logEnergyCadence(1, true);
    firstModule.diagnostics().logEnergyCadence(1, true);
    secondModule.diagnostics().logEnergyCadence(1, true);
    firstModule.diagnostics().logResidentEnergy(false, true);
    firstModule.diagnostics().logResidentEnergy(false, true);
    secondModule.diagnostics().logResidentEnergy(true, true);

    firstModule.diagnostics().hostWorkspace().cpuDihedralForces.resize(1);
    EXPECT_TRUE(secondModule.diagnostics().hostWorkspace().cpuDihedralForces.empty());
}

TEST_F(GaMDTest, Stage4PotentialStatisticsUseBoostedTotalPotential)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALED");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTD");

    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "para_nst 1\n";
        output << "reweight_nst 1\n";
        output << "sigma0P 6.0\n";
    }

    gaMDModule_.setCheckpointingThisStep(false);

    constexpr double c_step1Potential = 0.0;
    constexpr double c_step2Potential = 10.0;
    constexpr double c_step3Potential = 0.0;
    constexpr double c_step4Potential = 0.0;
    constexpr double c_dihedralEnergy = 0.0;

    gaMDModule_.prepareStep(1, 0);
    gaMDModule_.finalizeCurrentStep(1, 0, c_step1Potential, c_dihedralEnergy);

    gaMDModule_.prepareStep(2, 0);
    gaMDModule_.finalizeCurrentStep(2, 0, c_step2Potential, c_dihedralEnergy);

    gaMDModule_.prepareStep(3, 0);
    gaMDModule_.finalizeCurrentStep(3, 0, c_step3Potential, c_dihedralEnergy);

    gaMDModule_.prepareStep(4, 0);
    gaMDModule_.finalizeCurrentStep(4, 0, c_step4Potential, c_dihedralEnergy);

    const auto columns = readLastNumericColumns("gamd-para.dat");
    ASSERT_GE(columns.size(), 10U);
    EXPECT_EQ(columns[1], 4.0);
    EXPECT_EQ(columns[2], 4.0);

    // Amber updates stage-4 total-potential statistics using the current boosted
    // total potential. With the setup above, steps 3-4 each have a boost of 5.
    EXPECT_REAL_EQ_TOL(columns[5], 5.0, absoluteTolerance(1e-12));
}

TEST_F(GaMDTest, CheckpointReplayDoesNotDuplicateTextOutputForSavedStep)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALED");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTP");
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_BOOSTD");

    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "para_nst 1\n";
        output << "reweight_nst 1\n";
        output << "sigma0P 6.0\n";
    }

    constexpr double c_dihedralEnergy = 0.0;

    gaMDModule_.setCheckpointingThisStep(false);
    gaMDModule_.prepareStep(1, 0);
    gaMDModule_.finalizeCurrentStep(1, 0, 0.0, c_dihedralEnergy);
    gaMDModule_.prepareStep(2, 0);
    gaMDModule_.finalizeCurrentStep(2, 0, 10.0, c_dihedralEnergy);
    gaMDModule_.prepareStep(3, 0);
    gaMDModule_.finalizeCurrentStep(3, 0, 0.0, c_dihedralEnergy);

    gaMDModule_.prepareStep(4, 0);
    gaMDModule_.setCheckpointingThisStep(true);
    gaMDModule_.writeRestartState(4);
    gaMDModule_.finalizeCurrentStep(4, 0, 0.0, c_dihedralEnergy);
    gaMDModule_.setCheckpointingThisStep(false);

    const Index paraRowsAfterCheckpoint     = countNumericRows("gamd-para.dat");
    const Index reweightRowsAfterCheckpoint = countNumericRows("gamd-reweight.dat");
    ASSERT_GT(paraRowsAfterCheckpoint, 0U);
    ASSERT_GT(reweightRowsAfterCheckpoint, 0U);

    auto paraColumns = readLastNumericColumns("gamd-para.dat");
    ASSERT_GT(paraColumns.size(), 1U);
    EXPECT_EQ(paraColumns[1], 4.0);

    auto reweightColumns = readLastNumericColumns("gamd-reweight.dat");
    ASSERT_GT(reweightColumns.size(), 1U);
    EXPECT_EQ(reweightColumns[1], 4.0);

    // Simulate an abrupt stop after text output advanced beyond the last
    // committed checkpoint, including a torn final record.
    gaMDModule_.prepareStep(5, 0);
    gaMDModule_.finalizeCurrentStep(5, 0, 0.0, c_dihedralEnergy);
    gaMDModule_.prepareStep(6, 0);
    gaMDModule_.finalizeCurrentStep(6, 0, 0.0, c_dihedralEnergy);

    // Simulate interruption after GaMD committed a newer restart but before
    // GROMACS replaced the older .cpt. The rotated previous restart must keep
    // the older checkpoint recoverable.
    gaMDModule_.setCheckpointingThisStep(true);
    gaMDModule_.writeRestartState(6);
    gaMDModule_.setCheckpointingThisStep(false);
    ASSERT_TRUE(std::filesystem::exists("gamd-restart-prev.dat"));
    {
        std::ofstream paraOutput("gamd-para.dat", std::ios::app);
        paraOutput << "1 7 partial\n";
        std::ofstream reweightOutput("gamd-reweight.dat", std::ios::app);
        reweightOutput << "1 7 partial\n";
    }
    EXPECT_GT(countNumericRows("gamd-para.dat"), paraRowsAfterCheckpoint);
    EXPECT_GT(countNumericRows("gamd-reweight.dat"), reweightRowsAfterCheckpoint);

    gaMDModule_.resetStateForTesting();
    gaMDModule_.prepareStep(4, 0);
    gaMDModule_.finalizeCurrentStep(4, 0, 0.0, c_dihedralEnergy);

    EXPECT_EQ(countNumericRows("gamd-para.dat"), paraRowsAfterCheckpoint);
    EXPECT_EQ(countNumericRows("gamd-reweight.dat"), reweightRowsAfterCheckpoint);

    gaMDModule_.prepareStep(5, 0);
    gaMDModule_.finalizeCurrentStep(5, 0, 0.0, c_dihedralEnergy);

    EXPECT_EQ(countNumericRows("gamd-para.dat"), paraRowsAfterCheckpoint + 1);
    EXPECT_EQ(countNumericRows("gamd-reweight.dat"), reweightRowsAfterCheckpoint + 1);

    paraColumns = readLastNumericColumns("gamd-para.dat");
    ASSERT_GT(paraColumns.size(), 1U);
    EXPECT_EQ(paraColumns[1], 5.0);

    reweightColumns = readLastNumericColumns("gamd-reweight.dat");
    ASSERT_GT(reweightColumns.size(), 1U);
    EXPECT_EQ(reweightColumns[1], 5.0);
}

TEST_F(GaMDTest, IndependentModuleInstancesDoNotShareStepResults)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "para_nst 0\n";
        output << "reweight_nst 1\n";
        output << "sigma0P 6.0\n";
    }

    GaMDMDModule secondModule;
    gaMDModule_.prepareStep(1, 0);
    gaMDModule_.finalizeCurrentStep(1, 0, 12.0, 3.0);
    secondModule.prepareStep(1, 0);

    EXPECT_DOUBLE_EQ(12.0, gaMDModule_.stepResult().potentialEnergy);
    EXPECT_DOUBLE_EQ(0.0, secondModule.stepResult().potentialEnergy);
    EXPECT_DOUBLE_EQ(1.0, secondModule.stepResult().scaleP);
}

TEST_F(GaMDTest, ResidentProductionEnergyStagesOnlyForHostConsumers)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "para_nst 50\n";
        output << "reweight_nst 50\n";
        output << "sigma0P 6.0\n";
    }

    gaMDModule_.setCheckpointingThisStep(false);
    gaMDModule_.prepareStep(2, 0);
    EXPECT_TRUE(gaMDModule_.requiresHostEnergyThisStep(2));

    gaMDModule_.prepareStep(49, 0);
    EXPECT_FALSE(gaMDModule_.requiresHostEnergyThisStep(49));

    gaMDModule_.prepareStep(50, 0);
    EXPECT_TRUE(gaMDModule_.requiresHostEnergyThisStep(50));

    gaMDModule_.prepareStep(51, 0);
    gaMDModule_.setCheckpointingThisStep(true);
    EXPECT_TRUE(gaMDModule_.requiresHostEnergyThisStep(51));
    gaMDModule_.setCheckpointingThisStep(false);
}

TEST_F(GaMDTest, BufferedProductionOutputDefersTextCadenceButNotCheckpointing)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    gmxUnsetenv("GMX_GAMD_FORCE_OVERRIDE_SCALEP");
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "para_nst 50\n";
        output << "reweight_nst 50\n";
        output << "sigma0P 6.0\n";
    }

    gaMDModule_.setCheckpointingThisStep(false);
    gaMDModule_.setBufferedProductionOutputEnabled(true);
    gaMDModule_.prepareStep(50, 0);
    EXPECT_TRUE(gaMDModule_.bufferedProductionOutputEnabled());
    EXPECT_FALSE(gaMDModule_.requiresHostEnergyThisStep(50));

    gaMDModule_.setCheckpointingThisStep(true);
    gaMDModule_.prepareStep(51, 0);
    EXPECT_TRUE(gaMDModule_.requiresHostEnergyThisStep(51));
    gaMDModule_.setCheckpointingThisStep(false);
    gaMDModule_.setBufferedProductionOutputEnabled(false);
}

TEST_F(GaMDTest, GaMDCpuUpdateSelectsCpuReference)
{
    EXPECT_EQ(GaMDExecutionMode::CpuReference,
              selectGaMDExecutionMode(true,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      false,
                                      PressureCoupling::ParrinelloRahman,
                                      PressureCouplingType::Anisotropic,
                                      1));
}

TEST_F(GaMDTest, CpuGaMDRejectsDomainDecompositionAndMts)
{
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, false, false, false, false, true, false, false, PressureCoupling::No, PressureCouplingType::Isotropic, 0),
            InconsistentInputError);
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, false, false, false, false, false, false, true, PressureCoupling::No, PressureCouplingType::Isotropic, 0),
            InconsistentInputError);
}

TEST_F(GaMDTest, NonGaMDGpuUpdateRetainsNormalExecutionMode)
{
    EXPECT_EQ(GaMDExecutionMode::CpuReference,
              selectGaMDExecutionMode(false,
                                      false,
                                      false,
                                      false,
                                      true,
                                      true,
                                      true,
                                      true,
                                      PressureCoupling::ParrinelloRahman,
                                      PressureCouplingType::Anisotropic,
                                      1));
}

#if GMX_GPU_CUDA
TEST_F(GaMDTest, GaMDGpuExecutionModeSelectsResidentPathAutomatically)
{
    EXPECT_EQ(GaMDExecutionMode::GpuResident, selectSupportedGaMDExecutionMode());
    EXPECT_STREQ("GPU resident", gaMDExecutionModeName(GaMDExecutionMode::GpuResident));
}

TEST_F(GaMDTest, GaMDGpuExecutionModeRejectsUnsupportedWorkloads)
{
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, false, true, true, true, false, false, false, PressureCoupling::No, PressureCouplingType::Isotropic, 0),
            InconsistentInputError);
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, true, false, true, true, false, false, false, PressureCoupling::No, PressureCouplingType::Isotropic, 0),
            InconsistentInputError);
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, true, true, false, true, false, false, false, PressureCoupling::No, PressureCouplingType::Isotropic, 0),
            InconsistentInputError);
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, true, true, true, true, true, false, false, PressureCoupling::No, PressureCouplingType::Isotropic, 0),
            InconsistentInputError);
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, true, true, true, true, false, false, true, PressureCoupling::No, PressureCouplingType::Isotropic, 0),
            InconsistentInputError);
    EXPECT_THROW_GMX(selectGaMDExecutionMode(true,
                                             true,
                                             true,
                                             true,
                                             true,
                                             false,
                                             false,
                                             false,
                                             PressureCoupling::ParrinelloRahman,
                                             PressureCouplingType::Isotropic,
                                             0),
                     InconsistentInputError);
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, true, true, true, true, false, false, false, PressureCoupling::CRescale, PressureCouplingType::Anisotropic, 0),
            InconsistentInputError);
    EXPECT_THROW_GMX(
            selectGaMDExecutionMode(
                    true, true, true, true, true, false, false, false, PressureCoupling::No, PressureCouplingType::Isotropic, 10),
            InconsistentInputError);
}

TEST_F(GaMDTest, GaMDGpuExecutionModeAcceptsSupportedCRescaleTypes)
{
    for (const auto type : { PressureCouplingType::Isotropic, PressureCouplingType::SemiIsotropic })
    {
        EXPECT_EQ(GaMDExecutionMode::GpuResident,
                  selectGaMDExecutionMode(
                          true, true, true, true, true, false, false, false, PressureCoupling::CRescale, type, 0));
    }
}
#else
TEST_F(GaMDTest, GaMDGpuExecutionModeRejectsNonCudaBuild)
{
    EXPECT_THROW_GMX(selectSupportedGaMDExecutionMode(), InconsistentInputError);
}
#endif

TEST_F(GaMDTest, IrestProductionCheckpointCanContinueWithoutDuplicateRows)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "para_nst 1\n";
        output << "reweight_nst 1\n";
        output << "sigma0P 6.0\n";
    }

    for (long step = 1; step <= 4; step++)
    {
        gaMDModule_.prepareStep(step, 0);
        double potentialEnergy = 0.0;
        if (step == 2)
        {
            potentialEnergy = 10.0;
        }
        gaMDModule_.finalizeCurrentStep(step, 0, potentialEnergy, 0.0);
    }
    gaMDModule_.setCheckpointingThisStep(true);
    gaMDModule_.writeRestartState(4);
    gaMDModule_.setCheckpointingThisStep(false);

    gaMDModule_.resetStateForTesting();
    std::filesystem::remove("gamd-para.dat");
    std::filesystem::remove("gamd-reweight.dat");
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "irest_gamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "para_nst 1\n";
        output << "reweight_nst 1\n";
        output << "sigma0P 6.0\n";
    }

    gaMDModule_.prepareStep(0, 0);
    gaMDModule_.finalizeCurrentStep(0, 0, 0.0, 0.0);
    gaMDModule_.prepareStep(1, 0);
    gaMDModule_.setCheckpointingThisStep(true);
    gaMDModule_.writeRestartState(1);
    gaMDModule_.finalizeCurrentStep(1, 0, 0.0, 0.0);
    gaMDModule_.setCheckpointingThisStep(false);

    const Index paraRowsAtCheckpoint     = countNumericRows("gamd-para.dat");
    const Index reweightRowsAtCheckpoint = countNumericRows("gamd-reweight.dat");
    ASSERT_EQ(paraRowsAtCheckpoint, 1U);
    ASSERT_EQ(reweightRowsAtCheckpoint, 1U);

    gaMDModule_.resetStateForTesting();
    gaMDModule_.prepareStep(1, 0);
    gaMDModule_.finalizeCurrentStep(1, 0, 0.0, 0.0);

    EXPECT_EQ(countNumericRows("gamd-para.dat"), paraRowsAtCheckpoint);
    EXPECT_EQ(countNumericRows("gamd-reweight.dat"), reweightRowsAtCheckpoint);
}

TEST_F(GaMDTest, RestartReconciliationPreservesOriginalAfterWriteFailure)
{
#if defined(__linux__) && GTEST_HAS_DEATH_TEST
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    std::string            restart = c_validGaMDRestart;
    restart.replace(
            restart.find("saved_step 10"), std::string("saved_step 10").size(), "saved_step 65");
    writeGaMDProductionInput(restart);

    // 每行固定为 128 字节，让写入在刷新之前就触发文件大小限制。
    std::string originalOutput;
    for (int step = 1; step <= 65; ++step)
    {
        std::string row = "1 " + std::to_string(step) + " 0 0 1 1 1 0 0";
        row.resize(127, ' ');
        originalOutput += row + '\n';
    }
    const std::filesystem::path reweightPath = "gamd-reweight.dat";
    {
        std::ofstream output(reweightPath);
        output << originalOutput;
    }

    // 限制只作用于 death test 子进程，不改变其余测试的文件写入。
    const struct rlimit fileSizeLimit = { 4096, 4096 };
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            {
                std::signal(SIGXFSZ, SIG_IGN);
                GMX_RELEASE_ASSERT(setrlimit(RLIMIT_FSIZE, &fileSizeLimit) == 0,
                                   "Could not configure the test file-size limit");
                gaMDModule_.prepareStep(65, 0);
            },
            "Failed to flush and synchronize GaMD file");
    EXPECT_EQ(TextReader::readFileToString(reweightPath), originalOutput);
#else
    GTEST_SKIP() << "This test requires Linux file-size limits and death tests.";
#endif
}

TEST_F(GaMDTest, RestartRejectsMalformedRecordsBeforeRestoringState)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory           workingDirectory(directory);
    const std::array<std::string, 4> suffixes{ "saved_step 10\n",
                                               "unexpected 1\n",
                                               "statP 10 0 5 2 0 2 10 1 0.1\n",
                                               "version 3 trailing\n" };
    const std::array<const char*, 4> errors{ "Duplicate", "Unknown", "Duplicate", "Duplicate" };
    for (Index suffixIndex = 0; suffixIndex < ssize(suffixes); suffixIndex++)
    {
        writeGaMDProductionInput(c_validGaMDRestart + suffixes[suffixIndex]);
        GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), errors[suffixIndex]);
    }
    std::string trailing = c_validGaMDRestart;
    trailing.replace(
            trailing.find("version 3\n"), std::string("version 3\n").size(), "version 3 trailing\n");
    writeGaMDProductionInput(trailing);
    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "Unexpected trailing token");

    std::string missingFlag = c_validGaMDRestart;
    missingFlag.erase(missingFlag.find("production_restart 1\n"),
                      std::string("production_restart 1\n").size());
    writeGaMDProductionInput(missingFlag);
    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "requires production_restart");

    std::string invalidFlag = c_validGaMDRestart;
    invalidFlag.replace(invalidFlag.find("production_restart 1"),
                        std::string("production_restart 1").size(),
                        "production_restart 2");
    writeGaMDProductionInput(invalidFlag);
    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0),
                                  "production_restart must be 0 or 1");
}

TEST_F(GaMDTest, RestartReadsAllSupportedVersions)
{
    TestFileManager fileManager;
    for (int version = 1; version <= 3; version++)
    {
        const auto directory = fileManager.getOutputTempDirectory()
                               / TestFileManager::getTestSpecificFileNameRoot() / std::to_string(version);
        ScopedWorkingDirectory workingDirectory(directory);
        std::string            restart = c_validGaMDRestart;
        restart.replace(restart.find("version 3"),
                        std::string("version 3").size(),
                        "version " + std::to_string(version));
        if (version < 3)
        {
            restart.erase(restart.find("production_restart 1\n"),
                          std::string("production_restart 1\n").size());
        }
        if (version == 1)
        {
            restart.erase(restart.find("windowP 0 0 0\n"), std::string("windowP 0 0 0\n").size());
            restart.erase(restart.find("windowD 0 0 0\n"), std::string("windowD 0 0 0\n").size());
        }
        writeGaMDProductionInput(restart);
        gaMDModule_.resetStateForTesting();
        const GaMDStepPlan& plan = gaMDModule_.prepareStep(0, 0);
        EXPECT_EQ(plan.stage, GaMDStage::Production);
        EXPECT_DOUBLE_EQ(plan.thresholdP, 10.0);
    }
}

TEST_F(GaMDTest, RestartSelectsOnlyAValidatedMatchingSnapshot)
{
    TestFileManager fileManager;
    const auto      directory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory workingDirectory(directory);
    writeGaMDProductionInput(c_validGaMDRestart + "saved_step 10\n");
    {
        std::ofstream previous("gamd-restart-prev.dat");
        previous << c_validGaMDRestart;
    }
    const GaMDStepPlan& recovered = gaMDModule_.prepareStep(10, 0);
    EXPECT_EQ(recovered.stage, GaMDStage::Production);
    EXPECT_DOUBLE_EQ(recovered.thresholdP, 10.0);

    // 当前快照有效且匹配时，无关的旧备份不能阻止正常续跑。
    gaMDModule_.resetStateForTesting();
    writeGaMDProductionInput(c_validGaMDRestart);
    {
        std::ofstream previous("gamd-restart-prev.dat");
        previous << "corrupted backup\n";
    }
    EXPECT_EQ(gaMDModule_.prepareStep(10, 0).stage, GaMDStage::Production);
}

TEST_F(GaMDTest, RestartRetains64BitCounters)
{
    TestFileManager fileManager;
    const auto      workDirectory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDirectory);
    {
        std::ofstream input("gamd.in");
        input << "igamd 1\nirest_gamd 1\niE 1\niEP 1\nreweight_nst 50\nsigma0P 6.0\n";
        std::ofstream restart("gamd-restart.dat");
        restart << "version 3\nsaved_step 5000000000\nproduction_restart 1\n";
        restart << "params 1 1 1 1 10 20 10 20 10 50 25.104 25.104\n";
        restart << "windowP 4000000000 5 2\nwindowD 0 0 0\n";
        restart << "statP 10 0 5 2 0 4000000000 10 1 0.1\n";
        restart << "statD -1e99 1e99 0 0.001 0 0 0 0 0\n";
    }

    // 超过 32 位的步数和窗口计数必须完整地读回并再次写出。
    gaMDModule_.prepareStep(INT64_C(5000000000), 0);
    gaMDModule_.setCheckpointingThisStep(true);
    gaMDModule_.writeRestartState(INT64_C(5000000000));

    int64_t       restoredStep        = -1;
    int64_t       restoredWindowCount = -1;
    std::ifstream restart("gamd-restart.dat");
    std::string   line;
    while (std::getline(restart, line))
    {
        std::istringstream columns(line);
        std::string        key;
        columns >> key;
        if (key == "saved_step")
        {
            columns >> restoredStep;
        }
        else if (key == "windowP")
        {
            columns >> restoredWindowCount;
        }
    }
    EXPECT_EQ(restoredStep, INT64_C(5000000000));
    EXPECT_EQ(restoredWindowCount, INT64_C(4000000000));
}

TEST_F(GaMDTest, StrictInputRejectsMissingAndTrailingValues)
{
    TestFileManager fileManager;
    const auto      workDirectory =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDirectory);
    {
        std::ofstream input("gamd.in");
        input << "igamd\n";
    }
    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "Missing or invalid value");
    {
        std::ofstream input("gamd.in");
        input << "igamd 1 unexpected\n";
    }
    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "Unexpected trailing token");
}

TEST_F(GaMDTest, StrictInputRejectsUnknownParameter)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "reweight_nst 1\n";
        output << "sigma0P 6.0\n";
        output << "misspelled_parameter 7\n";
    }

    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "Unknown GaMD parameter");
}

TEST_F(GaMDTest, StrictInputRejectsDuplicateParameter)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "igamd 3\n";
    }

    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "Duplicate GaMD parameter");
}

TEST_F(GaMDTest, StrictInputRejectsInvalidRange)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gaMDModule_.resetStateForTesting();
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "iE 1\n";
        output << "iEP 1\n";
        output << "ntcmdprep 0\n";
        output << "ntcmd 2\n";
        output << "ntebprep 0\n";
        output << "nteb 2\n";
        output << "ntave 2\n";
        output << "reweight_nst 0\n";
        output << "sigma0P 6.0\n";
    }

    GMX_EXPECT_DEATH_IF_SUPPORTED(gaMDModule_.prepareStep(0, 0), "reweight_nst must be positive");
}

} // namespace
} // namespace test
} // namespace gmx
