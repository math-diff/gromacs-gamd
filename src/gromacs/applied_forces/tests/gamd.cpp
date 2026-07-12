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

#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/applied_forces/gamd/gamdforceprovider.h"

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

class ScopedGaMDGpuEnvironment
{
public:
    explicit ScopedGaMDGpuEnvironment(const char* value)
    {
        if (const char* originalValue = std::getenv("GMX_GAMD_GPU"))
        {
            originalValue_ = originalValue;
        }
        if (value == nullptr)
        {
            gmxUnsetenv("GMX_GAMD_GPU");
        }
        else
        {
            gmxSetenv("GMX_GAMD_GPU", value, 1);
        }
    }

    ~ScopedGaMDGpuEnvironment()
    {
        if (originalValue_.has_value())
        {
            gmxSetenv("GMX_GAMD_GPU", originalValue_->c_str(), 1);
        }
        else
        {
            gmxUnsetenv("GMX_GAMD_GPU");
        }
    }

private:
    std::optional<std::string> originalValue_;
};

class ScopedGaMDBufferedOutputEnvironment
{
public:
    explicit ScopedGaMDBufferedOutputEnvironment(const char* value)
    {
        if (const char* originalValue = std::getenv("GMX_GAMD_GPU_BUFFERED_OUTPUT"))
        {
            originalValue_ = originalValue;
        }
        if (value == nullptr)
        {
            gmxUnsetenv("GMX_GAMD_GPU_BUFFERED_OUTPUT");
        }
        else
        {
            gmxSetenv("GMX_GAMD_GPU_BUFFERED_OUTPUT", value, 1);
        }
    }

    ~ScopedGaMDBufferedOutputEnvironment()
    {
        if (originalValue_.has_value())
        {
            gmxSetenv("GMX_GAMD_GPU_BUFFERED_OUTPUT", originalValue_->c_str(), 1);
        }
        else
        {
            gmxUnsetenv("GMX_GAMD_GPU_BUFFERED_OUTPUT");
        }
    }

private:
    std::optional<std::string> originalValue_;
};

GaMDExecutionMode selectSupportedGaMDExecutionMode()
{
    return selectGaMDExecutionMode(true, true, true, true, true, false, false, false, false, 0);
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

size_t countNumericRows(const std::filesystem::path& path)
{
    return readNumericRows(path).size();
}

TEST(GaMDTest, Stage4PotentialStatisticsUseBoostedTotalPotential)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
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

    gamdSetCheckpointingThisStep(false);

    constexpr double c_step1Potential = 0.0;
    constexpr double c_step2Potential = 10.0;
    constexpr double c_step3Potential = 0.0;
    constexpr double c_step4Potential = 0.0;
    constexpr double c_dihedralEnergy = 0.0;

    gamdPrepareStep(1, 0);
    gamdFinalizeCurrentStep(1, 0, c_step1Potential, c_dihedralEnergy);

    gamdPrepareStep(2, 0);
    gamdFinalizeCurrentStep(2, 0, c_step2Potential, c_dihedralEnergy);

    gamdPrepareStep(3, 0);
    gamdFinalizeCurrentStep(3, 0, c_step3Potential, c_dihedralEnergy);

    gamdPrepareStep(4, 0);
    gamdFinalizeCurrentStep(4, 0, c_step4Potential, c_dihedralEnergy);

    const auto columns = readLastNumericColumns("gamd-para.dat");
    ASSERT_GE(columns.size(), 10U);
    EXPECT_EQ(columns[1], 4.0);
    EXPECT_EQ(columns[2], 4.0);

    // Amber updates stage-4 total-potential statistics using the current boosted
    // total potential. With the setup above, steps 3-4 each have a boost of 5.
    EXPECT_REAL_EQ_TOL(columns[5], 5.0, absoluteTolerance(1e-12));
}

TEST(GaMDTest, CheckpointReplayDoesNotDuplicateTextOutputForSavedStep)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
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

    gamdSetCheckpointingThisStep(false);
    gamdPrepareStep(1, 0);
    gamdFinalizeCurrentStep(1, 0, 0.0, c_dihedralEnergy);
    gamdPrepareStep(2, 0);
    gamdFinalizeCurrentStep(2, 0, 10.0, c_dihedralEnergy);
    gamdPrepareStep(3, 0);
    gamdFinalizeCurrentStep(3, 0, 0.0, c_dihedralEnergy);

    gamdPrepareStep(4, 0);
    gamdSetCheckpointingThisStep(true);
    gamdWriteRestartState(4);
    gamdFinalizeCurrentStep(4, 0, 0.0, c_dihedralEnergy);
    gamdSetCheckpointingThisStep(false);

    const size_t paraRowsAfterCheckpoint     = countNumericRows("gamd-para.dat");
    const size_t reweightRowsAfterCheckpoint = countNumericRows("gamd-reweight.dat");
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
    gamdPrepareStep(5, 0);
    gamdFinalizeCurrentStep(5, 0, 0.0, c_dihedralEnergy);
    gamdPrepareStep(6, 0);
    gamdFinalizeCurrentStep(6, 0, 0.0, c_dihedralEnergy);

    // Simulate interruption after GaMD committed a newer restart but before
    // GROMACS replaced the older .cpt. The rotated previous restart must keep
    // the older checkpoint recoverable.
    gamdSetCheckpointingThisStep(true);
    gamdWriteRestartState(6);
    gamdSetCheckpointingThisStep(false);
    ASSERT_TRUE(std::filesystem::exists("gamd-restart-prev.dat"));
    {
        std::ofstream paraOutput("gamd-para.dat", std::ios::app);
        paraOutput << "1 7 partial\n";
        std::ofstream reweightOutput("gamd-reweight.dat", std::ios::app);
        reweightOutput << "1 7 partial\n";
    }
    EXPECT_GT(countNumericRows("gamd-para.dat"), paraRowsAfterCheckpoint);
    EXPECT_GT(countNumericRows("gamd-reweight.dat"), reweightRowsAfterCheckpoint);

    gamdResetStateForTesting();
    gamdPrepareStep(4, 0);
    gamdFinalizeCurrentStep(4, 0, 0.0, c_dihedralEnergy);

    EXPECT_EQ(countNumericRows("gamd-para.dat"), paraRowsAfterCheckpoint);
    EXPECT_EQ(countNumericRows("gamd-reweight.dat"), reweightRowsAfterCheckpoint);

    gamdPrepareStep(5, 0);
    gamdFinalizeCurrentStep(5, 0, 0.0, c_dihedralEnergy);

    EXPECT_EQ(countNumericRows("gamd-para.dat"), paraRowsAfterCheckpoint + 1);
    EXPECT_EQ(countNumericRows("gamd-reweight.dat"), reweightRowsAfterCheckpoint + 1);

    paraColumns = readLastNumericColumns("gamd-para.dat");
    ASSERT_GT(paraColumns.size(), 1U);
    EXPECT_EQ(paraColumns[1], 5.0);

    reweightColumns = readLastNumericColumns("gamd-reweight.dat");
    ASSERT_GT(reweightColumns.size(), 1U);
    EXPECT_EQ(reweightColumns[1], 5.0);
}

TEST(GaMDTest, CurrentStepGpuCompatibilityAllowsGpuBondedAndGpuUpdate)
{
    gamdResetStateForTesting();

    EXPECT_EQ(nullptr, gmx::currentStepGaMDGpuIncompatibilityReason(false, false));
    EXPECT_EQ(nullptr, gmx::currentStepGaMDGpuIncompatibilityReason(false, true));
    EXPECT_EQ(nullptr, gmx::currentStepGaMDGpuIncompatibilityReason(true, false));
    EXPECT_EQ(nullptr, gmx::currentStepGaMDGpuIncompatibilityReason(true, true));
}

TEST(GaMDTest, ResidentProductionEnergyStagesOnlyForHostConsumers)
{
    ScopedGaMDBufferedOutputEnvironment bufferedOutput(nullptr);
    TestFileManager                     fileManager;
    const auto                          workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
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

    gamdSetCheckpointingThisStep(false);
    gamdPrepareStep(2, 0);
    EXPECT_TRUE(gamdRequiresHostEnergyThisStep(2));

    gamdPrepareStep(49, 0);
    EXPECT_FALSE(gamdRequiresHostEnergyThisStep(49));

    gamdPrepareStep(50, 0);
    EXPECT_TRUE(gamdRequiresHostEnergyThisStep(50));

    gamdPrepareStep(51, 0);
    gamdSetCheckpointingThisStep(true);
    EXPECT_TRUE(gamdRequiresHostEnergyThisStep(51));
    gamdSetCheckpointingThisStep(false);
}

TEST(GaMDTest, BufferedProductionOutputDefersTextCadenceButNotCheckpointing)
{
    ScopedGaMDBufferedOutputEnvironment bufferedOutput("1");
    TestFileManager                     fileManager;
    const auto                          workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
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

    gamdSetCheckpointingThisStep(false);
    gamdPrepareStep(50, 0);
    EXPECT_TRUE(gamdBufferedProductionOutputEnabled());
    EXPECT_FALSE(gamdRequiresHostEnergyThisStep(50));

    gamdSetCheckpointingThisStep(true);
    EXPECT_TRUE(gamdRequiresHostEnergyThisStep(51));
    gamdSetCheckpointingThisStep(false);
}

TEST(GaMDTest, GaMDGpuExecutionModeDefaultsToCpuReference)
{
    ScopedGaMDGpuEnvironment environment(nullptr);

    EXPECT_EQ(GaMDExecutionMode::CpuReference, selectSupportedGaMDExecutionMode());
}

TEST(GaMDTest, GaMDGpuExecutionModeCanBeForcedToCpuReference)
{
    ScopedGaMDGpuEnvironment environment("0");

    EXPECT_EQ(GaMDExecutionMode::CpuReference,
              selectGaMDExecutionMode(true, false, false, false, false, true, true, true, true, 1));
}

TEST(GaMDTest, NonGaMDRunIgnoresGpuExecutionModeRequest)
{
    ScopedGaMDGpuEnvironment environment("invalid");

    EXPECT_EQ(GaMDExecutionMode::CpuReference,
              selectGaMDExecutionMode(false, false, false, false, false, true, true, true, true, 1));
}

TEST(GaMDTest, GaMDGpuExecutionModeRejectsInvalidRequest)
{
    ScopedGaMDGpuEnvironment environment("true");

    GMX_EXPECT_DEATH_IF_SUPPORTED(selectSupportedGaMDExecutionMode(),
                                  "GMX_GAMD_GPU must be exactly 0 or 1");
}

#if GMX_GPU_CUDA
TEST(GaMDTest, GaMDGpuExecutionModeSelectsScalarSynchronizedCudaPath)
{
    ScopedGaMDGpuEnvironment environment("1");

    EXPECT_EQ(GaMDExecutionMode::GpuScalarSynchronized, selectSupportedGaMDExecutionMode());
    EXPECT_STREQ("GPU scalar-synchronized",
                 gaMDExecutionModeName(GaMDExecutionMode::GpuScalarSynchronized));
}

TEST(GaMDTest, GaMDGpuExecutionModeRejectsUnsupportedWorkloads)
{
    ScopedGaMDGpuEnvironment environment("1");

    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, false, true, true, true, false, false, false, false, 0),
            "requires -nb gpu");
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, true, false, true, true, false, false, false, false, 0),
            "requires -pme gpu");
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, true, true, false, true, false, false, false, false, 0),
            "requires -bonded gpu");
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, true, true, true, false, false, false, false, false, 0),
            "requires -update gpu");
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, true, true, true, true, true, false, false, false, 0),
            "without domain decomposition");
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, true, true, true, true, false, false, true, false, 0),
            "does not support MTS");
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, true, true, true, true, false, false, false, true, 0),
            "does not yet support pressure coupling");
    GMX_EXPECT_DEATH_IF_SUPPORTED(
            selectGaMDExecutionMode(true, true, true, true, true, false, false, false, false, 10),
            "requires nstfout=0");
}
#else
TEST(GaMDTest, GaMDGpuExecutionModeRejectsNonCudaBuild)
{
    ScopedGaMDGpuEnvironment environment("1");

    GMX_EXPECT_DEATH_IF_SUPPORTED(selectSupportedGaMDExecutionMode(), "requires a CUDA build");
}
#endif

TEST(GaMDTest, IrestProductionCheckpointCanContinueWithoutDuplicateRows)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
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
        gamdPrepareStep(step, 0);
        gamdFinalizeCurrentStep(step, 0, (step == 2 ? 10.0 : 0.0), 0.0);
    }
    gamdSetCheckpointingThisStep(true);
    gamdWriteRestartState(4);
    gamdSetCheckpointingThisStep(false);

    gamdResetStateForTesting();
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

    gamdPrepareStep(0, 0);
    gamdFinalizeCurrentStep(0, 0, 0.0, 0.0);
    gamdPrepareStep(1, 0);
    gamdSetCheckpointingThisStep(true);
    gamdWriteRestartState(1);
    gamdFinalizeCurrentStep(1, 0, 0.0, 0.0);
    gamdSetCheckpointingThisStep(false);

    const size_t paraRowsAtCheckpoint     = countNumericRows("gamd-para.dat");
    const size_t reweightRowsAtCheckpoint = countNumericRows("gamd-reweight.dat");
    ASSERT_EQ(paraRowsAtCheckpoint, 1U);
    ASSERT_EQ(reweightRowsAtCheckpoint, 1U);

    gamdResetStateForTesting();
    gamdPrepareStep(1, 0);
    gamdFinalizeCurrentStep(1, 0, 0.0, 0.0);

    EXPECT_EQ(countNumericRows("gamd-para.dat"), paraRowsAtCheckpoint);
    EXPECT_EQ(countNumericRows("gamd-reweight.dat"), reweightRowsAtCheckpoint);
}

TEST(GaMDTest, StrictInputRejectsUnknownParameter)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
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

    GMX_EXPECT_DEATH_IF_SUPPORTED(gamdPrepareStep(0, 0), "Unknown GaMD parameter");
}

TEST(GaMDTest, StrictInputRejectsDuplicateParameter)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
    {
        std::ofstream output("gamd.in");
        output << "igamd 1\n";
        output << "igamd 3\n";
    }

    GMX_EXPECT_DEATH_IF_SUPPORTED(gamdPrepareStep(0, 0), "Duplicate GaMD parameter");
}

TEST(GaMDTest, StrictInputRejectsInvalidRange)
{
    TestFileManager fileManager;
    const auto      workDir =
            fileManager.getOutputTempDirectory() / TestFileManager::getTestSpecificFileNameRoot();
    ScopedWorkingDirectory scopedWorkingDirectory(workDir);

    gamdResetStateForTesting();
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

    GMX_EXPECT_DEATH_IF_SUPPORTED(gamdPrepareStep(0, 0), "reweight_nst must be positive");
}

} // namespace
} // namespace test
} // namespace gmx
