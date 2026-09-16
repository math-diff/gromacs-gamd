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

#include "gromacs/taskassignment/decidesimulationworkload.h"

#include "config.h"

#include <cstdlib>

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/ewald/pme.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/taskassignment/decidegpuusage.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/logger.h"

#include "testutils/setenv.h"

namespace gmx
{
namespace test
{
namespace
{

class ScopedEnvironmentVariable
{
public:
    ScopedEnvironmentVariable(const char* name, const char* value) : name_(name)
    {
        if (const char* originalValue = std::getenv(name))
        {
            originalValue_ = originalValue;
        }
        if (value == nullptr)
        {
            gmxUnsetenv(name);
        }
        else
        {
            gmxSetenv(name, value, 1);
        }
    }

    ~ScopedEnvironmentVariable()
    {
        if (originalValue_.has_value())
        {
            gmxSetenv(name_.c_str(), originalValue_->c_str(), 1);
        }
        else
        {
            gmxUnsetenv(name_.c_str());
        }
    }

private:
    std::string                name_;
    std::optional<std::string> originalValue_;
};

TEST(DecideSimulationWorkloadTest, UsesGpuForceBufferOpsOnForceOnlySteps)
{
    SimulationWorkload simulationWork;
    simulationWork.computeNonbonded            = true;
    simulationWork.useGpuNonbonded             = true;
    simulationWork.useGpuFBufferOpsWhenAllowed = true;
    DomainLifetimeWorkload domainWork;
    std::vector<MtsLevel>  mtsLevels;

    const StepWorkload stepWork = setupStepWorkload(
            GMX_FORCE_NONBONDED | GMX_FORCE_FORCES, mtsLevels, 0, domainWork, simulationWork);

    EXPECT_TRUE(stepWork.computeNonbondedForces);
    EXPECT_TRUE(stepWork.useGpuFBufferOps);
}

TEST(DecideSimulationWorkloadTest, EnergyAndVirialWorkStagesGpuResultsToHostByDefault)
{
    SimulationWorkload     simulationWork;
    DomainLifetimeWorkload domainWork;
    std::vector<MtsLevel>  mtsLevels;

    const StepWorkload energyWork =
            setupStepWorkload(GMX_FORCE_ENERGY, mtsLevels, 0, domainWork, simulationWork);
    const StepWorkload virialWork =
            setupStepWorkload(GMX_FORCE_VIRIAL, mtsLevels, 0, domainWork, simulationWork);
    const StepWorkload forceOnlyWork =
            setupStepWorkload(GMX_FORCE_FORCES, mtsLevels, 0, domainWork, simulationWork);

    EXPECT_TRUE(energyWork.stageGpuEnergyAndVirialToHost);
    EXPECT_TRUE(virialWork.stageGpuEnergyAndVirialToHost);
    EXPECT_FALSE(forceOnlyWork.stageGpuEnergyAndVirialToHost);
}

TEST(DecideSimulationWorkloadTest, GpuGaMDUsesGpuForceBufferOpsOnVirialSteps)
{
    SimulationWorkload simulationWork;
    simulationWork.computeNonbonded            = true;
    simulationWork.useGpuNonbonded             = true;
    simulationWork.useGpuFBufferOpsWhenAllowed = true;
    simulationWork.gamdExecutionMode           = GaMDExecutionMode::GpuResident;
    DomainLifetimeWorkload domainWork;
    std::vector<MtsLevel>  mtsLevels;

    const StepWorkload stepWork = setupStepWorkload(
            GMX_FORCE_NONBONDED | GMX_FORCE_FORCES | GMX_FORCE_VIRIAL, mtsLevels, 0, domainWork, simulationWork);

    EXPECT_TRUE(stepWork.computeVirial);
    EXPECT_TRUE(stepWork.useGpuFBufferOps);
}

TEST(DecideSimulationWorkloadTest, CpuForceWorkKeepsGaMDVirialReductionOnCpu)
{
    SimulationWorkload simulationWork;
    simulationWork.computeNonbonded            = true;
    simulationWork.useGpuNonbonded             = true;
    simulationWork.useGpuFBufferOpsWhenAllowed = true;
    simulationWork.gamdExecutionMode           = GaMDExecutionMode::GpuResident;
    DomainLifetimeWorkload domainWork;
    domainWork.haveCpuLocalForceWork = true;
    std::vector<MtsLevel> mtsLevels;

    const StepWorkload stepWork = setupStepWorkload(
            GMX_FORCE_NONBONDED | GMX_FORCE_FORCES | GMX_FORCE_VIRIAL, mtsLevels, 0, domainWork, simulationWork);

    EXPECT_TRUE(stepWork.computeVirial);
    EXPECT_FALSE(stepWork.useGpuFBufferOps);
}

TEST(DecideSimulationWorkloadTest, CpuUpdateGaMDUsesReferenceModeWithoutMdGpuGraph)
{
    const MDLogger          logger;
    DevelopmentFeatureFlags devFlags;
    devFlags.enableCudaGraphs = true;
    t_inputrec inputrec;

    const bool useGpuNonbonded = GMX_GPU_CUDA || GMX_GPU_SYCL;
    if (useGpuNonbonded)
    {
        const SimulationWorkload baselineWork = createSimulationWorkload(
                logger, inputrec, false, false, devFlags, false, false, false, true, PmeRunMode::None, false, true, false, false, false);
        EXPECT_TRUE(baselineWork.useMdGpuGraph);
    }

    inputrec.bDoGaMD                  = true;
    const SimulationWorkload gamdWork = createSimulationWorkload(logger,
                                                                 inputrec,
                                                                 false,
                                                                 false,
                                                                 devFlags,
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 useGpuNonbonded,
                                                                 PmeRunMode::None,
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 false,
                                                                 false);

    EXPECT_EQ(GaMDExecutionMode::CpuReference, gamdWork.gamdExecutionMode);
    EXPECT_FALSE(gamdWork.useGpuUpdate);
    EXPECT_FALSE(gamdWork.useMdGpuGraph);
}

#if GMX_GPU_CUDA
TEST(DecideSimulationWorkloadTest, EnvironmentGaMDRejectsCpuBondedWithGpuUpdate)
{
    const ScopedEnvironmentVariable enableGaMD("GMX_USE_GAMD", "1");
    const MDLogger                  logger;
    const DevelopmentFeatureFlags   devFlags;
    const t_inputrec                inputrec;

    // 从真实入口请求旧的混合路径，应当直接拒绝，而不是退回 CPU 修正。
    EXPECT_THROW(
            createSimulationWorkload(
                    logger, inputrec, false, false, devFlags, false, false, false, true, PmeRunMode::GPU, false, true, false, false, false),
            InconsistentInputError);
}

TEST(DecideSimulationWorkloadTest, ResidentGpuGaMDAllowsMdGpuGraph)
{
    const MDLogger          logger;
    DevelopmentFeatureFlags devFlags;
    devFlags.enableCudaGraphs = true;
    t_inputrec inputrec;
    inputrec.bDoGaMD = true;

    const SimulationWorkload gamdWork = createSimulationWorkload(
            logger, inputrec, false, false, devFlags, false, false, false, true, PmeRunMode::GPU, true, true, false, false, false);

    EXPECT_EQ(GaMDExecutionMode::GpuResident, gamdWork.gamdExecutionMode);
    EXPECT_TRUE(gamdWork.useGpuUpdate);
    EXPECT_TRUE(gamdWork.useMdGpuGraph);
}
#endif

} // namespace
} // namespace test
} // namespace gmx
