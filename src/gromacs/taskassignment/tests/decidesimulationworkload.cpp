/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 */
#include "gmxpre.h"

#include "gromacs/taskassignment/decidesimulationworkload.h"

#include "config.h"

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/ewald/pme.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/multipletimestepping.h"
#include "gromacs/taskassignment/decidegpuusage.h"
#include "gromacs/utility/logger.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(DecideSimulationWorkloadTest, UsesGpuForceBufferOpsWhenNoHostPostProcessingNeedsForces)
{
    SimulationWorkload simulationWork;
    simulationWork.computeNonbonded                       = true;
    simulationWork.useGpuNonbonded                        = true;
    simulationWork.useGpuFBufferOpsWhenAllowed            = true;
    simulationWork.requireCpuForceBufferForPostProcessing = false;
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

TEST(DecideSimulationWorkloadTest, HostPostProcessingDisablesGpuForceBufferOpsThisStep)
{
    SimulationWorkload simulationWork;
    simulationWork.computeNonbonded                       = true;
    simulationWork.useGpuNonbonded                        = true;
    simulationWork.useGpuFBufferOpsWhenAllowed            = true;
    simulationWork.requireCpuForceBufferForPostProcessing = true;
    DomainLifetimeWorkload domainWork;
    std::vector<MtsLevel>  mtsLevels;

    const StepWorkload stepWork = setupStepWorkload(
            GMX_FORCE_NONBONDED | GMX_FORCE_FORCES, mtsLevels, 0, domainWork, simulationWork);

    EXPECT_TRUE(stepWork.computeNonbondedForces);
    EXPECT_FALSE(stepWork.useGpuFBufferOps);
}

TEST(DecideSimulationWorkloadTest, GpuGaMDUsesGpuForceBufferOpsOnVirialSteps)
{
    SimulationWorkload simulationWork;
    simulationWork.computeNonbonded            = true;
    simulationWork.useGpuNonbonded             = true;
    simulationWork.useGpuFBufferOpsWhenAllowed = true;
    simulationWork.gamdExecutionMode           = GaMDExecutionMode::GpuScalarSynchronized;
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
    simulationWork.gamdExecutionMode           = GaMDExecutionMode::GpuScalarSynchronized;
    DomainLifetimeWorkload domainWork;
    domainWork.haveCpuLocalForceWork = true;
    std::vector<MtsLevel> mtsLevels;

    const StepWorkload stepWork = setupStepWorkload(
            GMX_FORCE_NONBONDED | GMX_FORCE_FORCES | GMX_FORCE_VIRIAL, mtsLevels, 0, domainWork, simulationWork);

    EXPECT_TRUE(stepWork.computeVirial);
    EXPECT_FALSE(stepWork.useGpuFBufferOps);
}

TEST(DecideSimulationWorkloadTest, GaMDHostPostProcessingDisablesMdGpuGraph)
{
    if (!(GMX_GPU_CUDA || GMX_GPU_SYCL))
    {
        GTEST_SKIP() << "GPU graph scheduling requires a GPU build";
    }

    const MDLogger          logger;
    DevelopmentFeatureFlags devFlags;
    devFlags.enableCudaGraphs = true;
    t_inputrec inputrec;

    const SimulationWorkload baselineWork = createSimulationWorkload(
            logger, inputrec, false, false, devFlags, false, false, false, true, PmeRunMode::None, false, true, false, false, false);

    EXPECT_TRUE(baselineWork.useMdGpuGraph);

    inputrec.bDoGaMD                  = true;
    const SimulationWorkload gamdWork = createSimulationWorkload(
            logger, inputrec, false, false, devFlags, false, false, false, true, PmeRunMode::None, false, true, false, false, false);

    EXPECT_TRUE(gamdWork.requireCpuForceBufferForPostProcessing);
    EXPECT_FALSE(gamdWork.useMdGpuGraph);
}

} // namespace
} // namespace test
} // namespace gmx
