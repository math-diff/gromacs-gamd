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
 * \brief Analytical checks for the shared GaMD scientific core.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/gamd/gamdmath.h"

#include <cmath>

#include <array>

#include <gtest/gtest.h>

namespace gmx
{
namespace test
{
namespace
{

TEST(GaMDMathTest, BoostModesHaveIndependentAnalyticalValues)
{
    GaMDBiasParameters   parameters{ 1, GaMDStage::Production, 10.0, 0.1, 5.0, 0.2 };
    const GaMDStepResult potential = evaluateGaMDBias(parameters, 8.0, 2.0);
    EXPECT_DOUBLE_EQ(potential.boostP, 0.2);
    EXPECT_DOUBLE_EQ(potential.scaleP, 0.8);
    EXPECT_DOUBLE_EQ(potential.boostD, 0.0);
    EXPECT_DOUBLE_EQ(potential.scaleD, 1.0);

    parameters.boostType          = 2;
    const GaMDStepResult dihedral = evaluateGaMDBias(parameters, 8.0, 2.0);
    EXPECT_NEAR(dihedral.boostD, 0.9, 1e-15);
    EXPECT_NEAR(dihedral.scaleD, 0.4, 1e-15);
    EXPECT_DOUBLE_EQ(dihedral.boostP, 0.0);
    EXPECT_DOUBLE_EQ(dihedral.scaleP, 1.0);

    parameters.boostType      = 3;
    const GaMDStepResult dual = evaluateGaMDBias(parameters, 8.0, 2.0);
    // 双重偏置必须在 8.9 而不是 8.0 上计算总势能偏置。
    EXPECT_NEAR(dual.boostP, 0.0605, 1e-15);
    EXPECT_NEAR(dual.scaleP, 0.89, 1e-15);
    EXPECT_NEAR(dual.boostD, 0.9, 1e-15);
    EXPECT_NEAR(dual.scaleD, 0.4, 1e-15);
    EXPECT_DOUBLE_EQ(dual.potentialEnergy, 8.0);
    EXPECT_DOUBLE_EQ(dual.dihedralEnergy, 2.0);
}

TEST(GaMDMathTest, ThresholdAndPreparationStagesHaveNoBias)
{
    GaMDBiasParameters parameters{ 3, GaMDStage::Production, 10.0, 0.1, 5.0, 0.2 };
    for (double energyOffset : std::array<double, 2>{ 0.0, 1.0 })
    {
        const GaMDStepResult result =
                evaluateGaMDBias(parameters, 10.0 + energyOffset, 5.0 + energyOffset);
        EXPECT_DOUBLE_EQ(result.totalBoost(), 0.0);
        EXPECT_DOUBLE_EQ(result.scaleP, 1.0);
        EXPECT_DOUBLE_EQ(result.scaleD, 1.0);
    }
    const std::array<GaMDStage, 3> neutralStages = { GaMDStage::Inactive,
                                                     GaMDStage::ConventionalPreparation,
                                                     GaMDStage::ConventionalStatistics };
    for (std::size_t stageIndex = 0; stageIndex < neutralStages.size(); ++stageIndex)
    {
        parameters.stage            = neutralStages[stageIndex];
        const GaMDStepResult result = evaluateGaMDBias(parameters, 8.0, 2.0);
        EXPECT_DOUBLE_EQ(result.totalBoost(), 0.0);
        EXPECT_DOUBLE_EQ(result.scaleP, 1.0);
        EXPECT_DOUBLE_EQ(result.scaleD, 1.0);
    }
}

TEST(GaMDMathTest, BiasGradientMatchesForceComposition)
{
    constexpr double coordinate = 1.0;
    constexpr double increment  = 1e-6;
    for (int boostType = 1; boostType <= 3; boostType++)
    {
        GaMDBiasParameters   parameters{ boostType, GaMDStage::Production, 10.0, 0.05, 5.0, 0.1 };
        const GaMDStepResult result = evaluateGaMDBias(
                parameters, 2.0 * coordinate * coordinate, 0.5 * coordinate * coordinate);
        const double         lowerCoordinate = coordinate - increment;
        const double         upperCoordinate = coordinate + increment;
        const GaMDStepResult lower           = evaluateGaMDBias(parameters,
                                                      2.0 * lowerCoordinate * lowerCoordinate,
                                                      0.5 * lowerCoordinate * lowerCoordinate);
        const GaMDStepResult upper           = evaluateGaMDBias(parameters,
                                                      2.0 * upperCoordinate * upperCoordinate,
                                                      0.5 * upperCoordinate * upperCoordinate);
        // 直接对完整偏置势能求数值导数，独立检查力的链式组合。
        const double numericalForce = -(upper.potentialEnergy + upper.totalBoost()
                                        - lower.potentialEnergy - lower.totalBoost())
                                      / (2.0 * increment);
        const double composedForce = result.scaleP * (-4.0 * coordinate)
                                     + result.scaleP * (result.scaleD - 1.0) * (-coordinate);
        EXPECT_NEAR(composedForce, numericalForce, 1e-7);
    }
}

TEST(GaMDMathTest, EvaluatorLeavesSafetyFloorToTheCaller)
{
    const GaMDBiasParameters parameters{ 1, GaMDStage::Production, 10.0, 0.2, 0.0, 0.0 };
    const GaMDStepResult     result = evaluateGaMDBias(parameters, 0.0, 0.0);
    EXPECT_DOUBLE_EQ(result.scaleP, -1.0);
    EXPECT_DOUBLE_EQ(result.boostP, 10.0);
    EXPECT_DOUBLE_EQ(c_gaMDScaleFloor, 0.01);
}

TEST(GaMDMathTest, DihedralEnergySumsExactlyThePartitionTypes)
{
    // 二面角分区是科学定义，改动它必须同时有意识地更新这里的期望。
    const std::array<int, 4> expectedTypes = { F_PDIHS, F_RBDIHS, F_FOURDIHS, F_CMAP };
    EXPECT_EQ(c_gaMDDihedralInteractionTypes, expectedTypes);

    // 每类能量取不同的 2 的幂，求和结果唯一对应参与求和的类型集合。
    std::array<double, F_NRE> energies;
    for (int forceType = 0; forceType < F_NRE; forceType++)
    {
        energies[forceType] = std::ldexp(1.0, forceType);
    }
    double partitionEnergy = 0.0;
    for (const int forceType : c_gaMDDihedralInteractionTypes)
    {
        partitionEnergy += energies[forceType];
    }
    EXPECT_DOUBLE_EQ(gaMDDihedralEnergy(energies.data()), partitionEnergy);
}

TEST(GaMDMathTest, WindowsUsePopulationVarianceAndCanBeResumed)
{
    GaMDEnergyStatistics statistics;
    GaMDWindowStatistics window;
    updateGaMDStatistics(2.0, compat::make_not_null(&statistics), compat::make_not_null(&window));
    updateGaMDStatistics(4.0, compat::make_not_null(&statistics), compat::make_not_null(&window));
    GaMDWindowStatistics resumedWindow = window;
    updateGaMDStatistics(6.0, compat::make_not_null(&statistics), compat::make_not_null(&resumedWindow));
    updateGaMDStatistics(8.0, compat::make_not_null(&statistics), compat::make_not_null(&resumedWindow));
    finalizeWindowStatistics(compat::make_not_null(&resumedWindow), compat::make_not_null(&statistics));
    EXPECT_EQ(statistics.count, 4);
    EXPECT_DOUBLE_EQ(statistics.minimum, 2.0);
    EXPECT_DOUBLE_EQ(statistics.maximum, 8.0);
    EXPECT_DOUBLE_EQ(statistics.average, 5.0);
    EXPECT_DOUBLE_EQ(statistics.deviation, std::sqrt(5.0));
    EXPECT_EQ(resumedWindow.count, 0);
    EXPECT_DOUBLE_EQ(resumedWindow.mean, 0.0);
    EXPECT_DOUBLE_EQ(resumedWindow.m2, 0.0);
    finalizeWindowStatistics(compat::make_not_null(&resumedWindow), compat::make_not_null(&statistics));
    EXPECT_DOUBLE_EQ(statistics.average, 5.0);
    EXPECT_DOUBLE_EQ(statistics.deviation, std::sqrt(5.0));
}

TEST(GaMDMathTest, ThresholdPoliciesIncludeUpperBoundFallback)
{
    GaMDEnergyStatistics statistics;
    statistics.minimum   = 0.0;
    statistics.maximum   = 10.0;
    statistics.average   = 5.0;
    statistics.deviation = 2.0;
    calculateThresholdAndForceConstant(1, 1.5, compat::make_not_null(&statistics));
    EXPECT_DOUBLE_EQ(statistics.threshold, 10.0);
    EXPECT_DOUBLE_EQ(statistics.k0, 1.0);
    EXPECT_DOUBLE_EQ(statistics.forceConstant, 0.1);
    calculateThresholdAndForceConstant(2, 1.5, compat::make_not_null(&statistics));
    EXPECT_DOUBLE_EQ(statistics.threshold, 20.0);
    EXPECT_DOUBLE_EQ(statistics.k0, 0.5);
    EXPECT_DOUBLE_EQ(statistics.forceConstant, 0.05);
    calculateThresholdAndForceConstant(2, 0.5, compat::make_not_null(&statistics));
    EXPECT_DOUBLE_EQ(statistics.threshold, 10.0);
    EXPECT_DOUBLE_EQ(statistics.k0, 0.5);
    EXPECT_DOUBLE_EQ(statistics.forceConstant, 0.05);
}

TEST(GaMDMathTest, DegenerateWindowPreservesThePreviousBias)
{
    GaMDEnergyStatistics statistics;
    GaMDWindowStatistics window;
    statistics.threshold     = 20.0;
    statistics.forceConstant = 0.05;
    updateGaMDStatistics(4.0, compat::make_not_null(&statistics), compat::make_not_null(&window));
    finalizeWindowStatistics(compat::make_not_null(&window), compat::make_not_null(&statistics));
    calculateThresholdAndForceConstant(1, 1.0, compat::make_not_null(&statistics));
    EXPECT_DOUBLE_EQ(statistics.deviation, 0.0);
    EXPECT_DOUBLE_EQ(statistics.threshold, 20.0);
    EXPECT_DOUBLE_EQ(statistics.forceConstant, 0.05);
}

} // namespace
} // namespace test
} // namespace gmx
