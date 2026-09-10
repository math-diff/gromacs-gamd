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
 * \brief Host-side GaMD statistics and adaptive threshold calculations.
 */
#include "gmxpre.h"

#include "gamdmath.h"

#include <cmath>

#include <algorithm>

namespace gmx
{

void calculateThresholdAndForceConstant(int thresholdMode, double sigmaLimit, GaMDEnergyStatistics& statistics)
{
    if (statistics.maximum <= statistics.minimum || statistics.deviation < 1e-6
        || statistics.maximum == statistics.average)
    {
        return;
    }

    // 保留原来的上下界选择和回退条件，避免重构改变平衡阶段的参数。
    if (thresholdMode == 1)
    {
        statistics.threshold                 = statistics.maximum;
        const double normalizedForceConstant = (sigmaLimit / statistics.deviation)
                                               * (statistics.maximum - statistics.minimum)
                                               / (statistics.maximum - statistics.average);
        statistics.k0 = std::min(1.0, normalizedForceConstant);
    }
    else if (thresholdMode == 2)
    {
        const double upperForceConstant = (1.0 - sigmaLimit / statistics.deviation)
                                          * (statistics.maximum - statistics.minimum)
                                          / std::max(1e-6, statistics.average - statistics.minimum);
        if (upperForceConstant > 0.0 && upperForceConstant <= 1.0)
        {
            statistics.k0 = upperForceConstant;
            statistics.threshold =
                    statistics.minimum + (statistics.maximum - statistics.minimum) / statistics.k0;
        }
        else
        {
            statistics.threshold                 = statistics.maximum;
            const double normalizedForceConstant = (sigmaLimit / statistics.deviation)
                                                   * (statistics.maximum - statistics.minimum)
                                                   / (statistics.maximum - statistics.average);
            statistics.k0 = std::min(1.0, normalizedForceConstant);
        }
    }
    statistics.forceConstant = statistics.k0 / std::max(1e-6, statistics.maximum - statistics.minimum);
}

void updateGaMDStatistics(double energy, GaMDEnergyStatistics& statistics, GaMDWindowStatistics& window)
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

    // 用原有的逐样本算法累计均值和平方偏差，续跑时仍可保存完整窗口。
    window.count++;
    const double previousDifference = energy - window.mean;
    window.mean += previousDifference / window.count;
    const double updatedDifference = energy - window.mean;
    window.m2 += previousDifference * updatedDifference;
}

void finalizeWindowStatistics(GaMDWindowStatistics& window, GaMDEnergyStatistics& statistics)
{
    if (window.count <= 0)
    {
        return;
    }

    statistics.average = window.mean;
    if (window.count > 1)
    {
        statistics.deviation = std::sqrt(window.m2 / window.count);
    }
    else
    {
        statistics.deviation = 0.0;
    }
    window = {};
}

} // namespace gmx
