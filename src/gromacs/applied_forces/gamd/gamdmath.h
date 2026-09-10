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
 * \brief Pure GaMD bias and population-statistics calculations.
 *
 * The evaluator is shared by CPU force correction, CUDA production, and
 * deferred output. It does not read files, modify statistics, or clamp scales.
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDMATH_H
#define GMX_APPLIED_FORCES_GAMD_GAMDMATH_H

#include <cstdint>

#include "gromacs/applied_forces/gamd/gamdtypes.h"

namespace gmx
{

//! Existing force-weight safety floor; applied after evaluating the bias.
constexpr double c_gaMDScaleFloor = 0.01;

/*! \brief Evaluate the existing GaMD potential and its raw force weights.
 *
 * \param[in] parameters Validated boost mode, stage, thresholds, and constants.
 * \param[in] potentialEnergy Unbiased total potential energy, in kJ/mol.
 * \param[in] dihedralEnergy Unbiased dihedral-component energy, in kJ/mol.
 * \returns Bias energies in kJ/mol and unclamped dimensionless force weights.
 *
 * In dual mode the potential boost is evaluated at potentialEnergy + boostD.
 * The final force is scaleP * F + scaleP * (scaleD - 1) * F_dihedral.
 * Validation, safety-floor diagnostics, and output units belong to callers.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
        inline GaMDStepResult
        evaluateGaMDBias(const GaMDBiasParameters& parameters, double potentialEnergy, double dihedralEnergy)
{
    GaMDStepResult result;
    result.potentialEnergy = potentialEnergy;
    result.dihedralEnergy  = dihedralEnergy;
    if (parameters.stage < 3)
    {
        return result;
    }

    // 先算二面角偏置，双重偏置的总势能计算需要用到这个结果。
    if ((parameters.boostType == 2 || parameters.boostType == 3) && dihedralEnergy < parameters.thresholdD)
    {
        const double energyDifference = parameters.thresholdD - dihedralEnergy;
        result.boostD                 = 0.5 * parameters.kD * energyDifference * energyDifference;
        result.scaleD                 = 1.0 - parameters.kD * energyDifference;
    }

    double totalEnergyForPotentialBoost = potentialEnergy;
    if (parameters.boostType == 3)
    {
        totalEnergyForPotentialBoost += result.boostD;
    }

    if ((parameters.boostType == 1 || parameters.boostType == 3)
        && totalEnergyForPotentialBoost < parameters.thresholdP)
    {
        const double energyDifference = parameters.thresholdP - totalEnergyForPotentialBoost;
        result.boostP                 = 0.5 * parameters.kP * energyDifference * energyDifference;
        result.scaleP                 = 1.0 - parameters.kP * energyDifference;
    }
    return result;
}

//! Persistent statistics and bias parameters for one energy component.
struct GaMDEnergyStatistics
{
    double  maximum       = -1e99;
    double  minimum       = 1e99;
    double  average       = 0.0;
    double  deviation     = 0.001;
    double  m2            = 0.0; //!< Retained for legacy restart compatibility.
    int64_t count         = 0;
    double  threshold     = 0.0;
    double  k0            = 0.0;
    double  forceConstant = 0.0;
};

//! Welford accumulator for the current averaging window.
struct GaMDWindowStatistics
{
    int64_t count = 0;
    double  mean  = 0.0;
    double  m2    = 0.0;
};

//! Accumulate one sample without changing the currently applied bias.
void updateGaMDStatistics(double energy, GaMDEnergyStatistics& statistics, GaMDWindowStatistics& window);

//! Finish a nonempty window using population variance (M2 / count), then reset it.
void finalizeWindowStatistics(GaMDWindowStatistics& window, GaMDEnergyStatistics& statistics);

//! Update E and k using the existing iE=1/2 policy; retain them for degenerate input.
void calculateThresholdAndForceConstant(int thresholdMode, double sigmaLimit, GaMDEnergyStatistics& statistics);

} // namespace gmx

#endif
