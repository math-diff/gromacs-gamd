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
 * \brief Declares shared GaMD bias and population-statistics calculations.
 *
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDMATH_H
#define GMX_APPLIED_FORCES_GAMD_GAMDMATH_H

#include <cstdint>

#include <array>

#include "gromacs/applied_forces/gamd/gamdtypes.h"
#include "gromacs/compat/pointers.h"
#include "gromacs/topology/ifunc.h"

namespace gmx
{

//! \cond internal
//! \addtogroup module_applied_forces
//! \{

//! Existing force-weight safety floor; applied after evaluating the bias.
constexpr double c_gaMDScaleFloor = 0.01;

/*! \brief Interaction types that form the GaMD dihedral component.
 *
 * This is the scientific definition of the dihedral partition: changing it
 * changes both the bias and the meaning of saved statistics. The CPU
 * recomputation, the GPU support check and gaMDDihedralEnergy() follow this
 * table. The GPU listed-force dispatch and the raw CPU trace in the native
 * bonded code must select the same types; their comments point here.
 */
constexpr std::array<int, 4> c_gaMDDihedralInteractionTypes = { F_PDIHS, F_RBDIHS, F_FOURDIHS, F_CMAP };

/*! \brief Returns the dihedral-component energy from per-interaction energy terms.
 *
 * Adds the types of c_gaMDDihedralInteractionTypes in table order. The sum is
 * written out so that host and device code perform the same additions.
 */
template<typename Real>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
        inline Real
        gaMDDihedralEnergy(const Real* energies)
{
    return energies[F_PDIHS] + energies[F_RBDIHS] + energies[F_FOURDIHS] + energies[F_CMAP];
}

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
    if (parameters.stage < GaMDStage::BoostPreparation)
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
    //! Largest sampled energy in kJ/mol.
    double maximum = -1e99;
    //! Smallest sampled energy in kJ/mol.
    double minimum = 1e99;
    //! Mean from the most recently completed window, in kJ/mol.
    double average = 0.0;
    //! Population standard deviation in kJ/mol.
    double deviation = 0.001;
    double m2        = 0.0; //!< Retained for legacy restart compatibility.
    //! Number of accumulated samples.
    int64_t count = 0;
    //! Applied threshold energy in kJ/mol.
    double threshold = 0.0;
    //! Dimensionless normalized force constant.
    double k0 = 0.0;
    //! Applied force constant in inverse kJ/mol.
    double forceConstant = 0.0;
};

//! Welford accumulator for the current averaging window.
struct GaMDWindowStatistics
{
    //! Number of accumulated samples.
    int64_t count = 0;
    //! Current Welford window mean in kJ/mol.
    double mean = 0.0;
    //! Current sum of squared deviations in squared kJ/mol.
    double m2 = 0.0;
};

//! Accumulate one sample without changing the currently applied bias.
void updateGaMDStatistics(double                                  energy,
                          compat::not_null<GaMDEnergyStatistics*> statistics,
                          compat::not_null<GaMDWindowStatistics*> window);

//! Finish a nonempty window using population variance (M2 / count), then reset it.
void finalizeWindowStatistics(compat::not_null<GaMDWindowStatistics*> window,
                              compat::not_null<GaMDEnergyStatistics*> statistics);

//! Update E and k using the existing iE=1/2 policy; retain them for degenerate input.
void calculateThresholdAndForceConstant(int                                     thresholdMode,
                                        double                                  sigmaLimit,
                                        compat::not_null<GaMDEnergyStatistics*> statistics);

//! \}
//! \endcond

} // namespace gmx

#endif
