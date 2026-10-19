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
/*! \libinternal \file
 * \brief Declares the typed GaMD plans, results and device-state layout.
 *
 * \inlibraryapi
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDTYPES_H
#define GMX_APPLIED_FORCES_GAMD_GAMDTYPES_H

#include <cstdint>

namespace gmx
{

//! \cond libapi
//! \addtogroup module_applied_forces
//! \{

//! 阶段名称用于代码判断，固定数值继续用于已有的文本输出。
enum class GaMDStage : int
{
    //! No active GaMD work.
    Inactive = 0,
    //! Unbiased preparation without statistics.
    ConventionalPreparation = 1,
    //! Unbiased sampling used to initialize the boost.
    ConventionalStatistics = 2,
    //! Preparation under the initialized bias.
    BoostPreparation = 3,
    //! Adaptive bias and population-statistics updates.
    BoostStatistics = 4,
    //! Production with fixed bias parameters.
    Production = 5
};

//! 给设备数组的六个位置命名，保持已有 CUDA 缓冲区布局不变。
enum GaMDDeviceStateIndex
{
    //! Raw total potential-energy slot.
    c_gaMDPotentialEnergy = 0,
    //! Raw dihedral-energy slot.
    c_gaMDDihedralEnergy = 1,
    //! Total-force-weight slot.
    c_gaMDPotentialScale = 2,
    //! Dihedral-force-weight slot.
    c_gaMDDihedralScale = 3,
    //! Potential-bias-energy slot.
    c_gaMDPotentialBoost = 4,
    //! Dihedral-bias-energy slot.
    c_gaMDDihedralBoost = 5,
    //! Number of floats in the device bias state.
    c_gaMDStateSize = 6
};

/*! \libinternal
 * \brief Host result produced after current-step GaMD energy accounting.
 */
struct GaMDStepResult
{
    //! Raw total potential energy in kJ/mol.
    double potentialEnergy = 0;
    //! Raw dihedral-partition energy in kJ/mol.
    double dihedralEnergy = 0;
    //! Dimensionless total-force weight.
    double scaleP = 1;
    //! Dimensionless dihedral-force weight.
    double scaleD = 1;
    //! Potential bias energy in kJ/mol.
    double boostP = 0;
    //! Dihedral bias energy in kJ/mol.
    double boostD = 0;

    //! Returns the sum of potential and dihedral bias energies in kJ/mol.
    double totalBoost() const { return boostP + boostD; }
};

/*! \libinternal
 * \brief Bias parameters shared by CPU, CUDA, and deferred output.
 */
struct GaMDBiasParameters
{
    //! Legacy boost selection: potential, dihedral or dual (1, 2 or 3).
    int boostType = 0;
    //! Stage controlling statistics and bias application.
    GaMDStage stage = GaMDStage::Inactive;
    //! Potential threshold in kJ/mol.
    double thresholdP = 0;
    //! Potential force constant in inverse kJ/mol.
    double kP = 0;
    //! Dihedral threshold in kJ/mol.
    double thresholdD = 0;
    //! Dihedral force constant in inverse kJ/mol.
    double kD = 0;
};

/*! \libinternal
 * \brief Work description prepared at the start of one GaMD step.
 */
struct GaMDStepPlan
{
    //! Simulation step represented by this work description.
    int64_t step = -1;
    //! Bias parameters of this step.
    GaMDBiasParameters bias;
    //! Whether host state must account for this step.
    bool requiresHostEnergy = false;
};

/*! \libinternal
 * \brief Fixed diagnostic inputs shared by the CPU and CUDA execution paths.
 */
struct GaMDForceOverrides
{
    //! Whether diagnostic force or boost overrides were explicitly requested.
    bool enabled = false;
    //! Configured total-force weight for a diagnostic override.
    double totalScale = 1;
    //! Configured dihedral-force weight for a diagnostic override.
    double dihedralScale = 1;
    //! Configured potential bias energy in kJ/mol.
    double potentialBoost = 0;
    //! Configured dihedral bias energy in kJ/mol.
    double dihedralBoost = 0;
};

//! \}
//! \endcond

} // namespace gmx

#endif
