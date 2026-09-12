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
 * \brief Declares typed data exchanged at GaMD integration boundaries.
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDTYPES_H
#define GMX_APPLIED_FORCES_GAMD_GAMDTYPES_H

#include <cstdint>

namespace gmx
{

//! 阶段名称用于代码判断，固定数值继续用于已有的文本输出。
enum class GaMDStage : int
{
    Inactive                = 0,
    ConventionalPreparation = 1,
    ConventionalStatistics  = 2,
    BoostPreparation        = 3,
    BoostStatistics         = 4,
    Production              = 5
};

//! 给设备数组的六个位置命名，保持已有 CUDA 缓冲区布局不变。
enum GaMDDeviceStateIndex
{
    c_gaMDPotentialEnergy = 0,
    c_gaMDDihedralEnergy  = 1,
    c_gaMDPotentialScale  = 2,
    c_gaMDDihedralScale   = 3,
    c_gaMDPotentialBoost  = 4,
    c_gaMDDihedralBoost   = 5,
    c_gaMDStateSize       = 6
};

/*! \internal
 * \brief Immutable work description prepared at the start of one GaMD step.
 */
struct GaMDStepPlan
{
    int64_t   step               = -1;
    int       boostType          = 0;
    GaMDStage stage              = GaMDStage::Inactive;
    int       debug              = 0;
    double    thresholdP         = 0;
    double    kP                 = 0;
    double    thresholdD         = 0;
    double    kD                 = 0;
    bool      enabled            = false;
    bool      checkpointing      = false;
    bool      writesParameters   = false;
    bool      writesReweight     = false;
    bool      requiresHostEnergy = false;
    bool      gpuGraphEligible   = false;
};

/*! \internal
 * \brief Host result produced after current-step GaMD energy accounting.
 */
struct GaMDStepResult
{
    double potentialEnergy = 0;
    double dihedralEnergy  = 0;
    double scaleP          = 1;
    double scaleD          = 1;
    double boostP          = 0;
    double boostD          = 0;

    double totalBoost() const { return boostP + boostD; }
};

/*! \internal
 * \brief Bias parameters shared by CPU, CUDA, and deferred output.
 */
struct GaMDBiasParameters
{
    int       boostType  = 0;
    GaMDStage stage      = GaMDStage::Inactive;
    double    thresholdP = 0;
    double    kP         = 0;
    double    thresholdD = 0;
    double    kD         = 0;
};

/*! \internal
 * \brief Fixed diagnostic inputs shared by the CPU and CUDA execution paths.
 */
struct GaMDForceOverrides
{
    bool   enabled        = false;
    double totalScale     = 1;
    double dihedralScale  = 1;
    double potentialBoost = 0;
    double dihedralBoost  = 0;
};

} // namespace gmx

#endif
