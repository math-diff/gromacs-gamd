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
 * \brief Validates the supported GaMD execution settings.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gamdexecution.h"

#include "config.h"

#include "gromacs/utility/exceptions.h"

//! \cond internal
namespace gmx
{

const char* gaMDExecutionModeName(GaMDExecutionMode mode)
{
    switch (mode)
    {
        case GaMDExecutionMode::CpuReference: return "CPU reference";
        case GaMDExecutionMode::GpuResident: return "GPU resident";
    }
    return "unknown";
}

GaMDExecutionMode selectGaMDExecutionMode(bool                 useGaMD,
                                          bool                 useGpuNonbonded,
                                          bool                 useGpuPme,
                                          bool                 useGpuBonded,
                                          bool                 useGpuUpdate,
                                          bool                 havePpDomainDecomposition,
                                          bool                 haveSeparatePmeRank,
                                          bool                 useMts,
                                          PressureCoupling     pressureCoupling,
                                          PressureCouplingType pressureCouplingType,
                                          int                  nstfout)
{
    if (!useGaMD)
    {
        return GaMDExecutionMode::CpuReference;
    }

    // CPU 和 GPU 路径都必须先拒绝尚未实现的全局能量/力归约。
    if (havePpDomainDecomposition)
    {
        GMX_THROW(InconsistentInputError(
                "GaMD currently supports one PP rank without domain decomposition."));
    }
    if (useMts)
    {
        GMX_THROW(InconsistentInputError("GaMD does not support MTS."));
    }
    if (!useGpuUpdate)
    {
        return GaMDExecutionMode::CpuReference;
    }

    if (!GMX_GPU_CUDA)
    {
        GMX_THROW(InconsistentInputError("GPU-update GaMD currently requires a CUDA build."));
    }
    if (!useGpuNonbonded)
    {
        GMX_THROW(InconsistentInputError("GPU-update GaMD requires -nb gpu."));
    }
    if (!useGpuPme)
    {
        GMX_THROW(InconsistentInputError(
                "GPU-update GaMD requires -pme gpu for the current production target."));
    }
    if (!useGpuBonded)
    {
        GMX_THROW(InconsistentInputError("GPU-update GaMD requires -bonded gpu."));
    }
    if (haveSeparatePmeRank)
    {
        GMX_THROW(InconsistentInputError(
                "GPU-update GaMD currently supports one PP rank without domain decomposition "
                "or a separate PME rank."));
    }
    if (pressureCoupling != PressureCoupling::No && pressureCoupling != PressureCoupling::CRescale)
    {
        GMX_THROW(InconsistentInputError(
                "GPU-update GaMD supports only no pressure coupling or C-rescale pressure "
                "coupling."));
    }
    if (pressureCoupling == PressureCoupling::CRescale && pressureCouplingType != PressureCouplingType::Isotropic
        && pressureCouplingType != PressureCouplingType::SemiIsotropic)
    {
        GMX_THROW(InconsistentInputError(
                "GPU-update GaMD with C-rescale supports only isotropic or semiisotropic "
                "pressure coupling."));
    }
    if (nstfout != 0)
    {
        GMX_THROW(InconsistentInputError(
                "GPU-update GaMD currently requires nstfout=0 because corrected-force output "
                "staging is not implemented."));
    }

    return GaMDExecutionMode::GpuResident;
}

} // namespace gmx
//! \endcond
