/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 */
#include "gmxpre.h"

#include "gamdexecution.h"

#include "config.h"

#include "gromacs/utility/fatalerror.h"

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
    if (!useGaMD || !useGpuUpdate)
    {
        return GaMDExecutionMode::CpuReference;
    }

    if (!GMX_GPU_CUDA)
    {
        gmx_fatal(FARGS, "GPU-update GaMD currently requires a CUDA build.");
    }
    if (!useGpuNonbonded)
    {
        gmx_fatal(FARGS, "GPU-update GaMD requires -nb gpu.");
    }
    if (!useGpuPme)
    {
        gmx_fatal(FARGS, "GPU-update GaMD requires -pme gpu for the current production target.");
    }
    if (!useGpuBonded)
    {
        gmx_fatal(FARGS, "GPU-update GaMD requires -bonded gpu.");
    }
    if (havePpDomainDecomposition || haveSeparatePmeRank)
    {
        gmx_fatal(FARGS,
                  "GPU-update GaMD currently supports one PP rank without domain decomposition "
                  "or a separate PME rank.");
    }
    if (useMts)
    {
        gmx_fatal(FARGS, "GPU-update GaMD does not support MTS.");
    }
    if (pressureCoupling != PressureCoupling::No && pressureCoupling != PressureCoupling::CRescale)
    {
        gmx_fatal(FARGS,
                  "GPU-update GaMD supports only no pressure coupling or C-rescale pressure "
                  "coupling.");
    }
    if (pressureCoupling == PressureCoupling::CRescale && pressureCouplingType != PressureCouplingType::Isotropic
        && pressureCouplingType != PressureCouplingType::SemiIsotropic)
    {
        gmx_fatal(FARGS,
                  "GPU-update GaMD with C-rescale supports only isotropic or semiisotropic "
                  "pressure coupling.");
    }
    if (nstfout != 0)
    {
        gmx_fatal(FARGS,
                  "GPU-update GaMD currently requires nstfout=0 because corrected-force output "
                  "staging is not implemented.");
    }

    return GaMDExecutionMode::GpuResident;
}

} // namespace gmx
