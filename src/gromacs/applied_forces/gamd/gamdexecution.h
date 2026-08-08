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
/*! \internal \file
 * \brief Declares the single GaMD execution-mode selection boundary.
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDEXECUTION_H
#define GMX_APPLIED_FORCES_GAMD_GAMDEXECUTION_H

#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/simulation_workload.h"

namespace gmx
{

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
                                          int                  nstfout);

const char* gaMDExecutionModeName(GaMDExecutionMode mode);

} // namespace gmx

#endif
