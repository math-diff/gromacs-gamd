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
 * \brief Declares typed data exchanged at GaMD integration boundaries.
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDTYPES_H
#define GMX_APPLIED_FORCES_GAMD_GAMDTYPES_H

#include <cstdint>

namespace gmx
{

/*! \internal
 * \brief Immutable work description prepared at the start of one GaMD step.
 */
struct GaMDStepPlan
{
    int64_t step               = -1;
    int     boostType          = 0;
    int     stage              = 0;
    int     debug              = 0;
    double  thresholdP         = 0;
    double  kP                 = 0;
    double  thresholdD         = 0;
    double  kD                 = 0;
    bool    enabled            = false;
    bool    checkpointing      = false;
    bool    writesParameters   = false;
    bool    writesReweight     = false;
    bool    requiresHostEnergy = false;
    bool    gpuGraphEligible   = false;
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
 * \brief Stable host parameters consumed by the GPU production-scale kernel.
 */
struct GaMDGpuProductionParameters
{
    int    igamd      = 0;
    int    stage      = 0;
    double thresholdP = 0;
    double kP         = 0;
    double thresholdD = 0;
    double kD         = 0;
};

} // namespace gmx

#endif
