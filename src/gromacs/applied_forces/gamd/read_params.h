/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 */

#pragma once

#include "gromacs/mdtypes/gamd_params.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/fileio/warninp.h"   // ← 正确路径！（fileio 而不是 utility）

namespace gmx {

//! Check GaMD parameters for consistency (stub)
void checkGaMDParams(const GaMDParams& params, const t_inputrec& ir, WarningHandler* wi)
{
    (void)params;
    (void)ir;
    (void)wi;
}

} // namespace gmx
