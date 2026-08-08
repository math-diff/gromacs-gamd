/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 */

#pragma once

#include "gromacs/utility/real.h"

namespace gmx {

enum class GaMDBoostType {
    None,
    Lower,
    Upper,
    Dual,
    DihedralOnly,
    TotalOnly
};

struct GaMDParams {
    GaMDBoostType boostType = GaMDBoostType::Dual;
    real          sigma0    = 0.2;   // 目标标准差比例

    GaMDParams() = default;   // 极简默认构造，先让编译通过
};

} // namespace gmx
