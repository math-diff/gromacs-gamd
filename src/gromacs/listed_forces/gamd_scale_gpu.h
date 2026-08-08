/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026 The GROMACS Authors
 */
#ifndef GMX_LISTED_FORCES_GAMD_SCALE_GPU_H
#define GMX_LISTED_FORCES_GAMD_SCALE_GPU_H

#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gputraits.h"

namespace gmx
{

/*! \brief Evaluates production GaMD scales from device raw energies.
 *
 * Double precision is intentionally confined to this scalar-only kernel so
 * that force-producing CUDA kernels remain single precision.
 */
void launchGamdProductionScaleKernel(DeviceBuffer<float> gamdState,
                                     int                 igamd,
                                     int                 stage,
                                     double              thresholdP,
                                     double              kP,
                                     double              thresholdD,
                                     double              kD,
                                     const DeviceStream& deviceStream);

} // namespace gmx

#endif // GMX_LISTED_FORCES_GAMD_SCALE_GPU_H
