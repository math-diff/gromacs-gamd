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

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"

#include "gamdgpu.h"

namespace gmx
{

class GaMDGpuExecutor::Impl
{
};

GaMDGpuExecutor::GaMDGpuExecutor(bool, const DeviceContext&, const DeviceStream&, gmx_wallcycle*) :
    impl_(nullptr)
{
    // 非 CUDA 构建保留完整接口，但绝不能进入 CUDA 专用执行路径。
    gmx_fatal(FARGS, "The GaMD GPU executor currently requires a CUDA build.");
}

GaMDGpuExecutor::~GaMDGpuExecutor() = default;

bool GaMDGpuExecutor::dihedralShadowEnabled() const
{
    return false;
}

bool GaMDGpuExecutor::dihedralBufferEnabled() const
{
    return false;
}

bool GaMDGpuExecutor::energyEnabled() const
{
    return false;
}

bool GaMDGpuExecutor::energyDiagnosticsEnabled() const
{
    return false;
}

bool GaMDGpuExecutor::scaleFromDeviceEnabled() const
{
    return false;
}

void GaMDGpuExecutor::bindListedForceBuffers(ArrayRef<const int>,
                                             const InteractionDefinitions&,
                                             DeviceBuffer<Float4>,
                                             DeviceBuffer<Float3>,
                                             DeviceBuffer<Float3>,
                                             DeviceBuffer<float>,
                                             DeviceBuffer<float>,
                                             DeviceBuffer<float>)
{
}

void GaMDGpuExecutor::setPbc(const PbcAiuc&) {}

GaMDBondedDeviceBuffers GaMDGpuExecutor::prepareBondedForceAccumulation()
{
    return {};
}

void GaMDGpuExecutor::finishBondedForceAccumulation() {}

DeviceBuffer<float> GaMDGpuExecutor::pmeEnergyStagingBuffer()
{
    return nullptr;
}

GpuEventSynchronizer* GaMDGpuExecutor::pmeEnergyReadyEvent()
{
    return nullptr;
}

void GaMDGpuExecutor::launchEnergyReduction(int, int, double, double, double, double, bool) {}

std::array<double, 6> GaMDGpuExecutor::energyDiagnostics()
{
    return {};
}

std::vector<std::array<real, F_NRE>> GaMDGpuExecutor::takeEnergyHistory()
{
    return {};
}

void GaMDGpuExecutor::recordForceVirialSample() {}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::takeForceVirialHistory(int)
{
    return {};
}

void GaMDGpuExecutor::copyDihedralShadowForces(ArrayRef<RVec>) {}

void GaMDGpuExecutor::copyDihedralVirial(std::array<real, DIM * DIM>*) {}

void GaMDGpuExecutor::launchShortRangeVirialReduction() {}

void GaMDGpuExecutor::copyShortRangeVirial(std::array<real, DIM * DIM>*) {}

void GaMDGpuExecutor::applyForceCorrectionShadow(ArrayRef<const RVec>, real, real, ArrayRef<RVec>)
{
}

void GaMDGpuExecutor::applyForceCorrection(DeviceBuffer<RVec>, real, real, GpuEventSynchronizer*) {}

GpuEventSynchronizer* GaMDGpuExecutor::forcesReadyEvent()
{
    return nullptr;
}

} // namespace gmx
