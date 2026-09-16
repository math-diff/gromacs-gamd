/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2018- The GROMACS Authors
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
 * \brief Provides the non-CUDA implementation of the GaMD device interface.
 *
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"

#include "gamdgpu.h"

//! \cond internal
namespace gmx
{

class GaMDGpuExecutor::Impl
{
};

GaMDGpuExecutor::GaMDGpuExecutor(bool, const DeviceContext&, const DeviceStream&, gmx_wallcycle*) :
    impl_(nullptr)
{
    // 非 CUDA 构建保留完整接口，但绝不能进入 CUDA 专用执行路径。
    GMX_THROW(NotImplementedError("The GaMD GPU executor currently requires a CUDA build."));
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

void GaMDGpuExecutor::launchEnergyReduction(const GaMDBiasParameters&, const GaMDForceOverrides&, bool)
{
}

GaMDStepResult GaMDGpuExecutor::energyDiagnostics()
{
    return {};
}

GaMDHistoryBatch GaMDGpuExecutor::takeHistory()
{
    return {};
}

void GaMDGpuExecutor::recordForceVirialSample() {}

void GaMDGpuExecutor::initializeUpdateHistory(int, const DeviceStream&) {}

void GaMDGpuExecutor::bindUpdateBuffers(DeviceBuffer<Float3>,
                                        ArrayRef<const real>,
                                        ArrayRef<const unsigned short>)
{
}

void GaMDGpuExecutor::recordConstraintVirial(DeviceBuffer<float>, bool, DeviceBuffer<float>, bool, real)
{
}

void GaMDGpuExecutor::recordKineticEnergy(bool) {}

std::vector<std::array<real, DIM * DIM>> GaMDGpuExecutor::previousHalfStepKineticEnergy()
{
    return {};
}

void GaMDGpuExecutor::copyDihedralForces(ArrayRef<RVec>) {}

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
//! \endcond
