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
 * \brief Implements the CPU/GPU comparisons beside the GaMD force correction.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gamdforceshadow.h"

#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/mdlib/calcvir.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"

#include "gamddiagnostics.h"
#include "gamdgpu.h"

//! \cond internal
namespace gmx
{

void GaMDForceShadow::captureBeforeGpuCorrection(int64_t step,
                                                 const InteractionDefinitions& interactionDefinitions,
                                                 const matrix              box,
                                                 ArrayRefWithPadding<RVec> coordinates,
                                                 ArrayRef<const RVec>      forces,
                                                 const int                 numLocalAtoms,
                                                 ArrayRef<const real>      freeEnergyParameters,
                                                 t_forcerec*               forceRecord,
                                                 t_nrnb*                   operationCounter,
                                                 const StepWorkload&       stepWork,
                                                 const real                totalScale,
                                                 const real                dihedralCorrectionScale,
                                                 const std::array<real, DIM * DIM>& deviceDihedralVirial,
                                                 GpuEventSynchronizer* rawForcesReady,
                                                 GaMDGpuExecutor*      gpuExecutor,
                                                 const MDLogger&       logger)
{
    // 只有本步施加二面角修正时，才用 CPU 重算的二面角力和维里对照设备结果。
    if (stepWork.stageGpuEnergyAndVirialToHost && dihedralCorrectionScale != 0.0)
    {
        deviceDihedralForces_.resize(forces.size());
        gpuExecutor->copyDihedralForces(deviceDihedralForces_);

        // GPU update leaves the host coordinate buffer stale on ordinary steps.
        // Refresh it only for this explicit CPU/GPU diagnostic comparison.
        forceRecord->stateGpu->copyCoordinatesFromGpu(coordinates.unpaddedArrayRef(), AtomLocality::Local);
        forceRecord->stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);

        t_pbc        pbc;
        const t_pbc* pbcPtr = nullptr;
        if (forceRecord->bMolPBC)
        {
            set_pbc_dd(&pbc, forceRecord->pbcType, nullptr, TRUE, box);
            pbcPtr = &pbc;
        }

        calculateGaMDDihedralListedForces(interactionDefinitions,
                                          coordinates,
                                          forces.ssize(),
                                          pbcPtr,
                                          forceRecord->fcdata.get(),
                                          operationCounter,
                                          freeEnergyParameters,
                                          nullptr,
                                          &cpuReference_);

        validateGaMDGpuShadowForces(
                logger, "live-dihedral", step, cpuReference_.forces, deviceDihedralForces_, 0.005, 0.0001);

        tensor cpuDihedralVirial = { { 0 } };
        calc_vir(c_numShiftVectors,
                 as_rvec_array(forceRecord->shift_vec.data()),
                 as_rvec_array(cpuReference_.shiftForces.data()),
                 cpuDihedralVirial,
                 forceRecord->pbcType == PbcType::Screw,
                 box);
        f_calc_vir(0,
                   numLocalAtoms,
                   as_rvec_array(coordinates.unpaddedArrayRef().data()),
                   as_rvec_array(cpuReference_.forces.data()),
                   cpuDihedralVirial,
                   box);
        validateGaMDGpuShadowVirial(logger, step, cpuDihedralVirial, deviceDihedralVirial);
    }

    // 设备修正会原地覆盖力，原始力必须在修正前取出；这次等待是缩减力事件的第二个消费者。
    ArrayRef<const RVec> rawForces = forces;
    if (stepWork.useGpuFBufferOps)
    {
        rawForces_.resize(forces.size());
        rawForcesReady->waitForEvent();
        forceRecord->stateGpu->copyForcesFromGpu(rawForces_, AtomLocality::Local);
        forceRecord->stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
        rawForces = makeConstArrayRef(rawForces_);
    }
    referenceCorrectedForces_.assign(rawForces.begin(), rawForces.end());
    for (Index atomIndex = 0; atomIndex < forces.ssize(); ++atomIndex)
    {
        referenceCorrectedForces_[atomIndex] *= totalScale;
        if (dihedralCorrectionScale != 0.0)
        {
            referenceCorrectedForces_[atomIndex] +=
                    dihedralCorrectionScale * cpuReference_.forces[atomIndex];
        }
    }
}

void GaMDForceShadow::compareAfterGpuCorrection(const int64_t           step,
                                                const StepWorkload&     stepWork,
                                                StatePropagatorDataGpu* stateGpu,
                                                GaMDGpuExecutor*        gpuExecutor,
                                                GaMDDiagnostics*        diagnostics,
                                                const bool              isMainRank)
{
    // 上面的等待和设备修正已各消费一次缩减力事件，把预期消费次数恢复为 1。
    if (stepWork.useGpuFBufferOps)
    {
        stateGpu->setFReducedOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
    }
    // The event has a second consumer in diagnostic mode; GPU update consumes the other.
    gpuExecutor->forcesReadyEvent()->waitForEvent();
    correctedForces_.resize(referenceCorrectedForces_.size());
    stateGpu->copyForcesFromGpu(correctedForces_, AtomLocality::Local);
    stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
    validateGaMDGpuShadowForces(diagnostics->logger(),
                                "live-force-correction",
                                step,
                                referenceCorrectedForces_,
                                correctedForces_,
                                0.005,
                                0.0001);
    diagnostics->dumpFinalForces(step, correctedForces_, isMainRank);
}

void GaMDForceShadow::captureBeforeCpuCorrection(ArrayRef<const RVec> forces)
{
    // 一次性对照完成前，每步都保留修正前的原始力。
    if (!correctionCompared_)
    {
        rawForces_.assign(forces.begin(), forces.end());
    }
}

void GaMDForceShadow::compareAfterCpuCorrection(const int64_t        step,
                                                ArrayRef<const RVec> correctedForces,
                                                ArrayRef<const RVec> cpuDihedralForces,
                                                const real           totalScale,
                                                const real           dihedralCorrectionScale,
                                                GaMDGpuExecutor*     gpuExecutor,
                                                const MDLogger&      logger)
{
    // 影子模式已拒绝虚拟位点，CPU 缓冲里仍是未经展开的原始二面角力。
    if (!cpuDihedralForces.empty() && !dihedralCompared_)
    {
        deviceDihedralForces_.resize(cpuDihedralForces.size());
        gpuExecutor->copyDihedralForces(deviceDihedralForces_);
        validateGaMDGpuShadowForces(
                logger, "dihedral", step, cpuDihedralForces, deviceDihedralForces_, 0.1, 0.01);
        dihedralCompared_ = true;
    }
    if (!correctionCompared_)
    {
        correctedForces_.resize(correctedForces.size());
        gpuExecutor->applyForceCorrectionShadow(
                rawForces_, totalScale, dihedralCorrectionScale, correctedForces_);
        validateGaMDGpuShadowForces(
                logger, "force-correction", step, correctedForces, correctedForces_, 0.1, 0.01);
        correctionCompared_ = true;
    }
}

} // namespace gmx
//! \endcond
