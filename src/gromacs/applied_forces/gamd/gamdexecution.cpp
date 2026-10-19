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
 * \brief Selects GaMD execution and applies current-step CPU/GPU force corrections.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gamdexecution.h"

#include "config.h"

#include <algorithm>
#include <array>
#include <vector>

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/listed_forces/bonded.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/calcvir.h"
#include "gromacs/mdlib/enerdata_utils.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdlib/vsite.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/state_propagator_data_gpu.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "gamddiagnostics.h"
#include "gamdgpu.h"
#include "gamdmath.h"
#include "gamdmdmodule.h"

//! \cond internal
namespace gmx
{
namespace
{

//! Recomputes raw proper-dihedral, RB/Fourier and CMAP forces for CPU work.
void calculateGaMDDihedralListedForces(const InteractionDefinitions& interactionDefinitions,
                                       gmx::ArrayRefWithPadding<const gmx::RVec> coordinates,
                                       gmx::ArrayRef<gmx::RVec>                  forces,
                                       gmx::ArrayRef<gmx::RVec>                  shiftForces,
                                       const struct t_pbc*                       pbc,
                                       t_fcdata*                 forceCalculationData,
                                       t_nrnb*                   operationCounter,
                                       gmx::ArrayRef<const real> freeEnergyParameters,
                                       int*                      globalAtomIndices)
{
    const rvec* coordinateArray = as_rvec_array(coordinates.paddedArrayRef().data());

    GMX_RELEASE_ASSERT(forceCalculationData != nullptr,
                       "GaMD dihedral force calculation requires force calculation data");
    std::fill(shiftForces.begin(), shiftForces.end(), gmx::RVec(0.0_real, 0.0_real, 0.0_real));

    thread_local static std::vector<real> scratchForceBuffer;
    scratchForceBuffer.resize(forces.size() * sizeof(rvec4) / sizeof(real));
    std::fill(scratchForceBuffer.begin(), scratchForceBuffer.end(), 0.0_real);

    rvec4*     scratchForces        = reinterpret_cast<rvec4*>(scratchForceBuffer.data());
    rvec*      scratchShift         = as_rvec_array(shiftForces.data());
    real       freeEnergyDerivative = 0.0_real;
    const real bondedLambda =
            freeEnergyParameters[static_cast<int>(FreeEnergyPerturbationCouplingType::Bonded)];

    for (const int forceType : c_gaMDDihedralInteractionTypes)
    {
        const InteractionList& interactionList = interactionDefinitions.il[forceType];
        if (interactionList.empty())
        {
            continue;
        }

        if (forceType == F_CMAP)
        {
            cmap_dihs(interactionList.size(),
                      interactionList.iatoms.data(),
                      interactionDefinitions.iparams.data(),
                      &interactionDefinitions.cmap_grid,
                      coordinateArray,
                      scratchForces,
                      scratchShift,
                      pbc,
                      bondedLambda,
                      &freeEnergyDerivative,
                      {},
                      forceCalculationData,
                      nullptr,
                      nullptr,
                      globalAtomIndices);
        }
        else
        {
            calculateSimpleBond(forceType,
                                interactionList.size(),
                                interactionList.iatoms.data(),
                                interactionDefinitions.iparams.data(),
                                coordinateArray,
                                scratchForces,
                                scratchShift,
                                pbc,
                                bondedLambda,
                                &freeEnergyDerivative,
                                {},
                                forceCalculationData,
                                forceCalculationData->disres,
                                forceCalculationData->orires.get(),
                                globalAtomIndices,
                                BondedKernelFlavor::ForcesNoSimd);
        }

        inc_nrnb(operationCounter,
                 nrnbIndex(forceType),
                 interactionList.size() / (interaction_function[forceType].nratoms + 1));
    }

    // 输出会被完整覆盖；只需清零上面的累加缓冲。
    for (gmx::Index atomIndex = 0; atomIndex < forces.ssize(); ++atomIndex)
    {
        forces[atomIndex][XX] = scratchForces[atomIndex][XX];
        forces[atomIndex][YY] = scratchForces[atomIndex][YY];
        forces[atomIndex][ZZ] = scratchForces[atomIndex][ZZ];
    }
}

//! GPU 分支保留原来的事件、回传和修正顺序，诊断比较由诊断模块负责。
void applyGaMDGpuForceCorrection(int64_t                       step,
                                 const InteractionDefinitions& interactionDefinitions,
                                 const matrix                  box,
                                 ArrayRefWithPadding<RVec>     coordinates,
                                 ArrayRef<RVec>                forces,
                                 tensor                        forceVirial,
                                 int                           numLocalAtoms,
                                 ArrayRef<const real>          freeEnergyParameters,
                                 t_forcerec*                   forceRecord,
                                 t_nrnb*                       operationCounter,
                                 gmx_wallcycle*                wallCycle,
                                 GaMDMDModule&                 module,
                                 const StepWorkload&           stepWork,
                                 bool                          isMainRank)
{
    GaMDHostWorkspace&    gaMDHostWorkspace       = module.diagnostics().hostWorkspace();
    GaMDGpuExecutor*      gpuExecutor             = module.gpuExecutor();
    const GaMDStepResult& stepResult              = module.stepResult();
    real                  totalScale              = 1.0_real;
    real                  dihedralCorrectionScale = 0.0_real;
    if (stepWork.stageGpuEnergyAndVirialToHost)
    {
        totalScale              = static_cast<real>(stepResult.scaleP);
        dihedralCorrectionScale = static_cast<real>(stepResult.scaleP * (stepResult.scaleD - 1.0));
    }

    GMX_RELEASE_ASSERT(gpuExecutor != nullptr, "GPU GaMD requires its module-owned executor");
    GMX_RELEASE_ASSERT(forceRecord->stateGpu != nullptr && forceRecord->listedForcesGpu != nullptr,
                       "GPU GaMD force correction requires GPU state and listed-force objects");
    GMX_RELEASE_ASSERT(gpuExecutor->dihedralBufferEnabled(),
                       "GPU GaMD force correction requires the device dihedral-force buffer");

    const bool              validateLiveGpuCorrection = gpuExecutor->dihedralShadowEnabled();
    std::vector<gmx::RVec>& gamdGpuModeDihedralForces = gaMDHostWorkspace.gpuModeDihedralForces;
    std::vector<gmx::RVec>& gamdGpuModeDihedralShiftForces = gaMDHostWorkspace.gpuModeDihedralShiftForces;
    std::vector<gmx::RVec>&     gpuShadowDihedralForces = gaMDHostWorkspace.gpuShadowDihedralForces;
    std::array<real, DIM * DIM> gpuDihedralVirial       = {};

    const bool writesDihedralForces =
            isMainRank && module.diagnostics().dihedralForceOutputRequested(step);
    if (writesDihedralForces
        || (validateLiveGpuCorrection && stepWork.stageGpuEnergyAndVirialToHost
            && dihedralCorrectionScale != 0.0))
    {
        gpuShadowDihedralForces.resize(forces.size());
        gpuExecutor->copyDihedralForces(gpuShadowDihedralForces);
    }
    if (writesDihedralForces)
    {
        GMX_RELEASE_ASSERT(stepWork.stageGpuEnergyAndVirialToHost,
                           "GaMD dihedral diagnostics require current host scale values");
        module.diagnostics().dumpDihedralForces(
                step,
                makeConstArrayRef(gpuShadowDihedralForces).subArray(0, numLocalAtoms),
                stepResult.scaleP * stepResult.scaleD,
                isMainRank);
    }

    if (stepWork.stageGpuEnergyAndVirialToHost && totalScale != 1.0)
    {
        msmul(forceVirial, totalScale, forceVirial);
    }

    if (stepWork.stageGpuEnergyAndVirialToHost && dihedralCorrectionScale != 0.0)
    {
        wallcycle_start(wallCycle, WallCycleCounter::GaMDDihedral);
        gpuExecutor->copyDihedralVirial(&gpuDihedralVirial);

        if (validateLiveGpuCorrection)
        {
            // GPU update leaves the host coordinate buffer stale on ordinary steps.
            // Refresh it only for this explicit CPU/GPU diagnostic comparison.
            forceRecord->stateGpu->copyCoordinatesFromGpu(coordinates.unpaddedArrayRef(),
                                                          AtomLocality::Local);
            forceRecord->stateGpu->waitCoordinatesReadyOnHost(AtomLocality::Local);

            gamdGpuModeDihedralForces.resize(forces.ssize(), gmx::RVec(0.0_real, 0.0_real, 0.0_real));
            gamdGpuModeDihedralShiftForces.resize(gmx::c_numShiftVectors,
                                                  gmx::RVec(0.0_real, 0.0_real, 0.0_real));

            t_pbc        pbc;
            const t_pbc* pbcPtr = nullptr;
            if (forceRecord->bMolPBC)
            {
                set_pbc_dd(&pbc, forceRecord->pbcType, nullptr, TRUE, box);
                pbcPtr = &pbc;
            }

            calculateGaMDDihedralListedForces(interactionDefinitions,
                                              coordinates,
                                              gamdGpuModeDihedralForces,
                                              gamdGpuModeDihedralShiftForces,
                                              pbcPtr,
                                              forceRecord->fcdata.get(),
                                              operationCounter,
                                              freeEnergyParameters,
                                              nullptr);

            validateGaMDGpuShadowForces(module.diagnostics().logger(),
                                        "live-dihedral",
                                        step,
                                        gamdGpuModeDihedralForces,
                                        gpuShadowDihedralForces,
                                        0.005,
                                        0.0001);

            tensor cpuDihedralVirial = { { 0 } };
            calc_vir(c_numShiftVectors,
                     as_rvec_array(forceRecord->shift_vec.data()),
                     as_rvec_array(gamdGpuModeDihedralShiftForces.data()),
                     cpuDihedralVirial,
                     forceRecord->pbcType == PbcType::Screw,
                     box);
            f_calc_vir(0,
                       numLocalAtoms,
                       as_rvec_array(coordinates.unpaddedArrayRef().data()),
                       as_rvec_array(gamdGpuModeDihedralForces.data()),
                       cpuDihedralVirial,
                       box);
            validateGaMDGpuShadowVirial(
                    module.diagnostics().logger(), step, cpuDihedralVirial, gpuDihedralVirial);
        }
        wallcycle_stop(wallCycle, WallCycleCounter::GaMDDihedral);

        if (stepWork.computeVirial)
        {
            wallcycle_start(wallCycle, WallCycleCounter::GaMDForceScale);
            for (int rowIndex = 0; rowIndex < DIM; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < DIM; ++columnIndex)
                {
                    forceVirial[rowIndex][columnIndex] +=
                            dihedralCorrectionScale * gpuDihedralVirial[rowIndex * DIM + columnIndex];
                }
            }
            wallcycle_stop(wallCycle, WallCycleCounter::GaMDForceScale);
        }
    }

    std::vector<gmx::RVec>& referenceCorrectedForces  = gaMDHostWorkspace.referenceCorrectedForces;
    std::vector<gmx::RVec>& liveGpuCorrectedForces    = gaMDHostWorkspace.liveGpuCorrectedForces;
    std::vector<gmx::RVec>& rawReducedForcesForShadow = gaMDHostWorkspace.rawReducedForcesForShadow;

    GpuEventSynchronizer* rawForcesReady = nullptr;
    if (stepWork.useGpuFBufferOps)
    {
        rawForcesReady = forceRecord->stateGpu->fReducedOnDevice(AtomLocality::Local);
        if (validateLiveGpuCorrection)
        {
            rawReducedForcesForShadow.resize(forces.size());
            rawForcesReady->waitForEvent();
            forceRecord->stateGpu->copyForcesFromGpu(rawReducedForcesForShadow, AtomLocality::Local);
            forceRecord->stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
        }
    }
    else
    {
        forceRecord->stateGpu->copyForcesToGpu(forces, AtomLocality::Local);
        rawForcesReady = forceRecord->stateGpu->fReadyOnDevice(AtomLocality::Local);
    }
    if (validateLiveGpuCorrection)
    {
        ArrayRef<const RVec> rawReference = forces;
        if (stepWork.useGpuFBufferOps)
        {
            rawReference = makeConstArrayRef(rawReducedForcesForShadow);
        }
        referenceCorrectedForces.assign(rawReference.begin(), rawReference.end());
        liveGpuCorrectedForces.resize(forces.size());
        for (gmx::Index atomIndex = 0; atomIndex < forces.ssize(); ++atomIndex)
        {
            referenceCorrectedForces[atomIndex] *= totalScale;
            if (dihedralCorrectionScale != 0.0)
            {
                referenceCorrectedForces[atomIndex] +=
                        dihedralCorrectionScale * gamdGpuModeDihedralForces[atomIndex];
            }
        }
    }
    gpuExecutor->applyForceCorrection(
            forceRecord->stateGpu->getForces(), totalScale, dihedralCorrectionScale, rawForcesReady);
    if (stepWork.useGpuFBufferOps && validateLiveGpuCorrection)
    {
        forceRecord->stateGpu->setFReducedOnDeviceEventExpectedConsumptionCount(AtomLocality::Local, 1);
    }
    if (validateLiveGpuCorrection)
    {
        // The event has a second consumer in diagnostic mode; GPU update consumes the other.
        gpuExecutor->forcesReadyEvent()->waitForEvent();
        forceRecord->stateGpu->copyForcesFromGpu(liveGpuCorrectedForces, AtomLocality::Local);
        forceRecord->stateGpu->waitForcesReadyOnHost(AtomLocality::Local);
        validateGaMDGpuShadowForces(module.diagnostics().logger(),
                                    "live-force-correction",
                                    step,
                                    referenceCorrectedForces,
                                    liveGpuCorrectedForces,
                                    0.005,
                                    0.0001);
        module.diagnostics().dumpFinalForces(step, liveGpuCorrectedForces, isMainRank);
    }
}

//! CPU 分支按总力缩放、二面角修正、最终诊断的顺序执行。
void applyGaMDCpuForceCorrection(const t_commrec&              communicationRecord,
                                 int64_t                       step,
                                 const InteractionDefinitions& interactionDefinitions,
                                 const matrix                  box,
                                 ArrayRefWithPadding<RVec>     coordinates,
                                 ArrayRef<RVec>                forces,
                                 tensor                        forceVirial,
                                 int                           numLocalAtoms,
                                 ArrayRef<const real>          freeEnergyParameters,
                                 t_forcerec*                   forceRecord,
                                 VirtualSitesHandler*          virtualSites,
                                 t_nrnb*                       operationCounter,
                                 gmx_wallcycle*                wallCycle,
                                 GaMDMDModule&                 module,
                                 const StepWorkload&           stepWork)
{
    GaMDHostWorkspace&    gaMDHostWorkspace       = module.diagnostics().hostWorkspace();
    GaMDGpuExecutor*      gpuExecutor             = module.gpuExecutor();
    const GaMDStepResult& stepResult              = module.stepResult();
    real                  totalScale              = 1.0_real;
    real                  dihedralCorrectionScale = 0.0_real;
    if (stepWork.stageGpuEnergyAndVirialToHost)
    {
        totalScale              = static_cast<real>(stepResult.scaleP);
        dihedralCorrectionScale = static_cast<real>(stepResult.scaleP * (stepResult.scaleD - 1.0));
    }

    const bool writesDihedralForces =
            MAIN(&communicationRecord) && module.diagnostics().dihedralForceOutputRequested(step);
    // 影子与 DD 原子排序、虚拟位点的组合已在创建 GPU executor 时拒绝。
    const bool gpuShadowEnabled = (forceRecord->listedForcesGpu != nullptr && gpuExecutor != nullptr
                                   && gpuExecutor->dihedralShadowEnabled());
    const bool usesDomainAtomOrdering = haveDDAtomOrdering(communicationRecord);
    bool&                   correctionShadowCompared = gaMDHostWorkspace.correctionShadowCompared;
    std::vector<gmx::RVec>& rawForcesForGpuShadow = gaMDHostWorkspace.rawForcesForCorrectionShadow;
    std::vector<gmx::RVec>& correctedGpuShadowForces = gaMDHostWorkspace.correctedForcesForShadow;
    if (gpuShadowEnabled && !correctionShadowCompared)
    {
        rawForcesForGpuShadow.assign(forces.begin(), forces.end());
        correctedGpuShadowForces.resize(forces.size());
    }

    if (totalScale != 1.0)
    {
        wallcycle_start(wallCycle, WallCycleCounter::GaMDForceScale);
        const int numThreads =
                gmx_omp_nthreads_get_simple_rvec_task(ModuleMultiThread::Default, forces.ssize());
#pragma omp parallel for num_threads(numThreads) schedule(static)
        for (gmx::Index atomIndex = 0; atomIndex < forces.ssize(); ++atomIndex)
        {
            forces[atomIndex] *= totalScale;
        }

        msmul(forceVirial, totalScale, forceVirial);
        wallcycle_stop(wallCycle, WallCycleCounter::GaMDForceScale);
    }

    if (dihedralCorrectionScale != 0.0 || writesDihedralForces)
    {
        wallcycle_start(wallCycle, WallCycleCounter::GaMDDihedral);

        std::vector<gmx::RVec>& gamdDihedralForces      = gaMDHostWorkspace.cpuDihedralForces;
        std::vector<gmx::RVec>& gamdDihedralShiftForces = gaMDHostWorkspace.cpuDihedralShiftForces;
        gamdDihedralForces.resize(forces.ssize(), gmx::RVec(0.0_real, 0.0_real, 0.0_real));
        gamdDihedralShiftForces.resize(gmx::c_numShiftVectors, gmx::RVec(0.0_real, 0.0_real, 0.0_real));

        t_pbc        pbc;
        const t_pbc* pbcPtr            = nullptr;
        const IVec*  domainCells       = nullptr;
        int*         globalAtomIndices = nullptr;
        // 单 rank 的 CPU 运行也可能使用 DD 式排序，仍要传递原来的原子映射。
        if (usesDomainAtomOrdering)
        {
            domainCells       = &communicationRecord.dd->numCells;
            globalAtomIndices = communicationRecord.dd->globalAtomIndices.data();
        }
        if (forceRecord->bMolPBC)
        {
            set_pbc_dd(&pbc, forceRecord->pbcType, domainCells, TRUE, box);
            pbcPtr = &pbc;
        }

        calculateGaMDDihedralListedForces(interactionDefinitions,
                                          coordinates,
                                          gamdDihedralForces,
                                          gamdDihedralShiftForces,
                                          pbcPtr,
                                          forceRecord->fcdata.get(),
                                          operationCounter,
                                          freeEnergyParameters,
                                          globalAtomIndices);
        if (gpuShadowEnabled)
        {
            bool&                   shadowCompared  = gaMDHostWorkspace.dihedralShadowCompared;
            std::vector<gmx::RVec>& gpuShadowForces = gaMDHostWorkspace.gpuDihedralForcesForShadow;
            if (!shadowCompared)
            {
                gpuShadowForces.resize(gamdDihedralForces.size());
                gpuExecutor->copyDihedralForces(gpuShadowForces);
                validateGaMDGpuShadowForces(module.diagnostics().logger(),
                                            "dihedral",
                                            step,
                                            gamdDihedralForces,
                                            gpuShadowForces,
                                            0.1,
                                            0.01);
                shadowCompared = true;
            }
        }

        if (virtualSites)
        {
            using VirialHandling          = VirtualSitesHandler::VirialHandling;
            VirialHandling virialHandling = VirialHandling::None;
            if (stepWork.computeVirial)
            {
                virialHandling = VirialHandling::Pbc;
            }
            virtualSites->spreadForces(coordinates.unpaddedArrayRef(),
                                       gamdDihedralForces,
                                       virialHandling,
                                       gamdDihedralShiftForces,
                                       nullptr,
                                       operationCounter,
                                       box,
                                       wallCycle);
        }

        wallcycle_stop(wallCycle, WallCycleCounter::GaMDDihedral);

        if (writesDihedralForces)
        {
            ArrayRef<const int> outputAtomIndices;
            if (usesDomainAtomOrdering)
            {
                outputAtomIndices = arrayRefFromArray(globalAtomIndices, numLocalAtoms);
            }
            module.diagnostics().dumpDihedralForces(
                    step,
                    makeConstArrayRef(gamdDihedralForces).subArray(0, numLocalAtoms),
                    stepResult.scaleP * stepResult.scaleD,
                    MAIN(&communicationRecord),
                    outputAtomIndices);
        }

        if (dihedralCorrectionScale != 0.0)
        {
            wallcycle_start(wallCycle, WallCycleCounter::GaMDForceScale);
            const int numThreads =
                    gmx_omp_nthreads_get_simple_rvec_task(ModuleMultiThread::Default, forces.ssize());
#pragma omp parallel for num_threads(numThreads) schedule(static)
            for (gmx::Index atomIndex = 0; atomIndex < forces.ssize(); ++atomIndex)
            {
                forces[atomIndex] += dihedralCorrectionScale * gamdDihedralForces[atomIndex];
            }

            if (stepWork.computeVirial)
            {
                tensor dihedralVirial = { { 0 } };
                calc_vir(c_numShiftVectors,
                         as_rvec_array(forceRecord->shift_vec.data()),
                         as_rvec_array(gamdDihedralShiftForces.data()),
                         dihedralVirial,
                         forceRecord->pbcType == PbcType::Screw,
                         box);
                inc_nrnb(operationCounter, eNR_VIRIAL, c_numShiftVectors);

                f_calc_vir(0,
                           numLocalAtoms,
                           as_rvec_array(coordinates.unpaddedArrayRef().data()),
                           as_rvec_array(gamdDihedralForces.data()),
                           dihedralVirial,
                           box);
                inc_nrnb(operationCounter, eNR_VIRIAL, numLocalAtoms);

                for (int rowIndex = 0; rowIndex < DIM; ++rowIndex)
                {
                    for (int columnIndex = 0; columnIndex < DIM; ++columnIndex)
                    {
                        forceVirial[rowIndex][columnIndex] +=
                                dihedralCorrectionScale * dihedralVirial[rowIndex][columnIndex];
                    }
                }
            }
            wallcycle_stop(wallCycle, WallCycleCounter::GaMDForceScale);
        }
    }

    if (gpuShadowEnabled && !correctionShadowCompared)
    {
        gpuExecutor->applyForceCorrectionShadow(
                rawForcesForGpuShadow, totalScale, dihedralCorrectionScale, correctedGpuShadowForces);
        validateGaMDGpuShadowForces(
                module.diagnostics().logger(), "force-correction", step, forces, correctedGpuShadowForces, 0.1, 0.01);
        correctionShadowCompared = true;
    }
}

} // namespace


const char* gaMDExecutionModeName(GaMDExecutionMode mode)
{
    switch (mode)
    {
        case GaMDExecutionMode::Disabled: return "disabled";
        case GaMDExecutionMode::CpuReference: return "CPU reference";
        case GaMDExecutionMode::GpuResident: return "GPU resident";
    }
    return "unknown";
}

GaMDExecutionMode selectGaMDExecutionMode(const t_inputrec& inputrec,
                                          bool              useGpuNonbonded,
                                          bool              useGpuPme,
                                          bool              useGpuBonded,
                                          bool              useGpuUpdate,
                                          bool              havePpDomainDecomposition,
                                          bool              haveDDAtomOrdering,
                                          bool              haveSeparatePmeRank,
                                          bool              doRerun)
{
    // 只有 TPR 里的 gamd 开关能启用 GaMD。
    if (!inputrec.bDoGaMD)
    {
        return GaMDExecutionMode::Disabled;
    }

    // CPU 和 GPU 路径都必须先拒绝尚未实现的全局能量/力归约。
    if (havePpDomainDecomposition)
    {
        GMX_THROW(InconsistentInputError(
                "GaMD currently supports one PP rank without domain decomposition."));
    }
    if (inputrec.useMts)
    {
        GMX_THROW(InconsistentInputError("GaMD does not support MTS."));
    }
    // 每一步的 GaMD 状态只在 md 积分器的主循环里准备，其他积分器和 rerun 都不经过那里。
    if (inputrec.eI != IntegrationAlgorithm::MD)
    {
        GMX_THROW(InconsistentInputError(
                formatString("GaMD currently supports only integrator = md; got integrator = %s.",
                             enumValueToString(inputrec.eI))));
    }
    if (doRerun)
    {
        GMX_THROW(InconsistentInputError("GaMD does not support mdrun -rerun."));
    }
    // 偏置使用当前步的原始能量，所以每一步都必须计算能量。
    if (inputrec.nstcalcenergy != 1)
    {
        GMX_THROW(InconsistentInputError(formatString(
                "GaMD requires nstcalcenergy = 1 because the bias uses the raw energies of the "
                "current step; got nstcalcenergy = %d.",
                inputrec.nstcalcenergy)));
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
    const PressureCoupling     pressureCoupling     = inputrec.pressureCouplingOptions.epc;
    const PressureCouplingType pressureCouplingType = inputrec.pressureCouplingOptions.epct;
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
    if (inputrec.nstfout != 0)
    {
        GMX_THROW(InconsistentInputError(
                "GPU-update GaMD currently requires nstfout=0 because corrected-force output "
                "staging is not implemented."));
    }
    // GPU 路径假定状态缓冲按拓扑顺序排列；单 rank 的 DD 原子排序会打破这个前提。
    if (haveDDAtomOrdering)
    {
        GMX_THROW(InconsistentInputError(
                "GPU-update GaMD does not support domain-decomposition atom ordering, which is "
                "used with a single rank only when GMX_DD_SINGLE_RANK=1 is set. Unset it or use "
                "-update cpu."));
    }
    // 设备上的动能历史每个样本只为两个温度组留了位置。
    if (inputrec.opts.ngtc > 2)
    {
        GMX_THROW(InconsistentInputError(formatString(
                "GPU-update GaMD supports at most two temperature-coupling groups; got %d. Use "
                "-update cpu for more groups.",
                inputrec.opts.ngtc)));
    }

    return GaMDExecutionMode::GpuResident;
}

void applyGaMDCurrentStepCorrection(const t_commrec&              communicationRecord,
                                    int64_t                       step,
                                    const InteractionDefinitions& interactionDefinitions,
                                    const matrix                  box,
                                    ArrayRefWithPadding<RVec>     coordinates,
                                    ArrayRef<RVec>                forces,
                                    tensor                        forceVirial,
                                    int                           numLocalAtoms,
                                    gmx_enerdata_t*               energyData,
                                    ArrayRef<const real>          freeEnergyParameters,
                                    t_forcerec*                   forceRecord,
                                    VirtualSitesHandler*          virtualSites,
                                    t_nrnb*                       operationCounter,
                                    gmx_wallcycle*                wallCycle,
                                    const MdrunScheduleWorkload&  runScheduleWork)
{
    // GaMD 校验可能抛异常；不能把异常带出仍可能有设备工作的 legacy 求力入口。
    try
    {
        GaMDMDModule* module         = runScheduleWork.gaMDModule;
        const auto&   simulationWork = runScheduleWork.simulationWork;
        const auto& stepWork       = runScheduleWork.stepWork;

        // 不支持的配置已在 selectGaMDExecutionMode 里于动力学开始前拒绝。
        // 先完成同一步的能量记账，再把力修正交给对应执行分支。
        GaMDGpuExecutor* gpuExecutor = module->gpuExecutor();
        if (stepWork.stageGpuEnergyAndVirialToHost)
        {
            sum_epot(energyData->grpp, energyData->term.data());
            const double rawPotentialEnergy = energyData->term[F_EPOT];
            const double rawDihedralEnergy  = gaMDDihedralEnergy(energyData->term.data());
            module->finalizeCurrentStep(
                    step, communicationRecord.nodeid, rawPotentialEnergy, rawDihedralEnergy, wallCycle);
            if (forceRecord->listedForcesGpu != nullptr && gpuExecutor != nullptr
                && gpuExecutor->energyDiagnosticsEnabled())
            {
                validateGaMDGpuShadowEnergy(module->diagnostics().logger(),
                                            step,
                                            module->stepResult(),
                                            gpuExecutor->energyDiagnostics());
            }
        }
        else
        {
            GMX_RELEASE_ASSERT(
                    simulationWork.useGpuResidentGaMD() && forceRecord->listedForcesGpu != nullptr
                            && gpuExecutor != nullptr && gpuExecutor->scaleFromDeviceEnabled(),
                    "GaMD can skip host energy staging only with device scale correction");
        }

        if (simulationWork.useGpuResidentGaMD())
        {
            applyGaMDGpuForceCorrection(step,
                                        interactionDefinitions,
                                        box,
                                        coordinates,
                                        forces,
                                        forceVirial,
                                        numLocalAtoms,
                                        freeEnergyParameters,
                                        forceRecord,
                                        operationCounter,
                                        wallCycle,
                                        *module,
                                        stepWork,
                                        MAIN(&communicationRecord));
        }
        else
        {
            applyGaMDCpuForceCorrection(communicationRecord,
                                        step,
                                        interactionDefinitions,
                                        box,
                                        coordinates,
                                        forces,
                                        forceVirial,
                                        numLocalAtoms,
                                        freeEnergyParameters,
                                        forceRecord,
                                        virtualSites,
                                        operationCounter,
                                        wallCycle,
                                        *module,
                                        stepWork);
        }
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

} // namespace gmx
//! \endcond
