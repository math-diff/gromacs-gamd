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

#include <array>
#include <vector>

#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/nrnb.h"
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
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "gamddiagnostics.h"
#include "gamddihedralforces.h"
#include "gamdforceshadow.h"
#include "gamdgpu.h"
#include "gamdmath.h"
#include "gamdmdmodule.h"

//! \cond internal
namespace gmx
{
namespace
{

//! GPU 分支保留原来的事件、回传和修正顺序；影子诊断只在修正前后各调用一次。
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
                                 real                          totalScale,
                                 real                          dihedralCorrectionScale)
{
    GaMDGpuExecutor*      gpuExecutor = module.gpuExecutor();
    GaMDForceShadow*      shadow      = module.forceShadow();
    const GaMDStepResult& stepResult  = module.stepResult();

    GMX_RELEASE_ASSERT(gpuExecutor != nullptr, "GPU GaMD requires its module-owned executor");
    GMX_RELEASE_ASSERT(forceRecord->stateGpu != nullptr && forceRecord->listedForcesGpu != nullptr,
                       "GPU GaMD force correction requires GPU state and listed-force objects");
    GMX_RELEASE_ASSERT(gpuExecutor->dihedralBufferEnabled(),
                       "GPU GaMD force correction requires the device dihedral-force buffer");

    // 二面角快照只在选中的这一步回传设备二面角力，和影子诊断各自回传。
    if (module.diagnostics().dihedralForceOutputRequested(step))
    {
        GMX_RELEASE_ASSERT(stepWork.stageGpuEnergyAndVirialToHost,
                           "GaMD dihedral diagnostics require current host scale values");
        std::vector<RVec> deviceDihedralForces(forces.size());
        gpuExecutor->copyDihedralForces(deviceDihedralForces);
        module.diagnostics().dumpDihedralForces(
                step,
                makeConstArrayRef(deviceDihedralForces).subArray(0, numLocalAtoms),
                stepResult.scaleP * stepResult.scaleD);
    }

    if (stepWork.stageGpuEnergyAndVirialToHost && totalScale != 1.0)
    {
        msmul(forceVirial, totalScale, forceVirial);
    }

    tensor deviceDihedralVirial = { { 0 } };
    if (stepWork.stageGpuEnergyAndVirialToHost && dihedralCorrectionScale != 0.0)
    {
        wallcycle_start(wallCycle, WallCycleCounter::GaMDDihedral);
        gpuExecutor->copyDihedralVirial(deviceDihedralVirial);
        wallcycle_stop(wallCycle, WallCycleCounter::GaMDDihedral);

        if (stepWork.computeVirial)
        {
            wallcycle_start(wallCycle, WallCycleCounter::GaMDForceScale);
            // 二面角修正对维里的贡献也按同一个修正系数加到力维里上。
            tensor dihedralVirialCorrection;
            msmul(deviceDihedralVirial, dihedralCorrectionScale, dihedralVirialCorrection);
            m_add(forceVirial, dihedralVirialCorrection, forceVirial);
            wallcycle_stop(wallCycle, WallCycleCounter::GaMDForceScale);
        }
    }

    GpuEventSynchronizer* rawForcesReady = nullptr;
    if (stepWork.useGpuFBufferOps)
    {
        rawForcesReady = forceRecord->stateGpu->fReducedOnDevice(AtomLocality::Local);
    }
    else
    {
        forceRecord->stateGpu->copyForcesToGpu(forces, AtomLocality::Local);
        rawForcesReady = forceRecord->stateGpu->fReadyOnDevice(AtomLocality::Local);
    }

    // 影子在修正前取参照、修正后比对；这两个位置决定了影子模式下事件的消费顺序。
    if (shadow != nullptr)
    {
        shadow->captureBeforeGpuCorrection(step,
                                           interactionDefinitions,
                                           box,
                                           coordinates,
                                           forces,
                                           numLocalAtoms,
                                           freeEnergyParameters,
                                           forceRecord,
                                           operationCounter,
                                           stepWork,
                                           totalScale,
                                           dihedralCorrectionScale,
                                           deviceDihedralVirial,
                                           rawForcesReady,
                                           gpuExecutor,
                                           module.diagnostics().logger());
    }
    // 设备修正读取设备上的偏置状态，主机算出的缩放只用于上面的维里和影子对照。
    gpuExecutor->applyForceCorrection(forceRecord->stateGpu->getForces(), rawForcesReady);
    if (shadow != nullptr)
    {
        shadow->compareAfterGpuCorrection(
                step, stepWork, forceRecord->stateGpu, gpuExecutor, &module.diagnostics());
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
                                 const StepWorkload&           stepWork,
                                 real                          totalScale,
                                 real                          dihedralCorrectionScale)
{
    const GaMDStepResult& stepResult = module.stepResult();

    const bool writesDihedralForces   = module.diagnostics().dihedralForceOutputRequested(step);
    const bool usesDomainAtomOrdering = haveDDAtomOrdering(communicationRecord);
    // 影子要用 GPU bonded 算出的设备二面角力；
    // 它与 DD 原子排序、虚拟位点的组合已在创建 GPU executor 时拒绝。
    GaMDForceShadow* shadow = nullptr;
    if (forceRecord->listedForcesGpu != nullptr)
    {
        shadow = module.forceShadow();
    }
    if (shadow != nullptr)
    {
        shadow->captureBeforeCpuCorrection(forces);
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

    // 本步没有计算 CPU 二面角力时保持为空，影子据此跳过二面角对照。
    ArrayRef<const RVec> cpuDihedralForces;
    if (dihedralCorrectionScale != 0.0 || writesDihedralForces)
    {
        wallcycle_start(wallCycle, WallCycleCounter::GaMDDihedral);

        GaMDDihedralForceBuffers& dihedralBuffers = module.cpuDihedralForceBuffers();
        t_pbc                     pbc;
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
                                          forces.ssize(),
                                          pbcPtr,
                                          forceRecord->fcdata.get(),
                                          operationCounter,
                                          freeEnergyParameters,
                                          globalAtomIndices,
                                          &dihedralBuffers);
        std::vector<gmx::RVec>& gamdDihedralForces      = dihedralBuffers.forces;
        std::vector<gmx::RVec>& gamdDihedralShiftForces = dihedralBuffers.shiftForces;

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

                // 二面角修正对维里的贡献按同一个修正系数加到力维里上。
                msmul(dihedralVirial, dihedralCorrectionScale, dihedralVirial);
                m_add(forceVirial, dihedralVirial, forceVirial);
            }
            wallcycle_stop(wallCycle, WallCycleCounter::GaMDForceScale);
        }
        cpuDihedralForces = makeConstArrayRef(gamdDihedralForces);
    }

    if (shadow != nullptr)
    {
        shadow->compareAfterCpuCorrection(step,
                                          forces,
                                          cpuDihedralForces,
                                          totalScale,
                                          dihedralCorrectionScale,
                                          module.gpuExecutor(),
                                          module.diagnostics().logger());
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
        GaMDGpuExecutor* gpuExecutor             = module->gpuExecutor();
        real             totalScale              = 1.0_real;
        real             dihedralCorrectionScale = 0.0_real;
        if (stepWork.stageGpuEnergyAndVirialToHost)
        {
            sum_epot(energyData->grpp, energyData->term.data());
            const double rawPotentialEnergy = energyData->term[F_EPOT];
            const double rawDihedralEnergy  = gaMDDihedralEnergy(energyData->term.data());
            module->finalizeCurrentStep(step, rawPotentialEnergy, rawDihedralEnergy, wallCycle);
            if (forceRecord->listedForcesGpu != nullptr && gpuExecutor != nullptr
                && gpuExecutor->energyDiagnosticsEnabled())
            {
                validateGaMDGpuShadowEnergy(module->diagnostics().logger(),
                                            step,
                                            module->stepResult(),
                                            gpuExecutor->energyDiagnostics());
            }
            // 两条力修正路径都按本步主机结果算一次缩放：总力乘 scaleP，二面角力再加 scaleP*(scaleD-1)。
            const GaMDStepResult& stepResult = module->stepResult();
            totalScale                       = static_cast<real>(stepResult.scaleP);
            dihedralCorrectionScale = static_cast<real>(stepResult.scaleP * (stepResult.scaleD - 1.0));
        }
        else
        {
            GMX_RELEASE_ASSERT(simulationWork.useGpuResidentGaMD(),
                               "GaMD can skip host energy staging only on the GPU-resident path");
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
                                        totalScale,
                                        dihedralCorrectionScale);
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
                                        stepWork,
                                        totalScale,
                                        dihedralCorrectionScale);
        }
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
}

} // namespace gmx
//! \endcond
