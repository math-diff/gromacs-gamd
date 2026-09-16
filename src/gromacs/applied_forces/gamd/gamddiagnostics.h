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
/*! \libinternal \file
 * \brief Declares GaMD diagnostics, logging and reusable host buffers.
 *
 * \inlibraryapi
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDDIAGNOSTICS_H
#define GMX_APPLIED_FORCES_GAMD_GAMDDIAGNOSTICS_H

#include <cstdint>
#include <cstdio>

#include <array>
#include <string>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fileptr.h"
#include "gromacs/utility/logger.h"

#include "gamdtypes.h"

namespace gmx
{

//! \cond libapi
//! \addtogroup module_applied_forces
//! \{

//! 对照当前步的主机和设备结果，在比较误差前拒绝非有限数值。
void validateGaMDGpuShadowEnergy(const MDLogger&       logger,
                                 int64_t               step,
                                 const GaMDStepResult& reference,
                                 const GaMDStepResult& candidate);
//! Compares finite CPU/GPU virial tensors and logs the maximum error.
void validateGaMDGpuShadowVirial(const MDLogger&                    logger,
                                 int64_t                            step,
                                 const tensor                       reference,
                                 const std::array<real, DIM * DIM>& candidate);

//! Compare shadow forces and reject non-finite or out-of-tolerance results.
void validateGaMDGpuShadowForces(const MDLogger&      logger,
                                 const char*          label,
                                 int64_t              step,
                                 ArrayRef<const RVec> reference,
                                 ArrayRef<const RVec> candidate,
                                 double               maxAbsoluteTolerance,
                                 double               rmsTolerance);

/*! \libinternal
 * \brief Reusable host buffers whose lifetime is tied to one GaMD simulation.
 */
struct GaMDHostWorkspace
{
    //! CPU dihedral reference for live GPU validation.
    std::vector<RVec> gpuModeDihedralForces;
    //! Periodic shift-force reference for the dihedral virial.
    std::vector<RVec> gpuModeDihedralShiftForces;
    //! Downloaded device dihedral forces for validation or snapshots.
    std::vector<RVec> gpuShadowDihedralForces;
    //! CPU combination of raw and dihedral forces for comparison.
    std::vector<RVec> referenceCorrectedForces;
    //! Downloaded corrected device forces.
    std::vector<RVec> liveGpuCorrectedForces;
    //! Device-reduced forces captured before live correction.
    std::vector<RVec> rawReducedForcesForShadow;

    //! Raw CPU forces supplied to the GPU shadow correction.
    std::vector<RVec> rawForcesForCorrectionShadow;
    //! GPU shadow result compared with corrected CPU forces.
    std::vector<RVec> correctedForcesForShadow;
    //! Whether the one-time CPU-path force comparison has run.
    bool correctionShadowCompared = false;

    //! Reusable raw dihedral buffer for CPU force correction.
    std::vector<RVec> cpuDihedralForces;
    //! Reusable shift forces for CPU dihedral virial correction.
    std::vector<RVec> cpuDihedralShiftForces;
    //! Device dihedral reference used by the CPU execution path.
    std::vector<RVec> gpuDihedralForcesForShadow;
    //! Whether the one-time CPU-path dihedral comparison has run.
    bool dihedralShadowCompared = false;
};

/*! \libinternal
 * \brief Owns optional GaMD diagnostics for one simulation instance.
 */
class GaMDDiagnostics final
{
public:
    //! Opens configured diagnostic streams; throws on invalid settings or I/O.
    GaMDDiagnostics();
    //! Closes owned diagnostic streams and reports failures without propagating exceptions.
    ~GaMDDiagnostics();

    //! 文件句柄由当前实例独占，复制会让两个实例关闭同一文件。
    GaMDDiagnostics(const GaMDDiagnostics&) = delete;
    //! Disables assignment of diagnostic files and per-simulation state.
    GaMDDiagnostics& operator=(const GaMDDiagnostics&) = delete;

    //! 保存引擎提供的日志目标；默认日志对象不输出消息，便于独立测试。
    void setLogger(const MDLogger& logger);
    //! 返回当前模拟的日志接口。
    const MDLogger& logger() const { return logger_; }

    //! Returns reusable force buffers owned by this simulation instance.
    GaMDHostWorkspace& hostWorkspace();

    //! Read once on first GaMD use; ordinary simulations never request overrides.
    const GaMDForceOverrides& forceOverrides();

    //! Records the step number used by raw CPU evaluation tracing.
    void prepareDihedralStep(int64_t step);
    //! Tests whether the selected step still needs its one snapshot.
    bool dihedralForceOutputRequested(int64_t step) const;
    //! 写出同一步的加权二面角力；有原子重排序时使用拓扑编号。
    void dumpDihedralForces(int64_t              step,
                            ArrayRef<const RVec> rawForces,
                            double               forceWeight,
                            bool                 isMainRank,
                            ArrayRef<const int>  globalAtomIndices = {});
    //! Writes four raw CPU contributions; the OpenMP caller must contain I/O errors.
    void addRawDihedralForces(int        atomIndexI,
                              const rvec forceI,
                              int        atomIndexJ,
                              const rvec forceJ,
                              int        atomIndexK,
                              const rvec forceK,
                              int        atomIndexL,
                              const rvec forceL);
    //! Flushes raw tracing and rejects an unfulfilled selected-step snapshot.
    void finishDihedralStep(int64_t step, bool isMainRank);
    //! Writes the selected final-force buffer in its current atom order.
    void dumpFinalForces(int64_t step, ArrayRef<const RVec> forces, bool isMainRank);

    //! Logs the current-step energy requirement once on the main rank.
    void logEnergyCadence(int nstcalcenergy, bool isMainRank);
    //! Logs resident host-staging conditions once on the main rank.
    void logResidentEnergy(bool useBufferedOutput, bool isMainRank);
    //! Returns true once when workload diagnostics are requested.
    bool shouldPrintGpuWorkloadDiagnostics();

private:
    //! Writes one raw CPU force contribution, throwing on write failure.
    void addRawDihedralForce(int atomIndex, const rvec force);

    GaMDHostWorkspace hostWorkspace_;
    MDLogger          logger_;

    bool               forceOverridesRead_ = false;
    GaMDForceOverrides forceOverrides_;

    FilePtr     rawDihedralFile_;
    std::string rawDihedralFilename_;
    int64_t     rawDihedralCurrentStep_ = -1;

    FilePtr     dihedralForceFile_;
    std::string dihedralForceFilename_;
    int64_t     dihedralForceStep_        = -1;
    int64_t     dihedralForceWrittenStep_ = -1;
    int         dihedralForceBegin_       = 0;
    int         dihedralForceCount_       = -1;

    FilePtr     finalForceFile_;
    std::string finalForceFilename_;
    int64_t     finalForceStep_  = -1;
    int         finalForceBegin_ = 0;
    int         finalForceCount_ = -1;

    bool nstcalcenergyMessagePrinted_   = false;
    bool residentEnergyMessagePrinted_  = false;
    bool gpuWorkloadDiagnosticsEnabled_ = false;
    bool gpuWorkloadDiagnosticsPrinted_ = false;
};

//! \}
//! \endcond

} // namespace gmx

#endif
