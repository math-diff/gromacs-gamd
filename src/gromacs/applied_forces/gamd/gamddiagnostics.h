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
 * \brief Declares instance-owned GaMD diagnostic state and host scratch buffers.
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

#include "gamdtypes.h"

namespace gmx
{

//! 对照当前步的主机和设备结果，在比较误差前拒绝非有限数值。
void validateGaMDGpuShadowEnergy(int64_t step, const GaMDStepResult& reference, const GaMDStepResult& candidate);
void validateGaMDGpuShadowVirial(int64_t                            step,
                                 const tensor                       reference,
                                 const std::array<real, DIM * DIM>& candidate);

//! Compare shadow forces and reject non-finite or out-of-tolerance results.
void validateGaMDGpuShadowForces(const char*          label,
                                 int64_t              step,
                                 ArrayRef<const RVec> reference,
                                 ArrayRef<const RVec> candidate,
                                 double               maxAbsoluteTolerance,
                                 double               rmsTolerance);

/*! \internal
 * \brief Reusable host buffers whose lifetime is tied to one GaMD simulation.
 */
struct GaMDHostWorkspace
{
    std::vector<RVec> gpuModeDihedralForces;
    std::vector<RVec> gpuModeDihedralShiftForces;
    std::vector<RVec> gpuShadowDihedralForces;
    std::vector<RVec> referenceCorrectedForces;
    std::vector<RVec> liveGpuCorrectedForces;
    std::vector<RVec> rawReducedForcesForShadow;

    std::vector<RVec> rawForcesForCorrectionShadow;
    std::vector<RVec> correctedForcesForShadow;
    bool              correctionShadowCompared = false;

    std::vector<RVec> cpuDihedralForces;
    std::vector<RVec> cpuDihedralShiftForces;
    std::vector<RVec> gpuDihedralForcesForShadow;
    bool              dihedralShadowCompared = false;
};

/*! \internal
 * \brief Owns optional GaMD diagnostics for one simulation instance.
 */
class GaMDDiagnostics final
{
public:
    GaMDDiagnostics();
    ~GaMDDiagnostics();

    GaMDHostWorkspace& hostWorkspace();

    //! Read once on first GaMD use; ordinary simulations never request overrides.
    const GaMDForceOverrides& forceOverrides();

    void prepareDihedralStep(int64_t step);
    bool dihedralForceOutputRequested(int64_t step) const;
    //! 写出同一步的加权二面角力；有原子重排序时使用拓扑编号。
    void dumpDihedralForces(int64_t              step,
                            ArrayRef<const RVec> rawForces,
                            double               forceWeight,
                            bool                 isMainRank,
                            ArrayRef<const int>  globalAtomIndices = {});
    void addRawDihedralForces(int        atomIndexI,
                              const rvec forceI,
                              int        atomIndexJ,
                              const rvec forceJ,
                              int        atomIndexK,
                              const rvec forceK,
                              int        atomIndexL,
                              const rvec forceL);
    void finishDihedralStep(int64_t step, bool isMainRank);
    void dumpFinalForces(int64_t step, ArrayRef<const RVec> forces, bool isMainRank);

    bool shouldPrintNstcalcenergyMessage();
    bool shouldPrintResidentEnergyMessage();
    bool shouldPrintGpuWorkloadDiagnostics();

private:
    void addRawDihedralForce(int atomIndex, const rvec force);

    GaMDHostWorkspace hostWorkspace_;

    bool               forceOverridesRead_ = false;
    GaMDForceOverrides forceOverrides_;

    FILE*       rawDihedralFile_ = nullptr;
    std::string rawDihedralFilename_;
    int64_t     rawDihedralCurrentStep_ = -1;

    FILE*       dihedralForceFile_ = nullptr;
    std::string dihedralForceFilename_;
    int64_t     dihedralForceStep_        = -1;
    int64_t     dihedralForceWrittenStep_ = -1;
    int         dihedralForceBegin_       = 0;
    int         dihedralForceCount_       = -1;

    FILE*       finalForceFile_ = nullptr;
    std::string finalForceFilename_;
    int64_t     finalForceStep_  = -1;
    int         finalForceBegin_ = 0;
    int         finalForceCount_ = -1;

    bool nstcalcenergyMessagePrinted_   = false;
    bool residentEnergyMessagePrinted_  = false;
    bool gpuWorkloadDiagnosticsEnabled_ = false;
    bool gpuWorkloadDiagnosticsPrinted_ = false;
};

} // namespace gmx

#endif
