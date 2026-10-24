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
 * \brief Declares the CPU/GPU comparisons that run beside the GaMD force correction.
 *
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDFORCESHADOW_H
#define GMX_APPLIED_FORCES_GAMD_GAMDFORCESHADOW_H

#include <cstdint>

#include <array>
#include <vector>

#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

#include "gamddihedralforces.h"

class GpuEventSynchronizer;
class InteractionDefinitions;
struct t_forcerec;
struct t_nrnb;

namespace gmx
{

//! \cond internal
//! \addtogroup module_applied_forces
//! \{

class GaMDDiagnostics;
class GaMDGpuExecutor;
class MDLogger;
class StatePropagatorDataGpu;
class StepWorkload;

/*! \internal
 * \brief Compares the GaMD force correction with an independent reference.
 *
 * The module creates this object only when the GPU dihedral shadow is enabled.
 * The force correction calls a capture entry before it corrects the forces and
 * the matching compare entry afterwards; this object owns every buffer of these
 * comparisons.
 *
 * The positions of the GPU-path entries are part of the shadow-mode event
 * contract. sim_util.cpp expects two consumers of the reduced-force event, and
 * the executor creates its forces-ready event with two consumers. The capture
 * waits for the reduced forces before the correction consumes them; the
 * compare restores one expected consumer and waits for the corrected forces,
 * which GPU update also consumes.
 */
class GaMDForceShadow
{
public:
    /*! \brief Takes the GPU-path references before the device correction.
     *
     * When this step applies the dihedral correction, compares a CPU
     * recomputation of the dihedral forces and virial with the device results.
     * Always keeps the host reference of the corrected forces.
     */
    void captureBeforeGpuCorrection(int64_t                       step,
                                    const InteractionDefinitions& interactionDefinitions,
                                    const matrix                  box,
                                    ArrayRefWithPadding<RVec>     coordinates,
                                    ArrayRef<const RVec>          forces,
                                    int                           numLocalAtoms,
                                    ArrayRef<const real>          freeEnergyParameters,
                                    t_forcerec*                   forceRecord,
                                    t_nrnb*                       operationCounter,
                                    const StepWorkload&           stepWork,
                                    real                          totalScale,
                                    real                          dihedralCorrectionScale,
                                    const tensor                  deviceDihedralVirial,
                                    GpuEventSynchronizer*         rawForcesReady,
                                    GaMDGpuExecutor*              gpuExecutor,
                                    const MDLogger&               logger);
    //! Compares the corrected device forces with the reference and writes the requested final forces.
    void compareAfterGpuCorrection(int64_t                 step,
                                   const StepWorkload&     stepWork,
                                   StatePropagatorDataGpu* stateGpu,
                                   GaMDGpuExecutor*        gpuExecutor,
                                   GaMDDiagnostics*        diagnostics);
    //! Keeps the raw CPU forces until the one-time correction comparison has run.
    void captureBeforeCpuCorrection(ArrayRef<const RVec> forces);
    /*! \brief Compares the CPU-path dihedral and corrected forces with the device once each.
     *
     * \p cpuDihedralForces is empty on steps without a CPU dihedral evaluation.
     */
    void compareAfterCpuCorrection(int64_t              step,
                                   ArrayRef<const RVec> correctedForces,
                                   ArrayRef<const RVec> cpuDihedralForces,
                                   real                 totalScale,
                                   real                 dihedralCorrectionScale,
                                   GaMDGpuExecutor*     gpuExecutor,
                                   const MDLogger&      logger);

private:
    //! CPU recomputation of the dihedral forces on the GPU path.
    GaMDDihedralForceBuffers cpuReference_;
    //! Downloaded raw device dihedral forces.
    std::vector<RVec> deviceDihedralForces_;
    //! Raw forces captured before the correction.
    std::vector<RVec> rawForces_;
    //! Host reference of the corrected forces on the GPU path.
    std::vector<RVec> referenceCorrectedForces_;
    //! Device-corrected forces returned for comparison.
    std::vector<RVec> correctedForces_;
    //! Whether the one-time CPU-path dihedral comparison has run.
    bool dihedralCompared_ = false;
    //! Whether the one-time CPU-path correction comparison has run.
    bool correctionCompared_ = false;
};

//! \}
//! \endcond

} // namespace gmx

#endif
