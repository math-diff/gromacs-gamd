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
 * \brief Declares GaMD execution selection and the legacy force-correction entry point.
 *
 * \inlibraryapi
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDEXECUTION_H
#define GMX_APPLIED_FORCES_GAMD_GAMDEXECUTION_H

#include <cstdint>

#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/simulation_workload.h"

class InteractionDefinitions;
struct gmx_enerdata_t;
struct gmx_wallcycle;
struct t_commrec;
struct t_forcerec;
struct t_nrnb;

namespace gmx
{

class VirtualSitesHandler;

//! \cond libapi
//! \addtogroup module_applied_forces
//! \{

/*! \brief Selects the GaMD execution mode or throws InconsistentInputError.
 *
 * This is the single setup-time check of the supported GaMD configurations,
 * so unsupported input is rejected before the first dynamics step.
 *
 * Some limitations of the GPU path are not repeated here because no run can
 * reach them: GPU update already rejects virtual sites, Ewald surface
 * correction and orientation restraints (the only sources of whole-molecule
 * transforms), and PME rejects screw PBC. Topology support of the GPU force
 * path (FOURDIHS, perturbed dihedrals) needs the local topology and is checked
 * when the listed-force buffers are bound, before the first force evaluation.
 * Combinations with the GPU shadow diagnostics are checked where the
 * diagnostics run.
 */
GaMDExecutionMode selectGaMDExecutionMode(const t_inputrec& inputrec,
                                          bool              useGaMD,
                                          bool              useGpuNonbonded,
                                          bool              useGpuPme,
                                          bool              useGpuBonded,
                                          bool              useGpuUpdate,
                                          bool              havePpDomainDecomposition,
                                          bool              haveDDAtomOrdering,
                                          bool              haveSeparatePmeRank,
                                          bool              doRerun);

//! Returns a stable descriptive name for a GaMD execution mode.
const char* gaMDExecutionModeName(GaMDExecutionMode mode);

//! Applies current-step bias after ordinary force processing, preserving device event order.
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
                                    const MdrunScheduleWorkload&  runScheduleWork);

//! \}
//! \endcond

} // namespace gmx

#endif
