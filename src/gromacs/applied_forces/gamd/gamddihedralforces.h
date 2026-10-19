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
 * \brief Declares the CPU recomputation of GaMD dihedral-partition forces.
 *
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDDIHEDRALFORCES_H
#define GMX_APPLIED_FORCES_GAMD_GAMDDIHEDRALFORCES_H

#include <vector>

#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

class InteractionDefinitions;
struct t_fcdata;
struct t_nrnb;
struct t_pbc;

namespace gmx
{

//! \cond internal
//! \addtogroup module_applied_forces
//! \{

/*! \internal \brief Reusable host buffers of one CPU dihedral-partition force evaluation. */
struct GaMDDihedralForceBuffers
{
    //! Raw dihedral-partition forces in local atom order.
    std::vector<RVec> forces;
    //! Periodic shift forces for the dihedral virial.
    std::vector<RVec> shiftForces;
    //! rvec4 accumulation storage used by the native bonded kernels.
    std::vector<real> accumulation;
};

/*! \brief Recomputes raw dihedral-partition forces with the native CPU bonded kernels.
 *
 * Evaluates the interaction types of c_gaMDDihedralInteractionTypes and
 * overwrites the forces of \p numAtoms atoms and the shift forces in
 * \p buffers. The kernels add their flop counts to \p operationCounter.
 */
void calculateGaMDDihedralListedForces(const InteractionDefinitions&   interactionDefinitions,
                                       ArrayRefWithPadding<const RVec> coordinates,
                                       Index                           numAtoms,
                                       const t_pbc*                    pbc,
                                       t_fcdata*                       forceCalculationData,
                                       t_nrnb*                         operationCounter,
                                       ArrayRef<const real>            freeEnergyParameters,
                                       int*                            globalAtomIndices,
                                       GaMDDihedralForceBuffers*       buffers);

//! \}
//! \endcond

} // namespace gmx

#endif
