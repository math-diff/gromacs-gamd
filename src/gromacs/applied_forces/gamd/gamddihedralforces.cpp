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
 * \brief Recomputes GaMD dihedral-partition forces with the native CPU bonded kernels.
 *
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gamddihedralforces.h"

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/listed_forces/bonded.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/gmxassert.h"

#include "gamdmath.h"

//! \cond internal
namespace gmx
{

void calculateGaMDDihedralListedForces(const InteractionDefinitions&   interactionDefinitions,
                                       ArrayRefWithPadding<const RVec> coordinates,
                                       const Index                     numAtoms,
                                       const t_pbc*                    pbc,
                                       t_fcdata*                       forceCalculationData,
                                       t_nrnb*                         operationCounter,
                                       ArrayRef<const real>            freeEnergyParameters,
                                       int*                            globalAtomIndices,
                                       GaMDDihedralForceBuffers*       buffers)
{
    const rvec* coordinateArray = as_rvec_array(coordinates.paddedArrayRef().data());

    GMX_RELEASE_ASSERT(forceCalculationData != nullptr,
                       "GaMD dihedral force calculation requires force calculation data");
    // 原生 bonded 内核在 rvec4 缓冲和 shift 力上累加，每次都从零开始。
    buffers->shiftForces.assign(c_numShiftVectors, RVec(0.0_real, 0.0_real, 0.0_real));
    buffers->accumulation.assign(numAtoms * sizeof(rvec4) / sizeof(real), 0.0_real);

    rvec4*     accumulatedForces    = reinterpret_cast<rvec4*>(buffers->accumulation.data());
    rvec*      accumulatedShift     = as_rvec_array(buffers->shiftForces.data());
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
                      accumulatedForces,
                      accumulatedShift,
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
                                accumulatedForces,
                                accumulatedShift,
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

    // 输出缓冲整体覆盖为本次的累加结果。
    buffers->forces.resize(numAtoms);
    for (Index atomIndex = 0; atomIndex < numAtoms; ++atomIndex)
    {
        buffers->forces[atomIndex][XX] = accumulatedForces[atomIndex][XX];
        buffers->forces[atomIndex][YY] = accumulatedForces[atomIndex][YY];
        buffers->forces[atomIndex][ZZ] = accumulatedForces[atomIndex][ZZ];
    }
}

} // namespace gmx
//! \endcond
