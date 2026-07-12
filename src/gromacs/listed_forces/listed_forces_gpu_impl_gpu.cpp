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
 *
 * \brief Implements helper functions for GPU listed forces (bonded)
 *
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 *
 * \ingroup module_listed_forces
 */

#include "gmxpre.h"

#include <cstdlib>
#include <cstring>

#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpueventsynchronizer.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/utility/fatalerror.h"

#include "cmap_coefficients.h"
#include "listed_forces_gpu_impl.h"

// Number of GPU threads in a block
constexpr static int c_threadsPerBlock = 256;

namespace gmx
{

namespace
{

std::vector<float> makeCmapCoefficients(const gmx_cmap_t& cmapGrid)
{
    const int gridSpacing = cmapGrid.grid_spacing;
    if (gridSpacing <= 0 || cmapGrid.cmapdata.empty())
    {
        return {};
    }

    constexpr int c_numCoefficients = 16;
    const int     cellsPerMap       = gridSpacing * gridSpacing;
    std::vector<float> coefficients(cmapGrid.cmapdata.size() * cellsPerMap * c_numCoefficients);
    const float        gridWidthDegrees = 360.0F / gridSpacing;

    for (gmx::Index map = 0; map < gmx::ssize(cmapGrid.cmapdata); ++map)
    {
        const auto& values = cmapGrid.cmapdata[map].cmap;
        GMX_RELEASE_ASSERT(gmx::ssize(values) == 4 * cellsPerMap,
                           "CMAP grid has an unexpected number of values");

        for (int phi1 = 0; phi1 < gridSpacing; ++phi1)
        {
            const int phi1Next = (phi1 + 1) % gridSpacing;
            for (int phi2 = 0; phi2 < gridSpacing; ++phi2)
            {
                const int phi2Next = (phi2 + 1) % gridSpacing;
                const int positions[4] = { phi1 * gridSpacing + phi2,
                                           phi1Next * gridSpacing + phi2,
                                           phi1Next * gridSpacing + phi2Next,
                                           phi1 * gridSpacing + phi2Next };
                float tx[c_numCoefficients];
                for (int corner = 0; corner < 4; ++corner)
                {
                    tx[corner]      = values[positions[corner] * 4];
                    tx[corner + 4]  = values[positions[corner] * 4 + 1] * gridWidthDegrees;
                    tx[corner + 8]  = values[positions[corner] * 4 + 2] * gridWidthDegrees;
                    tx[corner + 12] = values[positions[corner] * 4 + 3] * gridWidthDegrees
                                      * gridWidthDegrees;
                }

                const int cell = map * cellsPerMap + phi1 * gridSpacing + phi2;
                for (int coefficient = 0; coefficient < c_numCoefficients; ++coefficient)
                {
                    float value = 0.0F;
                    for (int input = 0; input < c_numCoefficients; ++input)
                    {
                        value += c_cmapCoefficientMatrix[input * c_numCoefficients + coefficient]
                                 * tx[input];
                    }
                    coefficients[cell * c_numCoefficients + coefficient] = value;
                }
            }
        }
    }
    return coefficients;
}

} // namespace
// ---- ListedForcesGpu::Impl

static int chooseSubGroupSizeForDevice(const DeviceInformation& deviceInfo)
{
    if (deviceInfo.supportedSubGroupSizes.size() == 1)
    {
        return deviceInfo.supportedSubGroupSizes[0];
    }
    else if (deviceInfo.supportedSubGroupSizes.size() > 1)
    {
        switch (deviceInfo.deviceVendor)
        {
            case DeviceVendor::Intel: return 32;
            default:
                GMX_RELEASE_ASSERT(false, "Flexible sub-groups only supported for Intel GPUs");
                return 0;
        }
    }
    else
    {
        GMX_RELEASE_ASSERT(false, "Device has no known supported sub-group sizes");
        return 0;
    }
}

ListedForcesGpu::Impl::Impl(const gmx_ffparams_t& ffparams,
                            const float           electrostaticsScaleFactor,
                            const int             numEnergyGroupsForListedForces,
                            const DeviceContext&  deviceContext,
                            const DeviceStream&   deviceStream,
                            gmx_wallcycle*        wcycle) :
    deviceContext_(deviceContext), deviceStream_(deviceStream)
{
    GMX_RELEASE_ASSERT(numEnergyGroupsForListedForces == 1,
                       "Only a single energy group is supported with listed forces on GPU");

    GMX_RELEASE_ASSERT(deviceStream.isValid(),
                       "Can't run GPU version of bonded forces in stream that is not valid.");

    static_assert(
            c_threadsPerBlock >= c_numShiftVectors,
            "Threads per block in GPU bonded must be >= c_numShiftVectors for the virial kernel "
            "(calcVir=true)");

    wcycle_ = wcycle;

#if GMX_GPU_CUDA
    gamdDihedralShadowEnabled_ = (std::getenv("GMX_GAMD_GPU_DIH_SHADOW") != nullptr);
    const char* gamdGpuRequest = std::getenv("GMX_GAMD_GPU");
    gamdForceCorrectionEnabled_ =
            (gamdGpuRequest != nullptr && std::strcmp(gamdGpuRequest, "1") == 0);
    gamdDihedralBufferEnabled_ =
            gamdDihedralShadowEnabled_ || gamdForceCorrectionEnabled_;
    if (gamdForceCorrectionEnabled_)
    {
        gamdForcesReadyEvent_ = (gamdDihedralShadowEnabled_
                                         ? std::make_unique<GpuEventSynchronizer>(2, 2)
                                         : std::make_unique<GpuEventSynchronizer>());
    }
    if (gamdDihedralBufferEnabled_)
    {
        allocateDeviceBuffer(&d_gamdDihedralShiftForces_, c_numShiftVectors, deviceContext_);
        allocateDeviceBuffer(&d_gamdDihedralVirial_, DIM * DIM, deviceContext_);
        allocateDeviceBuffer(&d_gamdShortRangeVirial_, DIM * DIM, deviceContext_);
    }
#endif

    allocateDeviceBuffer(&d_forceParams_, ffparams.numTypes(), deviceContext_);
    // This could be an async transfer (if the source is pinned), so
    // long as it uses the same stream as the kernels and we are happy
    // to consume additional pinned pages.
    copyToDeviceBuffer(&d_forceParams_,
                       ffparams.iparams.data(),
                       0,
                       ffparams.numTypes(),
                       deviceStream_,
                       GpuApiCallBehavior::Sync,
                       nullptr);
    vTot_.resize(F_NRE);
    allocateDeviceBuffer(&d_vTot_, F_NRE, deviceContext_);
    clearDeviceBufferAsync(&d_vTot_, 0, F_NRE, deviceStream_);

    const std::vector<float> cmapCoefficients = makeCmapCoefficients(ffparams.cmap_grid);
    if (!cmapCoefficients.empty())
    {
        allocateDeviceBuffer(&d_cmapCoefficients_, cmapCoefficients.size(), deviceContext_);
        copyToDeviceBuffer(&d_cmapCoefficients_,
                           cmapCoefficients.data(),
                           0,
                           cmapCoefficients.size(),
                           deviceStream_,
                           GpuApiCallBehavior::Sync,
                           nullptr);
    }

    kernelParams_.electrostaticsScaleFactor = electrostaticsScaleFactor;
    kernelParams_.cmapGridSpacing            = ffparams.cmap_grid.grid_spacing;
    kernelParams_.cmapCoefficientsPerMap =
            16 * ffparams.cmap_grid.grid_spacing * ffparams.cmap_grid.grid_spacing;
    kernelBuffers_.d_forceParams            = d_forceParams_;
    kernelBuffers_.d_vTot                   = d_vTot_;
    kernelBuffers_.d_cmapCoefficients       = d_cmapCoefficients_;
    for (int i = 0; i < numFTypesOnGpu; i++)
    {
        kernelBuffers_.d_iatoms[i]       = nullptr;
        kernelParams_.fTypeRangeStart[i] = 0;
        kernelParams_.fTypeRangeEnd[i]   = -1;
    }

    const DeviceInformation& deviceInfo = deviceContext.deviceInfo();
    deviceSubGroupSize_                 = chooseSubGroupSizeForDevice(deviceInfo);
    GMX_RELEASE_ASSERT(std::find(deviceInfo.supportedSubGroupSizes.begin(),
                                 deviceInfo.supportedSubGroupSizes.end(),
                                 deviceSubGroupSize_)
                               != deviceInfo.supportedSubGroupSizes.end(),
                       "Device does not support selected sub-group size");

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];

    kernelLaunchConfig_.blockSize[0]     = c_threadsPerBlock;
    kernelLaunchConfig_.blockSize[1]     = 1;
    kernelLaunchConfig_.blockSize[2]     = 1;
    kernelLaunchConfig_.gridSize[0]      = (fTypeRangeEnd + c_threadsPerBlock) / c_threadsPerBlock;
    kernelLaunchConfig_.gridSize[1]      = 1;
    kernelLaunchConfig_.gridSize[2]      = 1;
    kernelLaunchConfig_.sharedMemorySize = c_numShiftVectors * sizeof(Float3);
}

ListedForcesGpu::Impl::~Impl()
{
    deviceStream_.synchronize();
    for (int fType : fTypesOnGpu)
    {
        if (d_iAtoms_[fType])
        {
            freeDeviceBuffer(&d_iAtoms_[fType]);
            d_iAtoms_[fType] = nullptr;
        }
    }

    freeDeviceBuffer(&d_forceParams_);
    freeDeviceBuffer(&d_vTot_);
    if (d_cmapCoefficients_ != nullptr)
    {
        freeDeviceBuffer(&d_cmapCoefficients_);
    }
    if (d_gamdDihedralForcesNbnxn_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralForcesNbnxn_);
    }
    if (d_gamdDihedralForcesState_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralForcesState_);
    }
    if (d_gamdDihedralShiftForces_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralShiftForces_);
    }
    if (d_gamdDihedralVirial_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdDihedralVirial_);
    }
    if (d_gamdShortRangeVirial_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdShortRangeVirial_);
    }
    if (d_gamdCorrectedForcesState_ != nullptr)
    {
        freeDeviceBuffer(&d_gamdCorrectedForcesState_);
    }
    if (d_nbnxnAtomOrder_ != nullptr)
    {
        freeDeviceBuffer(&d_nbnxnAtomOrder_);
    }
}

//! Return whether function type \p fType in \p idef has perturbed interactions
static bool fTypeHasPerturbedEntries(const InteractionDefinitions& idef, int fType)
{
    GMX_ASSERT(idef.ilsort == ilsortNO_FE || idef.ilsort == ilsortFE_SORTED,
               "Perturbed interactions should be sorted here");

    const InteractionList& ilist = idef.il[fType];

    return (idef.ilsort != ilsortNO_FE && idef.numNonperturbedInteractions[fType] != ilist.size());
}

//! Converts \p src with atom indices in state order to \p dest in nbnxn order
static void convertIlistToNbnxnOrder(const InteractionList& src,
                                     HostInteractionList*   dest,
                                     int                    numAtomsPerInteraction,
                                     ArrayRef<const int>    nbnxnAtomOrder)
{
    GMX_ASSERT(src.empty() || !nbnxnAtomOrder.empty(), "We need the nbnxn atom order");

    dest->iatoms.resize(src.size());

#pragma omp parallel for num_threads(gmx_omp_nthreads_get(ModuleMultiThread::Bonded)) schedule(static)
    for (int i = 0; i < src.size(); i += 1 + numAtomsPerInteraction)
    {
        dest->iatoms[i] = src.iatoms[i];
        for (int a = 0; a < numAtomsPerInteraction; a++)
        {
            dest->iatoms[i + 1 + a] = nbnxnAtomOrder[src.iatoms[i + 1 + a]];
        }
    }
}

//! Returns \p input rounded up to the closest multiple of \p factor.
static inline int roundUpToFactor(const int input, const int factor)
{
    GMX_ASSERT(factor > 0, "The factor to round up to must be > 0.");

    int remainder = input % factor;

    if (remainder == 0)
    {
        return (input);
    }
    return (input + (factor - remainder));
}

void ListedForcesGpu::Impl::updateHaveInteractions(const InteractionDefinitions& idef)
{
    haveInteractions_ = false;

    for (int fType : fTypesOnGpu)
    {
        /* Perturbation is not implemented in the GPU bonded kernels.
         * But instead of doing all interactions on the CPU, we can
         * still easily handle the types that have no perturbed
         * interactions on the GPU. */
        if (!idef.il[fType].empty() && !fTypeHasPerturbedEntries(idef, fType))
        {
            haveInteractions_ = true;
            return;
        }
    }
}


// TODO Consider whether this function should be a factory method that
// makes an object that is the only one capable of the device
// operations needed for the lifetime of an interaction list. This
// would be harder to misuse than ListedForcesGpu, and exchange the problem
// of naming this method for the problem of what to name the
// BondedDeviceInteractionListHandler type.

/*! Divides bonded interactions over threads and GPU.
 *  The bonded interactions are assigned by interaction type to GPU threads. The interaction
 *  types are assigned in blocks sized as kernelParams_.deviceSubGroupSize. The beginning and end
 *  (thread index) of each interaction type are stored in kernelParams_. Pointers to the relevant
 *  data structures on the GPU are also stored in kernelParams_.
 */
void ListedForcesGpu::Impl::updateInteractionListsAndDeviceBuffers(ArrayRef<const int> nbnxnAtomOrder,
                                                                   const InteractionDefinitions& idef,
                                                                   DeviceBuffer<Float4> d_xqPtr,
                                                                   DeviceBuffer<RVec>   d_fPtr,
                                                                   DeviceBuffer<RVec>   d_fShiftPtr)
{
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::GpuBondedListUpdate);

    if (gamdDihedralBufferEnabled_)
    {
        if (!idef.il[F_FOURDIHS].empty())
        {
            gmx_fatal(FARGS,
                      "The GaMD GPU force path does not support FOURDIHS. Use GMX_GAMD_GPU=0 or "
                      "add CUDA FOURDIHS support before using this topology.");
        }
        for (const int ftype : { F_PDIHS, F_RBDIHS, F_CMAP })
        {
            if (!idef.il[ftype].empty() && fTypeHasPerturbedEntries(idef, ftype))
            {
                gmx_fatal(FARGS,
                          "The GaMD GPU force path does not support perturbed %s interactions. "
                          "Use GMX_GAMD_GPU=0 for this topology.",
                          interaction_function[ftype].longname);
            }
        }
        GMX_RELEASE_ASSERT(!nbnxnAtomOrder.empty(),
                           "GaMD GPU dihedral forces require an atom-order mapping");
        const int nbnxnForceSize = *std::max_element(nbnxnAtomOrder.begin(), nbnxnAtomOrder.end()) + 1;
        reallocateDeviceBuffer(&d_gamdDihedralForcesNbnxn_,
                               nbnxnForceSize,
                               &gamdNbnxnForceSize_,
                               &gamdNbnxnForceCapacity_,
                               deviceContext_);
        reallocateDeviceBuffer(&d_gamdDihedralForcesState_,
                               nbnxnAtomOrder.ssize(),
                               &gamdStateForceSize_,
                               &gamdStateForceCapacity_,
                               deviceContext_);
        reallocateDeviceBuffer(&d_gamdCorrectedForcesState_,
                               nbnxnAtomOrder.ssize(),
                               &gamdCorrectedForceSize_,
                               &gamdCorrectedForceCapacity_,
                               deviceContext_);
        reallocateDeviceBuffer(&d_nbnxnAtomOrder_,
                               nbnxnAtomOrder.ssize(),
                               &nbnxnAtomOrderSize_,
                               &nbnxnAtomOrderCapacity_,
                               deviceContext_);
        copyToDeviceBuffer(&d_nbnxnAtomOrder_,
                           nbnxnAtomOrder.data(),
                           0,
                           nbnxnAtomOrder.ssize(),
                           deviceStream_,
                           GpuApiCallBehavior::Async,
                           nullptr);
    }

    bool haveGpuInteractions = false;
    int  fTypesCounter       = 0;

    for (int fType : fTypesOnGpu)
    {
        auto& iList = iLists_[fType];

        /* Perturbation is not implemented in the GPU bonded kernels.
         * But instead of doing all interactions on the CPU, we can
         * still easily handle the types that have no perturbed
         * interactions on the GPU. */
        if (!idef.il[fType].empty() && !fTypeHasPerturbedEntries(idef, fType))
        {
            haveGpuInteractions = true;

            convertIlistToNbnxnOrder(idef.il[fType], &iList, NRAL(fType), nbnxnAtomOrder);
        }
        else
        {
            iList.iatoms.clear();
        }

        // Update the device if necessary. This can leave some
        // allocations on the device when the host size decreases to
        // zero, which is OK, since we deallocate everything at the
        // end.
        if (iList.size() > 0)
        {
            int newListSize;
            reallocateDeviceBuffer(
                    &d_iAtoms_[fType], iList.size(), &newListSize, &d_iAtomsAlloc_[fType], deviceContext_);

            copyToDeviceBuffer(&d_iAtoms_[fType],
                               iList.iatoms.data(),
                               0,
                               iList.size(),
                               deviceStream_,
                               GpuApiCallBehavior::Async,
                               nullptr);
        }
        kernelParams_.fTypesOnGpu[fTypesCounter] = fType;
        int numBonds = iList.size() / (interaction_function[fType].nratoms + 1);
        kernelParams_.numFTypeBonds[fTypesCounter] = numBonds;
        kernelBuffers_.d_iatoms[fTypesCounter]     = d_iAtoms_[fType];
        if (fTypesCounter == 0)
        {
            kernelParams_.fTypeRangeStart[fTypesCounter] = 0;
        }
        else
        {
            kernelParams_.fTypeRangeStart[fTypesCounter] =
                    kernelParams_.fTypeRangeEnd[fTypesCounter - 1] + 1;
        }
        kernelParams_.fTypeRangeEnd[fTypesCounter] = kernelParams_.fTypeRangeStart[fTypesCounter]
                                                     + roundUpToFactor(numBonds, deviceSubGroupSize_) - 1;

        GMX_ASSERT(numBonds > 0
                           || kernelParams_.fTypeRangeEnd[fTypesCounter]
                                      <= kernelParams_.fTypeRangeStart[fTypesCounter],
                   "Invalid GPU listed forces setup. numBonds must be > 0 if there are threads "
                   "allocated to do work on that interaction function type.");
        GMX_ASSERT(
                kernelParams_.fTypeRangeStart[fTypesCounter] % deviceSubGroupSize_ == 0
                        && (kernelParams_.fTypeRangeEnd[fTypesCounter] + 1) % deviceSubGroupSize_ == 0,
                "The bonded interactions must be assigned to the GPU in blocks of sub-group size.");

        fTypesCounter++;
    }

    int fTypeRangeEnd               = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];
    kernelLaunchConfig_.gridSize[0] = (fTypeRangeEnd + c_threadsPerBlock) / c_threadsPerBlock;

    d_xq_     = d_xqPtr;
    d_f_      = d_fPtr;
    d_fShift_ = d_fShiftPtr;

    kernelBuffers_.d_forceParams = d_forceParams_;
    kernelBuffers_.d_vTot        = d_vTot_;
    kernelBuffers_.d_cmapCoefficients = d_cmapCoefficients_;

    GMX_RELEASE_ASSERT(haveGpuInteractions == haveInteractions_,
                       "inconsistent haveInteractions flags encountered.");

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::GpuBondedListUpdate);
}

void ListedForcesGpu::Impl::setPbc(PbcType pbcType, const matrix box, bool canMoleculeSpanPbc)
{
    PbcAiuc pbcAiuc;
    setPbcAiuc(canMoleculeSpanPbc ? numPbcDimensions(pbcType) : 0, box, &pbcAiuc);
    kernelParams_.pbcAiuc = pbcAiuc;
}

bool ListedForcesGpu::Impl::haveInteractions() const
{
    return haveInteractions_;
}

void ListedForcesGpu::Impl::launchEnergyTransfer()
{
    GMX_ASSERT(haveInteractions_,
               "No GPU bonded interactions, so no energies will be computed, so transfer should "
               "not be called");
    wallcycle_sub_start_nocount(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    // TODO add conditional on whether there has been any compute (and make sure host buffer doesn't contain garbage)
    float* h_vTot = vTot_.data();
    copyFromDeviceBuffer(h_vTot, &d_vTot_, 0, F_NRE, deviceStream_, GpuApiCallBehavior::Async, nullptr);
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
}

void ListedForcesGpu::Impl::waitAccumulateEnergyTerms(gmx_enerdata_t* enerd)
{
    GMX_ASSERT(haveInteractions_,
               "No GPU bonded interactions, so no energies will be computed or transferred, so "
               "accumulation should not occur");

    wallcycle_start(wcycle_, WallCycleCounter::WaitGpuBonded);
    deviceStream_.synchronize();
    wallcycle_stop(wcycle_, WallCycleCounter::WaitGpuBonded);

    for (int fType : fTypesOnGpu)
    {
        if (fType != F_LJ14 && fType != F_COUL14)
        {
            enerd->term[fType] += vTot_[fType];
        }
    }

    // Note: We do not support energy groups here
    gmx_grppairener_t* grppener = &enerd->grpp;
    grppener->energyGroupPairTerms[NonBondedEnergyTerms::LJ14][0] += vTot_[F_LJ14];
    grppener->energyGroupPairTerms[NonBondedEnergyTerms::Coulomb14][0] += vTot_[F_COUL14];
}

void ListedForcesGpu::Impl::clearEnergies()
{
    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpuPp);
    wallcycle_sub_start_nocount(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    clearDeviceBufferAsync(&d_vTot_, 0, F_NRE, deviceStream_);
    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpuPp);
}

bool ListedForcesGpu::Impl::gamdDihedralShadowEnabled() const
{
    return gamdDihedralShadowEnabled_;
}

bool ListedForcesGpu::Impl::gamdDihedralBufferEnabled() const
{
    return gamdDihedralBufferEnabled_;
}

void ListedForcesGpu::Impl::copyGamdDihedralShadowForces(ArrayRef<RVec> forces)
{
    GMX_RELEASE_ASSERT(gamdDihedralShadowEnabled_,
                       "GaMD GPU dihedral shadow-force copy requested while disabled");
    GMX_RELEASE_ASSERT(forces.ssize() == gamdStateForceSize_,
                       "GaMD GPU dihedral shadow-force host buffer has the wrong size");
    copyFromDeviceBuffer(reinterpret_cast<Float3*>(forces.data()),
                         &d_gamdDihedralForcesState_,
                         0,
                         gamdStateForceSize_,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
}

void ListedForcesGpu::Impl::copyGamdDihedralVirial(std::array<real, DIM * DIM>* virial)
{
    GMX_RELEASE_ASSERT(gamdDihedralBufferEnabled_,
                       "GaMD GPU dihedral virial copy requested while disabled");
    GMX_RELEASE_ASSERT(virial != nullptr, "GaMD GPU dihedral virial requires host storage");
    copyFromDeviceBuffer(virial->data(),
                         &d_gamdDihedralVirial_,
                         0,
                         virial->size(),
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
}

void ListedForcesGpu::Impl::copyGamdShortRangeVirial(std::array<real, DIM * DIM>* virial)
{
    GMX_RELEASE_ASSERT(gamdForceCorrectionEnabled_,
                       "GaMD GPU short-range virial copy requested while disabled");
    GMX_RELEASE_ASSERT(virial != nullptr, "GaMD GPU short-range virial requires host storage");
    copyFromDeviceBuffer(virial->data(),
                         &d_gamdShortRangeVirial_,
                         0,
                         virial->size(),
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
}

void ListedForcesGpu::Impl::applyGamdForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                                           real                 totalScale,
                                                           real                 dihedralCorrectionScale,
                                                           ArrayRef<RVec>       correctedForces)
{
    GMX_RELEASE_ASSERT(gamdDihedralShadowEnabled_,
                       "GaMD GPU force-correction shadow requested while disabled");
    GMX_RELEASE_ASSERT(rawForces.ssize() == gamdStateForceSize_
                               && correctedForces.ssize() == gamdStateForceSize_,
                       "GaMD GPU force-correction shadow buffers have the wrong size");
    copyToDeviceBuffer(&d_gamdCorrectedForcesState_,
                       reinterpret_cast<const Float3*>(rawForces.data()),
                       0,
                       gamdStateForceSize_,
                       deviceStream_,
                       GpuApiCallBehavior::Async,
                       nullptr);
    launchGamdForceCorrectionShadowKernel(totalScale, dihedralCorrectionScale);
    copyFromDeviceBuffer(reinterpret_cast<Float3*>(correctedForces.data()),
                         &d_gamdCorrectedForcesState_,
                         0,
                         gamdStateForceSize_,
                         deviceStream_,
                         GpuApiCallBehavior::Sync,
                         nullptr);
}

void ListedForcesGpu::Impl::applyGamdForceCorrection(DeviceBuffer<RVec>    forces,
                                                     real                  totalScale,
                                                     real                  dihedralCorrectionScale,
                                                     GpuEventSynchronizer* rawForcesReady)
{
    GMX_RELEASE_ASSERT(gamdForceCorrectionEnabled_,
                       "GaMD GPU force correction requested while disabled");
    GMX_RELEASE_ASSERT(forces != nullptr && rawForcesReady != nullptr,
                       "GaMD GPU force correction requires forces and a dependency event");
    launchGamdForceCorrectionKernel(
            forces, totalScale, dihedralCorrectionScale, rawForcesReady);
}

GpuEventSynchronizer* ListedForcesGpu::Impl::gamdForcesReadyEvent()
{
    GMX_RELEASE_ASSERT(gamdForcesReadyEvent_ != nullptr,
                       "GaMD corrected-force event requested while GPU correction is disabled");
    return gamdForcesReadyEvent_.get();
}

// ---- ListedForcesGpu

ListedForcesGpu::ListedForcesGpu(const gmx_ffparams_t& ffparams,
                                 const float           electrostaticsScaleFactor,
                                 const int             numEnergyGroupsForListedForces,
                                 const DeviceContext&  deviceContext,
                                 const DeviceStream&   deviceStream,
                                 gmx_wallcycle*        wcycle) :
    impl_(new Impl(ffparams, electrostaticsScaleFactor, numEnergyGroupsForListedForces, deviceContext, deviceStream, wcycle))
{
}

ListedForcesGpu::~ListedForcesGpu() = default;

void ListedForcesGpu::updateHaveInteractions(const InteractionDefinitions& idef)
{
    impl_->updateHaveInteractions(idef);
}

void ListedForcesGpu::updateInteractionListsAndDeviceBuffers(ArrayRef<const int> nbnxnAtomOrder,
                                                             const InteractionDefinitions& idef,
                                                             NBAtomDataGpu* nbnxmAtomDataGpu)
{
    impl_->updateInteractionListsAndDeviceBuffers(
            nbnxnAtomOrder, idef, nbnxmAtomDataGpu->xq, nbnxmAtomDataGpu->f, nbnxmAtomDataGpu->fShift);
}

void ListedForcesGpu::setPbc(PbcType pbcType, const matrix box, bool canMoleculeSpanPbc)
{
    impl_->setPbc(pbcType, box, canMoleculeSpanPbc);
}

bool ListedForcesGpu::haveInteractions() const
{
    return impl_->haveInteractions();
}

void ListedForcesGpu::setPbcAndlaunchKernel(PbcType                  pbcType,
                                            const matrix             box,
                                            bool                     canMoleculeSpanPbc,
                                            const gmx::StepWorkload& stepWork)
{
    setPbc(pbcType, box, canMoleculeSpanPbc);
    launchKernel(stepWork);
}

void ListedForcesGpu::launchEnergyTransfer()
{
    impl_->launchEnergyTransfer();
}

void ListedForcesGpu::waitAccumulateEnergyTerms(gmx_enerdata_t* enerd)
{
    impl_->waitAccumulateEnergyTerms(enerd);
}

void ListedForcesGpu::clearEnergies()
{
    impl_->clearEnergies();
}

bool ListedForcesGpu::gamdDihedralShadowEnabled() const
{
    return impl_->gamdDihedralShadowEnabled();
}

bool ListedForcesGpu::gamdDihedralBufferEnabled() const
{
    return impl_->gamdDihedralBufferEnabled();
}

void ListedForcesGpu::copyGamdDihedralShadowForces(ArrayRef<RVec> forces)
{
    impl_->copyGamdDihedralShadowForces(forces);
}

void ListedForcesGpu::copyGamdDihedralVirial(std::array<real, DIM * DIM>* virial)
{
    impl_->copyGamdDihedralVirial(virial);
}

void ListedForcesGpu::launchGamdShortRangeVirialReduction()
{
    impl_->launchGamdShortRangeVirialReduction();
}

void ListedForcesGpu::copyGamdShortRangeVirial(std::array<real, DIM * DIM>* virial)
{
    impl_->copyGamdShortRangeVirial(virial);
}

void ListedForcesGpu::applyGamdForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                                     real                 totalScale,
                                                     real                 dihedralCorrectionScale,
                                                     ArrayRef<RVec>       correctedForces)
{
    impl_->applyGamdForceCorrectionShadow(
            rawForces, totalScale, dihedralCorrectionScale, correctedForces);
}

void ListedForcesGpu::applyGamdForceCorrection(DeviceBuffer<RVec>    forces,
                                               real                  totalScale,
                                               real                  dihedralCorrectionScale,
                                               GpuEventSynchronizer* rawForcesReady)
{
    impl_->applyGamdForceCorrection(
            forces, totalScale, dihedralCorrectionScale, rawForcesReady);
}

GpuEventSynchronizer* ListedForcesGpu::gamdForcesReadyEvent()
{
    return impl_->gamdForcesReadyEvent();
}

} // namespace gmx
