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
 * \brief Declares GPU implementation class for GPU bonded
 * interactions.
 *
 * This header file is needed to include from both the device-side
 * kernels file, and the host-side management code.
 *
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */
#ifndef GMX_LISTED_FORCES_LISTED_FORCES_GPU_IMPL_H
#define GMX_LISTED_FORCES_LISTED_FORCES_GPU_IMPL_H

#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/pbcutil/pbc_aiuc.h"

#if GMX_GPU_SYCL
#    include "gromacs/gpu_utils/syclutils.h"
#endif

struct gmx_ffparams_t;
class DeviceContext;
class GpuEventSynchronizer;

namespace gmx
{

/*! \internal \brief Version of InteractionList that supports pinning */
struct HostInteractionList
{
    /*! \brief Returns the total number of elements in iatoms */
    int size() const { return iatoms.size(); }

    //! List of interactions, see \c HostInteractionLists
    HostVector<int> iatoms = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
};

/* \brief Bonded parameters and GPU pointers
 *
 * This is used to accumulate all the parameters and pointers so they can be passed
 * to the GPU as a single structure.
 *
 */
struct BondedGpuKernelParameters
{
    //! Periodic boundary data
    PbcAiuc pbcAiuc;
    //! Scale factor
    float electrostaticsScaleFactor;
    //! The bonded types on GPU
    int fTypesOnGpu[numFTypesOnGpu];
    //! The number of bonds for every function type
    int numFTypeBonds[numFTypesOnGpu];
    //! CMAP grid dimension and number of precomputed coefficients per map.
    int cmapGridSpacing        = 0;
    int cmapCoefficientsPerMap = 0;
    //! The start index in the range of each interaction type
    int fTypeRangeStart[numFTypesOnGpu];
    //! The end index in the range of each interaction type
    int fTypeRangeEnd[numFTypesOnGpu];
    BondedGpuKernelParameters()
    {
        matrix boxDummy = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        setPbcAiuc(0, boxDummy, &pbcAiuc);
        electrostaticsScaleFactor = 1.0F;
    }
};
struct BondedGpuKernelBuffers
{
    //! Force parameters (on GPU)
    DeviceBuffer<t_iparams> d_forceParams = nullptr;
    //! Total Energy (on GPU)
    DeviceBuffer<float> d_vTot = nullptr;
    //! Precomputed bicubic coefficients for all CMAP grid cells.
    DeviceBuffer<float> d_cmapCoefficients = nullptr;
    //! Interaction list atoms (on GPU)
    DeviceBuffer<t_iatom> d_iatoms[numFTypesOnGpu];
};

/*! \internal \brief Implements GPU bondeds */
class ListedForcesGpu::Impl
{
public:
    //! Constructor
    Impl(const gmx_ffparams_t& ffparams,
         float                 electrostaticsScaleFactor,
         int                   numEnergyGroupsForListedForces,
         const DeviceContext&  deviceContext,
         const DeviceStream&   deviceStream,
         gmx_wallcycle*        wcycle);
    //! \brief Destructor, non-default needed for freeing device-side buffers
    ~Impl();

    /*! \brief Update flag that tells whether there are bonded interactions suitable for the GPU.
     *
     * Intended to be called early during search steps so domainWork flags can be populated.
     */
    void updateHaveInteractions(const InteractionDefinitions& idef);

    /*! \brief Update lists of interactions from idef suitable for the GPU,
     * using the data structures prepared for PP work.
     *
     * Intended to be called after each neighbour search
     * stage. Copies the bonded interactions assigned to the GPU
     * to device data structures, and updates device buffers that
     * may have been updated after search. */
    void updateInteractionListsAndDeviceBuffers(ArrayRef<const int>           nbnxnAtomOrder,
                                                const InteractionDefinitions& idef,
                                                DeviceBuffer<Float4>          d_xqPtr,
                                                DeviceBuffer<RVec>            d_fPtr,
                                                DeviceBuffer<RVec>            d_fShiftPtr,
                                                DeviceBuffer<float>           d_nbLJEnergyPtr,
                                                DeviceBuffer<float>           d_nbElecEnergyPtr);
    /*! \brief
     * Update PBC data.
     *
     * Converts PBC data from t_pbc into the PbcAiuc format and stores the latter.
     *
     * \param[in] pbcType The type of the periodic boundary.
     * \param[in] box     The periodic boundary box matrix.
     * \param[in] canMoleculeSpanPbc  Whether one molecule can have atoms in different PBC cells.
     */
    void setPbc(PbcType pbcType, const matrix box, bool canMoleculeSpanPbc);

    /*! \brief Launches bonded kernel on a GPU */
    template<bool calcVir, bool calcEner>
    void launchKernel();
    /*! \brief Maps and reduces the GaMD dihedral buffers filled by the bonded kernel. */
    void finishGamdDihedralBuffers();
    /*! \brief Launches the diagnostic GaMD force-correction kernel. */
    void launchGamdForceCorrectionShadowKernel(real totalScale, real dihedralCorrectionScale);
    /*! \brief Launches a block-reduced virial kernel for nbnxn-order forces. */
    void launchGamdVirialReductionKernel(DeviceBuffer<Float3> forces,
                                         DeviceBuffer<Float3> shiftForces,
                                         DeviceBuffer<float>* virial,
                                         const char*          kernelName);
    /*! \brief Launches in-place GaMD force correction after a dependency event. */
    void launchGamdForceCorrectionKernel(DeviceBuffer<Float3>  forces,
                                         real                  totalScale,
                                         real                  dihedralCorrectionScale,
                                         GpuEventSynchronizer* rawForcesReady);
    /*! \brief Returns whether there are bonded interactions
     * assigned to the GPU */
    bool haveInteractions() const;
    /*! \brief Launches the transfer of computed bonded energies. */
    void launchEnergyTransfer();
    /*! \brief Waits on the energy transfer, and accumulates bonded energies to \c enerd. */
    void waitAccumulateEnergyTerms(gmx_enerdata_t* enerd);
    /*! \brief Clears the device side energy buffer */
    void clearEnergies();
    /*! \brief Returns whether the diagnostic GaMD dihedral shadow path is enabled. */
    bool gamdDihedralShadowEnabled() const;
    /*! \brief Returns whether device GaMD dihedral-force production is enabled. */
    bool gamdDihedralBufferEnabled() const;
    /*! \brief Returns whether the device GaMD energy/scale path is enabled. */
    bool gamdEnergyShadowEnabled() const;
    /*! \brief Returns whether host/device energy-state comparison is enabled. */
    bool gamdEnergyShadowDiagnosticsEnabled() const;
    /*! \brief Returns whether in-place correction consumes device GaMD scales. */
    bool gamdScaleFromDeviceEnabled() const;
    /*! \brief Returns PME reciprocal-energy staging owned by this consumer. */
    DeviceBuffer<float> gamdPmeEnergyStagingBuffer();
    /*! \brief Returns the event marked after PME reciprocal-energy staging. */
    GpuEventSynchronizer* gamdPmeEnergyReadyEvent();
    /*! \brief Reduces raw energies and evaluates production GaMD scales on device. */
    void launchGamdEnergyShadowReduction(int    igamd,
                                         int    stage,
                                         double thresholdP,
                                         double kP,
                                         double thresholdD,
                                         double kD,
                                         bool   recordEnergySample);
    /*! \brief Returns device VP, VD, scaleP, scaleD, boostP, and boostD. */
    std::array<double, 6> gamdEnergyShadowValues();
    /*! \brief Copies and clears device-resident deferred F_NRE energy samples. */
    std::vector<std::array<real, F_NRE>> takeGamdEnergyHistory();
    /*! \brief Records corrected force virial for the current deferred sample. */
    void recordGamdForceVirialSample();
    /*! \brief Copies device-resident corrected force-virial samples. */
    std::vector<std::array<real, DIM * DIM>> takeGamdForceVirialHistory(int numSamples);
    /*! \brief Copies the state-order diagnostic GaMD dihedral force buffer to the host. */
    void copyGamdDihedralShadowForces(ArrayRef<RVec> forces);
    /*! \brief Copies the device-reduced raw GaMD dihedral virial to the host. */
    void copyGamdDihedralVirial(std::array<real, DIM * DIM>* virial);
    /*! \brief Reduces the complete raw short-range virial after GPU NB/listed completion. */
    void launchGamdShortRangeVirialReduction();
    /*! \brief Copies the device-reduced complete raw short-range virial to the host. */
    void copyGamdShortRangeVirial(std::array<real, DIM * DIM>* virial);
    /*! \brief Applies the GaMD force formula in a diagnostic device buffer. */
    void applyGamdForceCorrectionShadow(ArrayRef<const RVec> rawForces,
                                        real                 totalScale,
                                        real                 dihedralCorrectionScale,
                                        ArrayRef<RVec>       correctedForces);
    /*! \brief Applies GaMD force correction in-place after raw GPU force reduction. */
    void applyGamdForceCorrection(DeviceBuffer<RVec>    forces,
                                  real                  totalScale,
                                  real                  dihedralCorrectionScale,
                                  GpuEventSynchronizer* rawForcesReady);
    /*! \brief Returns the event marked after in-place GaMD force correction. */
    GpuEventSynchronizer* gamdForcesReadyEvent();

private:
    static constexpr int c_gamdEnergyHistoryCapacity_ = 64;

    /*! \brief The interaction lists
     *
     * \todo This is potentially several pinned allocations, which
     * could contribute to exhausting such pages. */
    std::array<HostInteractionList, F_NRE> iLists_;

    //! Tells whether there are any interaction in iLists.
    bool haveInteractions_ = false;
    //! Interaction lists on the device.
    std::array<DeviceBuffer<t_iatom>, F_NRE> d_iAtoms_      = {};
    std::array<int, F_NRE>                   d_iAtomsAlloc_ = {};
    //! Bonded parameters for device-side use.
    DeviceBuffer<t_iparams> d_forceParams_ = nullptr;
    //! Precomputed bicubic coefficients for all CMAP grid cells.
    DeviceBuffer<float> d_cmapCoefficients_ = nullptr;
    //! Position-charge vector on the device.
    DeviceBuffer<Float4> d_xq_ = nullptr;
    //! Force vector on the device.
    DeviceBuffer<Float3> d_f_ = nullptr;
    //! Shift force vector on the device.
    DeviceBuffer<Float3> d_fShift_ = nullptr;
    //! Short-range LJ and electrostatic energy scalars owned by NBNXM.
    DeviceBuffer<float> d_nbLJEnergy_   = nullptr;
    DeviceBuffer<float> d_nbElecEnergy_ = nullptr;
    //! Diagnostic raw GaMD dihedral forces in nbnxn atom order.
    DeviceBuffer<Float3> d_gamdDihedralForcesNbnxn_ = nullptr;
    //! Diagnostic raw GaMD dihedral forces in state atom order.
    DeviceBuffer<Float3> d_gamdDihedralForcesState_ = nullptr;
    //! Raw GaMD dihedral shift forces.
    DeviceBuffer<Float3> d_gamdDihedralShiftForces_ = nullptr;
    //! Device-reduced raw GaMD dihedral virial.
    DeviceBuffer<float> d_gamdDihedralVirial_ = nullptr;
    //! Device-reduced complete raw short-range virial.
    DeviceBuffer<float> d_gamdShortRangeVirial_ = nullptr;
    //! Diagnostic total-force buffer corrected by the GaMD device kernel.
    DeviceBuffer<Float3> d_gamdCorrectedForcesState_ = nullptr;
    //! State-atom to nbnxn-atom index mapping.
    DeviceBuffer<int>   d_nbnxnAtomOrder_                   = nullptr;
    int                 gamdNbnxnForceSize_                 = 0;
    int                 gamdNbnxnForceCapacity_             = 0;
    int                 gamdStateForceSize_                 = 0;
    int                 gamdStateForceCapacity_             = 0;
    int                 gamdCorrectedForceSize_             = 0;
    int                 gamdCorrectedForceCapacity_         = 0;
    int                 nbnxnAtomOrderSize_                 = 0;
    int                 nbnxnAtomOrderCapacity_             = 0;
    bool                gamdDihedralShadowEnabled_          = false;
    bool                gamdDihedralBufferEnabled_          = false;
    bool                gamdForceCorrectionEnabled_         = false;
    bool                gamdEnergyShadowEnabled_            = false;
    bool                gamdEnergyShadowDiagnosticsEnabled_ = false;
    bool                gamdScaleFromDeviceEnabled_         = false;
    bool                gamdEnergyHistoryEnabled_           = false;
    DeviceBuffer<float> d_gamdPmeEnergy_                    = nullptr;
    DeviceBuffer<float> d_gamdEnergyShadow_                 = nullptr;
    DeviceBuffer<float> d_gamdEnergyHistory_                = nullptr;
    DeviceBuffer<float> d_gamdForceVirialHistory_           = nullptr;
    DeviceBuffer<int>   d_gamdEnergyHistoryCount_           = nullptr;
    HostVector<float> h_gamdEnergyShadow_ = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<float> h_gamdEnergyHistory_ = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<float>                     h_gamdForceVirialHistory_ = { {},
                                                                        gmx::HostAllocationPolicy(
                                                            gmx::PinningPolicy::PinnedIfSupported) };
    HostVector<int>                       h_gamdEnergyHistoryCount_ = { {},
                                                                        gmx::HostAllocationPolicy(
                                                          gmx::PinningPolicy::PinnedIfSupported) };
    std::unique_ptr<GpuEventSynchronizer> gamdPmeEnergyReadyEvent_;
    std::unique_ptr<GpuEventSynchronizer> gamdForcesReadyEvent_;
    //! \brief Host-side virial buffer
    HostVector<float> vTot_ = { {}, gmx::HostAllocationPolicy(gmx::PinningPolicy::PinnedIfSupported) };
    //! \brief Device-side total virial
    DeviceBuffer<float> d_vTot_ = nullptr;

    //! GPU context object
    const DeviceContext& deviceContext_;
    //! \brief Bonded GPU stream, not owned by this module
    const DeviceStream& deviceStream_;

    //! Parameters, passed to the GPU kernel
    BondedGpuKernelParameters kernelParams_;
    //! Buffers, used in the GPU kernel
    BondedGpuKernelBuffers kernelBuffers_;

    //! Device sub-group/warp size
    int deviceSubGroupSize_;

    //! GPU kernel launch configuration
    KernelLaunchConfig kernelLaunchConfig_;

    //! \brief Pointer to wallcycle structure.
    gmx_wallcycle* wcycle_;
};

} // namespace gmx

#endif // GMX_LISTED_FORCES_LISTED_FORCES_GPU_IMPL_H
