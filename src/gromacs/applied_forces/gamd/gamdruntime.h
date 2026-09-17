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
 * \brief Declares instance-owned GaMD statistics, planning and persistence.
 *
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_GAMD_GAMDRUNTIME_H
#define GMX_APPLIED_FORCES_GAMD_GAMDRUNTIME_H

#include <cstdint>
#include <cstdio>

#include <array>
#include <string>
#include <vector>

#include "gromacs/applied_forces/gamd/gamdmath.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fileptr.h"
#include "gromacs/utility/real.h"

struct gmx_wallcycle;

namespace gmx
{

//! \cond internal
//! \addtogroup module_applied_forces
//! \{

class GaMDDiagnostics;

/*! \internal \brief Validated legacy input parameters, with energies stored in kJ/mol. */
struct GaMDParameters
{
    //! Legacy boost selection: potential, dihedral or dual (1, 2 or 3).
    int boostType = 0;
    //! Legacy irest_gamd flag for using initialized production parameters.
    int restartMode = 0;
    //! Legacy iE field retained for input and restart compatibility.
    int thresholdMode = 1;
    //! iEP lower/upper threshold policy for the potential boost.
    int potentialMode = 1;
    //! iED lower/upper threshold policy for the dihedral boost.
    int dihedralMode = 1;
    //! Last conventional preparation step.
    int64_t conventionalPrepEnd = 200000;
    //! Last conventional statistics step.
    int64_t conventionalEnd = 1000000;
    //! Number of boost-preparation steps after conventional statistics.
    int64_t boostPrepLength = 200000;
    //! Total number of adaptive boost steps after conventional statistics.
    int64_t boostLength = 1000000;
    //! Number of samples per population-statistics window.
    int64_t averagingWindow = 50000;
    //! Logical step interval for bias records.
    int64_t reweightInterval = 1;
    //! Logical parameter-output interval; zero disables it.
    int64_t parameterInterval = 0;
    //! Potential-boost deviation limit in kJ/mol.
    double potentialSigmaLimit = 6.0 * 4.184;
    //! Dihedral-boost deviation limit in kJ/mol.
    double dihedralSigmaLimit = 6.0 * 4.184;
};

//! 输入和重启文件共用阶段校验；空字符串表示这些计数可以安全使用。
std::string gaMDStageParameterError(const GaMDParameters& parameters);

//! 拆分一行 legacy 输入；保留空行和两种行尾注释的既有语义。
std::vector<std::string> splitGaMDInputLine(const std::string& line);

/*! \internal
 * \brief Owns all mutable GaMD host state for one simulation.
 */
class GaMDRuntime
{
public:
    //! Constructs runtime state referring to diagnostics that outlive this object.
    explicit GaMDRuntime(GaMDDiagnostics& diagnostics) : diagnostics_(diagnostics) {}
    //! Closes owned output streams and contains any close failure at the legacy error boundary.
    ~GaMDRuntime();

    //! 统计状态和输出文件不能被另一个运行实例隐式共享。
    GaMDRuntime(const GaMDRuntime&) = delete;
    //! Disables assignment of owned runtime state and files.
    GaMDRuntime& operator=(const GaMDRuntime&) = delete;

    //! Loads validated input and prepares one step; throws on invalid input or I/O.
    const GaMDStepPlan& prepareStep(int64_t step, int rank, bool pressureCouplingThisStep);
    //! Returns the result from current-step host accounting.
    const GaMDStepResult& stepResult() const { return stepResult_; }
    //! Returns the immutable parameters used by CPU, CUDA and deferred output.
    GaMDBiasParameters biasParameters() const;
    //! Updates statistics once per step and evaluates the current bias.
    void finalizeCurrentStep(int64_t        step,
                             int            rank,
                             double         totalPotentialEnergy,
                             double         dihedralEnergy,
                             gmx_wallcycle* wallCycle);
    //! Reports the host-energy requirement for the prepared step.
    bool requiresHostEnergyThisStep(int64_t step) const;
    //! Configures deferred text reconstruction for resident production.
    void setBufferedProductionOutputEnabled(bool enabled);
    //! Reports whether buffering is active without force overrides.
    bool bufferedProductionOutputEnabled() const;
    //! Reconstructs a complete batch in step order; throws on output failure.
    void writeBufferedProductionOutput(ArrayRef<const int64_t> deferredSteps,
                                       ArrayRef<const std::array<real, F_NRE>> deferredEnergySamples,
                                       int64_t        currentStep,
                                       int            rank,
                                       gmx_wallcycle* wallCycle);
    //! Mirrors the MD checkpoint decision and updates host-staging requirements.
    void setCheckpointingThisStep(bool checkpointingThisStep);
    //! Commits eligible restart state before the MD energy buffers are reset.
    void writeRestartState(int64_t step);
    //! Logs once if the requested run cannot reach production.
    void warnIfRunTooShort(int64_t initStep, int64_t numSteps, int rank);

private:
    //! Reads gamd.in with strict conversion; throws FileIOError or InvalidInputError.
    void loadParameters(int rank);
    //! Selects a valid matching restart snapshot before restoring state.
    void loadRestartStateIfNeeded(int64_t step, int rank);
    //! Flushes outputs and atomically replaces the restart commit record.
    void saveRestartState(int64_t step, int rank);
    //! Maps a validated step to a GaMD stage, including production restart.
    GaMDStage determineStage(int64_t step) const;

    //! Returns the borrowed handle for main-rank reweight output.
    FILE* reweightFile(int rank);
    //! Returns the borrowed handle for main-rank parameter output.
    FILE* parameterFile(int rank);
    //! Writes one eligible parameter row in the legacy units and order.
    void writeParameterLine(int64_t step, int rank, bool flushOutput);
    //! Writes one eligible bias record with energies converted to kcal/mol.
    void writeReweightRecord(int64_t step, int rank, const GaMDStepResult& record, bool flushOutput);
    //! Writes the current host result at the configured output cadence.
    void writeReweightLine(int64_t step, int rank, bool flushOutput);
    //! Reports whether checkpoint reconciliation already retained this row.
    bool suppressTextOutputForStep(int64_t step) const;
    //! Closes owned outputs and throws FileIOError if closing fails.
    void closeOutputFiles();

    //! Rejects non-finite bias values and logs the existing force-weight floor.
    double validateAndClampScale(const char*                 label,
                                 double                      rawScale,
                                 double                      energy,
                                 const GaMDEnergyStatistics& statistics,
                                 int64_t                     step,
                                 int                         rank,
                                 bool*                       warnedAtFloor);

    GaMDDiagnostics&     diagnostics_;
    GaMDParameters       parameters_;
    bool                 parametersLoaded_ = false;
    GaMDEnergyStatistics potentialStatistics_;
    GaMDEnergyStatistics dihedralStatistics_;
    GaMDWindowStatistics potentialWindowStatistics_;
    GaMDWindowStatistics dihedralWindowStatistics_;

    GaMDStepPlan   stepPlan_;
    GaMDStepResult stepResult_;
    int64_t        lastPreparedStep_  = -1;
    int64_t        lastAccountedStep_ = -1;

    bool    restartLoadAttempted_            = false;
    bool    forceProductionRestart_          = false;
    int     currentRank_                     = 0;
    bool    checkpointingThisStep_           = false;
    bool    bufferedProductionOutputEnabled_ = false;
    int64_t suppressTextOutputStep_          = -1;
    bool    warnedPotentialScaleFloor_       = false;
    bool    warnedDihedralScaleFloor_        = false;
    bool    warnedAboutRunLength_            = false;
    int     debug_                           = 0;

    FilePtr reweightFile_;
    FilePtr parameterFile_;
};

//! \}
//! \endcond

} // namespace gmx

#endif
