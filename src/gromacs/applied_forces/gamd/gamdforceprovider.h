/*
 * Copyright 2026 The GROMACS Authors
 */
#pragma once

#include <cstdint>

#include "gromacs/mdtypes/gamd_params.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/topology/topology.h"

struct gmx_wallcycle;

namespace gmx
{

struct GaMDGpuProductionParameters
{
    int    igamd      = 0;
    int    stage      = 0;
    double thresholdP = 0;
    double kP         = 0;
    double thresholdD = 0;
    double kD         = 0;
};

void                        gamdPrepareStep(long step, int nodeid);
GaMDGpuProductionParameters gamdGpuProductionParameters();
bool                        gamdRequiresHostEnergyThisStep(long step);
void                        gamdWarnIfRunTooShort(int64_t initStep, int64_t nsteps, int nodeid);
void                        gamdFinalizeCurrentStep(long           step,
                                                    int            nodeid,
                                                    double         totalPotentialEnergy,
                                                    double         dihedralEnergy,
                                                    gmx_wallcycle* wcycle = nullptr);
void                        gamdSetCheckpointingThisStep(bool checkpointingThisStep);
void                        gamdWriteRestartState(long step);
void                        gamdResetStateForTesting();
const char* currentStepGaMDGpuIncompatibilityReason(bool useGpuUpdate, bool haveGpuBondedWork);
GaMDExecutionMode       selectGaMDExecutionMode(bool useGaMD,
                                                bool useGpuNonbonded,
                                                bool useGpuPme,
                                                bool useGpuBonded,
                                                bool useGpuUpdate,
                                                bool havePpDomainDecomposition,
                                                bool haveSeparatePmeRank,
                                                bool useMts,
                                                bool havePressureCoupling,
                                                int  nstfout);
const char*             gaMDExecutionModeName(GaMDExecutionMode mode);
class GaMDForceProvider final : public IForceProvider
{
public:
    // 删除了 t_state，彻底避开 runner.cpp 的编译错误
    GaMDForceProvider(const GaMDParams& params, const t_inputrec& ir, const gmx_mtop_t& mtop);

    void calculateForces(const ForceProviderInput& fpin, ForceProviderOutput* fpout) override;

    real getBoostPotential() const { return boostPotential_; }

private:
    GaMDParams params_;
    real       boostPotential_ = 0.0;
};

} // namespace gmx
