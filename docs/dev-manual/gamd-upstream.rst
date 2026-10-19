.. _gamd-upstream-review:

GaMD architecture and upstream review plan
==========================================

Scope and current status
------------------------

This is a maintainability refactor of a downstream GROMACS 2025.4 extension,
not an upstream merge approval. The initial refactor used ``5ff664c`` in the local
``gromacs-2025.4-guanfang`` repository. The production input/restart contract
and GPU schedule are intentionally preserved. Validation evidence belongs to
``.trellis/tasks/archive/2026-09/09-10-gamd-upstream-maintainability/validation.md`` in the
parent workspace; it must accompany any proposed patch series.

GROMACS main was inspected on 2026-09-10 at
``5ddcd2781e85070e16d474e4daa20fd7b6f0da86``. Its ``IMdpOptionProvider`` still
exposes ``initMdpTransform``, ``initMdpOptions``, and ``buildMdpOutput``. This
inspection is not a build or test of the full extension against that commit.

Initial refactor verification
-----------------------------

The refactor was checked on an RTX 4080 with CUDA 12.4, GCC 12.3, mixed
precision, one rank, and 28 OpenMP threads. The CUDA build passed six regression
suites; a separate CPU-only build passed the applied-forces suite. Seven
analytical math tests cover the shared core. Fourteen simulation invocations
cover CPU, non-GaMD, strict shadows, all supported continuation ensembles, and
CUDA memcheck (zero errors). Each 2000-step continuation produced exactly 40
output rows; parameter/restart files matched uninterrupted runs byte for byte.

The strict shadow recorded 102 samples: energy differences were zero, force
weight differences were below 3e-8, corrected-force max/RMS were below
0.000901/0.00000462, and dihedral virial max was below 0.00418, within the
existing mixed-precision limits.

Production-equivalent 100k measurements were 291.247 ns/day for the rebuilt NVT
baseline and 282.932/289.960 ns/day for two refactored runs; isotropic NPT was
287.526 versus 285.527 ns/day. These met the existing 3% non-regression gate.
The samples are too few to claim a speed improvement or statistical certainty.

Ownership and source map
------------------------

``MDModules`` owns one ``GaMDMDModule``, which owns ``GaMDRuntime``, diagnostics,
and the optional GPU executor. Keep that single ownership chain.

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - File
     - Responsibility
   * - ``gamdmath.h/.cpp``
     - Pure host/device bias evaluation, population statistics, and thresholds.
       No files, environment variables, logger, or simulator state.
   * - ``gamdcontroller.cpp``
     - Stage/step preparation, accounting, boundary validation, and buffered
       sample scheduling. Uses the same evaluator as CUDA.
   * - ``gamdinput.cpp``
     - Strict legacy parameter parsing and validation.
   * - ``gamdfileio.cpp``
     - Legacy restart/reconciliation and text output. Isolated so native
       checkpoint migration need not change scientific mathematics.
   * - ``gamd_scale_gpu.cu``
     - Existing device-state layout and launch wrapper around the shared math.
   * - ``gamdexecution.h/.cpp``
     - Execution selection and the legacy current-step correction entry. The
       CPU dihedral recomputation and CPU/GPU force sequences are private here.
   * - ``gamdgpu.*`` and ``gamddiagnostics.*``
     - Device buffers/events/launches and instance-owned diagnostics.

``GaMDBiasParameters`` is the shared immutable bias input. ``GaMDStepResult``
is reused for live and deferred results; no second six-field output record or
new strategy/factory hierarchy is needed.

Resource ownership, errors and documentation
---------------------------------------------

``GaMDRuntime`` and ``GaMDDiagnostics`` explicitly disable copying and own their
output streams through the existing ``FilePtr`` type. This also closes streams
if construction or input preparation fails partway through. The three shared
statistics functions use ``compat::not_null`` output pointers; their formulas
and population-variance convention are unchanged.

Legacy input and restart parsing use ``TextReader``, ``splitString``, and the
native strict numeric converters. Attached ``#``/``;`` comments, 64-bit
counters, restart versions 1--3, and NaN placeholder columns in parameter
history remain supported. Reconciliation preserves row whitespace and checks
write, flush, synchronization and close errors before replacing an output file.

Simulation-setup notifications supply ``MDLogger`` to the diagnostics owner.
Parameter summaries, execution messages and successful shadow comparisons use
its information target; warnings use its warning target. Verification tools
must inspect the configured simulation log, normally ``run.log`` with
``-deffnm run``, for the successful comparison records.

Internal parsing, I/O and numerical checks use the native input, file and
simulation-instability exception types. The module's legacy MD entry points,
the force-correction entry, diagnostic output callbacks, the GPU executor
initialization and the GPU buffer binding entry contain exceptions with the
existing fatal-error adapter. GPU buffer binding can be called after
asynchronous interaction-list uploads, and raw bonded diagnostics can execute
inside OpenMP; neither may unwind its caller. Executor initialization runs after
the runner has created GPU nonbonded objects; unwinding through
``Mdrunner::mdrunner`` would release them after the device stream and crash.
Destructors also contain close errors. Reinitialize an internal runtime after
failed input preparation; retrying a partially prepared instance is unsupported.

Doxygen files belong to the existing ``module_applied_forces`` group. Headers
used across modules declare library visibility, while implementation details
use the internal documentation section in both declarations and definitions.
The source-convention validation record is
``benchmarks/gamd-upstream-maintainability/gromacs-style-validation.md`` in the
parent workspace. Its scoped documentation checks do not certify a full
GROMACS documentation build or compatibility with current upstream main.

Scheduling and integration contracts
------------------------------------

``GaMDStage`` names the existing stage numbers. Input and restart candidates
share ordered/window-aligned stage validation and an overflow check before
forming ``ntcmd + nteb``. The six-float device bias state uses named indices;
its diagnostic host copy is returned as ``GaMDStepResult``. The GPU executor
stores one resident-mode flag and derives buffer/history capabilities from it
and the independent diagnostic switches.

``GaMDGpuExecutor::sc_historyCapacity`` is the single limit for device history
allocation and host scheduling. The MD loop checks its existing
``deferredPressureBoxes`` count before selecting host staging or graph replay;
a full batch is drained through ``takeHistory`` before appending another sample.
The host list advances on graph replay too, so this needs no additional counter
or per-step device readback.

The MD loop uses ``useGpuResidentGaMD`` for the selected execution mode and
``deferGaMDHostWork`` for steps that can postpone host statistics and readback.
CUDA Graph eligibility and replay retain their own scheduling checks. CPU
GaMD requires CPU update; GPU update requires the supported resident path.
Native CPU force work still controls its coordinate copies and event waits.

``do_force`` explicitly calls the module's restart writer immediately before
``reset_enerdata``. The native energy reset retains its one-argument interface
and only clears energy data. Constraint virial handling stays in
``UpdateConstrainGpu::Impl::integrate``: the deferred branch records device
history, while the immediate branch copies LINCS and SETTLE virials and applies
the existing scale in that order.

The legacy output synchronization helper checks ``ferror`` before treating a
file as committed. An earlier failed write can leave no pending buffer, making
the last ``fflush`` succeed; reconciliation must still stop before replacing
the original output. The regression test injects a file-size-limit failure
through the normal restart-loading entry and checks the original bytes.

``selectGaMDExecutionMode`` is the single setup-time check of supported GaMD
configurations; ``createSimulationWorkload`` calls it before dynamics. The MD
loop, ``do_force`` and the correction entry do not repeat its checks. Limits
already enforced by native GPU update and PME (virtual sites, whole-molecule
transforms, screw PBC) are not duplicated; relaxing one of those native rules
requires a matching GaMD check. Topology limits of the GPU force path need the
local topology and stay at GPU listed-force buffer binding.
GaMD is enabled only by the TPR flag. The execution mode (``Disabled``,
``CpuReference`` or ``GpuResident``) is the single source for whether and where
GaMD runs; the force path receives the module pointer only when GaMD is
enabled, so a non-null pointer means GaMD is active. ``GaMDDiagnostics`` reads
every diagnostic environment variable once, and the GPU executor receives the
shadow switches through its constructor. GPU-shadow combinations that a
CPU-update run cannot support are rejected when the executor is created.
The current-step correction entry performs energy accounting, then calls the
separate CPU or GPU sequence in ``gamdexecution.cpp``.
``sim_util.cpp`` retains the call at the original force-processing boundary.
The private CPU dihedral helper reuses the native bonded kernels and overwrites
its output buffer completely; only its accumulation buffers need clearing.
Numerical
energy, force, and virial comparisons live in ``gamddiagnostics.cpp`` and reject
non-finite values before aggregating errors. CPU bonded kernels produce raw
forces; the GaMD correction owns the final weights. Preserve the distinction
between single-rank DD-style atom ordering and actual PP domain decomposition.

Dihedral snapshots are emitted once from completed CPU/GPU force data through
``GaMDDiagnostics::dumpDihedralForces``. The CPU caller passes its topology atom
mapping; the GPU caller reuses ``copyDihedralForces``. Snapshot requests stage
current host weights at their selected step without enabling the full shadow
comparison. Bonded callbacks only retain the separately labelled raw CPU trace.
The diagnostics implementation shares checked stdio output for dihedral and
final-force snapshots and reports requested-file failures explicitly.

``EnergyOutput::deferGpuSamples`` marks energy, virial and kinetic bins together.
The completion routines retain independent sample-count checks, and replay
kinetic data, virials and energy in the original order. Native ``msmul`` handles
the host-side 3x3 virial scaling without additional allocation or transfer.

PME supplies its private energy buffer and producer stream to
``launchGaMDPmeEnergyStaging`` in ``gamdgpu.cu``. The existing kernel launch and
readiness event stay on that same stream. Execution announcements are emitted
once by ``GaMDDiagnostics`` through the simulation logger.

The raw-energy producers, force/virial ownership, constraints/update, standard
energy output, and MD scheduling still have necessary integration points.
Moving those interfaces into a GaMD directory would not remove their coupling.
A generic upstream interface must be designed with those subsystems' owners.

Decisions to discuss before proposing the full feature
------------------------------------------------------

1. **Scientific definition and admissible parameters.** Agree whether the
   dihedral component includes CMAP and how the 0.01 force floor is handled.
   The floor can break exact energy-gradient consistency; the current refactor
   explicitly preserves and documents it. Decide between rejecting such
   parameters, changing the potential consistently, or a reviewed alternative.
   Independent reference cases and restart definition/versioning are required
   for any scientific change. See :doc:`../reference-manual/special/gamd`.
2. **Native input and compatibility.** Prefer ``IMdpOptionProvider`` and the
   existing options/KVT machinery over further additions to ``t_inputrec``.
   Agree option names, units, component-threshold choices, and whether a legacy
   ``gamd.in`` importer is needed. Avoid silently treating an accepted but
   redundant legacy field as an effective algorithm selector.
3. **Checkpoint ownership and output.** Agree module checkpoint keys/versioning
   and main-rank/broadcast behavior using the existing MDModules checkpoint
   notifications (Colvars is a local example). The present text writer is a
   compatibility boundary, not an invented replacement for standard checkpoint
   storage. Decide how reweight output participates in append/continuation.
4. **GPU insertion interface.** Review a generic ordered device-observable and
   post-force transformation interface for energy, force, virial, and readiness
   events. Agree ownership, PME/NBNXM/listed-force boundaries, CUDA Graph replay,
   and C-rescale synchronization. Do not submit all GaMD-specific bridges as a
   supposedly isolated module or assume a PLUMED/gmxapi hook provides them.
5. **Supported matrix and CI.** Agree the first supported subset, unsupported
   mode diagnostics, CPU/double-precision expectations, CUDA/HIP/SYCL policy,
   MPI/DD scope, quick physics tests, sanitizer coverage, restart tests, and
   acceptable performance noise. Local CUDA/CPU tests do not replace upstream CI.
6. **Provenance and long-term ownership.** Verify authorship and contribution
   rights for every source portion, including any implementation derived from
   Amber or other projects. Supply the scientific citations and applicable
   license evidence, and identify who will maintain/review the feature.
7. **Current-main port and review sequence.** Agree a target main revision and
   reviewers before porting the complete interface surface. The current 2025.4
   build cannot certify compatibility with main or establish maintainer consent.

Proposed review sequence
------------------------

* First: scientific contract, pure math, analytical tests, and documentation.
* Second: agreed native MDP/KVT options and checkpoint state, with migration tests.
* Third: the smallest CPU implementation and end-to-end physics/continuation tests.
* Fourth: generic GPU integration interfaces reviewed with subsystem maintainers.
* Fifth: CUDA-resident execution, pressure coupling, buffered output, and
  performance evidence; other backends follow the agreed support policy.

Keep each change independently reviewable and state its prerequisite commits.
The baseline itself differs from pristine GROMACS in 81 files, so a refactor
inside the module does not by itself make the full feature a small merge request.
Do not erase useful local development history to pretend it is an upstream series.
Prepare the series separately after agreeing on its boundaries.

Draft developer-forum message
-----------------------------

The following draft has not been sent::

   Subject: Early design discussion: GaMD module and GPU post-force interfaces

   We maintain a GaMD extension based on GROMACS 2025.4. We have separated a
   pure bias/statistics core from legacy input/persistence and share the bias
   evaluator between CPU, CUDA, and deferred output. The present supported
   GPU target is single-rank CUDA mixed precision, with NVT and C-rescale
   isotropic/semiisotropic NPT.

   Before preparing an upstream series, we would appreciate guidance on:
   (1) the dihedral/CMAP definition and force-floor policy;
   (2) native MDP/KVT and module-checkpoint integration and legacy migration;
   (3) ordered GPU energy/force/virial interfaces and subsystem ownership;
   (4) the initial support/CI matrix and reviewers for a staged contribution.

   We can provide analytical energy-gradient/statistics tests, CPU/GPU
   numerical comparisons, continuation and memory-check results, and matched
   performance measurements. Independent physics reference cases and code
   provenance will be part of the review package. We are not requesting a
   single merge of the existing cross-subsystem patch.

   Would the proposed split be suitable, and which maintainers should review
   the input/checkpoint and GPU interface designs before implementation?

Official contribution guidance
-------------------------------

* `Contributing to GROMACS <https://manual.gromacs.org/current/dev-manual/contribute.html>`_
  requests early discussion, scientific verification, automated high-level
  tests, documentation, long-term maintenance, and manageable contributions.
* `Change management <https://manual.gromacs.org/current/dev-manual/change-management.html>`_
  describes issues, draft merge requests, main-branch integration, and CI/review.
* `Developer discussion forum <https://gromacs.bioexcel.eu/c/gromacs-developers/10>`_
  is the recommended place for the initial design discussion; check the current
  forum navigation before posting.
