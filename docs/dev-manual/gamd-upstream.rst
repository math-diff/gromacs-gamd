.. _gamd-upstream-review:

GaMD architecture and upstream review plan
==========================================

Scope and current status
------------------------

This is a maintainability refactor of a downstream GROMACS 2025.4 extension,
not an upstream merge approval. The baseline is ``5ff664c`` in the local
``gromacs-2025.4-guanfang`` repository. The production input/restart contract
and GPU schedule are intentionally preserved. Validation evidence belongs to
``.trellis/tasks/archive/2026-09/09-10-gamd-upstream-maintainability/validation.md`` in the
parent workspace; it must accompany any proposed patch series.

GROMACS main was inspected on 2026-09-10 at
``5ddcd2781e85070e16d474e4daa20fd7b6f0da86``. Its ``IMdpOptionProvider`` still
exposes ``initMdpTransform``, ``initMdpOptions``, and ``buildMdpOutput``. This
inspection is not a build or test of the full extension against that commit.

Local verification summary
--------------------------

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
   * - ``gamdgpu.*`` and ``gamddiagnostics.*``
     - Device buffers/events/launches and instance-owned diagnostics.

``GaMDBiasParameters`` is the shared immutable bias input. ``GaMDStepResult``
is reused for live and deferred results; no second six-field output record or
new strategy/factory hierarchy is needed.

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
