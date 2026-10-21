.. _gamd-downstream:

Gaussian accelerated molecular dynamics (GaMD)
==============================================

This page describes the GaMD extension in this downstream development tree.
It is based on GROMACS 2025.4 and is not a statement that the feature has been
accepted upstream. See :doc:`../../dev-manual/gamd-upstream` for integration
boundaries and the proposed contribution sequence.

The original method is described by Y. Miao, V. A. Feher, and J. A. McCammon,
*Journal of Chemical Theory and Computation* **11**, 3584--3595 (2015),
`doi:10.1021/acs.jctc.5b00436 <https://doi.org/10.1021/acs.jctc.5b00436>`_.
The implementation-specific component and safety policies are documented below.

Bias and force definition
-------------------------

For energy component :math:`V`, threshold :math:`E`, and non-negative force
constant :math:`k`, the bias is

.. math::

   B(V;E,k) = \begin{cases}
       \tfrac12 k(E-V)^2, & V < E,\\
       0, & V \ge E.
   \end{cases}

The corresponding raw force weight below the threshold is
:math:`s=1-k(E-V)` and is one above the threshold. Energies are in kJ/mol and
:math:`k` is in inverse kJ/mol. The core returns raw weights; the calling
CPU/CUDA boundary applies the existing safety policy.

``igamd=1`` boosts the total potential, ``igamd=2`` boosts the dihedral
component, and ``igamd=3`` applies the following ordered dual boost:

.. math::

   B_D &= B(V_D; E_D, k_D),\\
   B_P &= B(V_P+B_D; E_P, k_P),\\
   V^* &= V_P + B_D + B_P,\\
   \mathbf F^* &= s_P\mathbf F_P + s_P(s_D-1)\mathbf F_D.

Here :math:`\mathbf F_P` is the complete unbiased force, including the
dihedral contribution; :math:`\mathbf F_D` is the force of the selected
component. Applying the second term to a force that already excludes
:math:`\mathbf F_D` would be incorrect.

The present component is the sum of ``F_PDIHS``, ``F_RBDIHS``, ``F_FOURDIHS``,
and ``F_CMAP``. It is not Amber's proper-dihedral-only partition. Changing this
partition changes both the bias and the meaning of saved statistics, and needs
separate scientific validation and restart versioning.

.. warning::

   The existing runtime clamps each force weight to a minimum of 0.01 without
   changing the reported bias energy. If that floor is active, the applied
   force is not the exact gradient of the unmodified bias above. This refactor
   preserves that compatibility policy and warns on the CPU path. Its physical
   interpretation and an upstream policy must be agreed before submission;
   numerical equivalence to this fork is not independent scientific validation.

Statistics and stages
---------------------

Each averaging window uses Welford accumulation of count, mean, and M2. The
standard deviation is the population value ``sqrt(M2/count)``, not the sample
estimate with ``count-1``. Min/max and cumulative counts are stored separately.
An empty window does not replace previous statistics; a one-sample window has
zero deviation. Degenerate energy ranges retain the previous threshold and
force constant.

For lower-bound mode, ``E=Vmax`` and

.. math::

   k_0=\min\left(1,\frac{\sigma_0}{\sigma_V}
                    \frac{V_{\max}-V_{\min}}{V_{\max}-V_{\rm avg}}\right),
   \qquad k=\frac{k_0}{V_{\max}-V_{\min}}.

Upper-bound mode first evaluates

.. math::

   k_0=\left(1-\frac{\sigma_0}{\sigma_V}\right)
       \frac{V_{\max}-V_{\min}}{V_{\rm avg}-V_{\min}}.

If ``0 < k0 <= 1``, it uses ``E=Vmin+(Vmax-Vmin)/k0``; otherwise it falls back
to the lower-bound calculation. The implementation retains its existing
``1e-6`` denominator/variance guards. ``iEP`` and ``iED`` select the component
threshold policies. The legacy ``iE`` field remains validated and recorded for
compatibility; it does not override those component selections.

The five stages are conventional preparation, conventional statistics, boost
preparation, boosted statistics, and fixed-parameter production. The two
preparation stages do not update statistics. Stage four accumulates the
boosted total potential while retaining the existing dihedral statistics
convention. Thresholds update at the conventional-statistics boundary and
during stage four. ``irest_gamd=1`` resumes initialized parameters in production.

Input, output, and continuation
-------------------------------

Set ``gamd = yes`` in the MDP used by ``grompp``. This TPR flag is the only
switch that enables GaMD; there is no environment-variable alternative, so a
TPR written without it, including one from an unmodified ``grompp``, must be
regenerated with this branch. The TPR currently stores only the activation
flag; detailed parameters are read from ``gamd.in`` in the ``mdrun`` working
directory. Preserve that file together with the TPR.

The strict legacy parser rejects unknown/duplicate keys, missing values,
trailing non-comment tokens, invalid ranges, and unaligned averaging windows.
Input and restart snapshots share stage validation: ``ntave`` must be positive,
stage lengths must be ordered and window-aligned, and ``ntcmd + nteb`` must fit
in a signed 64-bit step counter. Invalid restart windows are rejected.
With ``irest_gamd=1``, stage controls in ``gamd.in`` may still be omitted or zero;
the validated restart supplies their stored values.
``sigma0P`` and ``sigma0D`` are provided in kcal/mol and converted to kJ/mol.
``gamd-para.dat`` uses kJ/mol; ``gamd-reweight.dat`` uses kcal/mol. The two boost
columns are separate contributions and must be combined when a total bias is
needed for reweighting.

The runtime preserves ``gamd-restart.dat`` versions 1--3 and the current
version-3 writer. It rejects duplicate/unknown fields, trailing non-comment
content, invalid values, and a version-3 record without ``production_restart``.
Each candidate is parsed into a complete snapshot before selection; a malformed
current file can only be replaced by a validated backup matching the checkpoint.
A valid matching current file does not depend on an unused older backup.
Parameters, statistics, window state, and saved step are validated before use. Text output is reconciled to the checkpoint without
replaying duplicate rows. Counters and their text formatting are 64 bit.
Reconciliation checks the stream error state as well as flushing and syncing;
a failed write stops replacement and preserves the original output file.

GPU production can buffer text/energy histories until a host consumer needs
them. This changes when rows are physically written, not their logical step
numbers or configured ``reweight_nst`` / ``para_nst`` cadence.
The fixed batch capacity is 4096 samples. A full batch forces host staging
before another deferred sample is recorded, including during CUDA Graph replay.
This also applies when ``nstenergy`` and ``nstlog`` exceed the batch capacity.

Supported execution
-------------------

Unsupported configurations are rejected once, before dynamics. All GaMD
execution modes require ``integrator = md``, ``nstcalcenergy = 1`` (the bias
uses the raw energies of the current step), and one PP rank without domain
decomposition; MTS and ``mdrun -rerun`` are rejected. CPU update does not
provide a distributed GaMD energy/force reduction. The CPU reference path is
selected with ``-update cpu``. Single-rank CPU runs may still use the engine's
DD-style atom ordering; their local periodic-boundary information and atom
mapping remain in use.
The validated GPU-resident target uses CUDA mixed precision and one GPU, with
``-nb gpu -pme gpu -bonded gpu -update gpu``. It requires ``nstfout=0`` and does
not support DD, separate PME ranks, MTS, more than two temperature-coupling
groups, or the single-rank DD-style atom ordering requested with
``GMX_DD_SINGLE_RANK=1``. The supported pressure modes are NVT and C-rescale
isotropic/semiisotropic NPT. GPU update itself rejects virtual sites, Ewald
surface correction and orientation restraints, and PME rejects screw PBC.
Topologies with ``F_FOURDIHS`` or perturbed dihedrals are rejected at the start
of the run, when the GPU listed-force buffers are set up. Coupling steps stage
the current corrected virial and required globals; eligible intervening steps
retain the resident/CUDA Graph path. Other builds/workloads are not implied to
be supported.

Diagnostic configuration
------------------------

Explicit numeric diagnostic environment variables use the existing GROMACS
strict converters. Empty, non-numeric, trailing-text, overflow, and non-finite
force override values are errors rather than implicit zeroes. Integer debug
indices and steps also use checked conversion. These switches select optional
diagnostics; production mode selection remains automatic. All diagnostic
variables, including ``GMX_GAMD_GPU_DIH_SHADOW`` and
``GMX_GAMD_GPU_ENERGY_SHADOW``, are read once when ``mdrun`` starts. With
``-update cpu`` the GPU dihedral shadow does not support DD-style atom ordering
or virtual sites; such runs stop before dynamics.

GaMD information messages and successful CPU/GPU shadow comparisons use the
standard simulation logger. With ``-deffnm run`` they are recorded in
``run.log``; warnings use the configured warning output as well. Scripts that
verify a shadow comparison should read the simulation log. Input, numerical
and requested-file errors still report their cause and terminate the run.

Force overrides are read once on first GaMD use and shared by CPU and CUDA.
They only affect GaMD-enabled simulations. The current-step force correction
applies the requested weights once to forces and virial; the integration loop
and individual bonded kernels do not apply them again. Forced diagnostics use
immediate output so a buffered-output flush cannot repeat the current record.
On CUDA, final-force diagnostic output uses the existing shadow-force transfer
to read the corrected device forces. These diagnostic transfers can add cost.

``GMX_GAMD_DIH_FORCE_DEBUG_FILE`` and ``GMX_GAMD_DIH_FORCE_DEBUG_STEP`` request
one dihedral force snapshot. Its value is ``scaleP * scaleD * F_dihedral``, the
dihedral contribution to the corrected total force. ``F_dihedral`` uses the
dihedral/CMAP partition defined above. Rows are written once in ascending,
zero-based topology atom order on both CPU and GPU. The optional
``GMX_GAMD_DIH_FORCE_DEBUG_BEGIN`` and ``GMX_GAMD_DIH_FORCE_DEBUG_COUNT`` select
that topology range; the defaults are zero and all remaining atoms.

GPU snapshots read the existing device dihedral-force buffer at the selected
step. They work without enabling the full GPU shadow comparison. Host staging
is requested for that step so the snapshot uses its current force weights.

``GMX_GAMD_FORCE_DEBUG_FILE`` retains the final-force dump in its current force
buffer's atom order. Explicit requests for either file report open, write,
flush, or close errors instead of silently disabling output. The former raw CPU
bonded-evaluation trace ``GMX_GAMD_DIH_RAW_DEBUG_FILE`` has been removed, so
the native bonded kernels carry no GaMD hooks; the variable is no longer read.

Verification
------------

``AppliedForcesUnitTest`` contains analytical single/dual-boost values,
finite-difference checks of the un-clamped biased potential against the force
composition, threshold branches, population statistics, restart counters, and
input failures. CPU and CUDA compile the same bias evaluator. Integration
validation additionally needs numerical GPU shadows, all supported continuation
ensembles, CUDA memory checking, and matched production-equivalent benchmarks.
Long-time trajectory equality is not a suitable sole criterion for stochastic
or chaotic simulations.
