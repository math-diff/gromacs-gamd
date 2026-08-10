# GROMACS 2025.4 with GPU-accelerated GaMD v1.1

This repository is an **unofficial, modified distribution of GROMACS 2025.4**
that adds Gaussian accelerated molecular dynamics (GaMD), including a
high-performance CUDA path.

It is not an official GROMACS release and is not maintained or endorsed by the
GROMACS development team. Report problems with this fork in this repository,
not in the upstream GROMACS issue tracker.

## What this version provides

- GaMD total-potential, dihedral, and dual-boost modes.
- A CPU-reference GaMD path selected with `-update cpu`.
- An automatic GPU-resident GaMD path selected with `-update gpu`.
- GPU force correction, device-side scale evaluation, resident energy and
  global histories, buffered GaMD output, and eligible CUDA Graphs.
- NVT simulations and C-rescale NPT simulations with isotropic or
  semiisotropic pressure coupling.
- Checkpoint-aware GaMD restart files and deterministic reconciliation of GaMD
  text output after continuation.

No GaMD performance-selection environment variables are required. A supported
run with `-update gpu` automatically enables the complete GPU-resident path.

## Supported production target

The validated high-performance configuration is:

- Linux, mixed precision, and an NVIDIA CUDA build.
- One PP rank and one GPU: `-ntmpi 1`.
- GPU nonbonded, PME, bonded, and update tasks.
- No domain decomposition, separate PME rank, or multiple time stepping.
- `nstfout = 0` because corrected-force trajectory output is not implemented.
- One or two temperature-coupling groups.
- No pressure coupling, or C-rescale with `isotropic` or `semiisotropic`
  coupling.

Unsupported GPU-update combinations fail before dynamics instead of silently
falling back to a different GaMD algorithm.

## Installation

### Requirements

- CMake 3.28 or newer.
- A C++17 compiler supported by GROMACS 2025.4.
- CUDA Toolkit 12.1 or newer for the GPU-resident path.
- An NVIDIA driver compatible with the selected CUDA Toolkit.
- FFTW single-precision libraries, or permission for CMake to download and
  build FFTW automatically.

The complete upstream build documentation remains available in
[`docs/install-guide/index.rst`](docs/install-guide/index.rst).

### Recommended CUDA build

The following installs into the current user's home directory and does not
overwrite another GROMACS installation:

```bash
git clone https://github.com/math-diff/gromacs-gamd.git
cd gromacs-gamd

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGMX_GPU=CUDA \
  -DGMX_MPI=OFF \
  -DGMX_THREAD_MPI=ON \
  -DGMX_OPENMP=ON \
  -DGMX_DOUBLE=OFF \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_CYCLE_SUBCOUNTERS=OFF \
  -DBUILD_TESTING=ON \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/gromacs-gamd"

cmake --build build -j 8
ctest --test-dir build --output-on-failure -j 4
cmake --install build

source "$HOME/.local/gromacs-gamd/bin/GMXRC"
gmx --version
```

Change `-j 8` and the test parallelism to suit the available CPU and memory.
`GMX_BUILD_OWN_FFTW=ON` downloads and builds FFTW; omit that option when a
compatible single-precision FFTW installation is already available.

For a system-wide installation, configure with a dedicated prefix such as
`-DCMAKE_INSTALL_PREFIX=/usr/local/gromacs-gamd`, then run only the installation
step with elevated privileges:

```bash
sudo cmake --install build
source /usr/local/gromacs-gamd/bin/GMXRC
```

Do not configure or compile as root.

### CPU-only build

GaMD also has a CPU-reference path. On a machine without CUDA, configure a
separate build with `-DGMX_GPU=OFF` and run with CPU task assignment:

```bash
gmx mdrun -deffnm gamd \
  -nb cpu -pme cpu -bonded cpu -update cpu
```

## Quick start

Start from a conventionally minimized and equilibrated GROMACS system. GaMD
does not replace normal system preparation or force-field validation.

### 1. Enable GaMD in the MDP file

Add the following settings to the MDP file used to create the GaMD TPR:

```ini
integrator    = md
gamd          = yes
nstcalcenergy = 1
nstfout       = 0
```

Keep the remaining thermostat, constraints, cutoffs, output cadence, and other
simulation settings appropriate for the physical system.

### 2. Create `gamd.in`

`gamd.in` is read by `mdrun` from its **current working directory**. It is not
embedded into the TPR by `grompp`.

The following is an illustrative fresh dual-boost setup. Stage lengths are MD
steps, not picoseconds, and must be chosen for the system and protocol rather
than copied blindly.

```text
igamd 3
irest_gamd 0
iE 1
iEP 1
iED 1
ntcmdprep 200000
ntcmd 1000000
ntebprep 200000
nteb 1000000
ntave 50000
reweight_nst 50
para_nst 50
sigma0P 6.0
sigma0D 6.0
```

The parser is strict: duplicate keys, unknown keys, invalid values, and
unexpected trailing tokens are rejected. Blank lines and lines beginning with
`#` or `;` are allowed.

### 3. Build the TPR

```bash
gmx grompp \
  -f gamd.mdp \
  -c equilibrated.gro \
  -p topol.top \
  -o gamd.tpr
```

Use `-t` and `-n` as needed for the normal GROMACS preparation workflow.

### 4. Run the high-performance GPU path

Run from the directory containing both `gamd.tpr` and `gamd.in`:

```bash
gmx mdrun \
  -s gamd.tpr \
  -deffnm gamd \
  -ntmpi 1 \
  -ntomp 8 \
  -pin on \
  -nb gpu \
  -pme gpu \
  -bonded gpu \
  -update gpu
```

Replace `-ntomp 8` with a value suitable for the CPU paired with the GPU. No
`GMX_GAMD_GPU*` variables and no `GMX_CUDA_GRAPH` variable are needed.

Confirm the selected path in `gamd.log`:

```bash
grep -E "GaMD execution mode|CUDA Graph eligibility" gamd.log
```

A supported run reports:

```text
GaMD execution mode: GPU resident.
CUDA Graph eligibility enabled automatically for GPU-resident GaMD.
```

### 5. CPU-reference comparison

To retain GPU force offload but execute the GaMD reference/update path on the
CPU, change only the update assignment:

```bash
gmx mdrun \
  -s gamd.tpr \
  -deffnm gamd-cpu-reference \
  -ntmpi 1 \
  -ntomp 8 \
  -pin on \
  -nb gpu \
  -pme gpu \
  -bonded gpu \
  -update cpu
```

This is a mixed CPU/GPU reference run, not a CPU-only run.

## GaMD modes and stages

### Boost modes

| `igamd` | Boost mode |
| ---: | --- |
| `1` | Total-potential boost |
| `2` | Dihedral boost |
| `3` | Dual boost: total potential plus dihedral |

`iE`, `iEP`, and `iED` must each be `1` or `2`. `iEP` and `iED` select the
potential and dihedral threshold formulations; the examples retain `iE = 1`
for input and restart compatibility.

`sigma0P` and `sigma0D` are entered in kcal/mol. They are converted internally
to kJ/mol. `sigma0P` is required for modes 1 and 3; `sigma0D` is required for
modes 2 and 3.

### Adaptive stages

For `irest_gamd 0`, one simulation advances through five stages:

| Stage | Step interval | Behavior |
| ---: | --- | --- |
| 1 | `step <= ntcmdprep` | Conventional-MD preparation |
| 2 | `ntcmdprep < step <= ntcmd` | Conventional-MD statistics |
| 3 | `ntcmd < step <= ntcmd + ntebprep` | Boost preparation |
| 4 | `ntcmd + ntebprep < step <= ntcmd + nteb` | Boost statistics and adaptive parameters |
| 5 | `step > ntcmd + nteb` | Production with the trained parameters |

The planned run must extend beyond `ntcmd + nteb` to contain production
sampling. `ntcmdprep`, `ntcmd`, `ntebprep`, and `nteb` must all be multiples of
`ntave`, with:

```text
0 <= ntcmdprep < ntcmd
0 <= ntebprep  < nteb
ntave > 0
```

`reweight_nst` must be positive. `para_nst = 0` disables parameter-history
output; a positive value selects its step interval.

## NPT with C-rescale

The GPU-resident path supports only C-rescale with isotropic or semiisotropic
coupling. All other pressure-coupling algorithms and types are unsupported for
`-update gpu` GaMD.

Isotropic example:

```ini
pcoupl          = C-rescale
pcoupltype      = isotropic
nstpcouple      = 100
tau-p           = 5.0
compressibility = 4.5e-5
ref-p           = 1.0
```

Semiisotropic example:

```ini
pcoupl          = C-rescale
pcoupltype      = semiisotropic
nstpcouple      = 100
tau-p           = 5.0
compressibility = 4.5e-5 4.5e-5
ref-p           = 1.0 1.0
```

These pressure values are examples, not universal recommendations. Use values
appropriate for the simulated system. On a pressure-coupling step, the current
GaMD-corrected virial is staged for C-rescale; CUDA Graph replay continues on
eligible intervening steps.

## Restart and continuation

GaMD files use fixed names in the current working directory. Use a separate
directory for each independent simulation or replica.

### Continue an interrupted segment

Keep the matching GROMACS checkpoint and all GaMD files together, then use the
normal GROMACS continuation command in the same directory:

```bash
gmx mdrun \
  -deffnm gamd \
  -cpi gamd.cpt \
  -append \
  -ntmpi 1 \
  -ntomp 8 \
  -pin on \
  -nb gpu \
  -pme gpu \
  -bonded gpu \
  -update gpu
```

Do not manually concatenate `gamd-reweight.dat` or `gamd-para.dat`. The restart
loader reconciles those files with the saved GaMD step and prevents duplicate
rows after continuation.

### Start a new production segment from trained GaMD statistics

Use a **clean output directory** containing:

- A production TPR prepared by the normal GROMACS continuation workflow.
- The completed equilibration/adaptive `gamd-restart.dat`.
- A production `gamd.in` with `irest_gamd 1`.

The stage counters may be omitted in production-restart mode because their
provenance is loaded from the restart file. A minimal dual-boost production
input is:

```text
igamd 3
irest_gamd 1
iE 1
iEP 1
iED 1
reweight_nst 50
para_nst 50
sigma0P 6.0
sigma0D 6.0
```

The boost mode, threshold formulations, and sigma values must match the restart
state. Do not copy old `gamd-reweight.dat` or `gamd-para.dat` into a new
production directory unless they belong to the same checkpoint continuation.

### Restart files

| File | Purpose |
| --- | --- |
| `gamd-restart.dat` | Current committed GaMD statistics and parameters |
| `gamd-restart-prev.dat` | Previous committed state, used if checkpoint and GaMD replacement were interrupted between commits |
| `gamd-restart.dat.tmp` | Temporary file used during atomic replacement; it is not the primary restart input |

Keep `gamd-restart.dat` and `gamd-restart-prev.dat` with the matching `.cpt`
when moving or archiving a running simulation.

## Output files

| File | Contents |
| --- | --- |
| `gamd-reweight.dat` | Stepwise unboosted energies, force weights, and boost energies; energy columns are in kcal/mol |
| `gamd-para.dat` | GaMD stage and parameter history; energy columns are in kJ/mol |
| `gamd-restart.dat` | Restartable GaMD statistics and trained parameters |
| `gamd-restart-prev.dat` | Previous restart commit for interruption recovery |
| `gamd.log`, `gamd.edr`, `gamd.xtc`, `gamd.cpt`, etc. | Standard GROMACS output selected by `-deffnm gamd` |

The GaMD-specific filenames are fixed and are not renamed by `-deffnm`.

## Current limitations

- The GPU-resident production path is CUDA mixed precision only.
- Use exactly one PP rank (`-ntmpi 1`) and one GPU.
- Domain decomposition, separate PME ranks, MTS, and multi-GPU GaMD are not
  supported by the validated path.
- GPU-update GaMD requires `-nb gpu -pme gpu -bonded gpu -update gpu`.
- Corrected force trajectory output is unavailable, so `nstfout` must be zero.
- C-rescale isotropic and semiisotropic are the only supported NPT modes.
- The resident kinetic-history implementation supports at most two
  temperature-coupling groups.

### Scientific-definition note

In v1.1, the GaMD "dihedral" boost group contains proper periodic,
Ryckaert-Bellemans/Fourier, and CMAP energy and forces. CPU and GPU paths use
the same definition, but this is not numerically identical to an Amber-style
proper-dihedral-only partition for a topology that contains CMAP terms.

Do not reuse `gamd-restart.dat` statistics between implementations or versions
that use different energy partitions. Validate the chosen definition for the
force field and scientific protocol before production use.

## Troubleshooting

### `gamd.in` was not found

Run `mdrun` from the directory that contains `gamd.in`, or copy the file into
the intended run directory. Supplying `gamd = yes` to `grompp` does not embed
the text file in the TPR.

### A fresh run found existing GaMD files

Use a clean directory for an independent run. Existing GaMD output is accepted
only when it can be matched safely to a checkpoint continuation.

### The log reports `CPU reference`

`-update cpu` intentionally selects the CPU-reference GaMD path. Use the full
GPU command shown above to request the resident path.

### The run does not reach production

Ensure the final MD step is greater than `ntcmd + nteb`. The program emits a
warning when the planned run ends before stage 5.

### Restart saved-step mismatch

Keep the GROMACS `.cpt`, `gamd-restart.dat`, and
`gamd-restart-prev.dat` from the same run together. Do not mix restart files
from different replicas or production segments.

## Citation, license, and support

Please cite both GROMACS and the GaMD method when using this software in
research. GROMACS citation metadata is provided in [`CITATION.cff`](CITATION.cff).
The original GaMD method is described in:

- Miao, Y.; Feher, V. A.; McCammon, J. A. *Gaussian Accelerated Molecular
  Dynamics: Unconstrained Enhanced Sampling and Free Energy Calculation.*
  Journal of Chemical Theory and Computation (2015).
  [DOI: 10.1021/acs.jctc.5b00436](https://doi.org/10.1021/acs.jctc.5b00436)

This derived work is distributed under the GNU Lesser General Public License,
version 2.1 or later. See [`COPYING`](COPYING) and [`AUTHORS`](AUTHORS).

For questions or bug reports specific to this fork, open an issue at
<https://github.com/math-diff/gromacs-gamd/issues>.
