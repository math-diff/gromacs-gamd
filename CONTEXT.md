# Gaussian accelerated molecular dynamics

Domain language for GaMD biasing, diagnostic comparisons, and simulation histories.

## Language

**Bias**:
An added potential used to modify the sampled energy landscape. Its associated force weights determine how the physical force contributions are combined.

**Force override**:
A fixed set of diagnostic force weights and bias energies used only in a GaMD-enabled simulation. Each weight is applied once; an ordinary simulation is unaffected.
_Avoid_: Production mode, additional force scaling

**Deferred sample batch**:
Energy, force virial, constraint virial, and kinetic records for the same ordered simulation steps, held until their next output consumer.
_Avoid_: Independent histories when referring to the complete set

**GaMD stage**:
Conventional preparation, conventional statistics, boost preparation, boost statistics, or production. `GaMDStage` names these states in code; the text protocol retains values 1 through 5, with 0 reserved for an inactive plan.

**Device bias state**:
The fixed six-float layout containing potential/dihedral energies, their force weights, and their boosts. Named indices describe the device layout; host diagnostics receive a `GaMDStepResult` copied from that state.
_Avoid_: Energy shadow when referring to the production device buffer

**Dihedral force snapshot**:
One selected step's dihedral contribution to the corrected total force: `scaleP * scaleD * F_dihedral`. The file uses zero-based topology atom indices on CPU and GPU. Raw CPU evaluation traces are a separate record and may contain repeated evaluations.
