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
