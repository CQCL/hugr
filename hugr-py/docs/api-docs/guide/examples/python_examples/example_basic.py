import tket.extensions as ext
import tket_exts

from hugr import cli
from hugr.build.tracked_dfg import TrackedDfg
from hugr.package import Package

# HUGR extensions needed for this example.
quantum = ext.quantum
measure = ext.measurement

# Initialize a DFG (dataflow graph) that is able to track wires by indices.
circ = TrackedDfg()

# Add operations that allocate qubits and track the resulting wires.
q0 = circ.add(quantum.qAlloc()).out(0)
q1 = circ.add(quantum.qAlloc()).out(0)
circ.track_wires([q0, q1])

# Apply quantum operations on the tracked qubit wires.
circ.add(quantum.H(0))
circ.add(quantum.CX(0, 1))
# Use `extend` to add multiple ops at the same time.
circ.extend(quantum.measure_free(0), measure.read(0))

# Connect all tracked wires to the output of the DFG.
circ.set_tracked_outputs()

# Validation and visualization.
package = Package(
    modules=[circ.hugr], extensions=list(tket_exts.tket_registry().extensions)
)
cli.validate(package.to_bytes())

# Uncomment to generate a DOT file for visualization.
# with Path("example-basic.dot").open("w") as f:
#     f.write(str(circ.hugr.render_dot()))
