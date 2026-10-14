import tket.extensions as ext
import tket_exts

from hugr import cli
from hugr.build.tracked_dfg import TrackedDfg
from hugr.package import Package

# HUGR extensions needed for this example.
quantum = ext.quantum
measure = ext.measurement

circ = TrackedDfg()

# Allocate and entangle qubits, passing wires by name instead of index.
data = circ.add(quantum.qAlloc()).out(0)
ancilla = circ.add(quantum.qAlloc()).out(0)
ancilla = circ.add(quantum.H(ancilla)).out(0)
data, ancilla = circ.add(quantum.CX(data, ancilla))

# Measure the ancilla qubit to get a `Bool` wire that can be used in the conditional.
measurement = circ.add(quantum.measure_free(ancilla))
result = circ.add(measure.read(measurement))

# The first argument passed to the conditional builder is the `Bool` controlling branch
# choice, followed by any other values you need within this graph region.
with circ.add_if(result, data) as if_:
    # Either apply an X gate in the `True` branch...
    flipped = if_.add(quantum.X(if_.input_node[0]))
    if_.set_outputs(flipped)

with if_.add_else() as else_:
    # ... or in the `False` branch, do nothing.
    else_.set_outputs(else_.input_node[0])

# Set the output of the conditional as the output of the DFG.
circ.set_outputs(if_.conditional_node[0])

# Validation and visualization.
package = Package(
    modules=[circ.hugr], extensions=list(tket_exts.tket_registry().extensions)
)
cli.validate(package.to_bytes())

# Uncomment to generate a DOT file for visualization.
# with Path("example-control-flow1.dot").open("w") as f:
#     f.write(str(circ.hugr.render_dot()))
