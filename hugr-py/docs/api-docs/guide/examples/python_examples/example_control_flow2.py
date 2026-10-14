import tket.extensions as ext
import tket_exts

from hugr import cli, ops, tys
from hugr.build.tracked_dfg import TrackedDfg
from hugr.package import Package

# HUGR extensions needed for this example.
quantum = ext.quantum
measure = ext.measurement

circ = TrackedDfg()

# Only allocate the data qubit outside the loop.
data = circ.add(quantum.qAlloc())

# Define a type alias for the specific `Either` type we will use to control the tail
# loop, with one variant representing the `Continue` case for retrying and the other
# the `Break` case in the case of success, each carrying a qubit.
either_ty = tys.Either([tys.Qubit], [tys.Qubit])

# The first argument to the tail loop builder is a list of wires that are only inputs;
# the second argument is a list of wires that are both inputs and outputs of the loop.
with circ.add_tail_loop([data], []) as loop:
    [loop_data] = loop.inputs()
    # Allocate a fresh ancilla in each iteration and entangle it with the data qubit.
    ancilla = loop.add(quantum.qAlloc()).out(0)
    ancilla = loop.add(quantum.H(ancilla)).out(0)
    loop_data, ancilla = loop.add(quantum.CX(loop_data, ancilla))

    measurement = loop.add(quantum.measure_free(ancilla))
    result = loop.add(measure.read(measurement))

    # Same conditional as before, but this time tag the output qubit in order to
    # use it in the loop condition which decides whether to do another iteration.
    with loop.add_if(result, loop_data) as if_:
        flipped = if_.add(quantum.X(if_.input_node[0]))
        tagged_break = if_.add(ops.Break(either_ty)(flipped))
        if_.set_outputs(tagged_break)

    with if_.add_else() as else_:
        tagged_cont = else_.add(ops.Continue(either_ty)(else_.input_node[0]))
        else_.set_outputs(tagged_cont)

    # Set the conditional output (the data qubit tagged with either `Continue` or
    # `Break`).
    loop.set_loop_outputs(if_.conditional_node[0])

# Set the final loop output as the output of the DFG.
circ.set_outputs(*loop.outputs())

# Validation and visualization.
package = Package(
    modules=[circ.hugr], extensions=list(tket_exts.tket_registry().extensions)
)
cli.validate(package.to_bytes())

# Uncomment to generate a DOT file for visualization.
# with Path("example-control-flow2.dot").open("w") as f:
#     f.write(str(circ.hugr.render_dot()))
