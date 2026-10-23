import tket.extensions as ext
import tket_exts

from hugr import cli, ops, tys
from hugr.build.tracked_dfg import TrackedDfg
from hugr.package import Package
from hugr.std.collections.array import EXTENSION as ARRAY_EXTENSION
from hugr.std.collections.array import Array

quantum = ext.quantum
measure = ext.measurement


# Create wrapper functions around the array extension which instantiate the array
# operations we require given the array element type and length.
def new_array_op(elem_ty: tys.Type, length: int) -> ops.ExtOp:
    # An argument that can instantiate a type parameter indicating a natural number.
    length_arg = tys.BoundedNatArg(length)
    # An argument that can instantiate a type parameter indicating a type.
    elem_arg = tys.TypeTypeArg(elem_ty)
    # There is an existing class wrapper around the array type.
    arr_ty = Array(elem_ty, length)
    return ARRAY_EXTENSION.get_op("new_array").instantiate(
        [length_arg, elem_arg], tys.FunctionType([elem_ty] * length, [arr_ty])
    )


# The `scan` op allows us to map a function over an array.
# Let it take either an `int` or type argument as we will need both in the code below.
def array_scan_op(
    elem_ty: tys.Type, new_elem_ty: tys.Type, length: int | tys.TypeArg
) -> ops.ExtOp:
    length_arg = tys.BoundedNatArg(length) if isinstance(length, int) else length
    ty_args = [
        length_arg,
        tys.TypeTypeArg(elem_ty),
        tys.TypeTypeArg(new_elem_ty),
        tys.ListArg([]),  # We ignore the accumulators.
    ]
    ins: list[tys.Type] = [
        Array(elem_ty, length_arg),
        tys.FunctionType([elem_ty], [new_elem_ty]),
    ]
    outs: list[tys.Type] = [Array(new_elem_ty, length_arg)]
    return ARRAY_EXTENSION.get_op("scan").instantiate(
        ty_args, tys.FunctionType(ins, outs)
    )


# The `unpack` op allows us to split an array into individual elements.
def array_unpack_op(elem_ty: tys.Type, length: int) -> ops.ExtOp:
    length_arg = tys.BoundedNatArg(length)
    elem_arg = tys.TypeTypeArg(elem_ty)
    arr_ty = Array(elem_ty, length)
    return ARRAY_EXTENSION.get_op("unpack").instantiate(
        [length_arg, elem_arg], tys.FunctionType([arr_ty], [elem_ty] * length)
    )


circ = TrackedDfg()

module = circ.module_root_builder()

with module.define_function("prepare", [tys.Qubit, tys.Qubit]) as prepare:
    p_data, p_ancilla = prepare.inputs()
    p_ancilla = prepare.add(quantum.H(p_ancilla)).out(0)
    p_data, p_ancilla = prepare.add(quantum.CX(p_data, p_ancilla))
    prepare.set_outputs(p_data, p_ancilla)

with module.define_function("correct", [tys.Qubit]) as correct:
    (c_data,) = correct.inputs()
    c_data = correct.add(quantum.X(c_data)).out(0)
    correct.set_outputs(c_data)

n_param = tys.BoundedNatParam()

# Define a polymorphic function by passing a type parameter and referring to it with a
# variable argument inside of the array type (using de Bruijn indices).
with module.define_function(
    "correct_all",
    [Array(tys.Qubit, tys.VariableArg(0, n_param))],
    type_params=[n_param],
) as correct_all:
    (qs,) = correct_all.inputs()
    correct_fn = correct_all.load_function(correct.parent_node)
    qs = correct_all.add_op(
        array_scan_op(tys.Qubit, tys.Qubit, tys.VariableArg(0, n_param)), qs, correct_fn
    ).out(0)
    correct_all.set_outputs(qs)


data = circ.add(quantum.qAlloc())
either_ty = tys.Either([tys.Qubit], [tys.Qubit])

with circ.add_tail_loop([data], []) as loop:
    [loop_data] = loop.inputs()
    ancilla = loop.add(quantum.qAlloc()).out(0)

    loop_data, ancilla = loop.call(prepare.parent_node, loop_data, ancilla)

    measurement = loop.add(quantum.measure_free(ancilla))
    result = loop.add(measure.read(measurement))

    with loop.add_if(result, loop_data) as if_:
        one_qubit_arr_ty = Array(tys.Qubit, 1)

        # Using the functions we defined above, create a new array containing the
        # single qubit from the loop input, pass it to the `correct_all` function,
        # and then unpack the resulting array to get the corrected qubit.
        arr = if_.add_op(new_array_op(tys.Qubit, 1), if_.input_node[0])
        arr = if_.call(
            correct_all.parent_node,
            arr,
            instantiation=tys.FunctionType([one_qubit_arr_ty], [one_qubit_arr_ty]),
            type_args=[tys.BoundedNatArg(1)],
        )
        (flipped,) = if_.add_op(array_unpack_op(tys.Qubit, 1), arr)

        tagged_break = if_.add(ops.Break(either_ty)(flipped))
        if_.set_outputs(tagged_break)

    with if_.add_else() as else_:
        tagged_cont = else_.add(ops.Continue(either_ty)(else_.input_node[0]))
        else_.set_outputs(tagged_cont)

    loop.set_loop_outputs(if_.conditional_node[0])

circ.set_outputs(*loop.outputs())

# Validation and visualization.
package = Package(
    modules=[circ.hugr], extensions=list(tket_exts.tket_registry().extensions)
)
cli.validate(package.to_bytes())

# Uncomment to generate a DOT file for visualization.
# with Path("example-functions2.dot").open("w") as f:
#     f.write(str(circ.hugr.render_dot()))
