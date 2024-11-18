from guppylang.decorator import guppy
from guppylang.std.quantum import discard, measure, qubit, cx

# The conditional dependent on unknown input obstructs a static circuit here.
# The qubit histories of q1 and q2 at the cx are both ambiguous. Ambiguity in
# either is enough to obstruct the extraction of a static circuit.

## BEGIN
# The conditional dependent on unknown input obstructs a static circuit here.
# The qubit histories of q1 and q2 at the cx are both ambiguous. Ambiguity in
# either is enough to obstruct the extraction of a static circuit.
@guppy
def main(random: bool) -> None:
    q1,q2 = qubit(),qubit()

    if random:
        q1,q2 = q2,q1

    cx(q1,q2)

    discard(q1)
    discard(q2)
## END

print(guppy.get_module().compile().to_executable_package().package.to_json())
