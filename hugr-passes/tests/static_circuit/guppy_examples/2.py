from guppylang.decorator import guppy
from guppylang.std.quantum import discard, measure, qubit, cx


@guppy
def main(random: bool) -> None:
    q1,q2 = qubit(),qubit()

    if random:
        q1,q2 = q2,q1

    cx(q1,q2)

    discard(q1)
    discard(q2)

print(guppy.get_module().compile().to_executable_package().package.to_json())
