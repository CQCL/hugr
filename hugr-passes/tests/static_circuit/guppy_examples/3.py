from guppylang.decorator import guppy
from guppylang.std.quantum import discard, measure, qubit, cx, angle,rx

## BEGIN
# This currently doesn't work, for reasons I don't yet understand.  I expect to
# be able to infer coherent histories here
@guppy
def main(random: bool) -> None:
    q1,q2 = qubit(),qubit()

    cx(q1,q2)

    if random:
        rx(q1, angle(1.0))
    else:
        rx(q1, angle(0.5))

    discard(q1)
    discard(q2)
## END

print(guppy.get_module().compile().to_executable_package().package.to_json())
