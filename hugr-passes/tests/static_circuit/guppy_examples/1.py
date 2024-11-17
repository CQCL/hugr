from guppylang.decorator import guppy
from guppylang.std.quantum import discard, measure, qubit, cx

## BEGIN
@guppy
def main() -> None:
    q1,q2 = qubit(),qubit()

    cx(q1,q2)

    discard(q1)
    discard(q2)
## END

print(guppy.get_module().compile().to_executable_package().package.to_json())
