from guppylang.decorator import guppy
from guppylang.module import GuppyModule
from guppylang.prelude import quantum
from guppylang.prelude.quantum import measure, phased_x, qubit

mod = GuppyModule("main")
mod.load(quantum)

@guppy(mod)
def is_even(x: int) -> bool:
    q = qubit()
    return measure(h(q))


@guppy(mod)
def is_odd(x: int) -> bool:
    if x == 0:
        return False
    return is_even(x - 1)

if __name__ == "__main__":
    print(mod.compile().serialize())
