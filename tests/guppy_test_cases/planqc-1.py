from guppylang.decorator import guppy
from guppylang.module import GuppyModule
from guppylang.prelude import quantum
from guppylang.prelude.quantum import measure, qubit, h, rz

mod = GuppyModule("main")
mod.load(quantum)

@guppy(mod)
def rx(q: qubit, x: float) -> qubit:
  # Implement Rx via Rz rotation
  return h(rz(h(q), x))


@guppy(mod)
def main() -> bool:
  q = qubit()
  z = rx(q,1.5)
  return measure(z)

if __name__ == "__main__":
    print(mod.compile().serialize())

