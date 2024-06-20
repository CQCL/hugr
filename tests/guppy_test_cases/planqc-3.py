from guppylang.decorator import guppy
from guppylang.module import GuppyModule
from guppylang.prelude import quantum
from guppylang.prelude.quantum import measure, qubit, cx, h, z, x, t, tdg, discard

mod = GuppyModule("main")
mod.load(quantum)

@guppy(mod)
def rus(q: qubit, tries: int) -> qubit:
  i = 0;
  while i < tries:
    # Prepare ancillary qubits
    a, b = h(qubit()), h(qubit())

    b, a = cx(b, tdg(a))
    if not measure(t(a)):
      # First part failed; try again
      discard(b)
      continue

    q, b = cx(z(t(q)), b)
    if measure(t(b)):
      # Success, we are done
      break

    # Otherwise, apply correction
    q = x(q)
    i = i + 1

  return q

@guppy(mod)
def main() -> bool:
    q = qubit() # todo initialise into an interesting state
    return measure(rus(q,100))

if __name__ == "__main__":
    print(mod.compile().serialize())
