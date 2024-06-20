from guppylang.decorator import guppy
from guppylang.module import GuppyModule
from guppylang.prelude import quantum
from guppylang.prelude.quantum import measure, qubit, cx, h, z, x

mod = GuppyModule("main")
mod.load(quantum)

@guppy(mod)
def teleport(
  src: qubit, tgt: qubit
) -> qubit:
  # Entangle qubits with ancilla
  tmp, tgt = cx(h(qubit()), tgt)
  src, tmp = cx(src, tmp)
  # Apply classical corrections
  if measure(h(src)):
    tgt = z(tgt)
  if measure(tmp):
    tgt = x(tgt)
  return tgt

@guppy(mod)
def main() -> bool:
  q1,q2 = qubit(), qubit() # TODO initialise into some interesting state
  return measure(teleport(q1,q2))

if __name__ == "__main__":
    print(mod.compile().serialize())
