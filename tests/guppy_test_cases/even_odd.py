from guppylang.decorator import guppy
from guppylang.module import GuppyModule

main = GuppyModule("main")


@guppy(main)
def is_even(x: int) -> bool:
    if x == 0:
        return True
    return is_odd(x - 1)


@guppy(main)
def is_odd(x: int) -> bool:
    if x == 0:
        return False
    return is_even(x - 1)

if __name__ == "__main__":
    print(main.compile().serialize())
