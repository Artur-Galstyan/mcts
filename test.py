import numpy as np
from beartype.typing import NamedTuple


class Tree:
    nums: np.ndarray

    def __init__(self, nums):
        self.nums = nums


class State(NamedTuple):
    tree: Tree


t = Tree(np.zeros(shape=(5, 1)))


def bp(tree: Tree) -> None:
    state = State(tree)

    state.tree.nums[0] = 1000


bp(t)

print(t.nums)
