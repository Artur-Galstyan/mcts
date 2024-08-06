import numpy as np
from beartype.typing import NamedTuple
from typing_extensions import Callable


class BanditEnvironment:
    def __init__(self):
        self.tree = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: []}
        self.current_state = 0

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        if self.current_state in [3, 4, 5, 6]:
            return self.current_state, 0, True, {}

        if action < 0 or action >= len(self.tree[self.current_state]):
            raise ValueError("Invalid action")

        self.current_state = self.tree[self.current_state][action]

        done = self.current_state in [3, 4, 5, 6]
        reward = 1 if self.current_state == 6 else 0

        return self.current_state, reward, done, {}

    def render(self):
        print(f"Current state: {self.current_state}")

    @staticmethod
    def get_future_value(state):
        if state == 2:
            return 0.5
        elif state == 6:
            return 1
        else:
            return 0


def ucb1(avg_node_value, visits_parent, visits_node, exploration_exploitation_factor=2):
    return avg_node_value + exploration_exploitation_factor * np.sqrt(
        np.log(visits_parent) / visits_node
    )


ROOT = 0
UNVISITED_NODE = -1
NO_PARENT = -1


class Tree:
    def __init__(self, n_actions: int, n_nodes: int):
        self.n_actions = n_actions
        self.n_nodes = n_nodes

        self.nodes = np.full(shape=(self.n_nodes,), fill_value=UNVISITED_NODE)
        self.nodes[0] = ROOT

        self.node_visits = np.zeros(shape=(self.n_nodes))
        self.node_values = np.zeros(shape=(self.n_nodes))

        self.children = np.full(
            shape=(self.n_nodes, self.n_actions), fill_value=UNVISITED_NODE
        )


def simulate(tree: Tree, max_depth: int, inner_simulation_fn: Callable):
    class SearchState(NamedTuple):
        node: int
        next_node: int
        action: int
        depth: int
        continue_: bool

    def _simulate(state: SearchState):
        current_node = state.next_node
        action = inner_simulation_fn(tree, current_node, state.depth)
        next_node = tree.children[current_node, action]
        should_continue = state.depth + 1 < max_depth and next_node == UNVISITED_NODE

        return SearchState(
            node=current_node,
            next_node=next_node,
            action=action,
            depth=state.depth + 1,
            continue_=should_continue,
        )

    state = SearchState(
        node=NO_PARENT, next_node=ROOT, action=NO_PARENT, depth=0, continue_=True
    )

    while state.continue_:
        state = _simulate(state)


def expand():
    pass
