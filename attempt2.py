from functools import partial

import numpy as np
from beartype.typing import NamedTuple
from typing_extensions import Callable


class BanditEnvironment:
    """
        This game tree looks like this:

            0
           / \\
           1   2
          / \\ / \\
         3   4 5  6
    """

    def __init__(self):
        self.tree = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: []}
        self.current_state = np.array(0)

    def reset(self):
        self.current_state = np.array(0)
        return self.current_state

    def set_state(self, state):
        assert state in [0, 1, 2, 3, 4, 5, 6]
        self.current_state = state

    def step(self, action):
        if self.current_state in [3, 4, 5, 6]:
            return self.current_state, 0, True

        if action < 0 or action >= len(self.tree[int(self.current_state)]):
            raise ValueError("Invalid action")

        self.current_state = self.tree[int(self.current_state)][action]

        done = self.current_state in [3, 4, 5, 6]
        reward = 1 if self.current_state == 6 else 0

        return self.current_state, reward, done

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


class SimulationSearchState(NamedTuple):
    node: int
    next_node: int
    action: int
    depth: int
    continue_: bool


ROOT = 0
UNVISITED_NODE = -1
NO_PARENT = -1


class RootFnOutput(NamedTuple):
    root_state: np.ndarray


class Tree:
    def __init__(self, n_actions: int, n_nodes: int, root_fn_output: RootFnOutput):
        self.n_actions = n_actions
        self.n_nodes = n_nodes

        self.nodes = np.full(shape=(self.n_nodes,), fill_value=UNVISITED_NODE)
        self.nodes[0] = ROOT

        self.node_visits = np.zeros(shape=(self.n_nodes))
        self.node_values = np.zeros(shape=(self.n_nodes))

        self.children = np.full(
            shape=(self.n_nodes, self.n_actions), fill_value=UNVISITED_NODE
        )

        self.states = {ROOT: root_fn_output.root_state}
        self.parents = np.full(shape=(self.n_nodes), fill_value=NO_PARENT)
        self.action_from_parent = np.full(shape=(self.n_nodes), fill_value=NO_PARENT)
        self.children_rewards = np.zeros(shape=(self.n_nodes, self.n_actions))
        self.children_discounts = np.zeros(shape=(self.n_nodes, self.n_actions))
        self.children_visits = np.zeros(shape=(self.n_nodes, self.n_actions))
        self.children_values = np.zeros(shape=(self.n_nodes, self.n_actions))


def simulate(
    tree: Tree, max_depth: int, inner_simulation_fn: Callable
) -> SimulationSearchState:
    def _simulate(state: SimulationSearchState):
        current_node = int(state.next_node)
        action = inner_simulation_fn(tree, current_node, state.depth)
        # for debugging
        children_value = tree.children_values[current_node, action]
        next_node = tree.children[current_node, action]
        print(
            f"SIMULATION: {current_node=}, {action=}, {next_node=}, {children_value=}"
        )
        should_continue = state.depth + 1 < max_depth and next_node != UNVISITED_NODE

        return SimulationSearchState(
            node=current_node,
            next_node=next_node,
            action=action,
            depth=state.depth + 1,
            continue_=should_continue,
        )

    state = SimulationSearchState(
        node=NO_PARENT, next_node=ROOT, action=NO_PARENT, depth=0, continue_=True
    )

    while state.continue_:
        state = _simulate(state)

    return state


class StepFnReturn(NamedTuple):
    value: float
    discount: float
    reward: float
    state: np.ndarray


def expand(
    tree: Tree,
    parent_node: int,
    action: int,
    recurrent_step_fn: Callable[[int, np.ndarray], StepFnReturn],
):
    if tree.children[parent_node][action] == UNVISITED_NODE:
        next_node_id = np.max(tree.nodes) + 1
    else:
        next_node_id = tree.children[parent_node][action]

    tree.nodes[next_node_id] = next_node_id
    tree.children[parent_node][action] = next_node_id

    state = tree.states[parent_node]

    value, discount, reward, next_state = recurrent_step_fn(action, state)
    tree.states[next_node_id] = next_state
    tree.node_values[next_node_id] = value
    tree.node_visits[next_node_id] += 1
    tree.children_rewards[parent_node, action] = reward
    tree.children_discounts[parent_node, action] = discount

    tree.parents[next_node_id] = parent_node
    tree.action_from_parent[next_node_id] = action
    print(f"EXPAND {next_node_id=}")
    return tree, next_node_id


n_actions = 2
n_simulations = 30


tree = Tree(
    n_actions, n_simulations, RootFnOutput(root_state=BanditEnvironment().reset())
)
max_depth = 2


def inner_simulation_fn(tree: Tree, node: int, depth: int):
    children = tree.children[node]
    parent = node

    best_action = NO_PARENT
    best_ucb = float("-inf")
    for action in range(tree.n_actions):
        child_index = children[action]
        if child_index == UNVISITED_NODE:
            return action
        else:
            ucb = ucb1(
                avg_node_value=tree.node_values[child_index],
                visits_parent=tree.node_visits[parent],
                visits_node=tree.node_visits[child_index],
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
    return best_action


def stepper(action: int, state: np.ndarray, env: BanditEnvironment) -> StepFnReturn:
    env.set_state(state)
    discount = 0.8
    next_state, reward, done = env.step(action)
    value = env.get_future_value(next_state)
    return StepFnReturn(
        value=value, state=np.array(next_state), discount=discount, reward=reward
    )


step_fn_partial = partial(stepper, env=BanditEnvironment())


class BackpropagationLoopState(NamedTuple):
    value: float
    node_index: int


def backpropagate(tree: Tree, leaf_node: int) -> Tree:
    def _backpropagate(state: BackpropagationLoopState) -> BackpropagationLoopState:
        print(f"BACKPROPAGATION {state.node_index}")
        parent = tree.parents[state.node_index]
        parent_visits = tree.node_visits[parent]
        action = tree.action_from_parent[state.node_index]
        reward = tree.children_rewards[parent, action]
        discount = tree.children_discounts[parent, action]
        leaf_value = reward + discount * state.value
        parent_value = (tree.node_values[parent] * parent_visits + leaf_value) / (
            parent_visits + 1.0
        )
        tree.node_values[parent] = parent_value
        tree.node_visits[parent] += 1
        tree.children_visits[parent, action] += 1
        tree.children_values[parent, action] = tree.node_values[state.node_index]

        return BackpropagationLoopState(node_index=parent, value=float(leaf_value))

    state = BackpropagationLoopState(
        node_index=leaf_node, value=tree.node_values[leaf_node]
    )
    while state.node_index != ROOT:
        state = _backpropagate(state)

    print(f"BACKPROPAGATION {state.node_index}")

    return tree


tree = Tree(
    n_actions, n_simulations, RootFnOutput(root_state=BanditEnvironment().reset())
)


def search(tree: Tree, n_iterations: int):
    for iteration in range(n_iterations):
        print(f"{iteration=}")
        out = simulate(tree, max_depth, inner_simulation_fn)
        tree, leaf_node = expand(
            tree, out.node, action=out.action, recurrent_step_fn=step_fn_partial
        )

        tree = backpropagate(tree, leaf_node)


def main():
    search(tree, n_iterations=30)


if __name__ == "__main__":
    main()
