from functools import partial
from typing import Callable, NamedTuple

import numpy as np
from beartype.typing import Any

from attempt2 import BanditEnvironment

UNVISITED_NODE = -1
ROOT_INDEX = 0


def ucb1(avg_node_value, visits_parent, visits_node, exploration_exploitation_factor=2):
    return avg_node_value + exploration_exploitation_factor * np.sqrt(
        np.log(visits_parent) / visits_node
    )


class Node:
    index: int
    value: float
    visits: int

    reward: float
    discount: float
    parent: "Node | None"
    children: dict[int, "Node"]

    embedding: Any

    def __init__(self, parent: "Node | None", index: int = -1) -> None:
        self.index = index
        self.parent = parent
        self.value, self.visits, self.reward, self.discount = 0, 0, 0, 0
        self.children = dict()

    def get_child(self, action: int) -> "Node | None":
        if action not in self.children:
            return None
        else:
            return self.children[action]

    def action_from_parent(self) -> int | None:
        parent = self.parent

        if parent is None:
            return None
        else:
            for action in parent.children:
                if parent.children[action].index == self.index:
                    return action
        return None

    def __repr__(self) -> str:
        return f"[Index: {self.index}, Parent: {self.parent.index if self.parent is not None else None}, Value: {np.round(self.value, 2)}, Visits: {self.visits}, Reward: {np.round(self.reward, 2)}]"


class Tree:
    root_node: Node

    n_actions: int

    nodes: list[Node]

    def __init__(self, root_node: Node, n_actions: int) -> None:
        self.root_node = root_node
        self.n_actions = n_actions
        self.nodes = [self.root_node]

    def get_max_node_index(self):
        return max(self.nodes, key=lambda x: x.index).index


def inner_simulation_fn(tree: Tree, node: Node, depth: int):
    best_action = -1
    best_ucb = float("-inf")
    for action in range(tree.n_actions):
        child = node.get_child(action)
        if not child:
            return action
        else:
            ucb = ucb1(
                avg_node_value=child.value,
                visits_parent=node.visits,
                visits_node=child.visits,
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
    return best_action


class StepFnReturn(NamedTuple):
    value: float
    discount: float
    reward: float
    state: np.ndarray


class SimulationSearchState(NamedTuple):
    node: Node
    next_node: Node | None
    action: int
    depth: int
    proceed: bool


def simulate(
    tree: Tree, max_depth: int, inner_simulation_fn: Callable
) -> SimulationSearchState:
    def _simulate(state: SimulationSearchState) -> SimulationSearchState:
        current_node = state.next_node
        assert current_node is not None

        action = inner_simulation_fn(tree, current_node, state.depth)
        next_node = current_node.get_child(action)
        print(f"Simulation: {current_node=}, {action=}, {next_node=}")
        proceed = state.depth + 1 < max_depth and next_node is not None
        return SimulationSearchState(
            node=current_node,
            next_node=next_node,
            action=action,
            depth=state.depth + 1,
            proceed=proceed,
        )

    state = SimulationSearchState(
        tree.root_node, next_node=tree.root_node, action=0, depth=0, proceed=True
    )

    while state.proceed:
        state = _simulate(state)
    return state


def expand(tree: Tree, node: Node, action: int, recurrent_step_fn: Callable):
    leaf_node = node.get_child(action)
    if leaf_node is None:
        leaf_node = Node(parent=node, index=tree.get_max_node_index() + 1)
        node.children[action] = leaf_node

    embedding = node.embedding
    value, discount, reward, next_state = recurrent_step_fn(action, embedding)

    leaf_node.embedding = next_state
    leaf_node.value = value
    leaf_node.visits += 1
    leaf_node.reward = reward
    leaf_node.discount = discount

    print(f"Expanded new leaf_node: {leaf_node}")

    tree.nodes.append(leaf_node)
    return leaf_node


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
    node: Node


def backpropagate(leaf_node: Node):
    def _backpropagate(state: BackpropagationLoopState) -> BackpropagationLoopState:
        parent = state.node.parent
        assert parent is not None

        parent_visits = parent.visits
        reward = state.node.reward
        discount = state.node.discount

        leaf_value = reward + discount * state.value
        parent_value = (parent.value * parent_visits + leaf_value) / (
            parent_visits + 1.0
        )

        parent.value = parent_value
        parent.visits += 1

        return BackpropagationLoopState(node=parent, value=float(leaf_value))

    state = BackpropagationLoopState(value=leaf_node.value, node=leaf_node)

    while state.node.parent is not None:
        state = _backpropagate(state)

    return state


def main():
    env = BanditEnvironment()
    root_node = Node(parent=None, index=ROOT_INDEX)
    root_node.embedding = env.reset()

    max_depth = 2
    n_actions = 2

    n_iterations = 10000

    tree = Tree(root_node=root_node, n_actions=n_actions)

    for i in range(n_iterations):
        sim_out = simulate(
            tree, max_depth=max_depth, inner_simulation_fn=inner_simulation_fn
        )
        leaf_node = expand(tree, sim_out.node, sim_out.action, step_fn_partial)
        backpropagate(leaf_node)

    node = tree.root_node
    print(f"{node=}")
    while True:
        best_action = max(node.children, key=lambda x: node.children[x].value)
        print(f"Best action: {best_action}")
        obs, reward, done = env.step(best_action)
        print(f"Observation: {obs}, Reward: {reward}")
        node = node.children[best_action]
        print(f"{node=}")
        if done:
            break


if __name__ == "__main__":
    main()
