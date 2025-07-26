from dataclasses import dataclass

from beartype.typing import Any, Callable, NamedTuple

NO_PARENT = -1
UNVISITED = -1
ROOT_INDEX = 0


@dataclass
class Tree:
    parent_indices: list[int]
    children_indices: list[list[int]]
    action_from_parent: list[int]

    n_s: list[int]
    n_sa: list[list[int]]

    v_s: list[float]
    q_sa: list[list[float]]
    r_sa: list[list[float]]

    dones: list[bool]
    states: dict[int, Any]


class RootFnOutput(NamedTuple):
    state: Any


class PolicyInput(NamedTuple):
    tree: Tree
    node_index: int
    depth: int


class PolicyReturn(NamedTuple):
    action: int


class SelectionOutput(NamedTuple):
    parent_index: int
    action: int


class StepFnInput(NamedTuple):
    state: Any
    action: int


class StepFnReturn(NamedTuple):
    value: float
    reward: float
    done: bool
    state: Any


class LeafNode(NamedTuple):
    node_index: int
    action: int


class BackpropagationState(NamedTuple):
    tree: Tree
    idx: int
    value: float


class SelectionState(NamedTuple):
    node_index: int
    next_node_index: int
    depth: int
    action: int
    proceed: bool


def generate_tree(n_nodes: int, n_actions: int, root_fn_output: RootFnOutput) -> Tree:
    parent_indices = [NO_PARENT for _ in range(n_nodes)]
    action_from_parent = [NO_PARENT for _ in range(n_nodes)]
    children_indices = [[UNVISITED for __ in range(n_actions)] for _ in range(n_nodes)]

    n_s = [0 for _ in range(n_nodes)]
    v_s = [0.0 for _ in range(n_nodes)]

    q_sa = [[0.0 for __ in range(n_actions)] for _ in range(n_nodes)]
    n_sa = [[0 for __ in range(n_actions)] for _ in range(n_nodes)]
    r_sa = [[0.0 for __ in range(n_actions)] for _ in range(n_nodes)]
    dones = [False for _ in range(n_nodes)]

    states = {ROOT_INDEX: root_fn_output.state}

    return Tree(
        parent_indices=parent_indices,
        children_indices=children_indices,
        action_from_parent=action_from_parent,
        n_s=n_s,
        v_s=v_s,
        dones=dones,
        q_sa=q_sa,
        n_sa=n_sa,
        r_sa=r_sa,
        states=states,
    )


def selection(
    tree: Tree,
    max_depth: int,
    policy_fn: Callable[[PolicyInput], PolicyReturn],
) -> SelectionOutput:
    def _selection(state: SelectionState) -> SelectionState:
        node_index = state.next_node_index
        if tree.dones[node_index]:
            return SelectionState(
                node_index=state.node_index,
                next_node_index=node_index,
                depth=state.depth,
                action=state.action,
                proceed=False,
            )

        action_selection_output = policy_fn(PolicyInput(tree, node_index, state.depth))
        next_node_index = tree.children_indices[node_index][
            action_selection_output.action
        ]
        child_exists = next_node_index != UNVISITED
        max_depth_not_exceeded = state.depth + 1 < max_depth
        proceed = child_exists and max_depth_not_exceeded

        return SelectionState(
            node_index,
            next_node_index,
            state.depth + 1,
            action_selection_output.action,
            proceed,
        )

    state = SelectionState(
        node_index=NO_PARENT,
        next_node_index=ROOT_INDEX,
        depth=0,
        action=UNVISITED,
        proceed=True,
    )

    while state.proceed:
        state = _selection(state)

    return SelectionOutput(state.node_index, state.action)


def expansion(
    tree: Tree,
    selection_output: SelectionOutput,
    next_node_index: int,
    step_fn: Callable[[StepFnInput], StepFnReturn],
) -> LeafNode:
    parent_index, action = selection_output
    assert tree.children_indices[parent_index][action] == UNVISITED, (
        f"Can only expand unvisited nodes, got {tree.children_indices[parent_index][action]=}"
    )
    state = tree.states[parent_index]
    value, reward, done, next_state = step_fn(StepFnInput(state=state, action=action))
    tree.children_indices[parent_index][action] = next_node_index
    tree.action_from_parent[next_node_index] = action
    tree.parent_indices[next_node_index] = parent_index
    tree.v_s[next_node_index] = value
    tree.n_s[next_node_index] = 1
    tree.dones[next_node_index] = done
    tree.r_sa[parent_index][action] = reward
    tree.states[next_node_index] = next_state

    return LeafNode(
        node_index=next_node_index,
        action=action,
    )


def backpropagate(tree: Tree, leaf_index: int) -> Tree:
    def _backpropagate(state: BackpropagationState) -> BackpropagationState:
        tree, idx, value = state
        parent = tree.parent_indices[idx]
        action = tree.action_from_parent[idx]

        reward = tree.r_sa[parent][action]

        parent_value = tree.v_s[parent]
        parent_visits = tree.n_s[parent]

        leaf_value = reward + state.value
        parent_value = (parent_value * parent_visits + leaf_value) / (
            parent_visits + 1.0
        )

        tree.v_s[parent] = parent_value
        tree.n_s[parent] = parent_visits + 1

        tree.q_sa[parent][action] = tree.v_s[idx]
        tree.n_sa[parent][action] = tree.n_sa[parent][action] + 1

        next_state = BackpropagationState(idx=parent, value=leaf_value, tree=tree)

        return next_state

    state = BackpropagationState(idx=leaf_index, value=tree.v_s[leaf_index], tree=tree)

    while state.idx != ROOT_INDEX:
        state = _backpropagate(state)

    return state.tree


class MCTS:
    @staticmethod
    def search(
        n_actions: int,
        root_fn: Callable[[], RootFnOutput],
        policy_fn: Callable[[PolicyInput], PolicyReturn],
        step_fn: Callable[[StepFnInput], StepFnReturn],
        max_depth: int,
        n_iterations: int,
    ):
        node_index_counter = 0
        tree = generate_tree(
            n_nodes=n_iterations + 1, n_actions=n_actions, root_fn_output=root_fn()
        )

        for iteration in range(n_iterations):
            selection_output = selection(tree, max_depth, policy_fn)

            if (
                tree.children_indices[selection_output.parent_index][
                    selection_output.action
                ]
                == UNVISITED
            ):
                node_index_counter += 1
                leaf_node = expansion(
                    tree, selection_output, node_index_counter, step_fn
                )
            else:
                child_idx = tree.children_indices[selection_output.parent_index][
                    selection_output.action
                ]
                leaf_node = LeafNode(
                    node_index=child_idx,
                    action=selection_output.action,
                )

            tree = backpropagate(tree, leaf_node.node_index)

        return tree


import math
import random

import gymnasium as gym


def ucb1_fn_factory(
    exploration_constant: float,
) -> Callable[[PolicyInput], PolicyReturn]:
    def ucb1_action_selection_fn(
        policy_input: PolicyInput,
    ) -> PolicyReturn:
        tree = policy_input.tree
        node_index = policy_input.node_index
        n_actions = len(tree.children_indices[node_index])

        parent_visits = tree.n_s[node_index]
        if parent_visits == 0:
            return PolicyReturn(action=random.randint(0, n_actions - 1))

        best_action = -1
        max_ucb_score = -float("inf")

        for action in range(n_actions):
            child_visits = tree.n_sa[node_index][action]

            if child_visits == 0:
                ucb_score = float("inf")
            else:
                exploitation_score = tree.q_sa[node_index][action]
                exploration_score = exploration_constant * math.sqrt(
                    math.log(parent_visits) / child_visits
                )
                ucb_score = exploitation_score + exploration_score

            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_action = action

        return PolicyReturn(action=best_action)

    return ucb1_action_selection_fn


def root_fn_factory(env_name: str) -> Callable[[], RootFnOutput]:
    def root_fn() -> RootFnOutput:
        env = gym.make(env_name)
        initial_state, info = env.reset()
        env.close()
        return RootFnOutput(state=initial_state)

    return root_fn


def step_fn_factory(env_name: str) -> Callable[[StepFnInput], StepFnReturn]:
    env = gym.make(env_name)
    goal_pos = (3, 3)

    def get_pos(state: int) -> tuple[int, int]:
        return (state // 4, state % 4)

    def estimate_value(state: int) -> float:
        state_pos = get_pos(state)
        distance = abs(state_pos[0] - goal_pos[0]) + abs(state_pos[1] - goal_pos[1])
        return 1.0 / (1.0 + distance)

    def step_fn(step_input: StepFnInput) -> StepFnReturn:
        env.reset()
        env.unwrapped.s = step_input.state
        next_state, reward, done, truncated, info = env.step(step_input.action)
        is_terminal = done or truncated

        value = 0.0
        if done and float(reward) > 0:
            value = reward
        elif not is_terminal:
            value = estimate_value(next_state)

        return StepFnReturn(
            value=float(value),
            reward=float(reward),
            done=is_terminal,
            state=next_state,
        )

    return step_fn


def random_action_selection_fn(
    policy_input: PolicyInput,
) -> PolicyReturn:
    n_actions = len(policy_input.tree.children_indices[0])
    action = random.randint(0, n_actions - 1)
    return PolicyReturn(action=action)


def find_best_action(tree: Tree) -> int:
    root_child_visits = tree.n_sa[ROOT_INDEX]
    best_action = max(range(len(root_child_visits)), key=lambda i: root_child_visits[i])
    return best_action


def main():
    ENV_NAME = "FrozenLake-v1"
    N_ACTIONS = 4
    N_ITERATIONS = 100
    MAX_DEPTH = 15
    EXPLORATION_CONSTANT = 1.2

    root_fn = root_fn_factory(ENV_NAME)
    step_fn = step_fn_factory(ENV_NAME)
    ucb1_fn = ucb1_fn_factory(EXPLORATION_CONSTANT)

    final_tree = MCTS.search(
        n_actions=N_ACTIONS,
        root_fn=root_fn,
        policy_fn=ucb1_fn,
        step_fn=step_fn,
        max_depth=MAX_DEPTH,
        n_iterations=N_ITERATIONS,
    )

    best_action = find_best_action(final_tree)
    action_map = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    print(f"Search complete after {N_ITERATIONS} iterations.")
    print(f"Root visit counts: {final_tree.n_sa[ROOT_INDEX]}")
    print(f"Root values: {[f'{v:.3f}' for v in final_tree.q_sa[ROOT_INDEX]]}")
    print(f"Best action from root: {action_map[best_action]} ({best_action})")


if __name__ == "__main__":
    main()
