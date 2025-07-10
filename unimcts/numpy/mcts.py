from dataclasses import dataclass

from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Bool, Float, Int

import numpy as np

NO_PARENT = -1
UNVISITED = -1
ROOT_INDEX = 0


@dataclass
class Tree:
    parent_indices: Int[np.ndarray, "n_nodes"]
    children_indices: Int[np.ndarray, "n_nodes n_actions"]
    action_from_parent: Int[np.ndarray, "n_nodes"]

    node_visits: Int[np.ndarray, "n_nodes"]
    node_values: Float[np.ndarray, "n_nodes"]
    node_is_terminal: Bool[np.ndarray, "n_nodes"]

    children_values: Float[np.ndarray, "n_nodes n_actions"]
    children_visits: Int[np.ndarray, "n_nodes n_actions"]
    children_rewards: Float[np.ndarray, "n_nodes n_actions"]

    embeddings: dict[Int, Any]


class RootFnOutput(NamedTuple):
    embedding: Any


class ActionSelectionInput(NamedTuple):
    tree: Tree
    node_index: int
    depth: Int[np.ndarray, ""]


class ActionSelectionReturn(NamedTuple):
    action: Int[np.ndarray, ""]


class SelectionOutput(NamedTuple):
    parent_index: int
    action: Int[np.ndarray, ""]


class StepFnInput(NamedTuple):
    embedding: Any
    action: Int[np.ndarray, ""]


class StepFnReturn(NamedTuple):
    """
    Return type for the step function that processes a state and action.

    Attributes:
        value: Estimated value from this state (via rollout/NN/heuristic)
        reward: Immediate reward for taking action
        done: Boolean indicating whether this is a terminal state
        embedding: Next state representation after taking the action
    """

    value: Float[np.ndarray, ""]
    reward: Float[np.ndarray, ""]
    done: Bool[np.ndarray, ""]
    embedding: Any


class BatchedStepFnInput(NamedTuple):
    embeddings: list[Any]
    actions: Int[np.ndarray, "batch_size 1"]


class BatchedStepFnReturn(NamedTuple):
    returns: list[StepFnReturn]


class LeafNode(NamedTuple):
    node_index: int
    action: Int[np.ndarray, ""]


class BackpropagationState(NamedTuple):
    tree: Tree
    idx: int
    value: Float[np.ndarray, ""]


class SelectionState(NamedTuple):
    node_index: int
    next_node_index: int
    depth: Int[np.ndarray, ""]
    action: Int[np.ndarray, ""]
    proceed: Bool[np.ndarray, ""]


def generate_tree(n_nodes: int, n_actions: int, root_fn_output: RootFnOutput) -> Tree:
    parent_indices = np.full(shape=(n_nodes), fill_value=NO_PARENT)
    action_from_parent = np.full(shape=(n_nodes), fill_value=NO_PARENT)
    children_indices = np.full(shape=(n_nodes, n_actions), fill_value=UNVISITED)

    node_visits = np.zeros(shape=(n_nodes), dtype=np.int32)
    node_values = np.zeros(shape=(n_nodes))

    children_values = np.zeros(shape=(n_nodes, n_actions))
    children_visits = np.zeros(shape=(n_nodes, n_actions), dtype=np.int32)
    children_rewards = np.zeros(shape=(n_nodes, n_actions))
    node_is_terminal = np.zeros(shape=(n_nodes), dtype=bool)

    embeddings = {ROOT_INDEX: root_fn_output.embedding}
    node_visits[ROOT_INDEX] = 1

    return Tree(
        parent_indices=parent_indices,
        children_indices=children_indices,
        action_from_parent=action_from_parent,
        node_visits=node_visits,
        node_values=node_values,
        node_is_terminal=node_is_terminal,
        children_values=children_values,
        children_visits=children_visits,
        children_rewards=children_rewards,
        embeddings=embeddings,
    )


def selection(
    tree: Tree,
    max_depth: int,
    inner_action_selection_fn: Callable[[ActionSelectionInput], ActionSelectionReturn],
) -> SelectionOutput:
    def _selection(state: SelectionState) -> SelectionState:
        node_index = state.next_node_index
        if tree.node_is_terminal[node_index]:
            return SelectionState(
                node_index=state.node_index,
                next_node_index=node_index,
                depth=state.depth,
                action=state.action,
                proceed=np.array(False),
            )

        action_selection_output = inner_action_selection_fn(
            ActionSelectionInput(tree, node_index, state.depth)
        )
        next_node_index = tree.children_indices[
            node_index, action_selection_output.action
        ]
        visited = next_node_index != np.array(UNVISITED)
        max_depth_not_exceeded = state.depth + 1 < max_depth
        proceed = np.logical_and(visited, max_depth_not_exceeded)

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
        depth=np.array(0),
        action=np.array(UNVISITED),
        proceed=np.array(True),
    )

    while state.proceed:
        state = _selection(state)

    return SelectionOutput(state.node_index, state.action)


def expansion(
    tree: Tree,
    selection_output: SelectionOutput,
    next_node_index: int,
    step_fn_return: StepFnReturn,
) -> LeafNode:
    parent_index, action = selection_output
    assert tree.children_indices[parent_index, action] == UNVISITED, (
        f"Can only expand unvisited nodes, got {tree.children_indices[parent_index, action]=}"
    )
    value, reward, done, next_state = step_fn_return

    tree.children_indices[parent_index, action] = next_node_index
    tree.action_from_parent[next_node_index] = action
    tree.parent_indices[next_node_index] = parent_index
    tree.node_values[next_node_index] = value
    tree.node_visits[next_node_index] = 1
    tree.node_is_terminal[next_node_index] = done
    tree.children_rewards[parent_index, action] = reward
    tree.embeddings[next_node_index] = next_state

    return LeafNode(
        node_index=next_node_index,
        action=action,
    )


def backpropagate(tree: Tree, leaf_index: int) -> Tree:
    def _backpropagate(state: BackpropagationState) -> BackpropagationState:
        tree, idx, value = state
        parent = tree.parent_indices[idx]
        action = tree.action_from_parent[idx]

        reward = tree.children_rewards[parent, action]

        parent_value = tree.node_values[parent]
        parent_visits = tree.node_visits[parent]

        leaf_value = reward + state.value
        parent_value = (parent_value * parent_visits + leaf_value) / (
            parent_visits + 1.0
        )

        tree.node_values[parent] = parent_value
        tree.node_visits[parent] = parent_visits + 1

        tree.children_values[parent, action] = tree.node_values[idx]
        tree.children_visits[parent, action] = tree.children_visits[parent, action] + 1

        next_state = BackpropagationState(idx=parent, value=leaf_value, tree=tree)

        return next_state

    state = BackpropagationState(
        idx=leaf_index, value=tree.node_values[leaf_index], tree=tree
    )

    while state.idx != ROOT_INDEX:
        state = _backpropagate(state)

    return state.tree


class MCTS:
    @staticmethod
    def search(
        n_actions: int,
        root_fn: Callable[[], RootFnOutput],
        inner_action_selection_fn: Callable[
            [ActionSelectionInput], ActionSelectionReturn
        ],
        batched_step_fn: Callable[[BatchedStepFnInput], BatchedStepFnReturn],
        max_depth: int,
        n_iterations: int,
        batch_size: int = 1,
    ):
        node_index_counter = 0
        tree = generate_tree(
            n_nodes=n_iterations + 1, n_actions=n_actions, root_fn_output=root_fn()
        )

        total_iterations_done = 0
        while total_iterations_done < n_iterations:
            dedupes = set()
            pending_expansions: list[tuple[SelectionOutput, int]] = []
            leaf_nodes: list[LeafNode] = []

            attempts = 0
            while (
                len(pending_expansions) + len(leaf_nodes) < batch_size
                and attempts < batch_size * 3
                and total_iterations_done + len(pending_expansions) + len(leaf_nodes)
                < n_iterations
            ):
                attempts += 1
                selection_output = selection(tree, max_depth, inner_action_selection_fn)

                if (
                    selection_output.parent_index,
                    int(selection_output.action),
                ) in dedupes:
                    continue
                elif (
                    tree.children_indices[
                        selection_output.parent_index, selection_output.action
                    ]
                    == UNVISITED
                ):
                    node_index_counter += 1
                    pending_expansions.append((selection_output, node_index_counter))
                    dedupes.add(
                        (
                            selection_output.parent_index,
                            int(selection_output.action),
                        )
                    )
                else:
                    child_idx = tree.children_indices[
                        selection_output.parent_index, selection_output.action
                    ]
                    leaf_nodes.append(
                        LeafNode(node_index=child_idx, action=selection_output.action)
                    )

            if len(pending_expansions) > 0:
                actions = np.zeros(shape=(len(pending_expansions), 1))
                embeddings = []

                for i, p in enumerate(pending_expansions):
                    selection_output, _ = p
                    action = selection_output.action
                    embedding = tree.embeddings[selection_output.parent_index]
                    actions[i] = action
                    embeddings.append(embedding)

                batched_step_fn_return = batched_step_fn(
                    BatchedStepFnInput(
                        actions=actions,
                        embeddings=embeddings,
                    )
                )

                for b, p in zip(batched_step_fn_return.returns, pending_expansions):
                    selection_output, next_node_index = p
                    leaf_nodes.append(
                        expansion(
                            tree=tree,
                            selection_output=selection_output,
                            next_node_index=next_node_index,
                            step_fn_return=b,
                        )
                    )

            for leaf_node in leaf_nodes:
                tree = backpropagate(tree, leaf_node.node_index)

            total_iterations_done += len(leaf_nodes)

        return tree
