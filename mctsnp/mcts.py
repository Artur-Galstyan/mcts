import copy
import functools
from dataclasses import dataclass

import numpy as np
from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Bool, Float, Int, PyTree

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

    embeddings: dict

    def __str__(self):
        """
        Returns a formatted string representation of the tree.
        This provides a more readable view than the default __repr__.
        """
        # Format tree statistics
        stats = []
        stats.append("Tree Statistics:")

        # Count visited nodes
        visited_nodes = sum(self.node_visits > 0)
        stats.append(f"  Visited nodes: {visited_nodes}/{len(self.node_visits)}")

        # Add root info if it has been visited
        if self.node_visits[ROOT_INDEX] > 0:
            stats.append(f"  Root visits: {self.node_visits[ROOT_INDEX]}")
            stats.append(f"  Root value: {self.node_values[ROOT_INDEX]:.4f}")

        # Count expanded children
        expanded_children = sum(np.any(self.children_indices != UNVISITED, axis=1))
        stats.append(f"  Nodes with children: {expanded_children}")

        # Format compact tree structure
        structure = []
        structure.append("Tree Structure:")

        def _format_node(node_idx, depth=0, max_depth=2):
            if depth > max_depth:
                return []

            indent = "  " * depth
            node_lines = []

            # Skip unvisited nodes
            if self.node_visits[node_idx] == 0 and node_idx != ROOT_INDEX:
                return node_lines

            # Format node info
            node_text = f"{indent}Node {node_idx}"
            if node_idx == ROOT_INDEX:
                node_text += " (ROOT)"
            else:
                action = self.action_from_parent[node_idx]
                node_text += f" (via action {action})"

            visits = self.node_visits[node_idx]
            value = self.node_values[node_idx]
            node_text += f", Visits: {visits}, Value: {value:.4f}"

            node_lines.append(node_text)

            # Add children (if any and if we haven't reached max depth)
            if depth < max_depth:
                children = [
                    (a, child_idx)
                    for a, child_idx in enumerate(self.children_indices[node_idx])
                    if child_idx != UNVISITED
                ]

                for action, child_idx in children:
                    child_visits = self.children_visits[node_idx, action]
                    # Skip unvisited children
                    if child_visits == 0:
                        continue

                    child_value = self.children_values[node_idx, action]
                    child_text = f"{indent}  └── Action {action}: → Node {child_idx}, Visits: {child_visits}, Value: {child_value:.4f}"
                    node_lines.append(child_text)

                    # Recursively add the child's children
                    child_lines = _format_node(child_idx, depth + 2, max_depth)
                    node_lines.extend(child_lines)

            return node_lines

        structure.extend(_format_node(ROOT_INDEX))
        return "\n".join(stats + [""] + structure)

    def __repr__(self):
        lines = ["Tree("]
        parent_shape = self.parent_indices.shape
        children_shape = self.children_indices.shape

        non_default_parents = np.sum(self.parent_indices != NO_PARENT)
        non_default_children = np.sum(self.children_indices != UNVISITED)
        non_zero_visits = np.sum(self.node_visits > 0)

        lines.append(
            f"  parent_indices: shape={parent_shape}, non_default={non_default_parents},"
        )
        lines.append(
            f"  children_indices: shape={children_shape}, non_default={non_default_children},"
        )
        lines.append(
            f"  node_visits: shape={self.node_visits.shape}, non_zero={non_zero_visits},"
        )

        lines.append(
            f"  embeddings: {{{', '.join(f'{k}' for k in self.embeddings.keys())}}})"
        )

        return "\n".join(lines)


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
    step_fn: Callable[[StepFnInput], StepFnReturn],
) -> LeafNode:
    parent_index, action = selection_output
    assert tree.children_indices[parent_index, action] == UNVISITED, (
        f"Can only expand unvisited nodes, got {tree.children_indices[parent_index, action]=}"
    )
    embedding = tree.embeddings[parent_index]
    value, reward, done, next_state = step_fn(
        StepFnInput(embedding=embedding, action=action)
    )
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
        step_fn: Callable[[StepFnInput], StepFnReturn],
        max_depth: int,
        n_iterations: int,
    ):
        node_index_counter = 0
        tree = generate_tree(
            n_nodes=n_iterations + 1, n_actions=n_actions, root_fn_output=root_fn()
        )

        for iteration in range(n_iterations):
            selection_output = selection(tree, max_depth, inner_action_selection_fn)

            if (
                tree.children_indices[
                    selection_output.parent_index, selection_output.action
                ]
                == UNVISITED
            ):
                node_index_counter += 1
                leaf_node = expansion(
                    tree, selection_output, node_index_counter, step_fn
                )
            else:
                child_idx = tree.children_indices[
                    selection_output.parent_index, selection_output.action
                ]
                leaf_node = LeafNode(
                    node_index=child_idx,
                    action=selection_output.action,
                )

            tree = backpropagate(tree, leaf_node.node_index)

        return tree
