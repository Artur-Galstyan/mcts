import multiprocessing
import multiprocessing.pool
import queue
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Bool, Float, Int

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
    virtual_losses: Int[np.ndarray, "n_nodes"]
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


class BatchedStepFnInput(NamedTuple):
    embeddings: list[Any]
    actions: Int[np.ndarray, "batch_size"]


class StepFnReturn(NamedTuple):
    value: Float[np.ndarray, ""]
    reward: Float[np.ndarray, ""]
    done: Bool[np.ndarray, ""]
    embedding: Any


class BatchedStepFnReturn(NamedTuple):
    value: Float[np.ndarray, "batch_size"]
    reward: Float[np.ndarray, "batch_size"]
    done: Bool[np.ndarray, "batch_size"]
    embedding: list[Any]


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
    virtual_losses = np.zeros(shape=(n_nodes), dtype=np.int32)
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
        virtual_losses=virtual_losses,
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
    step_result: StepFnReturn,
) -> LeafNode:
    parent_index, action = selection_output
    assert tree.children_indices[parent_index, action] == UNVISITED
    value, reward, done, next_state = step_result
    tree.children_indices[parent_index, action] = next_node_index
    tree.action_from_parent[next_node_index] = action
    tree.parent_indices[next_node_index] = parent_index
    tree.node_values[next_node_index] = value
    tree.node_visits[next_node_index] = 1
    tree.node_is_terminal[next_node_index] = done
    tree.children_rewards[parent_index, action] = reward
    tree.embeddings[next_node_index] = next_state
    return LeafNode(node_index=next_node_index, action=action)


def backpropagate(tree: Tree, leaf_index: int) -> None:
    def _backpropagate(state: BackpropagationState) -> BackpropagationState:
        tree, idx, _ = state
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


def apply_virtual_loss(tree: Tree, leaf_node_index: int):
    idx = leaf_node_index
    if idx == NO_PARENT:
        return
    while idx != ROOT_INDEX:
        tree.virtual_losses[idx] += 1
        idx = tree.parent_indices[idx]
    tree.virtual_losses[ROOT_INDEX] += 1


def remove_virtual_loss(tree: Tree, leaf_node_index: int):
    idx = leaf_node_index
    if idx == NO_PARENT:
        return
    while idx != ROOT_INDEX:
        tree.virtual_losses[idx] -= 1
        idx = tree.parent_indices[idx]
    tree.virtual_losses[ROOT_INDEX] -= 1


def inference_worker(
    n_workers: int,
    step_queue: queue.Queue,
    step_fn: Callable[[BatchedStepFnInput], BatchedStepFnReturn],
    results_dict: dict[int, StepFnReturn],
    batch_timeout_ms: int,
):
    while True:
        try:
            first_request = step_queue.get()
            if first_request is None:
                break

            batch_requests = [first_request]
            deadline = time.time() + (batch_timeout_ms / 1000)
            while len(batch_requests) < n_workers and time.time() < deadline:
                try:
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        break
                    request = step_queue.get(timeout=remaining_time)
                    if request is None:
                        step_queue.put(None)
                        break
                    batch_requests.append(request)
                except queue.Empty:
                    break

            request_ids = []
            step_inputs = []

            for req_id, step_input in batch_requests:
                request_ids.append(req_id)
                step_inputs.append(step_input)

            embeddings = [inp.embedding for inp in step_inputs]
            actions = np.array([inp.action for inp in step_inputs])
            batched_input = BatchedStepFnInput(embeddings=embeddings, actions=actions)
            batched_output = step_fn(batched_input)

            for i, request_id in enumerate(request_ids):
                result = StepFnReturn(
                    value=batched_output.value[i],
                    reward=batched_output.reward[i],
                    done=batched_output.done[i],
                    embedding=batched_output.embedding[i],
                )
                results_dict[request_id] = result
        except Exception:
            continue


def run_simulation(args):
    (
        iteration,
        tree,
        step_queue,
        results_dict,
        max_depth,
        inner_action_selection_fn,
    ) = args

    selection_output = selection(tree, max_depth, inner_action_selection_fn)
    parent_index = selection_output.parent_index
    action = selection_output.action
    apply_virtual_loss(tree, parent_index)
    if tree.children_indices[parent_index, action] == UNVISITED:
        request_id = iteration
        embedding = tree.embeddings[parent_index]
        step_input = StepFnInput(embedding=embedding, action=action)
        step_queue.put((request_id, step_input))

        while request_id not in results_dict:
            time.sleep(0.0001)
        step_result = results_dict.pop(request_id)

        if tree.children_indices[parent_index, action] == UNVISITED:
            node_index_counter = iteration + 1
            leaf_node = expansion(
                tree, selection_output, node_index_counter, step_result
            )
        else:
            child_idx = tree.children_indices[parent_index, action]
            leaf_node = LeafNode(node_index=child_idx, action=action)
    else:
        child_idx = tree.children_indices[parent_index, action]
        leaf_node = LeafNode(node_index=child_idx, action=action)
    remove_virtual_loss(tree, parent_index)
    backpropagate(tree, leaf_node.node_index)


class MCTS:
    @staticmethod
    def search(
        n_actions: int,
        root_fn: Callable[[], RootFnOutput],
        inner_action_selection_fn: Callable[
            [ActionSelectionInput], ActionSelectionReturn
        ],
        step_fn: Callable[[BatchedStepFnInput], BatchedStepFnReturn],
        max_depth: int,
        n_iterations: int,
        n_workers: int,
        batch_timeout_ms: int = 1,
    ):
        assert n_workers >= 1
        manager = multiprocessing.Manager()
        step_queue = manager.Queue()
        results_dict = manager.dict()
        tree = generate_tree(
            n_nodes=n_iterations + 1, n_actions=n_actions, root_fn_output=root_fn()
        )
        inference_thread = multiprocessing.Process(
            target=inference_worker,
            args=(n_workers, step_queue, step_fn, results_dict, batch_timeout_ms),
            daemon=True,
        )
        inference_thread.start()
        with multiprocessing.pool.ThreadPool(processes=n_workers) as pool:
            simulation_args = (
                tree,
                step_queue,
                results_dict,
                max_depth,
                inner_action_selection_fn,
            )
            args_for_map = [(i,) + simulation_args for i in range(n_iterations)]
            pool.map(run_simulation, args_for_map)
        step_queue.put(None)
        inference_thread.join(timeout=1)
        return tree
