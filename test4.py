import copy
import functools
import multiprocessing
import multiprocessing.pool
import queue
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Bool, Float, Int

# ==============================================================================
# MCTS Core Implementation
# ==============================================================================

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


def inference_worker(n_workers, step_queue, step_fn, results_dict):
    while True:
        try:
            first_request = step_queue.get()
            if first_request is None:
                break

            batch_requests = [first_request]
            while len(batch_requests) < n_workers:
                try:
                    request = step_queue.get_nowait()
                    if request is None:
                        step_queue.put(None)
                        break
                    batch_requests.append(request)
                except queue.Empty:
                    break

            request_ids = [req[0] for req in batch_requests]
            step_inputs = [req[1] for req in batch_requests]

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
            args=(n_workers, step_queue, step_fn, results_dict),
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


# ==============================================================================
# Test Script for FrozenLake
# ==============================================================================


def root_fn(initial_obs: int) -> RootFnOutput:
    return RootFnOutput(embedding=int(initial_obs))


def ucb_action_selection(input: ActionSelectionInput) -> ActionSelectionReturn:
    tree = input.tree
    node = input.node_index
    children_indices = tree.children_indices[node]
    unvisited_actions = np.where(children_indices == UNVISITED)[0]
    if len(unvisited_actions) > 0:
        action = np.random.choice(unvisited_actions)
        return ActionSelectionReturn(action=np.array(action))

    parent_visits = tree.node_visits[node] + tree.virtual_losses[node]

    ucb_scores = np.full(len(children_indices), -np.inf, dtype=float)

    for action_idx, child_node_idx in enumerate(children_indices):
        child_visits = tree.children_visits[node, action_idx]
        if child_visits > 0:
            child_value = tree.children_values[node, action_idx]

            exploration_term = 2.0 * np.sqrt(np.log(parent_visits) / child_visits)
            ucb_scores[action_idx] = child_value + exploration_term

    action = np.argmax(ucb_scores)
    return ActionSelectionReturn(action=np.array(action))


def batched_step_fn(input: BatchedStepFnInput) -> BatchedStepFnReturn:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    values, rewards, dones, next_embeddings = [], [], [], []

    for embedding, action in zip(input.embeddings, input.actions):
        env.reset()
        env.unwrapped.s = embedding
        next_obs, reward, terminated, truncated, _ = env.step(int(action))

        if terminated and reward > 0:
            value = 1.0
        elif terminated and reward == 0:
            value = -1.0
        else:
            value = 0.0

        values.append(value)
        rewards.append(float(reward))
        dones.append(terminated or truncated)
        next_embeddings.append(int(next_obs))

    return BatchedStepFnReturn(
        value=np.array(values),
        reward=np.array(rewards),
        done=np.array(dones),
        embedding=next_embeddings,
    )


def get_best_path_no_cycles(tree: Tree):
    path = []
    node = ROOT_INDEX
    visited_states = {tree.embeddings[ROOT_INDEX]}
    for _ in range(20):
        if tree.node_is_terminal[node]:
            break
        child_values = tree.children_values[node]
        child_visits = tree.children_visits[node]
        child_indices = tree.children_indices[node]
        if np.all(child_visits == 0):
            break
        action_scores = np.full_like(child_values, -np.inf)
        for action, value in enumerate(child_values):
            if child_visits[action] > 0:
                next_node_idx = child_indices[action]
                if next_node_idx != UNVISITED:
                    next_state = tree.embeddings.get(next_node_idx)
                    if next_state not in visited_states:
                        action_scores[action] = value
        if np.all(action_scores == -np.inf):
            break
        best_action = np.argmax(action_scores)
        path.append(int(best_action))
        next_node = child_indices[best_action]
        if next_node == UNVISITED:
            break
        next_state = tree.embeddings.get(next_node)
        visited_states.add(next_state)
        node = int(next_node)
    return path


def debug_node(tree, node_idx, depth=0):
    if depth > 2:
        return
    state = tree.embeddings.get(node_idx, "?")
    visits = tree.node_visits[node_idx]
    value = tree.node_values[node_idx]
    terminal = tree.node_is_terminal[node_idx]
    indent = "  " * depth
    print(
        f"{indent}Node {node_idx}: state={state}, visits={visits}, value={value:.3f}, terminal={terminal}"
    )
    if not terminal and visits > 0:
        for action in range(len(tree.children_visits[node_idx])):
            child_visits = tree.children_visits[node_idx, action]
            if child_visits > 0:
                child_idx = tree.children_indices[node_idx, action]
                child_value = tree.children_values[node_idx, action]
                print(
                    f"{indent}  Action {action} ({['L', 'D', 'R', 'U'][action]}): visits={child_visits}, value={child_value:.3f}"
                )
                if depth < 2:
                    debug_node(tree, child_idx, depth + 1)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
    obs, _ = env.reset(seed=42)

    print("Running MCTS on FrozenLake")
    print("Goal: Find path from S to G avoiding holes (H)")
    print(f"Initial state: {obs}\n")

    start_time = time.time()

    n_iterations = 50_000
    n_workers = 1

    tree = MCTS.search(
        n_actions=env.action_space.n,
        max_depth=20,
        n_iterations=n_iterations,
        n_workers=n_workers,
        root_fn=functools.partial(root_fn, initial_obs=obs),
        inner_action_selection_fn=ucb_action_selection,
        step_fn=batched_step_fn,
    )

    end_time = time.time()

    print(
        f"Search duration was {end_time - start_time:.3f} seconds, i.e. each iteration took {(end_time - start_time) / n_iterations * 1000:.3f} ms ({n_workers} workers)"
    )

    print("Tree statistics:")
    print(f"  Total nodes: {len(tree.embeddings)}")
    print(f"  Root visits: {tree.node_visits[ROOT_INDEX]}")
    print(f"  Root value: {tree.node_values[ROOT_INDEX]:.3f}")

    # print("\nTree structure from root:")
    # debug_node(tree, ROOT_INDEX)

    best_path = get_best_path_no_cycles(tree)
    print(f"\nBest path found (avoiding cycles): {best_path}")

    if best_path:
        env.reset(seed=42)
        print("\nExecuting path:")
        print("\nInitial state:")
        print(env.render())
        print("Player starts at state 0 (top-left corner)")
        total_reward = 0
        state = 0
        for i, action in enumerate(best_path):
            # print(f"\nStep {i + 1}: Action {['Left', 'Down', 'Right', 'Up'][action]}")
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs
            total_reward += reward
            # print(env.render())
            # print(f"Player at state {state} (row {state // 4}, col {state % 4})")
            # print(f"Reward: {reward}")
            if terminated or truncated:
                if reward == 0:
                    print("💀 Fell in a hole!")
                else:
                    print("🎉 Reached the goal!")
                break
        print(f"\nFinal state: {state}")
        print(f"Total reward: {total_reward}")
        print(f"Result: {'SUCCESS!' if total_reward > 0 else 'Failed'}")
    else:
        print("\nNo path found!")

    print("\nExpected optimal paths: [2,2,1,1,1,2] or [1,1,2,2,1,2]")
    env.close()
