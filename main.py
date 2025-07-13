import functools
import time

import gymnasium as gym
import numpy as np
from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Bool, Float, Int
from unimcts import (
    MCTS,
    ROOT_INDEX,
    UNVISITED,
    ActionSelectionInput,
    ActionSelectionReturn,
    BatchedStepFnInput,
    BatchedStepFnReturn,
    RootFnOutput,
    Tree,
)

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

        if terminated and reward > 0:  # pyright: ignore
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

    n_iterations = 20_000
    n_workers = 16

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
    # print(f"\nBest path found (avoiding cycles): {best_path}")

    if best_path:
        env.reset(seed=42)
        # print("\nExecuting path:")
        # print("\nInitial state:")
        # print(env.render())
        # print("Player starts at state 0 (top-left corner)")
        total_reward = 0
        state = 0
        for i, action in enumerate(best_path):
            # print(f"\nStep {i + 1}: Action {['Left', 'Down', 'Right', 'Up'][action]}")
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs
            total_reward += reward  # pyright: ignore
            # print(env.render())
            # print(f"Player at state {state} (row {state // 4}, col {state % 4})")
            # print(f"Reward: {reward}")
            if terminated or truncated:
                if reward == 0:
                    print("💀 Fell in a hole!")
                else:
                    print("🎉 Reached the goal!")
                break
        # print(f"\nFinal state: {state}")
        # print(f"Total reward: {total_reward}")
        # print(f"Result: {'SUCCESS!' if total_reward > 0 else 'Failed'}")
    else:
        print("\nNo path found!")

    print("\nExpected optimal paths: [2,2,1,1,1,2] or [1,1,2,2,1,2]")
    env.close()
