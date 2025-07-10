import copy
import functools
import time

import gymnasium as gym
import numpy as np
from unimcts.numpy import (
    MCTS,
    ROOT_INDEX,
    UNVISITED,
    BatchedStepFnInput,
    BatchedStepFnReturn,
    RootFnOutput,
    StepFnReturn,
    Tree,
)
from unimcts.numpy.strategies import ucb_action_selection


def root_fn(initial_obs: int) -> RootFnOutput:
    return RootFnOutput(embedding=int(initial_obs))


def batched_step_fn(input: BatchedStepFnInput, env: gym.Env) -> BatchedStepFnReturn:
    returns = []

    for i in range(len(input.actions)):
        env_copy = copy.deepcopy(env)
        env_copy.unwrapped.s = input.embeddings[i]

        next_obs, reward, terminated, truncated, _ = env_copy.step(
            int(input.actions[i, 0])
        )

        if terminated and reward > 0:  # pyright: ignore
            value = 1.0
        elif terminated and reward == 0:
            value = -1.0
        else:
            value = 0.0

        returns.append(
            StepFnReturn(
                value=np.array(value),
                reward=np.array(float(reward)),
                embedding=int(next_obs),
                done=np.array(terminated or truncated),
            )
        )

    return BatchedStepFnReturn(returns=returns)


def get_best_path_no_cycles(tree: Tree):
    path = []
    node = ROOT_INDEX
    visited_states = {tree.embeddings[ROOT_INDEX]}

    for _ in range(20):
        if tree.node_is_terminal[node]:
            break

        visits = tree.children_visits[node]
        indices = tree.children_indices[node]

        if np.all(visits == 0):
            break

        # Find best action that doesn't create a cycle
        action_scores = []
        for action in range(len(visits)):
            if visits[action] == 0:
                action_scores.append(-1)
            else:
                next_node = indices[action]
                if next_node != UNVISITED:
                    next_state = tree.embeddings.get(next_node)
                    if next_state in visited_states:
                        action_scores.append(-1)  # Penalize cycles
                    else:
                        action_scores.append(visits[action])
                else:
                    action_scores.append(-1)

        if all(score == -1 for score in action_scores):
            # No valid moves, just take most visited
            action = np.argmax(visits)
        else:
            action = np.argmax(action_scores)

        path.append(int(action))

        next_node = indices[action]
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


# Setup
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
obs, _ = env.reset(seed=42)

print("Running MCTS on FrozenLake")
print("Goal: Find path from S to G avoiding holes (H)")
print(f"Initial state: {obs}\n")

# Run MCTS with more iterations
start = time.time()
tree = MCTS.search(
    n_actions=env.action_space.n,
    max_depth=20,
    root_fn=functools.partial(root_fn, initial_obs=obs),
    inner_action_selection_fn=ucb_action_selection,
    batched_step_fn=functools.partial(batched_step_fn, env=env),
    n_iterations=10000,
    batch_size=8,  # Add batch size for efficiency
)
end = time.time()
print(f"Search duration = {end - start}")

print("Tree statistics:")
print(f"  Total nodes: {len(tree.embeddings)}")
print(f"  Root visits: {tree.node_visits[ROOT_INDEX]}")
print(f"  Root value: {tree.node_values[ROOT_INDEX]:.3f}")

# Get and test best path
best_path = get_best_path_no_cycles(tree)
print(f"\nBest path found (avoiding cycles): {best_path}")

if best_path:
    env.reset()
    print("\nExecuting path:")
    print("\nInitial state:")
    print(env.render())
    print("Player starts at state 0 (top-left corner)")

    total_reward = 0
    state = 0

    for i, action in enumerate(best_path):
        print(f"\nStep {i + 1}: Action {['Left', 'Down', 'Right', 'Up'][action]}")
        obs, reward, terminated, truncated, _ = env.step(action)
        state = obs
        total_reward += reward  # pyright: ignore
        print(env.render())
        print(f"Player at state {state} (row {state // 4}, col {state % 4})")
        print(f"Reward: {reward}")

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
