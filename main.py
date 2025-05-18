import copy
import functools
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Bool, Float, Int, PyTree
from mctsnp import (
    MCTS,
    ActionSelectionInput,
    ActionSelectionReturn,
    RootFnOutput,
    StepFnInput,
    StepFnReturn,
)


def root_fn(env: gym.Env) -> RootFnOutput:
    obs, _ = env.reset()
    return RootFnOutput(embedding=obs)


def inner_action_selection_fn(
    action_selection_input: ActionSelectionInput, *, env: gym.Env
) -> ActionSelectionReturn:
    return ActionSelectionReturn(action=env.action_space.sample())


def step_fn(step_fn_input: StepFnInput, *, env: gym.Env) -> StepFnReturn:
    env_copy = copy.deepcopy(env)

    obs = step_fn_input.embedding
    env_copy.unwrapped.s = obs  # pyright: ignore

    next_obs, reward, terminated, truncated, info = env_copy.step(step_fn_input.action)

    value = np.array(np.random.uniform())

    return StepFnReturn(
        value=value,
        discount=np.array(0.9 if not (terminated or truncated) else 0.0),
        reward=np.array(reward),
        embedding=next_obs,
    )


env = gym.make("FrozenLake-v1")
action_space: int = env.action_space.n  # pyright: ignore
observation_space: int = env.observation_space.n  # pyright: ignore

obs, _ = env.reset(seed=44)

tree = MCTS.search(
    action_space,
    max_depth=500,
    root_fn=functools.partial(root_fn, env=env),
    inner_action_selection_fn=functools.partial(inner_action_selection_fn, env=env),
    step_fn=functools.partial(step_fn, env=env),
    n_iterations=500,
)
print(tree)
