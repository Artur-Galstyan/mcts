from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import pgx
from beartype import beartype
from beartype.typing import Any, Callable, Protocol, runtime_checkable
from jax._src.pjit import PytreeLeaf
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict
from typing_extensions import ParamSpec

# === Very simple environment ===
# The environment tree looks like this:
#
# 0
# | \
# 1  2
# | \ | \
# 3  4 5  6
# Only reaching state 5 gives a reward of 1. The rest of the states give a reward of 0.
# States 3, 4, and 6 are terminal states.

UNVISITED_NODE = -1
NO_PARENT = -1
ROOT_NODE = 0


class RandomWalkEnv:
    def __init__(self) -> None:
        pass

    def reset(self) -> Array:
        return jnp.array(0)

    def step(self, state: Array, action: Array) -> tuple[Array, Array, Bool]:
        assert state in [0, 1, 2], "Invalid state."
        assert action in [0, 1], "Invalid action."
        if state == 0:
            if action == 0:
                return (
                    jnp.array(1, dtype=jnp.int32),
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(False, dtype=jnp.bool),
                )
            else:
                return (
                    jnp.array(2, dtype=jnp.int32),
                    jnp.array(0),
                    jnp.array(False, dtype=jnp.bool),
                )
        elif state == 1:
            if action == 0:
                return (
                    jnp.array(3, dtype=jnp.int32),
                    jnp.array(0),
                    jnp.array(True, dtype=jnp.bool),
                )
            else:
                return (
                    jnp.array(4, dtype=jnp.int32),
                    jnp.array(0),
                    jnp.array(True, dtype=jnp.bool),
                )
        else:
            if action == 0:
                return (
                    jnp.array(5, dtype=jnp.int32),
                    jnp.array(1),
                    jnp.array(True, dtype=jnp.bool),
                )
            else:
                return (
                    jnp.array(6, dtype=jnp.int32),
                    jnp.array(0),
                    jnp.array(True, dtype=jnp.bool),
                )


def ucb1(
    avg_node_value: Float[Array, ""],
    visits_parent: Float[Array, ""],
    visits_node: Float[Array, ""],
    exploration_exploitation_factor: Float[Array, ""] = jnp.array(2.0),
) -> Float[Array, ""]:
    """
    Upper Confidence Bound 1 (UCB1) formula for MCTS.

    Args:
        avg_node_value: The average value of the current node. V(s)
        visits_parent: The number of visits of the parent node. n(s_parent)
        visits_node: The number of visits of the current node. n(s)
        exploration_exploitation_factor: The exploration-exploitation factor
            that balances between exploration and exploitation. C

    Returns:
        The UCB1 value of the current node. UCB1(s)
    """
    return avg_node_value + exploration_exploitation_factor * jnp.sqrt(
        jnp.log(visits_parent) / visits_node
    )


class RootFnOutput(BaseModel):
    params: PyTree
    value: Array
    state_embedding: PyTree

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RootFnCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> RootFnOutput: ...


class ActionSelectionRoot(Protocol):
    def __call__(self) -> Array: ...


class StepFnOutput(BaseModel):
    reward: Array
    discount: Array
    value: Array
    priors: Array

    state_embedding: PyTree

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StepFnCallable(Protocol):
    def __call__(
        self, params: PyTree, action: Array, state_embedding: PyTree, key: PRNGKeyArray
    ) -> StepFnOutput: ...


class Tree(BaseModel):
    node_values: Float[Array, "N"]
    node_visits: Int[Array, "N"]
    parent_indices: Int[Array, "N"]
    states: PyTree
    action_from_parent: Int[Array, "N"]
    children_index: Int[Array, "N A"]
    children_visits: Int[Array, "N A"]
    children_rewards: Float[Array, "N A"]
    children_values: Float[Array, "N A"]
    children_discounts: Float[Array, "N A"]
    children_priors: Float[Array, "N A"]

    root_fn_output: RootFnOutput

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, N: int, A: int, root_fn_output: RootFnOutput):
        self.node_visits = jnp.zeros(shape=(N,), dtype=jnp.int32)
        self.parent_indices = jnp.zeros(shape=(N,), dtype=jnp.int32).at[0].set(-1)
        self.action_from_parent = jnp.zeros(shape=(N,), dtype=jnp.int32).at[0].set(-1)
        self.children_index = jnp.ones(shape=(N, A), dtype=jnp.int32) * UNVISITED_NODE
        self.children_visits = jnp.zeros(shape=(N, A), dtype=jnp.int32)
        self.children_rewards = jnp.zeros(shape=(N, A))
        self.children_values = jnp.zeros(shape=(N, A))
        self.children_discounts = jnp.zeros(shape=(N, A))
        self.children_priors = jnp.zeros(shape=(N, A))
        self.root_fn_output = root_fn_output
        self.node_values = jnp.zeros(shape=(N,)).at[0].set(self.root_fn_output.value)


@beartype
class MCTS:
    n_simulations: int
    n_actions: int
    root_fn: RootFnCallable
    tree: Tree

    def __init__(
        self, n_simulations: int, n_actions: int, root_fn: RootFnCallable
    ) -> None:
        assert isinstance(
            root_fn(), RootFnOutput
        ), "The root_fn must return a RootFnOutput object."
        self.root_fn = root_fn
        self.n_simulations = n_simulations
        self.n_actions = n_actions
        self.tree = Tree(self.n_simulations + 1, self.n_actions, self.root_fn())

    def _simulation(self):
        pass


def get_root_fn():
    env = RandomWalkEnv()
    root = env.reset()
    value = jnp.array(0)
    params = jnp.array(0)
    return RootFnOutput(params=params, value=value, state_embedding=root)


mcts = MCTS(n_simulations=4, n_actions=2, root_fn=get_root_fn)
