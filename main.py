from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import pgx
from beartype import beartype
from beartype.typing import Any, Callable, NamedTuple, Protocol, runtime_checkable
from jax._src.pjit import PytreeLeaf
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
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

UNVISITED_NODE = jnp.array(-1)
NO_PARENT = jnp.array(-1)
ROOT_NODE = jnp.array(0)


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


# === END ===


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


@dataclass
class RootFnOutput:
    params: PyTree
    value: Array
    state_embedding: PyTree


class RootFnCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> RootFnOutput: ...


class ActionSelectionRoot(Protocol):
    def __call__(self) -> Array: ...


@dataclass
class StepFnOutput:
    reward: Array
    discount: Array
    value: Array
    priors: Array

    state_embedding: PyTree


class StepFnCallable(Protocol):
    def __call__(
        self, params: PyTree, action: Array, state_embedding: PyTree, key: PRNGKeyArray
    ) -> StepFnOutput: ...


@dataclass
class Tree:
    node_values: Float[Array, " N"]
    node_visits: Int[Array, " N"]
    parent_indices: Int[Array, " N"]
    states: PyTree
    action_from_parent: Int[Array, " N"]
    children_index: Int[Array, "N A"]
    children_visits: Int[Array, "N A"]
    children_rewards: Float[Array, "N A"]
    children_values: Float[Array, "N A"]
    children_discounts: Float[Array, "N A"]
    children_priors: Float[Array, "N A"]

    N: int
    A: int

    root_fn_output: RootFnOutput

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
        self.N = N
        self.A = A


class ActionSelectionFn(Protocol):
    def __call__(
        self,
        tree: Tree,
        current_node_index: Int[Array, ""],
        current_depth: Int[Array, ""],
    ) -> Array: ...


class SimulationState(NamedTuple):
    node_index: Int[Array, ""]
    action: Array
    next_node_index: Int[Array, ""]
    depth: Int[Array, ""]
    continue_simulation: Bool[Array, ""]

    key: PRNGKeyArray


@beartype
class MCTS:
    n_simulations: int
    n_actions: int
    max_depth: int
    root_fn_output: RootFnOutput
    tree: Tree

    def __init__(
        self,
        n_simulations: int,
        n_actions: int,
        root_fn: RootFnOutput,
        max_depth: int | None = None,
    ) -> None:
        self.root_fn_output = root_fn
        self.n_simulations = n_simulations
        self.n_actions = n_actions
        self.max_depth = max_depth if max_depth is not None else n_simulations
        self.tree = Tree(self.n_simulations + 1, self.n_actions, self.root_fn_output)

    def _simulation(
        self, action_selection_fn: ActionSelectionFn, key: PRNGKeyArray
    ) -> SimulationState:
        """
        Traverse the tree until an unvisited leaf node is reached (or `max_depth`
        was reached).
        """

        def body(state: SimulationState) -> SimulationState:
            key, subkey = jax.random.split(state.key)
            node_index = state.next_node_index
            action = action_selection_fn(
                self.tree,
                current_node_index=node_index,
                current_depth=state.depth,
            )

            next_node_index = self.tree.children_index[state.node_index, action]
            continue_simulation = jnp.logical_and(
                state.depth + 1 < self.max_depth, next_node_index == UNVISITED_NODE
            )
            next_state = SimulationState(
                node_index=node_index,
                action=action,
                next_node_index=next_node_index,
                depth=state.depth + 1,
                continue_simulation=continue_simulation,
                key=subkey,
            )
            return next_state

        initial_state = SimulationState(
            node_index=NO_PARENT,
            action=NO_PARENT,
            next_node_index=ROOT_NODE,
            depth=jnp.array(0, dtype=jnp.int32),
            continue_simulation=jnp.array(True),
            key=key,
        )

        end_state = jax.lax.while_loop(
            cond_fun=(lambda s: s.continue_simulation),
            body_fun=body,
            init_val=initial_state,
        )
        return end_state


def get_root_fn():
    env = RandomWalkEnv()
    root = env.reset()
    value = jnp.array(0)
    params = jnp.array(0)
    return RootFnOutput(params=params, value=value, state_embedding=root)


key = jax.random.key(2)

mcts = MCTS(n_simulations=4, n_actions=2, root_fn=get_root_fn())


def action_selected_fn(
    tree: Tree, current_node_index: Int[Array, ""], current_depth: Int[Array, ""]
) -> Array:
    visits_parent = tree.node_visits[current_node_index]
    children_values = tree.children_values[current_node_index]
    children_visits = tree.children_visits[current_node_index]

    ucbs = jax.vmap(ucb1, in_axes=(0, None, 0))(
        children_values, visits_parent, children_visits
    )
    return jnp.argmax(ucbs)


end_state = mcts._simulation(action_selection_fn=action_selected_fn, key=key)
print(end_state)
