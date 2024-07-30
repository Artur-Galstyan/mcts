import os
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import pgx
import treescope
from beartype import beartype
from beartype.typing import Any, Callable, NamedTuple, Protocol, runtime_checkable
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from loguru import logger


def jax_log(fmt: str, *args, **kwargs):
    jax.debug.callback(
        lambda *args, **kwargs: logger.info(fmt.format(*args, **kwargs)),
        *args,
        **kwargs,
    )


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
    n_actions: int = 2

    def __init__(self) -> None:
        pass

    def reset(self) -> Array:
        return jnp.array(0)

    def step(self, state: Array, action: Array) -> tuple[Array, Array, Bool]:
        def state_0(action):
            next_state = jnp.where(action == 0, 1, 2)
            reward = jnp.zeros_like(action)
            done = jnp.zeros_like(action, dtype=jnp.bool_)
            return next_state, reward, done

        def state_1(action):
            next_state = jnp.where(action == 0, 3, 4)
            reward = jnp.zeros_like(action)
            done = jnp.ones_like(action, dtype=jnp.bool_)
            return next_state, reward, done

        def state_other(action):
            next_state = jnp.where(action == 0, 5, 6)
            reward = jnp.where(action == 0, 1, 0)
            done = jnp.ones_like(action, dtype=jnp.bool_)
            return next_state, reward, done

        return jax.lax.switch(state, [state_0, state_1, state_other], action)


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


class RootFnOutput(NamedTuple):
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


class Tree(eqx.Module):
    node_values: Float[Array, " N"]
    node_visits: Int[Array, " N"]
    parent_indices: Int[Array, " N"]
    action_from_parent: Int[Array, " N"]
    children_index: Int[Array, "N A"]
    children_visits: Int[Array, "N A"]
    children_rewards: Float[Array, "N A"]
    children_values: Float[Array, "N A"]
    children_discounts: Float[Array, "N A"]
    children_priors: Float[Array, "N A"]

    N: int = eqx.field(static=True)
    A: int = eqx.field(static=True)

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


class SearchState(NamedTuple):
    tree: Tree
    state_embedding: PyTree
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

    def run(
        self,
        action_selection_fn: ActionSelectionFn,
        step_fn: StepFnCallable,
        key: PRNGKeyArray,
    ) -> Tree:
        logger.info("Running MCTS")

        def body(i: int, state: SearchState):
            key, simulation_key, expand_key, next_key = jax.random.split(state.key, 4)
            jax_log("MCTS.run.body: Search iteration {i}", i=i)
            end_state = self._simulation(action_selection_fn, simulation_key)
            parent_index, action = end_state.node_index, end_state.action
            jax_log(
                "MCTS.run.body: Simulation end state: parent_index={parent_index}, action={action}",
                parent_index=parent_index,
                action=action,
            )
            next_node_index = state.tree.children_index[parent_index, action]
            next_node_index = jnp.where(
                next_node_index == UNVISITED_NODE,
                i + 1,
                next_node_index,
            )
            jax_log(
                "MCTS.run.body: Next node index: {next_node_index}",
                next_node_index=next_node_index,
            )
            tree, next_state_embedding = self._expand(
                step_fn,
                state.tree,
                parent_index,
                action,
                next_node_index,
                state.state_embedding,
                expand_key,
            )
            return SearchState(tree, next_state_embedding, next_key)

        initial_state = SearchState(self.tree, self.root_fn_output.state_embedding, key)
        final_state = jax.lax.fori_loop(0, self.n_simulations, body, initial_state)
        return final_state.tree

    def _expand(
        self,
        step_fn: StepFnCallable,
        tree: Tree,
        parent_index: Int[Array, ""],
        action_to_expand: Int[Array, ""],
        next_node_index: Int[Array, ""],
        state_embedding: PyTree,
        key: PRNGKeyArray,
    ) -> tuple[Tree, PyTree]:
        key, subkey = jax.random.split(key)
        jax_log(
            "MCTS._expand: Expanding node {parent_index} with action {action_to_expand}",
            parent_index=parent_index,
            action_to_expand=action_to_expand,
        )
        step_fn_output = step_fn(
            tree.root_fn_output.params,
            action_to_expand,
            state_embedding,
            subkey,
        )
        # reward: Array
        # discount: Array
        # value: Array
        # priors: Array

        # state_embedding: PyTree

        jax_log(
            "MCTS._expand: Step function output: reward={reward}, discount={discount}, value={value}, priors={priors}, state_embedding={state_embedding}",
            reward=step_fn_output.reward,
            discount=step_fn_output.discount,
            value=step_fn_output.value,
            priors=step_fn_output.priors,
            state_embedding=state_embedding,
        )

        tree = eqx.tree_at(
            lambda t: t.node_values,
            tree,
            tree.node_values.at[next_node_index].set(step_fn_output.value),
        )

        tree = eqx.tree_at(
            lambda t: t.children_rewards,
            tree,
            tree.children_rewards.at[next_node_index, action_to_expand].set(
                step_fn_output.reward
            ),
        )

        tree = eqx.tree_at(
            lambda t: t.children_discounts,
            tree,
            tree.children_discounts.at[next_node_index, action_to_expand].set(
                step_fn_output.discount
            ),
        )

        tree = eqx.tree_at(
            lambda t: t.children_priors,
            tree,
            tree.children_priors.at[next_node_index, action_to_expand].set(
                step_fn_output.priors
            ),
        )

        tree = eqx.tree_at(
            lambda t: t.node_visits,
            tree,
            tree.node_visits.at[parent_index].add(1),
        )

        tree = eqx.tree_at(
            lambda t: t.children_index,
            tree,
            tree.children_index.at[parent_index, action_to_expand].set(next_node_index),
        )

        tree = eqx.tree_at(
            lambda t: t.parent_indices,
            tree,
            tree.parent_indices.at[next_node_index].set(parent_index),
        )

        tree = eqx.tree_at(
            lambda t: t.action_from_parent,
            tree,
            tree.action_from_parent.at[next_node_index].set(action_to_expand),
        )

        return tree, step_fn_output.state_embedding

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
            jax_log(
                "MCTS._simulation.body: Simulation depth {depth}", depth=state.depth
            )
            jax_log(
                "MCTS._simulation.body: Simulation node_index {node_index}",
                node_index=node_index,
            )

            action = action_selection_fn(
                self.tree,
                current_node_index=node_index,
                current_depth=state.depth,
            )

            jax_log("MCTS._simulation.body: Simulation action {action}", action=action)

            next_node_index = self.tree.children_index[state.node_index, action]

            jax_log(
                "MCTS._simulation.body: Simulation next_node_index {next_node_index}",
                next_node_index=next_node_index,
            )

            continue_simulation = jnp.logical_and(
                state.depth + 1 < self.max_depth, next_node_index != UNVISITED_NODE
            )

            jax_log(
                "MCTS._simulation.body: Simulation continue_simulation {continue_simulation}",
                continue_simulation=continue_simulation,
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


def rollout_randomly(
    env, current_state: jnp.ndarray, key: jnp.ndarray, max_steps: int = 1000
):
    def body_fun(carry, _):
        current_state, total_reward, key, done = carry
        key, subkey = jax.random.split(key)
        action = jax.random.randint(subkey, (), 0, env.n_actions)
        next_state, reward, new_done = env.step(current_state, action)

        # Update only if not already done
        current_state = jnp.where(done, current_state, next_state)
        total_reward = jnp.where(done, total_reward, total_reward + reward)
        done = jnp.logical_or(done, new_done)

        return (current_state, total_reward, key, done), None

    initial_carry = (current_state, jnp.array(0.0), key, jnp.array(False))
    (final_state, total_reward, final_key, done), _ = jax.lax.scan(
        body_fun, initial_carry, None, length=max_steps
    )

    return total_reward


def step_function(
    env, params: PyTree, action: Array, state_embedding: PyTree, key: PRNGKeyArray
) -> StepFnOutput:
    next_state, reward, done = env.step(state_embedding, action)
    value = rollout_randomly(env, next_state, key)
    discount = jnp.where(done, jnp.array(0.0), jnp.array(1.0))
    priors = jnp.array(0.5)
    return StepFnOutput(
        value=value,
        reward=reward,
        discount=discount,
        priors=priors,
        state_embedding=next_state,
    )


step_fn_partial = partial(step_function, RandomWalkEnv())
final_state = mcts.run(action_selected_fn, step_fn_partial, key)
# with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
#     contents = treescope.render_to_html(final_state)

# with open("/tmp/treescope_output.html", "w") as f:
#     f.write(contents)
#     webbrowser.open("file://" + os.path.realpath("/tmp/treescope_output.html"))
