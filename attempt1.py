import matplotlib.pyplot as plt
import numpy as np
from beartype.typing import Callable, NamedTuple


def ucb1(avg_node_value, visits_parent, visits_node, exploration_exploitation_factor=2):
    return avg_node_value + exploration_exploitation_factor * np.sqrt(
        np.log(visits_parent) / visits_node
    )


class StepReturn(NamedTuple):
    board: np.ndarray
    reward: int
    done: bool


n_actions = 9


def reset():
    return np.zeros((3, 3))


def action_to_index(action: int) -> tuple[int, int]:
    return action // 3, action % 3


def step(board: np.ndarray, action: int) -> StepReturn:
    turn = get_turn(board)
    action_index = action_to_index(action)
    board_copy = np.copy(board)
    if board_copy[action_index] == 0:
        board_copy[action_index] = turn
        if check_draw(board_copy):
            return StepReturn(board_copy, 0, True)
        if check_winner(board_copy):
            return StepReturn(board_copy, turn, True)
        return StepReturn(board_copy, 0, False)
    return StepReturn(board_copy, 0, False)


def random_action(board) -> int:
    actions = legal_actions(board)
    if len(actions) == 0:
        raise ValueError("No legal actions")
    return np.random.choice(actions)


def get_turn(board: np.ndarray):
    return 1 if np.sum(board) == 0 else -1


def check_draw(board: np.ndarray):
    return np.all(board != 0)


def check_winner(board: np.ndarray):
    for i in range(3):
        if board[i, 0] == board[i, 1] == board[i, 2] != 0:
            return True
        if board[0, i] == board[1, i] == board[2, i] != 0:
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return True
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return True
    return False


def legal_actions(board: np.ndarray):
    return np.where(board.flatten() == 0)[0]


def render(board: np.ndarray):
    # make white borders between cells
    plt.imshow(np.ones((3, 3, 3)))
    for i in range(1, 3):
        plt.plot([0, 3], [i, i], "k")
        plt.plot([i, i], [0, 3], "k")
    for i in range(3):
        for j in range(3):
            if board[i, j] == 1:
                plt.text(j + 0.5, i + 0.5, "1", fontsize=50, ha="center", va="center")
            elif board[i, j] == -1:
                plt.text(j + 0.5, i + 0.5, "-1", fontsize=50, ha="center", va="center")
    plt.axis("off")
    plt.show()


def rollout_randomly_until_done(board: np.ndarray, render_env: bool = False):
    ret = StepReturn(board, 0, False)
    if render_env:
        render(ret.board)
    while not ret.done:
        actions = legal_actions(ret.board)
        action = np.random.choice(actions)
        ret = step(ret.board, action)

        if render_env:
            render(ret.board)

    return ret.reward, ret.board


def play_sequence_of_actions(actions: list[int], render_env: bool = False):
    board = reset()
    ret = StepReturn(board, 0, False)
    if render_env:
        render(ret.board)
    for action in actions:
        ret = step(ret.board, action)
        if render_env:
            render(ret.board)
        if ret.done:
            break
    return ret.reward


def play_draw(render_env: bool = False):
    actions_to_play_a_draw = [4, 0, 1, 2, 5, 3, 6, 7, 8]
    return play_sequence_of_actions(actions_to_play_a_draw, render_env)


def play_winner_1(render_env: bool = False):
    actions_to_play_winner_1 = [4, 0, 1, 2, 5, 3, 6, 8, 7]
    return play_sequence_of_actions(actions_to_play_winner_1, render_env)


def play_winner_2(render_env: bool = False):
    actions_to_play_winner_2 = [2, 0, 1, 3, 4, 6]
    return play_sequence_of_actions(actions_to_play_winner_2, render_env)


ROOT_NODE = 0
UNVISITED_NODE = -1
NO_PARENT = -1


class StepFnOutput(NamedTuple):
    # reward: float
    embedding: np.ndarray
    value: float


class RootFnOutput(NamedTuple):
    embedding: np.ndarray


class Tree:
    def __init__(self, root_fn_output: RootFnOutput, n_actions: int, n_nodes: int):
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.nodes = np.ones(shape=(n_nodes), dtype=int) * UNVISITED_NODE
        self.nodes[ROOT_NODE] = ROOT_NODE
        self.node_visits = np.zeros(shape=(n_nodes), dtype=int)
        self.node_values = np.zeros(shape=(n_nodes), dtype=float)
        self.child_nodes = (
            np.ones(shape=(n_nodes, n_actions), dtype=int) * UNVISITED_NODE
        )
        self.child_values = np.zeros(shape=(n_nodes, n_actions))
        self.parent_nodes = np.ones(shape=(n_nodes), dtype=int) * NO_PARENT

        self.embeddings = {ROOT_NODE: root_fn_output.embedding}


class MCTS:
    action_selection_fn: Callable
    step_fn: Callable[[np.ndarray, int], StepFnOutput]
    root_fn: Callable[[], RootFnOutput]
    tree: Tree

    n_actions: int

    n_simulations: int

    def __init__(
        self,
        n_simulations: int,
        n_actions: int,
        action_selection_fn: Callable,
        step_fn: Callable[[np.ndarray, int], StepFnOutput],
        root_fn: Callable[[], RootFnOutput],
    ):
        self.tree = Tree(
            root_fn_output=root_fn(), n_actions=n_actions, n_nodes=n_simulations + 1
        )
        self.n_actions = n_actions
        self.n_simulations = n_simulations

        self.action_selection_fn = action_selection_fn
        self.step_fn = step_fn
        self.root_fn = root_fn

    def search(self, render_env: bool = False):
        for i in range(self.n_simulations):
            parent_node, action = self.simulate()
            self.expand(parent_node, action)
            if render_env:
                pass

    def simulate(self):
        # simulate until leaf node is reached

        class SearchState(NamedTuple):
            node: int

        def tree_traversal(state: SearchState) -> tuple[int, int]:
            node = state.node
            if UNVISITED_NODE in self.tree.child_nodes[node]:
                unvisited_actions = [
                    i
                    for i, child in enumerate(self.tree.child_nodes[node])
                    if child == UNVISITED_NODE
                ]
                return node, np.random.choice(unvisited_actions)
            else:
                action = self.action_selection_fn(self.tree, node)
                next_node = self.tree.child_nodes[node][action]
                return tree_traversal(SearchState(next_node))

        initial_state = SearchState(node=ROOT_NODE)
        final_state = tree_traversal(initial_state)
        return final_state

    def expand(self, node: int, action: int):
        self.tree.node_visits[node] += 1
        next_node = int(np.max(self.tree.nodes)) + 1
        self.tree.child_nodes[node][action] = next_node

        self.tree.nodes[next_node] = next_node

        step_fn_output = self.step_fn(self.tree.embeddings[node], action)
        self.tree.child_values[node][action] = step_fn_output.value

        self.tree.embeddings[next_node] = step_fn_output.embedding
        self.tree.parent_nodes[next_node] = node
        self.tree.node_values[next_node] = step_fn_output.value
        self.tree.node_visits[next_node] = 1

    def backward(self):
        pass


def inner_action_selection_fn(tree: Tree, node: int):
    parent_node = node
    best_action = None
    best_value = float("-inf")
    for action in range(len(tree.child_nodes[parent_node])):
        child = tree.child_nodes[parent_node][action]
        if child == UNVISITED_NODE:
            return action  # Immediately return an unvisited action

        value = ucb1(
            tree.node_values[child],
            visits_node=tree.node_visits[child],
            visits_parent=tree.node_visits[parent_node],
        )
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def recurrent_step_fn(embeddings: np.ndarray, action: int):
    step_fn_output = step(embeddings, action)
    value, final_board = rollout_randomly_until_done(step_fn_output.board)
    return StepFnOutput(step_fn_output.board, value)


def root_fn() -> RootFnOutput:
    return RootFnOutput(reset())


n_simulations = 30

mcts = MCTS(
    n_simulations,
    n_actions,
    action_selection_fn=inner_action_selection_fn,
    step_fn=recurrent_step_fn,
    root_fn=root_fn,
)


mcts.search()
