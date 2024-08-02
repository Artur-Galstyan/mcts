import matplotlib.pyplot as plt
import numpy as np
from beartype.typing import Any, Callable, NamedTuple


def ucb1(avg_node_value, visits_parent, visits_node, exploration_exploitation_factor=2):
    return avg_node_value + exploration_exploitation_factor * np.sqrt(
        np.log(visits_parent) / visits_node
    )


class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.turn = 1
        self.done = False

    def reset(self):
        self.board = np.zeros((3, 3))
        self.turn = 1
        self.done = False
        return self.board

    def step(self, action):
        if self.done:
            return self.board, 0, self.done, {}
        if self.board[action] == 0:
            self.board[action] = self.turn
            if self.check_draw():
                self.done = True
                return self.board, 0, self.done, {}
            if self.check_winner():
                self.done = True
                return self.board, self.turn, self.done, {}
            self.turn = -self.turn
        return self.board, 0, self.done, {}

    def check_draw(self):
        return np.all(self.board != 0)

    def check_winner(self):
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2] != 0:
                return True
            if self.board[0, i] == self.board[1, i] == self.board[2, i] != 0:
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return True
        return False

    def legal_actions(self):
        return np.where(self.board.flatten() == 0)[0]

    def render(self):
        # make white borders between cells
        plt.imshow(np.ones((3, 3, 3)))
        for i in range(1, 3):
            plt.plot([0, 3], [i, i], "k")
            plt.plot([i, i], [0, 3], "k")
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 1:
                    plt.text(
                        j + 0.5, i + 0.5, "X", fontsize=50, ha="center", va="center"
                    )
                elif self.board[i, j] == -1:
                    plt.text(
                        j + 0.5, i + 0.5, "O", fontsize=50, ha="center", va="center"
                    )
        plt.axis("off")
        plt.show()


env = TicTacToeEnv()


def rollout_randomly_until_done(env: TicTacToeEnv):
    while not env.done:
        actions = env.legal_actions()
        action = np.random.choice(actions)
        env.step((action // 3, action % 3))
    return env.turn


ROOT_NODE = 0
UNVISITED_NODE = -1


class StepFnOutput(NamedTuple):
    reward: float
    embedding: np.ndarray
    value: float


class Tree:
    nodes: list[int]
    node_visits: dict[int, int]
    child_nodes: dict[int, list[int]]
    child_values: dict[int, float]

    embeddings: dict[int, np.ndarray]

    def __init__(self):
        self.nodes = [ROOT_NODE]
        self.node_visits = {ROOT_NODE: 0}


class MCTS:
    action_selection_fn: Callable
    step_fn: Callable[[np.ndarray, int], StepFnOutput]

    tree: Tree

    n_actions: int

    def __init__(self, n_actions: int):
        self.tree = Tree()
        self.n_actions = n_actions

    def search(self):
        pass

    def simulate(self):
        # simulate until leaf node is reached

        class SearchState(NamedTuple):
            node: int

        initial_state = SearchState(node=ROOT_NODE)

    def expand(self, node: int, action: int):
        self.tree.node_visits[node] += 1
        self.tree.child_nodes[node] = [-1 for _ in range(self.n_actions)]
        self.tree.child_nodes[node][action] = len(self.tree.nodes)
        next_node = len(self.tree.nodes)

        self.tree.nodes.append(len(self.tree.nodes))

        step_fn_output = self.step_fn(self.tree.embeddings[node], action)

        self.tree.embeddings[next_node] = step_fn_output.embedding
        self.tree.child_values[next_node] = step_fn_output.value

    def backward(self):
        pass
