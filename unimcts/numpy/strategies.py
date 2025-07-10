from unimcts.numpy import UNVISITED, ActionSelectionInput, ActionSelectionReturn

import numpy as np


def ucb_action_selection(input: ActionSelectionInput) -> ActionSelectionReturn:
    tree = input.tree
    node = input.node_index

    visits = tree.children_visits[node]
    indices = tree.children_indices[node]

    unvisited = np.where(indices == UNVISITED)[0]

    if len(unvisited) > 0:
        action = np.random.choice(unvisited)
        return ActionSelectionReturn(action=np.array(action))

    values = tree.children_values[node]
    total_visits = tree.node_visits[node]

    exploration_term = 2.0 * np.sqrt(np.log(total_visits) / visits)
    ucb = values + exploration_term

    action = np.argmax(ucb)
    return ActionSelectionReturn(action=np.array(action))
