import numpy as np

from unimcts.mcts import NO_PARENT, ROOT_INDEX, UNVISITED, Tree


def tree_tostring(tree: Tree):
    """
    Returns a formatted string representation of the tree.
    This provides a more readable view than the default __repr__.
    """
    # Format tree statistics
    stats = []
    stats.append("Tree Statistics:")

    # Count visited nodes
    visited_nodes = sum(tree.node_visits > 0)
    stats.append(f"  Visited nodes: {visited_nodes}/{len(tree.node_visits)}")

    # Add root info if it has been visited
    if tree.node_visits[ROOT_INDEX] > 0:
        stats.append(f"  Root visits: {tree.node_visits[ROOT_INDEX]}")
        stats.append(f"  Root value: {tree.node_values[ROOT_INDEX]:.4f}")

    # Count expanded children
    expanded_children = sum(np.any(tree.children_indices != UNVISITED, axis=1))
    stats.append(f"  Nodes with children: {expanded_children}")

    # Format compact tree structure
    structure = []
    structure.append("Tree Structure:")

    def _format_node(node_idx, depth=0, max_depth=2):
        if depth > max_depth:
            return []

        indent = "  " * depth
        node_lines = []

        # Skip unvisited nodes
        if tree.node_visits[node_idx] == 0 and node_idx != ROOT_INDEX:
            return node_lines

        # Format node info
        node_text = f"{indent}Node {node_idx}"
        if node_idx == ROOT_INDEX:
            node_text += " (ROOT)"
        else:
            action = tree.action_from_parent[node_idx]
            node_text += f" (via action {action})"

        visits = tree.node_visits[node_idx]
        value = tree.node_values[node_idx]
        node_text += f", Visits: {visits}, Value: {value:.4f}"

        node_lines.append(node_text)

        # Add children (if any and if we haven't reached max depth)
        if depth < max_depth:
            children = [
                (a, child_idx)
                for a, child_idx in enumerate(tree.children_indices[node_idx])
                if child_idx != UNVISITED
            ]

            for action, child_idx in children:
                child_visits = tree.children_visits[node_idx, action]
                # Skip unvisited children
                if child_visits == 0:
                    continue

                child_value = tree.children_values[node_idx, action]
                child_text = f"{indent}  └── Action {action}: → Node {child_idx}, Visits: {child_visits}, Value: {child_value:.4f}"
                node_lines.append(child_text)

                # Recursively add the child's children
                child_lines = _format_node(child_idx, depth + 2, max_depth)
                node_lines.extend(child_lines)

        return node_lines

    structure.extend(_format_node(ROOT_INDEX))
    return "\n".join(stats + [""] + structure)


def tree_repr(tree):
    lines = ["Tree("]
    parent_shape = tree.parent_indices.shape
    children_shape = tree.children_indices.shape

    non_default_parents = np.sum(tree.parent_indices != NO_PARENT)
    non_default_children = np.sum(tree.children_indices != UNVISITED)
    non_zero_visits = np.sum(tree.node_visits > 0)

    lines.append(
        f"  parent_indices: shape={parent_shape}, non_default={non_default_parents},"
    )
    lines.append(
        f"  children_indices: shape={children_shape}, non_default={non_default_children},"
    )
    lines.append(
        f"  node_visits: shape={tree.node_visits.shape}, non_zero={non_zero_visits},"
    )

    lines.append(
        f"  embeddings: {{{', '.join(f'{k}' for k in tree.embeddings.keys())}}})"
    )

    return "\n".join(lines)
