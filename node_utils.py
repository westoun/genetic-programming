from typing import List
import random
from nodes import Node, Leaf, OperatorConstructor, LeafConstructor


def select_random_node(
    tree: Node, min_depth: int = None, max_depth: int = None
) -> Node:
    all_nodes = flatten_tree(tree)

    if min_depth is not None:
        all_nodes = [node for node in all_nodes if compute_depth(node) >= min_depth]

    if max_depth is not None:
        all_nodes = [node for node in all_nodes if compute_depth(node) <= max_depth]

    # all nodes should include at least one leaf node with depth 1
    assert len(all_nodes) > 0

    return random.choice(all_nodes)


def compute_depth(tree: Node) -> int:
    if issubclass(tree.__class__, Leaf):
        return 1
    else:
        return 1 + max(compute_depth(tree.child1), compute_depth(tree.child2))


def flatten_tree(tree: Node) -> List[Node]:
    if issubclass(tree.__class__, Leaf):
        return [tree]
    else:
        flattened_tree = []
        flattened_tree.extend(flatten_tree(tree.child1))
        flattened_tree.extend(flatten_tree(tree.child2))
        return flattened_tree


def generate_random_tree(
    OperatorConstructors: List[OperatorConstructor],
    LeafConstructors: List[LeafConstructor],
    min_depth: int = 1,
    max_depth: int = 6,
) -> Node:
    depth = random.randint(min_depth, max_depth)

    if depth == 1:
        return random.choice(LeafConstructors)()

    NodeConstructors = OperatorConstructors + LeafConstructors
    root = random.choice(NodeConstructors)()

    current_depth_nodes = [root]
    next_depth_nodes = []

    for i in range(depth - 1):

        for node in current_depth_nodes:
            if issubclass(node.__class__, Leaf):
                continue

            child1 = random.choice(NodeConstructors)()
            child2 = random.choice(NodeConstructors)()

            node.child1 = child1
            child1.parent = node

            node.child2 = child2
            child2.parent = node

            next_depth_nodes.append(child1)
            next_depth_nodes.append(child2)

        current_depth_nodes = next_depth_nodes
        next_depth_nodes = []

    for node in current_depth_nodes:
        if issubclass(node.__class__, Leaf):
            continue

        if node.child1 is None:
            child1 = random.choice(LeafConstructors)()

            node.child1 = child1
            child1.parent = node

        if node.child2 is None:
            child2 = random.choice(LeafConstructors)()

            node.child2 = child2
            child2.parent = node

    return root
