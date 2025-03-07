import random
from typing import Set, List, Callable, Type

from functions import add, subtract, multiply, divide, pow

from nodes import (
    Node,
    Leaf,
    Operator,
    LeafConstructor,
    OperatorConstructor,
    ValueListLeaf,
    ValueListLeafConstructor,
    RandomIntLeaf,
    RandomIntLeafConstructor,
)


def generate_random_tree(
    OperatorConstructors: List[OperatorConstructor],
    LeafConstructors: List[LeafConstructor],
    min_depth: int = 1,
    max_depth: int = 6,
) -> Node:
    depth = random.randint(min_depth, max_depth)

    if depth == 1:
        return random.choice(LeafConstructors)()

    previous_level_nodes: List[Node] = []
    current_level_nodes: List[Node] = []

    for i in range(depth):
        current_level_nodes = []

        current_level_node_count = 2 ** (depth - i - 1)

        if i == 0:
            for _ in range(current_level_node_count):
                node: Leaf = random.choice(LeafConstructors)()
                current_level_nodes.append(node)

        else:
            for j in range(current_level_node_count):
                node: Operator = random.choice(OperatorConstructors)()
                node.child1 = previous_level_nodes[2 * j]
                node.child2 = previous_level_nodes[2 * j + 1]
                current_level_nodes.append(node)

        previous_level_nodes = current_level_nodes

    assert len(current_level_nodes) == 1
    return current_level_nodes[0]


operators: List[OperatorConstructor] = [
    OperatorConstructor("add", add),
    OperatorConstructor("subtract", subtract),
    OperatorConstructor("multiply", multiply),
    OperatorConstructor("divide", divide),
    OperatorConstructor("pow", pow),
]
leaves = [
    ValueListLeafConstructor("x", [0, 1, -1, 2]),
    RandomIntLeafConstructor("c", min_value=-10, max_value=10),
]

tree = generate_random_tree(operators, leaves, min_depth=2, max_depth=3)
print(tree)

for i in range(4):
    print(tree(case_i=i))
