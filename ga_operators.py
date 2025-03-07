from typing import Set, List

from nodes import (
    Node,
    LeafConstructor,
    OperatorConstructor,
    select_random_node,
    compute_depth,
    generate_random_tree,
)


def remove_duplicates(population: List[Node]) -> List[Node]:
    return list(set(population))


def crossover(tree1: Node, tree2: Node) -> None:
    swap_point1: Node = select_random_node(tree1)

    swap_point_depth = compute_depth(swap_point1)

    swap_point2: Node = select_random_node(tree2, max_depth=swap_point_depth)

    # do nothing if swap point is root
    if swap_point1.parent is None or swap_point2.parent is None:
        return

    parent1 = swap_point1.parent
    parent2 = swap_point2.parent

    if not (parent1.child1 is swap_point1 or parent1.child2 is swap_point1) or not (
        parent2.child1 is swap_point2 or parent2.child2 is swap_point2
    ):
        raise AssertionError()

    # TODO: Refactor
    if parent1.child1 is swap_point1:
        parent1.child1 = swap_point2
        swap_point2.parent = parent1
    else:
        parent1.child2 = swap_point2
        swap_point2.parent = parent1

    if parent2.child1 is swap_point2:
        parent2.child1 = swap_point1
        swap_point1.parent = parent2
    else:
        parent2.child2 = swap_point1
        swap_point1.parent = parent2


# TODO: Avoid passing through parameters
def mutate(
    tree: Node,
    OperatorConstructors: List[OperatorConstructor],
    LeafConstructors: List[LeafConstructor],
) -> None:
    swap_point: Node = select_random_node(tree)
    swap_point_depth = compute_depth(swap_point)

    # do nothing if swap point is root
    if swap_point.parent is None:
        return

    random_tree = generate_random_tree(
        OperatorConstructors, LeafConstructors, max_depth=swap_point_depth
    )
    if swap_point.parent.child1 is swap_point:
        swap_point.parent.child1 = random_tree
        random_tree.parent = swap_point.parent

    elif swap_point.parent.child2 is swap_point:
        swap_point.parent.child2 = random_tree
        random_tree.parent = swap_point.parent

    else:
        raise AssertionError()


def select(population: List[Node], k: int) -> List[None]:
    population.sort(key=lambda tree: tree.fitness, reverse=False)
    return population[:k]
