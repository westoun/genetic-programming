import random
from copy import deepcopy
from typing import Set, List, Callable, Type
from statistics import mean
from functools import partial

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


def compute_depth(tree: Node) -> int:
    if issubclass(tree.__class__, Leaf):
        return 1
    else:
        return 1 + max(compute_depth(tree.child1), compute_depth(tree.child2))


def compute_fitness(tree: Node, targets: List[float]):
    deviations = []

    for i, target in enumerate(targets):
        y = tree(case_i=i)

        try:
            deviation = abs(y - target)
            deviations.append(deviation)
        except OverflowError:
            deviations.append(10000)

    tree.fitness = max(deviations)


def flatten_tree(tree: Node) -> List[Node]:
    if issubclass(tree.__class__, Leaf):
        return [tree]
    else:
        flattened_tree = []
        flattened_tree.extend(flatten_tree(tree.child1))
        flattened_tree.extend(flatten_tree(tree.child2))
        return flattened_tree


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


def select(population: List[Node], k: int, tourn_size: int = 2) -> List[None]:
    selection = []

    for _ in range(k):
        contestors = [random.choice(population) for _ in range(tourn_size)]
        scores = [tree.fitness for tree in contestors]
        min_score = min(scores)
        min_idx = scores.index(min_score)

        winner = contestors[min_idx]
        selection.append(deepcopy(winner))

    return selection


if __name__ == "__main__":
    GENERATIONS: int = 100
    POPULATION_SIZE: int = 100
    MUTATION_PROB: float = 0.3
    CROSSOVER_PROB: float = 0.5
    MAX_DEPTH: int = 4

    INPUTS: List[float] = [-2, -1, 0, 1, 2]
    TARGETS: List[float] = [1, -2, -3, -2, 1]

    OPERATORS: List[OperatorConstructor] = [
        OperatorConstructor("add", add),
        OperatorConstructor("subtract", subtract),
        OperatorConstructor("multiply", multiply),
        OperatorConstructor("divide", divide),
        OperatorConstructor("pow", pow),
    ]
    LEAVES = [
        ValueListLeafConstructor("x", INPUTS),
        RandomIntLeafConstructor("c", min_value=-10, max_value=10),
    ]

    population: List[Node] = [
        generate_random_tree(OPERATORS, LEAVES, min_depth=1, max_depth=MAX_DEPTH)
        for _ in range(POPULATION_SIZE)
    ]
    for tree in population:
        compute_fitness(tree, targets=TARGETS)

    for generation in range(GENERATIONS):

        offspring = deepcopy(population)

        for tree1, tree2 in zip(offspring[:-1], offspring[1:]):
            if random.random() < CROSSOVER_PROB:
                crossover(tree1, tree2)

        for tree in offspring:
            if random.random() < MUTATION_PROB:
                mutate(tree, OPERATORS, LEAVES)

        for tree in offspring:
            compute_fitness(tree, targets=TARGETS)

        population = select(population=population + offspring, k=POPULATION_SIZE)

        mean_fitness = mean([tree.fitness for tree in population])
        best_fitness = min([tree.fitness for tree in population])

        print("")
        print(f"Best fitness at gen={generation + 1}: {best_fitness}")
        print(f"Mean fitness at gen={generation + 1}: {mean_fitness}")

    population.sort(key=lambda tree: tree.fitness, reverse=False)
    for i in range(3):
        print("")
        print(f"{i + 1}. best tree (fitness={population[i].fitness}):")
        print(population[i])
