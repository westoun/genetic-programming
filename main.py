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
    generate_random_tree,
)

from ga_operators import crossover, mutate, select, remove_duplicates


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


if __name__ == "__main__":
    GENERATIONS: int = 100
    POPULATION_SIZE: int = 100
    MUTATION_PROB: float = 0.2
    CROSSOVER_PROB: float = 0.5
    MAX_DEPTH: int = 4

    FITNESS_THRESHOLD: float = 0.000005

    INPUTS: List[float] = [-2, -1, 0, 1, 2]
    TARGETS: List[float] = [1, -2, -3, -2, 1]

    OPERATORS: List[OperatorConstructor] = [
        OperatorConstructor("add", add),
        OperatorConstructor("subtract", subtract),
        OperatorConstructor("multiply", multiply),
        OperatorConstructor("divide", divide),
        # OperatorConstructor("pow", pow),
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

        # Shuffle to avoid crossover in the same proximity across
        # generations.
        random.shuffle(offspring)

        for tree1, tree2 in zip(offspring[:-1], offspring[1:]):
            if random.random() < CROSSOVER_PROB:
                crossover(tree1, tree2)

        for tree in offspring:
            if random.random() < MUTATION_PROB:
                mutate(tree, OPERATORS, LEAVES)

        for tree in offspring:
            compute_fitness(tree, targets=TARGETS)

        population = select(population=population + offspring, k=POPULATION_SIZE)

        population = remove_duplicates(population)
        for _ in range(POPULATION_SIZE - len(population)):
            new_tree = generate_random_tree(
                OPERATORS, LEAVES, min_depth=1, max_depth=MAX_DEPTH
            )
            compute_fitness(new_tree, targets=TARGETS)
            population.append(new_tree)

        mean_fitness = mean([tree.fitness for tree in population])
        best_fitness = min([tree.fitness for tree in population])

        print("")
        print(f"Best fitness at gen={generation + 1}: {best_fitness}")
        print(f"Mean fitness at gen={generation + 1}: {mean_fitness}")

        if best_fitness < FITNESS_THRESHOLD:
            print(
                f"\nBreaking at gen={generation + 1} since best fitness below threshold."
            )
            break

    population.sort(key=lambda tree: tree.fitness, reverse=False)
    for i in range(3):
        print("")
        print(f"{i + 1}. best tree (fitness={population[i].fitness}):")
        print(population[i])
