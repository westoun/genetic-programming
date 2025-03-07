import random
from copy import deepcopy
from typing import Set, List, Callable, Type, Tuple
from statistics import mean
from functools import partial

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from functions import add, subtract, multiply, divide

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
    OptimizableLeaf,
    OptimizableLeafConstructor,
)
from node_utils import generate_random_tree, update_params, extract_params

from ga_operators import crossover, mutate, select, remove_duplicates


def compute_fitness(tree: Node, targets: List[float]) -> float:
    try:
        total_deviation = 0

        for i, target in enumerate(targets):
            y = tree(case_i=i)

            total_deviation += abs(y - target)

        return total_deviation
    except KeyboardInterrupt:
        raise KeyboardInterrupt()
    except:
        return np.inf


def scipy_objective_function(
    params: List[float], tree: Node, targets: List[float]
) -> float:
    update_params(tree, params)
    fitness = compute_fitness(tree, targets)
    tree.fitness = fitness
    return fitness


def evaluate(tree: Node, targets: List[float]) -> None:
    initial_params = extract_params(tree)

    if len(initial_params) == 0:
        fitness = compute_fitness(tree, targets)
        tree.fitness = fitness
        return

    bounds = [(-1000, 1000) for _ in initial_params]

    objective_function = partial(scipy_objective_function, tree=tree, targets=targets)

    optimization_result: OptimizeResult = minimize(
        objective_function,
        x0=initial_params,
        method="Nelder-Mead",
        bounds=bounds,
        tol=0,
        options={"maxiter": 10, "disp": False},
    )

    best_params = optimization_result.x
    update_params(tree, best_params)


def load_experiment_data() -> Tuple[List[ValueListLeafConstructor], List[float]]:
    import json

    with open("experiment_results.json", "r") as results_file:
        experiments = json.load(results_file)

    circuit_counts = []
    gate_counts = []
    qubit_nums = []
    redundancies = []
    cache_sizes = []
    reordering_steps = []
    merging_rounds = []

    total_durations = []

    for experiment in experiments:
        circuit_counts.append(experiment["params"]["circuit_count"])
        gate_counts.append(experiment["params"]["gate_count"])
        qubit_nums.append(experiment["params"]["qubit_num"])
        redundancies.append(experiment["params"]["redundancy"])
        cache_sizes.append(experiment["params"]["cache_size"])
        reordering_steps.append(experiment["params"]["reordering_steps"])
        merging_rounds.append(experiment["params"]["merging_rounds"])

        total_durations.append(experiment["total_duration"])

    LeafConstructors = [
        ValueListLeafConstructor("circuit count", circuit_counts),
        ValueListLeafConstructor("gate count", gate_counts),
        ValueListLeafConstructor("qubit num", qubit_nums),
        ValueListLeafConstructor("redundancy", redundancies),
        ValueListLeafConstructor("circuit size", cache_sizes),
        ValueListLeafConstructor("reordering steps", reordering_steps),
        ValueListLeafConstructor("mergning rounds", merging_rounds),
    ]
    return LeafConstructors, total_durations


if __name__ == "__main__":
    GENERATIONS: int = 200
    POPULATION_SIZE: int = 1000
    MUTATION_PROB: float = 0.3
    CROSSOVER_PROB: float = 0.5
    MAX_DEPTH: int = 8

    FITNESS_THRESHOLD: float = 0.000005

    # INPUTS: List[float] = [-2, -1, 0, 1, 2]
    # TARGETS: List[float] = [1, -2, -3, -2, 1]

    OPERATORS: List[OperatorConstructor] = [
        OperatorConstructor("add", add),
        OperatorConstructor("subtract", subtract),
        OperatorConstructor("multiply", multiply),
        OperatorConstructor("divide", divide),
    ]
    LEAVES = [
        OptimizableLeafConstructor("opt_c0"),
        OptimizableLeafConstructor("opt_c1"),
        OptimizableLeafConstructor("opt_c2"),
        OptimizableLeafConstructor("opt_c3"),
        RandomIntLeafConstructor("rand_c0", -10, 10),
        RandomIntLeafConstructor("rand_c1", -100, 100),
        RandomIntLeafConstructor("rand_c2", -1000, 1000),
        RandomIntLeafConstructor("rand_c3", -10000, 10000),
    ]

    value_list_constructors, TARGETS = load_experiment_data()
    LEAVES.extend(value_list_constructors)

    population: List[Node] = [
        generate_random_tree(OPERATORS, LEAVES, min_depth=1, max_depth=MAX_DEPTH)
        for _ in range(POPULATION_SIZE)
    ]
    for tree in population:
        evaluate(tree, targets=TARGETS)

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
            evaluate(tree, targets=TARGETS)

        population = select(population=population + offspring, k=POPULATION_SIZE)

        # population = remove_duplicates(population)
        # for _ in range(POPULATION_SIZE - len(population)):
        #     new_tree = generate_random_tree(
        #         OPERATORS, LEAVES, min_depth=1, max_depth=MAX_DEPTH
        #     )
        #     evaluate(new_tree, targets=TARGETS)
        #     population.append(new_tree)

        mean_fitness = mean([tree.fitness for tree in population])
        best_fitness = min([tree.fitness for tree in population])

        if (generation + 1) % 5 == 0:
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
