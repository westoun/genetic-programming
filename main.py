import random
from typing import Set, List, Callable, Type


class Node:
    name: str

    def __call__(self, case_i: int = 0) -> float:
        raise NotImplementedError()

    def __repr__(self, tabs: int = 0) -> None:
        raise NotImplementedError()


class Operator(Node):
    name: str
    child1: Node
    child2: Node
    func: Callable[[float, float], float]

    def __init__(self, name: str, func: Callable[[float, float], float]):
        self.name = name
        self.func = func

    def __call__(self, case_i: int = 0) -> float:
        x1 = self.child1(case_i)
        x2 = self.child2(case_i)
        return self.func(x1, x2)

    def __repr__(self, tabs: int = 0) -> str:
        repr = "\t" * tabs + self.name + ":"
        repr += "\n" + self.child1.__repr__(tabs=tabs + 1)
        repr += "\n" + self.child2.__repr__(tabs=tabs + 1)
        return repr


class OperatorConstructor:
    name: str
    func: Callable[[float, float], float]

    def __init__(self, name: str, func: Callable[[float, float], float]):
        self.name = name
        self.func = func

    def __call__(self) -> Operator:
        return Operator(name=self.name, func=self.func)


class Leaf(Node):
    name: str

    def __call__(self, case_i: int) -> float:
        raise NotImplementedError()

    def __repr__(self, tabs: int = 0) -> str:
        repr = "\t" * tabs + self.name
        return repr


class LeafConstructor:

    def __call__(self) -> Leaf:
        raise NotImplementedError()


class ValueListLeaf(Leaf):
    name: str
    values: List[float]

    def __init__(self, name: str, values: List[float]):
        self.name = name
        self.values = values

    def __call__(self, case_i: int) -> float:
        return self.values[case_i]


class ValueListLeafConstructor(LeafConstructor):
    name: str
    values: List[float]

    def __init__(self, name: str, values: List[float]):
        self.name = name
        self.values = values

    def __call__(self) -> ValueListLeaf:
        return ValueListLeaf(self.name, self.values)


class RandomIntLeaf(Leaf):
    name: str
    value: int

    def __init__(self, name: str, min_value: int, max_value: int):
        self.name = name
        self.value = random.randint(min_value, max_value)

    def __call__(self, case_i: int) -> float:
        return self.value

    def __repr__(self, tabs: int = 0) -> str:
        repr = "\t" * tabs + f"{self.name}={self.value}"
        return repr


class RandomIntLeafConstructor(LeafConstructor):
    name: str
    min_value: int
    max_value: int

    def __init__(self, name: str, min_value: int, max_value: int):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self) -> RandomIntLeaf:
        return RandomIntLeaf(self.name, self.min_value, self.max_value)


class OptimizableLeaf(Leaf):
    name: str
    value: float

    def __init__(self, name: str, start_value: float = None):
        self.name = name

        if start_value is not None:
            self.value = start_value
        else:
            self.value = 0.5 - random.random()

    def __call__(self, case_i: int) -> float:
        return self.value


class OptimizableLeafConstructor(LeafConstructor):
    name: str
    start_value: float

    def __init__(self, name: str, start_value: float = None):
        self.name = name
        self.start_value = start_value

    def __call__(self) -> OptimizableLeaf:
        return OptimizableLeaf(self.name, self.start_value)


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


# TODO: Move to utils
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def subtract(a, b):
    return a - b


def divide(a, b):
    if b == 0:
        return 10000

    return a / b


def pow(a, b):
    return a**b


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
