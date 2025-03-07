import random
from typing import Set, List, Callable, Type, Tuple


class Node:
    name: str
    fitness: float

    def __call__(self, case_i: int = 0) -> float:
        raise NotImplementedError()

    def __repr__(self, tabs: int = 0) -> None:
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, node: "Node") -> bool:
        return self.__hash__() == node.__hash__()


class Operator(Node):
    name: str
    parent: Node = None
    child1: Node = None
    child2: Node = None
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
    parent: Node = None

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
