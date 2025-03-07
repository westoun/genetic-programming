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
    
    def __repr__(self, tabs: int = 0) -> str:
        repr = "\t" * tabs + f"{self.name}={round(self.value, 2)}"
        return repr


class OptimizableLeafConstructor(LeafConstructor):
    name: str
    start_value: float

    def __init__(self, name: str, start_value: float = None):
        self.name = name
        self.start_value = start_value

    def __call__(self) -> OptimizableLeaf:
        return OptimizableLeaf(self.name, self.start_value)
