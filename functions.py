def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def subtract(a, b):
    return a - b


def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 10000.0


def pow(a, b):
    try:
        return a**b
    except ZeroDivisionError:
        return 10000.0
    except OverflowError:
        return 10000.0