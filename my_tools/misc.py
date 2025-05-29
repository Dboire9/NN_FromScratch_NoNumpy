from math import exp

def my_sigmoid(x):
    """Sigmoid activation function."""
    from my_tools.VectorMatrixClass import Vector, Matrix
    if isinstance(x, Vector):
        return Vector([1 / (1 + exp(-1 * v)) for v in x])
    elif isinstance(x, Matrix):
        return Matrix([[1 / (1 + exp(-v)) for v in row] for row in x.rows])
    elif hasattr(x, "__iter__") and not isinstance(x, str):
        return type(x)([1 / (1 + exp(-1 * v)) for v in x])
    else:
        return 1 / (1 + exp(-1 * x))