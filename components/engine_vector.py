import numpy as np

class Unit:
    def __init__(self, data, _op='', _prev=[]):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = np.zeros(self.data.shape)
        self._backward = lambda: None
        self._op = _op
        self._prev = _prev

    def __repr__(self):
        return f"Unit(data={self.data})"

    # Math ops
    def __add__(self, other):
        if not isinstance(other, Unit):
            other = Unit(np.array(other))

        out = Unit(np.add(self.data, other.data), '+', [self, other])

        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Unit):
            other = Unit(np.array(other))

        out = Unit(np.multiply(self.data, other.data), '*', [self, other])

        def _backward():
            self.grad = self.grad + out.grad * other.data
            other.grad = other.grad + out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int or float"
        out = Unit((self.data ** other), '^', [self])

        def _backward():
            self.grad = self.grad + out.grad * other * self.data ** (other - 1)

        out._backward = _backward
        return out

    def __rpow__(self, other):
        return self ** other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def exp(self):
        out = Unit((np.exp(self.data)), 'exp', [self])

        def _backward():
            self.grad = self.grad + out.data * out.grad

        out._backward = _backward
        return out

    def dot(self, other):
        if not isinstance(other, Unit):
            other = Unit(np.array(other))

        return Unit(np.dot(self.data, other.data))

    # Non-linearity functions
    def sigmoid(self):
        out = Unit((1 / (1 + np.exp(-self.data))), 'sigmoid', [self])

        def _backward():
            self.grad = self.grad + out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Unit(((np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)), 'tanh', [self])

        def _backward():
            self.grad = self.grad + (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Unit(np.maximum(0, self.data), 'relu', [self])

        def _backward():
            self.grad = self.grad + (out.data > 0) * out.grad

        out._backward = _backward
        return out


    # Backward pass
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
            