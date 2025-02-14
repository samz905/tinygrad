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

    # Math ops - every operation must have a backward step to backprop the grads and every output must populate the _prev variables properly to backprop the grads to
    def __add__(self, other):
        if not isinstance(other, Unit):
            other = Unit(np.array(other))

        out = Unit(np.add(self.data, other.data), '+', [self, other])

        # We do += here to account for self or grad being part of multiple computational paths
        # For ex., if c = a + b happens in one path and h = a * e happens in another (different paths still leading to an output), then a will get chain rule'd from both c (self.grad += c.grad) and h (self.grad += e.value (deriv) * h.grad (chain rule)) 
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
            other = Unit(other)

        prod = np.dot(self.data, other.data)
        out = Unit(prod[0] if isinstance(prod, np.ndarray) else prod, 'dot product', [self, other])

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

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
            self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        return out

    
    @staticmethod
    def concat(units):
        data = np.array([u.data for u in units])
        out = Unit(data, 'concat', units)
        
        def _backward():
            for i, u in enumerate(units):
                # Since each unit has a grad of the same shape, we backprop the respective gradient to its corresponding unit in the data list of the output unit
                # Since each unit is just the output of the respective neuron in the layer, it now has its own gradient which can be backpropped to the neuron's inputs which as in nn are w, b and x 
                u.grad += out.grad[i]
        
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
