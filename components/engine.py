import math

class Unit:
    def __init__(self, value, _op='', _prev=[]):
        self.value = value
        self.grad = 0.0
        self._op = _op
        self._prev = _prev


    def __repr__(self):
        return str(f"Unit(value={self.value})")
    

    # Math ops
    def __add__(self, other):
        other = other if isinstance(other, Unit) else Unit(other)
        out = Unit((self.value + other.value), '+', [self, other])

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward  
        return out
    

    def __radd__(self, other):
        return self + other
    

    def __sub__(self , other):
        return self + (-other)
    

    def __mul__(self, other):
        other = other if isinstance(other, Unit) else Unit(other)
        out = Unit((self.value * other.value), '*', [self, other])

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

        out._backward = _backward
        return out


    def __rmul__(self, other):
        return self * other


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int or float"
        out = Unit((self.value ** other), '^', [self])

        def _backward():
            self.grad += out.grad * other * self.value ** (other - 1)

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
        out = Unit((math.exp(self.value)), 'exp', [self])

        def _backward():
            self.grad += out.value * out.grad

        out._backward = _backward
        return out
    

    # Non-linearity functions
    def sigmoid(self):
        out = Unit((1 / (1 + math.exp(-self.value))), 'sigmoid', [self])

        def _backward():
            self.grad += out.value * out.grad

        out._backward = _backward
        return out
    

    def tanh(self):
        out = Unit(((math.exp(2*self.value) - 1) / (math.exp(2 * self.value) + 1)), 'tanh', [self])

        def _backward():
            self.grad += (1 - out.value ** 2) * out.grad

        out._backward = _backward
        return out
    

    def relu(self):
        out = Unit(max(0, self.value), 'relu', [self])

        def _backward():
            self.grad += (out.value > 0) * out.grad

        out._backward = _backward
        return out
    

    # Backward pass
    def backward(self):
        # topological order all of the nodes or children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # This function is expected to be called on the output node to initiate backprop from there so o.backward() will initiate backprop. Remember that we are finding grads for each weight as the derivative of the output w.r.t to that weight and for the output itself, do/do = 1
        self.grad = 1.0

        for node in reversed(topo): # Reversed because the list is ordered from input layer to output layer and we wanna go in the backwards direction starting from the output for backprop
            node._backward()



# neuron1 = Unit(1.0)
# neuron2 = Unit(2.0)
# exp = neuron2.exp()
# print(exp)