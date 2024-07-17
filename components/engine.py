class Unit:
    def __init__(self, value):
        self.value = value
        self.grad = 0.0
        self._op = ''
        self._prev = []

    def __repr__(self):
        return str(f"Unit(value={self.value})")
    
    def __add__(self, other):
        other = other if isinstance(other, Unit) else Unit(other)
        out = Unit(self.value + other.value)
        out._op = '+'
        out._prev = [self, other]

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward  
        return out
    
    def __sub__(self , other):
        return self + -other
    
    def __mul__(self, other):
        other = other if isinstance(other, Unit) else Unit(other)
        out = Unit(self.value * other.value)
        out._op = '*'
        out._prev = [self, other]

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int or float"
        out = Unit(self.value ** other)
        out._op = '**'
        out._prev = [self]

        def _backward():
            self.grad += out.grad * other * self.value ** (other - 1)

        out._backward = _backward
        return out

neuron = Unit(1.0)
print(neuron)