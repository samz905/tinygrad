import numpy as np
from engine_vector import Unit

class Neuron:
    def __init__(self, nin):
        self.w = [Unit(np.random.random()) for _ in range(nin)]
        self.b = Unit(np.random.random(1))

    def parameters(self):
        return self.w + [self.b]
    
    def __call__(self, x):
        if not isinstance(x, list):
            x = [Unit(xi) for xi in x]
        activation = sum([wi * xi for wi, xi in zip(self.w, x)]) + self.b
        return activation.relu()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [params for neuron in self.neurons for params in neuron.parameters()]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    

class MLP:
    def __init__(self, nin, nout: list):
        total = [nin] + nout
        self.layers = [Layer(total[i], total[i + 1]) for i in range(len(nout))]

    def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x