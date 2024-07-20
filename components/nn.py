import numpy as np
from engine_vector import Unit

class Neuron:
    def __init__(self, nin):
        self.w = Unit(np.random.random(nin,))
        self.b = Unit(np.random.random(1))

    def parameters(self):
        return [self.w, self.b]
    
    def __call__(self, x):
        if not isinstance(x, Unit):
            x = Unit(x)
        activation = (self.w * x).sum() + self.b
        output = activation.relu().data[0]
        return output


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [params for neuron in self.neurons for params in neuron.parameters()]
    
    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    

class MLP:
    def __init__(self, nin, nout: list):
        total = [nin] + nout
        self.layers = [Layer(total[i], total[i + 1]) for i in range(len(nout))]

    def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = Unit(layer(x))
        return x
    