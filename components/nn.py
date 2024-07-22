import numpy as np
from engine_vector import Unit

class Neuron:
    def __init__(self, nin):
        self.w = Unit(np.random.rand(nin))
        self.b = Unit(np.random.rand(1)[0])

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x):        
        if not isinstance(x, Unit):
            x = Unit(x)
        activation = self.w.dot(x) + self.b
        return activation.relu()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]

        # The below step converts the list of neuron outputs into a single vectorised unit while incorporating backprop through the custom concat function defined for the Unit class
        # Each unit in the input of this concat operation is the output of the respective neuron in the layer stored as a list. Once each neuron gets its gradient, it backprops to the weights as well through w * x + b
        return outputs[0] if len(outputs) == 1 else Unit.concat(outputs) # Converts a list of units into a single vectorised unit


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x