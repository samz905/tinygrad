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
        # The output unit of the concat operation can only perform shape-preserving or supported operations only so its output also backprops the gradients in the same shape (data and grad always have the same shape for each unit)
        # Each input unit of the concat operation is the output of the respective neuron in the layer stored as a list. Once each neuron gets its gradient, it backprops to the weights as well through w * x + b (Each neuron's weight matrix has the same shape as the input feature matrix)
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
    
    def predict(self, batch):
        return [self(x) for x in batch]
    
    def loss(self, predictions, targets):
        return (sum((prediction - target)**2 for prediction, target in zip(predictions, targets)) / len(predictions))**0.5
    
    def train(self, batch, targets, iterations=1000, lr=0.0005):
        for _ in range(iterations):
            # Forward pass
            predictions = self.predict(batch)
            rms_loss = self.loss(predictions, targets)

            #Backprop
            rms_loss.backward()

            # Gradient descent
            for param in self.parameters():
                # print(f"param: {param} and param.grad: {param.grad}")
                param.data = param.data - lr * param.grad
                param.grad = np.zeros_like(param.data)

        return {
            "loss": rms_loss.data,
            "predictions": predictions
        }


# Trying out on a tiny dataset
n = MLP(3, [5, 4, 1])

xs = [
    [2, 3, -1],
    [3, -1, 0.5],
    [0.5, 1, 1],
    [1, 1, -1],
]

ys = [1, -1, -1, 1]

predictions = n.predict(xs)
loss_before_training = n.loss(predictions, ys)

print(f"Loss before training: {loss_before_training} and predictions: {predictions}")

loss_after_training = n.train(xs, ys, iterations=1000, lr=0.01)

print(f"Loss after training: {loss_after_training['loss']} and predictions: {loss_after_training['predictions']}")
            