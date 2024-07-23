## A tiny implementation of PyTorch's autograd engine
Based on Karpathy sensei's [micrograd](https://github.com/karpathy/micrograd) engine, this project implements from scratch, a minimalist neural network library featuring backpropagation through automatic differentiation on a dynamically built DAG. I wanted to take my learnings to the next level so I added a few improvements: 
1. Incorporated full support for vector values and mathematical operations like dot product and concat.
2. Convenient methods in the MLP class to predict outputs and train neural networks.

### Using tinygrad
Trying out on a tiny dataset
```
xs = [
    [2, 3, -1],
    [3, -1, 0.5],
    [0.5, 1, 1],
    [1, 1, -1],
]

ys = [1, -1, -1, 1]
```

Initialising the network
```
# 3 featured input shape, (5, 4, 1) neural network shape
n = MLP(3, [5, 4, 1])
```

Making predictions and calculating the RMS loss
```
predictions = n.predict(xs)
loss_before_training = n.loss(predictions, ys).data
print(f"Loss before training: {loss_before_training} and predictions: {predictions}")

# prints Loss before training: 14.646449126643851 and predictions: [17.067146324690277, 15.92326097064621, 15.238256397218107, 8.059954982580635]
```

Training the network
```
loss_after_training = n.train(xs, ys, iterations=1000, lr=0.01)
print(f"Loss after training: {loss_after_training['loss']} and predictions: {loss_after_training['predictions']}")

# prints Loss after training: 0.7102683487043046 and predictions: [1.0597375025781735, 0.0, 0.0, 0.8801837261876291]
```