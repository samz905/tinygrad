## A tiny implementation of PyTorch's autograd engine
Based on Karpathy sensei's [micrograd](https://github.com/karpathy/micrograd) engine, this project implements from scratch, a minimalist neural network library featuring backpropagation through automatic differentiation on a dynamically built DAG. I wanted to take my learnings to the next level so I added a few improvements: 
1. Incorporated full support for vector values and mathematical operations like dot product and concat.
2. Convenient methods in the MLP class to predict outputs and train neural networks.