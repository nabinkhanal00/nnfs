from typing import Any
from numpy._typing import NDArray
from dense import Dense
from activations import Layer, Tanh
from losses import mse, mse_prime

import numpy as np

X: NDArray[Any] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2, 1))
Y: NDArray[Any] = np.array(
    [[0], [1], [1], [0]],
).reshape((4, 1, 1))

network: list[Layer] = [Dense(2, 3), Tanh(), Dense(3, 1), Tanh()]

epochs: int = 10_000
learning_rate: float = 0.1

for e in range(epochs):
    error: float = 0
    for x, y in zip(X, Y):
        output = x
        for layer in network:
            output = layer.forward(output)

        error += mse(y, output)
        grad = mse_prime(y, output)

        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    error /= len(X)
    if e % 1000 == 999:
        print(f"{e+1}/{epochs}, error={error}")
