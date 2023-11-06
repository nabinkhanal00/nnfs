from typing import Any
from numpy._typing import NDArray
from dense import Dense
from activations import Layer, Tanh
from losses import mse, mse_prime
from network import train, predict

import numpy as np

X: NDArray[Any] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2, 1))
Y: NDArray[Any] = np.array(
    [[0], [1], [1], [0]],
).reshape((4, 1, 1))

network: list[Layer] = [Dense(2, 3), Tanh(), Dense(3, 1), Tanh()]

train(network, mse, mse_prime, X, Y, 100000, 0.01, False)

output = predict(network, [[0], [1]])

print(output)
