from __future__ import annotations

import pandas as pd
import numpy as np
import numpy.typing as npt

from layer import Layer
from utils import ReLU, d_ReLU, mse, d_mse, sigmoid, d_sigmoid


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, hw_1_init: bool = False):
        if hw_1_init:
            self.weights = np.ones((input_size, output_size))
            self.bias = np.zeros((1, output_size))
        else:
            self.weights = np.random.rand(input_size, output_size) - 0.5
            self.bias = np.random.rand(1, output_size) - 0.5

    def forward_prop(self, input_data: npt.ArrayLike) -> npt.ArrayLike:
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_prop(self, output_err: npt.ArrayLike, lr: float) -> npt.ArrayLike:
        input_err = np.dot(output_err, self.weights.T)
        weights_err = np.dot(self.input.T, output_err)
        print(f"{'*' * 13}\nDLoss/DW:\n{weights_err}\n{'*' * 13}\n")

        self.weights -= lr * weights_err
        self.bias -= lr * output_err
        return input_err


class ActivationLayer(Layer):
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    def forward_prop(self, input_data: npt.ArrayLike) -> npt.ArrayLike:
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_prop(self, output_err: npt.ArrayLike, lr) -> npt.ArrayLike:
        return self.d_activation(self.input) * output_err


class Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.d_loss = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def use(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss

    def predict(self, input_data: npt.ArrayLike) -> list:
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            result.append(output)

        return result

    def fit(self, x_train: npt.ArrayLike, y_train: npt.ArrayLike, epochs: int, lr: float):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                err += self.loss(y_train[j], output)

                error = self.d_loss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, lr)

            err /= samples
            print(f"Epoch {i + 1}/{epochs} >> error={err}")


def train_hw1_net(activation_function: str = 'ReLU') -> None:
    """
    Train the required net from HW1 Q4. During the training pipeline all required
    outputs will be printed.
    """
    assert activation_function in ACT_FUNCTIONS.keys(), \
        "The provided activation function is available"

    X_TRAIN = np.array([[[1, 2, -1]]])
    Y_TRAIN = np.array([[[0]]])
    ACT_FUNC = ACT_FUNCTIONS[activation_function]

    net_hw1 = Network()
    net_hw1.add(FullyConnectedLayer(3, 2, hw_1_init=True))
    net_hw1.add(ActivationLayer(*ACT_FUNC))
    net_hw1.add(FullyConnectedLayer(2, 2, hw_1_init=True))
    net_hw1.add(ActivationLayer(*ACT_FUNC))
    net_hw1.add(FullyConnectedLayer(2, 1, hw_1_init=True))

    net_hw1.use(mse, d_mse)
    net_hw1.fit(X_TRAIN, Y_TRAIN, epochs=EPOCHS, lr=LEARNING_RATE)

    ctr = 1
    for i, layer in enumerate(net_hw1.layers):
        if i in [0, 2, 4]:
            sep = f"{'-' * 13}"
            print(f"{sep}\n{' ' * 3}Layer {ctr}\n{sep}\nWeights:\n{layer.weights}\nBiases:\n{layer.bias}\n")
            ctr += 1


if __name__ == '__main__':
    ACT_FUNCTIONS = {
        'ReLU': (ReLU, d_ReLU),
        'sigmoid': (sigmoid, d_sigmoid)
    }

    LEARNING_RATE = 0.1
    EPOCHS = 1

    # ReLU
    train_hw1_net(activation_function='ReLU')

    # Sigmoid
    train_hw1_net(activation_function='sigmoid')
