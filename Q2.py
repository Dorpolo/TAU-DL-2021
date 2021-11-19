import numpy as np


def get_loss(Y_hat, Y):
    RSS = ((Y - Y_hat) ** 2).sum()
    return RSS


def get_d_loss(Y_hat, Y):
    return 2 * (Y - Y_hat)


class ReLU:
    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ


class Sigmoid:
    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))

    def backward(self, dA, Z):
        sig = self.forward(Z)
        return dA * sig * (1 - sig)


def initialize_weigths(layers, init_random=False, seed=42):
    if init_random:
        np.random.seed(seed)
    weights = {}

    for i, layer in enumerate(layers):
        layer_idx = i + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        if init_random:
            weights['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
            weights['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
        else:
            weights['W' + str(layer_idx)] = np.ones((layer_output_size, layer_input_size))
            weights['b' + str(layer_idx)] = 0

    return weights


def single_layer_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b

    if activation is not None:
        A = activation.forward(Z)
    else:
        A = Z.copy()

    return A, Z


def full_forward(X, weights, layers):
    history = {}
    A = X

    for idx, layer in enumerate(layers):
        layer_idx = idx + 1
        A_prev = A

        activ_function = None

        if layer["activation"] == 'relu':
            activ_function = ReLU()
        elif layer["activation"] == 'sigmoid':
            activ_function = Sigmoid()

        W = weights["W" + str(layer_idx)]
        b = weights["b" + str(layer_idx)]
        A, Z = single_layer_forward(A_prev, W, b, activ_function)

        history["A" + str(idx)] = A_prev
        history["Z" + str(layer_idx)] = Z

    return A, history


def single_layer_backward(dA, W, b, Z, A_prev, activation):
    m = A_prev.shape[1]

    if activation is not None:
        dZ = activation.backward(dA, Z)
    else:
        dZ = dA.copy()

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def full_backward(Y_hat, Y, history, weights, layers):
    grads_values = {}
    Y = Y.reshape(Y_hat.shape)

    dA_prev = get_d_loss(Y_hat, Y)

    for layer_idx_prev, layer in reversed(list(enumerate(layers))):
        layer_idx = layer_idx_prev + 1

        activ_function = None
        if layer["activation"] == 'relu':
            activ_function = ReLU()
        elif layer["activation"] == 'sigmoid':
            activ_function = Sigmoid()

        dA = dA_prev

        A_prev = history["A" + str(layer_idx_prev)]
        Z = history["Z" + str(layer_idx)]
        W = weights["W" + str(layer_idx)]
        b = weights["b" + str(layer_idx)]

        dA_prev, dW, db = single_layer_backward(
            dA, W, b, Z, A_prev, activ_function)

        grads_values["dW" + str(layer_idx)] = dW
        grads_values["db" + str(layer_idx)] = db

    return grads_values


def get_grad_values(X, Y, layers):
    weights = initialize_weigths(layers, init_random=False)
    Y_hat, history = full_forward(X, weights, layers)
    grads_values = full_backward(Y_hat, Y, history, weights, layers)

    for key, val in grads_values.items():
        print('Layer name: {}, grads: {}'.format(key, str(val.tolist())))


if __name__ == '__main__':
    layers = [
        {"input_dim": 3, "output_dim": 2, "activation": "relu"},
        {"input_dim": 2, "output_dim": 2, "activation": "relu"},
        {"input_dim": 2, "output_dim": 1, "activation": "None"}]

    Y = np.array([[0]])
    X = np.array([[1], [2], [-1]])
    get_grad_values(X, Y, layers)
