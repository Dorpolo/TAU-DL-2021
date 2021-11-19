import numpy as np


def ReLU(x):
    return np.clip(x, 0, None)


def d_ReLU(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def d_mse(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

