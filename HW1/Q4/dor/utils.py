import numpy as np


class MSE:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return ((y_pred - y_true) ** 2).mean()

    def backward(self):
        n = self.y_true.shape[0]
        self.gradient = 2. * (self.y_pred - self.y_true) / n
        return self.gradient


class Linear:
    def __init__(self, input_dim: int = 4, num_hidden: int = 1):
        self.weights = np.random.randn(input_dim, num_hidden)
        self.bias = np.ones(num_hidden)

    def __call__(self, x):
        self.x = x
        output = x @ self.weights + self.bias
        return output

    def backward(self, gradient):
        self.weights_gradient = self.x.T @ gradient
        self.bias_gradient = gradient.sum(axis=0)
        self.x_gradient = gradient @ self.weights.T
        return self.x_gradient

    def update(self, lr: float = 0.1, decay_rate: float = None, itr: int = 1) -> None:
        if decay_rate:
            lr = lr * np.power(decay_rate, itr)
        self.weights = self.weights - lr * self.weights_gradient
        self.bias = self.bias - lr * self.bias_gradient

    def update_moment(self, lr: float = 0.1, gamma: float = 0.8, v_w: float = 0, v_b: float = 0) -> tuple:
        v_w = gamma * v_w + lr * self.weights_gradient
        v_b = gamma * v_b + lr * self.bias_gradient
        self.weights = self.weights - v_w
        self.bias = self.bias - v_b
        return v_w, v_b


def beautiful_sep(job_name: str = '') -> str:
    n = 70
    n_side = int((n - 1 - len(job_name))/2)
    print(f"\n{'-'*n}\n{'-'*n_side} {job_name} {'-'*n_side}\n{'-'*n}")
