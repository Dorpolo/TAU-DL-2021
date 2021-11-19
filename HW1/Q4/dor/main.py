import random
import numpy as np

from utils import MSE, Linear


def gradient_descent(x,
                     y_true,
                     conv_th: float = 1e-5,
                     lr: float = 0.01,
                     gd_type: str = 'default',
                     decay_rate: float = None) -> None:
    diff, epoch = 100, 0
    epoch_loss, epoch_loss_diff = {}, {}

    loss = MSE()
    linear = Linear()

    while diff > conv_th:
        y_pred = linear(x)
        loss_value = loss(y_pred, y_true)
        epoch_loss.update({epoch: loss_value})
        diff = abs(epoch_loss[epoch] - epoch_loss[epoch - 1]) if len(epoch_loss) > 1 else abs(loss_value)
        epoch_loss_diff.update({epoch: diff})

        if epoch % 5 == 0:
            print(f'Epoch {epoch}, loss {loss_value},  diff {diff}')

        gradient_from_loss = loss.backward()
        linear.backward(gradient_from_loss)
        if 'exp' in gd_type:
            linear.update(lr, decay_rate=decay_rate if decay_rate else 0.8, itr=epoch)
        else:
            linear.update(lr)
        epoch += 1


def stochastic_gradient_descent(x,
                                y_true,
                                conv_th: float = 1e-5,
                                lr: float = 0.01,
                                batch_size: int = 100):
    batch_count = len(x) / batch_size
    diff, epoch = 100, 0
    epoch_loss, epoch_loss_diff = {}, {}

    loss = MSE()
    linear = Linear()

    while diff > conv_th and epoch < 100:
        data = list(zip(x, y_true))
        random.shuffle(data)
        X_tmp, Y_tmp = [np.asarray(item) for item in zip(*data)]
        for j in range(int(batch_count)):
            min_rng = int(j * batch_size)
            max_rng = int((j + 1) * batch_size)

            X_batch = X_tmp[min_rng:max_rng]
            Y_batch = Y_tmp[min_rng:max_rng]

            y_pred = linear(X_batch)
            loss_value = loss(y_pred, Y_batch)
            gradient_from_loss = loss.backward()
            linear.backward(gradient_from_loss)
            linear.update(lr)

        if epoch % 5 == 0:
            print(f'Epoch {epoch}, loss {loss_value},  diff {diff}')

        epoch += 1
        epoch_loss.update({epoch: loss_value})
        diff = abs(epoch_loss[epoch] - epoch_loss[epoch - 1]) \
            if len(epoch_loss) > 1 else abs(loss_value)
        epoch_loss_diff.update({epoch: diff})


def moment_gradient_descent(x,
                            y_true,
                            conv_th: float = 1e-5,
                            lr: float = 0.01,
                            gamma: float = 0.3):
    return NotImplementedError


if __name__ == '__main__':
    DIM = (10000, 4)
    MU, SIG = 0, 1
    Y_COEF = [1, -2, 3, -4]
    CONV_TH = 1e-5

    coef = np.array([Y_COEF])
    X = np.random.uniform(0, 1, DIM)
    eps = np.random.normal(MU, SIG, DIM[0])
    Y_TRUE = np.array([(X @ coef.T)[:, 0] + eps]).T

    LEARNING_RATE = 0.1
    EPOCHS = 1

    # Gradient Descent
    gradient_descent(gd_type='default',
                     x=X,
                     y_true=Y_TRUE,
                     lr=LEARNING_RATE,
                     conv_th=CONV_TH)

    # Exponential Gradient Descent
    gradient_descent(gd_type='exponential',
                     x=X,
                     y_true=Y_TRUE,
                     lr=LEARNING_RATE,
                     conv_th=CONV_TH,
                     decay_rate=0.8)

    # Stochastic Gradient Descent
    stochastic_gradient_descent(x=X,
                                y_true=Y_TRUE,
                                conv_th=CONV_TH,
                                lr=LEARNING_RATE,
                                batch_size=100)

    # Moment Gradient Descent
    moment_gradient_descent(x=X,
                            y_true=Y_TRUE)
