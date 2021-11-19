import random
import numpy as np


def get_loss(X, Y, m, b):
    Y_pred = np.dot(m, X.T) + b
    RSS = ((Y - Y_pred.T) ** 2).sum()
    return RSS


def get_loss_grad(X, Y, m, b):
    Y_pred = np.dot(m, X.T) + b
    g_m = (-2) * (np.dot(X.T, (Y - Y_pred.T))).sum(axis=1)
    g_b = (-2) * (Y - Y_pred.T).sum()
    return g_m, g_b


def single_update(X, Y, m, b, learning_rate):
    g_m, g_b = get_loss_grad(X, Y, m, b)
    m -= learning_rate * g_m
    b -= learning_rate * g_b
    return m, b, g_m, g_b


def single_update_moment(X, Y, m, b, vm_old, vb_old, gamma, learning_rate):
    g_m, g_b = get_loss_grad(X, Y, m, b)
    v_m = gamma * vm_old + learning_rate * g_m
    v_b = gamma * vb_old + learning_rate * g_b
    m -= v_m
    b -= v_b
    return m, b, g_m, g_b, v_m, v_b


def GD(X, Y, m, b, learning_rate=1e-5, epochs=None, conv_thresh=1e-5):
    m_history = [m]
    b_history = [b]
    loss_history = [get_loss(X, Y, m, b)]
    count = 0
    diff = 100

    while (count < epochs) and (diff > conv_thresh):
        if count % 10 == 0:
            print(count)
        count += 1
        m, b, g_m, g_b = single_update(X, Y, m, b, learning_rate)
        m_history.append(m)
        b_history.append(b)
        curr_loss = get_loss(X, Y, m, b)
        diff = np.abs(curr_loss - loss_history[-1])
        print("Epoch {0}: loss: {1}, m: {2}, b: {3} ".format(str(count), str(curr_loss), str(m[0].tolist()), str(b)))
        loss_history.append(curr_loss)

    return m, b, m_history, b_history


def Exp_GD(X, Y, m, b, learning_rate=1e-5, decay_rate=0.8, epochs=None, conv_thresh=1e-5):
    m_history = [m]
    b_history = [b]
    loss_history = [get_loss(X, Y, m, b)]
    count = 0
    diff = 100
    original_learning_rate = learning_rate

    while (count < epochs) and (diff > conv_thresh):
        if count % 10 == 0:
            print(count)
        count += 1
        m, b, g_m, g_b = single_update(X, Y, m, b, learning_rate)
        m_history.append(m)
        b_history.append(b)
        curr_loss = get_loss(X, Y, m, b)
        diff = np.abs(curr_loss - loss_history[-1])
        print("Epoch {0}: loss: {1}, m: {2}, b: {3} ".format(str(count), str(curr_loss), str(m[0].tolist()), str(b)))
        loss_history.append(curr_loss)
        learning_rate = original_learning_rate * (np.power(decay_rate, count))
    return m, b, m_history, b_history


def SGD(X, Y, m, b, learning_rate=1e-5, batch_size=1, epochs=None, conv_thresh=1e-5):
    num_batches = len(X) / batch_size
    m_history = [m]
    b_history = [b]
    loss_history = [get_loss(X, Y, m, b)]
    count = 0
    diff = 100

    while (count < epochs) and (diff > conv_thresh):
        if count % 10 == 0:
            print(count)
        count += 1
        data = list(zip(X, Y))
        random.shuffle(data)
        X_temp, Y_temp = zip(*data)
        X_temp = np.asarray(X_temp)
        Y_temp = np.asarray(Y_temp)
        for j in range(int(num_batches) + 1):
            X_batch = X_temp[int(j * batch_size):int((j + 1) * batch_size)]
            Y_batch = Y_temp[int(j * batch_size):int((j + 1) * batch_size)]
            m, b, g_m, g_b = single_update(X_batch, Y_batch, m, b, learning_rate)
            curr_loss = get_loss(X, Y, m, b)
            m_history.append(m)
            b_history.append(b)
            # print( "Epoch {0} batch {1}: loss: {2}, m: {3}, b: {4} ".format(str(count),str(j), str(curr_loss),
            # str(m[0].tolist()), str(b)))
        curr_loss = get_loss(X, Y, m, b)
        diff = np.abs(curr_loss - loss_history[-1])
        print("Epoch {0}: loss: {1}, m: {2}, b: {3} ".format(str(count), str(curr_loss), str(m[0].tolist()), str(b)))
        loss_history.append(curr_loss)

    return m, b, m_history, b_history


def moment_GD(X, Y, m, b, gamma, learning_rate=1e-5, epochs=None, conv_thresh=1e-8):
    m_history = [m]
    b_history = [b]
    vm_history = [0]
    vb_history = [0]
    loss_history = [get_loss(X, Y, m, b)]
    count = 0
    diff = 100

    while (count < epochs) and (diff > conv_thresh):
        if count % 10 == 0:
            print(count)
        count += 1
        m, b, g_m, g_b, v_m, v_b = single_update_moment(X, Y, m, b, vm_history[-1], vb_history[-1], gamma,
                                                        learning_rate)
        m_history.append(m)
        b_history.append(b)
        vm_history.append(v_m)
        vb_history.append(v_b)
        curr_loss = get_loss(X, Y, m, b)
        diff = np.abs(curr_loss - loss_history[-1])
        print("Epoch {0}: loss: {1}, m: {2}, b: {3} ".format(str(count), str(curr_loss), str(m[0].tolist()), str(b)))
        loss_history.append(curr_loss)

    return m, b, m_history, b_history


if __name__ == '__main__':
    X = np.random.uniform(0, 1, (10000, 4))
    eps = np.random.normal(0, 1, 10000)
    Y = X[:, 0] - 2 * X[:, 1] + 3 * X[:, 2] - 4 * X[:, 3] + eps
    Y = np.reshape(Y, (Y.shape[0], 1))

    start_m = np.ones((1, 4))
    start_b = 1

    ##Regular Gradient Descent
    GD(X, Y, start_m, start_b, learning_rate=1e-5, epochs=2000)

    ##Exponential decay GD
    # Exp_GD(X, Y, start_m, start_b, learning_rate=1e-4, decay_rate=0.95, epochs=2000)

    ##SGD
    # SGD(X, Y, start_m, start_b, learning_rate=1e-4,batch_size=64, epochs=2000)

    ##Mpmentum GD
    # moment_GD(X, Y, start_m, start_b,gamma=0.8, learning_rate=1e-5,epochs=2000)
