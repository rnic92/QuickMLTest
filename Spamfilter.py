import numpy as np
import math


def sigmoid(a):
    if a < -700:
        return .000001
    return 1/(1+math.exp(-a))


def activation(w, phi):
    return np.dot(np.transpose(w), basis(phi))


def basis(x):
    return np.append(1, x)


def gradientdescent(M, N, alpha, mean, w, data, labels):
    e = 1e-3
    step = 0.01
    current = w
    # stop = False
    maxiter = 20
    i = 1
    olderror = float('INF')
    curerror = float('INF')
    while np.linalg.norm(curerror) > e and i <= maxiter:
        grad = gradient(alpha, mean, current, N, data, labels)
        # print("iteration: ", i)
        # print("old Error", np.linalg.norm(olderror))
        # print("Step", step)
        prediction = current - step*grad
        curerror = objective(M, mean, alpha, prediction, N, data, labels)
        if(curerror < olderror):
            step *= 2
            current = prediction
            olderror = curerror
            i = i + 1
        else:
            step = step/2
    # print("iteration:", i-1)
    # print("current error:", curerror)
    return current


def objective(M, mean, alpha, w, N, data, labels):
    S0 = alpha*np.identity(w.size)
    temp = np.dot(np.transpose(w-mean), np.linalg.inv(S0))
    temp = np.dot(temp, (w-mean))
    obj = M/2*math.log(2*math.pi) + (1/2)\
        * math.log(np.linalg.norm(S0)) + 1/2 * temp
    for i in range(N):
        y = sigmoid(activation(w, data[i]))
        te = 1-y
        if(te == 0):
            te = 1e-20
        obj -= (labels[i]*math.log(y) + (1-labels[i])*math.log(te))
    return obj


def gradient(alpha, mean, w, N, data, labels):
    # grad = covariance^(-1)*(theta-m0) + sum(yn-t(n))*phi(n)
    # S0 = alpha * np.identity(w.size)
    grad = 1/alpha * (w-mean)
    for i in range(N):
        y = sigmoid(activation(w, data[i]))
        temp = (y-labels[i]) * basis(data[i])
        temp = temp[:, np.newaxis]
        grad += temp
    # print("gradient", grad.shape)
    return grad


def train(traind, trainl):
    alpha = 15
    D = traind[0].size  # feature vector size
    N = traind[:, 0].size  # sample size
    M = D + 1  # theta size 1 larger than feature vector
    theta = np.zeros(M)  # fill with random starting values
    theta = theta[:,  np.newaxis]  # fix matrix size D x 1
    mean0 = 0
    theta = gradientdescent(M, N, alpha, mean0, theta, traind, trainl)
    correct = 0
    for i in range(N):
        temp = sigmoid(activation(theta, traind[i]))
        if (temp > 0.5 and trainl[i] == 1):
            correct += 1
        elif(temp < 0.5 and trainl[i] == 0):
            correct += 1
    print(correct, "correct out of", trainl.size)
    print("error rate: ", 100-correct/N*100)
    return theta
