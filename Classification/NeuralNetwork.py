# coding: utf-8

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from Preprocessing.WeightsInitialization import *

from DataSets.LoadData import *


class NeuralNetwork(object):
    def __init__(self, layer_dims=(5, 1), learning_rate=0.005, max_iter=500, activate_fn='relu', init_weights_coef=0.01,
                 regularization="l2", lambd=0.5, keep_prob=0.9, random_state=None, verbose=True):
        """
        Initialize the model params
        :param layer_dims: The number of nodes in each layer
        :param learning_rate: The learning rate in the gradient descent
        :param max_iter: The maximum epochs of training model
        :param activate_fn: The activation function in the hidden layer
        :param regularization: The method of regularization
        :param lambd: The coefficient of regularization
        :param keep_prob: The coefficient of dropout regularization
        :param init_weights_coef: The scaling size of initializing weights
        :param random_state: Random state
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.activate_fn = activate_fn
        self.activate_fns = {
            "sigmoid": self.__sigmoid,
            "tanh": self.__tanh,
            "relu": self.__relu,
            "lrelu": self.__leaky_relu
        }
        self.derivative_activate_fns = {
            "sigmoid": self.__derivative_sigmoid,
            "tanh": self.__derivative_tanh,
            "relu": self.__derivative_relu,
            "lrelu": self.__derivative_leaky_relu
        }
        self.init_weights_coef = init_weights_coef
        self.regularization = regularization
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.random_state = random_state
        self.verbose = verbose
        self.weights = defaultdict()

        self.costs = []  # 记录训练过程的损失

    # activation functions and their derivatives
    def __sigmoid(self, z):
        s = 1.0 / (1 + np.exp(-z))
        return s

    def __derivative_sigmoid(self, z):
        return self.__sigmoid(z) * (1.0 - self.__sigmoid(z))

    def __tanh(self, z):
        s = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return s

    def __derivative_tanh(self, z):
        return 1 - np.power(self.__tanh(z), 2)

    def __relu(self, z):
        s = np.maximum(np.zeros_like(z), z)
        return s

    def __derivative_relu(self, z):
        return (z >= 0).astype(np.int64)

    def __leaky_relu(self, z):
        s = np.maximum(0.01 * z, z)
        return s

    def __derivative_leaky_relu(self, z):
        return np.maximum((z >= 0).astype(np.int64), np.full_like(z, 0.01, dtype=np.float64))

    def __initialize_weights(self, nx):
        np.random.seed(self.random_state)

        self.weights = defaultdict()
        self.layer_dims = [nx] + list(self.layer_dims)
        L = len(self.layer_dims)

        # Generate the weights of NN
        for l in range(1, L):
            self.weights["W" + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * self.init_weights_coef
            # self.weights["b" + str(l)] = np.random.randn(self.layer_dims[l], 1) * self.init_weights_coef
            self.weights["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert (self.weights["W" + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
            assert (self.weights["b" + str(l)].shape == (self.layer_dims[l], 1))

    def __forward(self, A, W, b, activation_fn, output_layer=False):
        """
        Implement the part of a layer's forward propagation
        :param A: activations from previous layer(or the input data)
        :param W: weights matrix (l x l-1)
        :param b: bias vetor (l, 1)
        :param activation_fn: The Activation function of each layer
        :param output_layer: If the current layer is the output layer
        :return: 
        """
        Z = np.dot(W, A) + b
        A_new = activation_fn(Z)
        D = np.ones_like(A_new)  # Mask

        # Implement the Inverted Dropout Regularization
        if self.regularization == "dropout" and not output_layer:
            D = np.random.rand(A_new.shape[0], A_new.shape[1]) < self.keep_prob
            A_new = np.multiply(A_new, D) / self.keep_prob

        assert (Z.shape == (W.shape[0], A.shape[1]))
        assert (A_new.shape == (W.shape[0], A.shape[1]))

        cache = (A, W, b, Z, D)

        return A_new, cache

    def __backward(self, dA, cache, derivative_activate_fn):
        """
        Implement the part of a layer's backward propagation
        :param dA: derivatives value of forward propagtion
        :param cache: A_prev, W, b, Z
        :return: 
        """
        A_prev, W, b, Z, D = cache

        m = A_prev.shape[1]

        # Mask
        dA = np.multiply(dA, D) / self.keep_prob

        dZ = dA * derivative_activate_fn(Z)
        dW = (1.0 / m) * np.dot(dZ, A_prev.T)
        db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)


        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        assert (dA_prev.shape == A_prev.shape)

        return dA_prev, dW, db

    def __compute_cost(self, a, y):
        """
        Compute the cost
        :param a: pred
        :param y: real tag
        :return: cost
        """
        y = y.reshape(1, -1)
        m = y.shape[1]

        cost = (-1.0 / m) * np.sum(np.multiply(y, np.log(a)) + np.multiply(1-y, np.log(1-a)))
        cost = np.squeeze(cost)

        # If need regularization
        if self.regularization == "l2":
            L = len(self.weights) // 2
            for l in range(1, L+1):
                cost += self.lambd/(2*m) * np.sum(np.square(self.weights["W" + str(l)]))

        return cost

    def __propagate(self, X, y):
        mx = X.shape[0]

        A = X.T
        caches = []
        # Forward Propagate
        L = len(self.weights) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.__forward(A_prev, self.weights["W" + str(l)], self.weights["b" + str(l)], self.activate_fns[self.activate_fn])
            caches.append(cache)

        # final output
        AL, cache = self.__forward(A, self.weights["W" + str(L)], self.weights["b" + str(L)], self.__sigmoid, output_layer=True)
        caches.append(cache)

        # Cost
        y = y.reshape(1, -1)
        cost = self.__compute_cost(AL, y)

        # Backward Propagate
        grads = defaultdict()

        # current cache
        dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))

        current_cache = caches[L-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.__backward(dAL, current_cache, self.__derivative_sigmoid)

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_tmp, dW_tmp, db_tmp = self.__backward(grads["dA" + str(l + 2)], current_cache, self.derivative_activate_fns[self.activate_fn])
            grads["dA" + str(l + 1)] = dA_tmp
            grads["dW" + str(l + 1)] = dW_tmp
            grads["db" + str(l + 1)] = db_tmp

        # Update the parameters
        decay_coef = 1  # if not using regularization method, there is no decay when back-propagation
        if self.regularization == "l2":
            decay_coef = 1 - self.learning_rate * self.lambd / mx

        for l in range(1, L + 1):
            assert (self.weights["W" + str(l)].shape == grads["dW" + str(l)].shape)
            assert (self.weights["b" + str(l)].shape == grads["db" + str(l)].shape)

            self.weights["W" + str(l)] = decay_coef * self.weights["W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
            self.weights["b" + str(l)] = self.weights["b" + str(l)] - self.learning_rate * grads["db" + str(l)]

        return cost

    def fit(self, X, y):
        assert (X.shape[0] == y.shape[0])

        self.layer_dims = [X.shape[1]] + list(self.layer_dims)
        self.weights = initialize_params_he(self.layer_dims)
        # self.__initialize_weights(X.shape[1])

        # train the model
        for i in range(self.max_iter):
            cost = self.__propagate(X, y)

            self.costs.append(cost)

            # print the log
            if self.verbose and i % 1000 == 0:
                print("Training times: {}, Training error: {}".format(i, cost))

    def __predict(self, X):
        mx = X.shape[0]

        A = X.T
        # Forward Propagate
        L = len(self.weights) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.__forward(A_prev, self.weights["W" + str(l)], self.weights["b" + str(l)],
                                      self.activate_fns[self.activate_fn])

        # final output
        AL, _ = self.__forward(A, self.weights["W" + str(L)], self.weights["b" + str(L)], self.__sigmoid)

        return AL.reshape(mx, 1).squeeze()

    def predict(self, X):
        pred = (self.__predict(X) >= 0.5).astype(np.int64)
        return pred

if __name__ == "__main__":
    # Load the data
    X, y = load_circles()

    train_X, dev_X, train_y, dev_y = train_test_split(X, y, test_size=0.4)


    # Train
    clf = NeuralNetwork(layer_dims=(8, 12, 1), activate_fn="relu", regularization="dropout", keep_prob=1, max_iter=20000, learning_rate=0.05, random_state=123)
    clf.fit(train_X, train_y)

    train_preds = clf.predict(train_X)
    print("Train Accuracy: {}%".format(100.0 * (train_preds.reshape(1, -1) == train_y.reshape(1, -1)).sum() / train_y.shape[0]))

    dev_preds = clf.predict(dev_X)
    print("Dev Accuracy: {}%".format(100.0 * (dev_preds.reshape(1, -1) == dev_y.reshape(1, -1)).sum() / dev_y.shape[0]))