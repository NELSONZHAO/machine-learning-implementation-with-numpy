# coding: utf-8
import numpy as np
from DataSets.LoadData import load_planar


class SingleHiddenNeuralNetworkClassifier(object):
    def __init__(self, hidden_num=32, learning_rate=0.005, max_iter=500, activate_fn='sigmoid', init_weights_coef=0.01,
                 random_state=None, verbose=True):
        """
        Initialize the model params
        :param hidden_num: The number of nodes in the hidden layer
        :param learning_rate: The learning rate in the gradient descent
        :param max_iter: The maximum epochs of training model
        :param activate_fn: The activation function in the hidden layer
        :param init_weights_coef: The scaling size of initializing weights
        :param random_state: Random state
        """
        self.hidden_num = hidden_num
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
        self.random_state = random_state
        self.verbose = verbose

        np.random.seed(self.random_state)

        self.costs = []  # 记录训练过程的损失

    # 激活函数及导数
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

    def __initialize_weights(self, input_shape, output_shape):
        # 随机状态
        np.random.seed(self.random_state)

        self.W1 = np.random.randn(self.hidden_num, input_shape) * self.init_weights_coef
        self.b1 = np.zeros((self.hidden_num, 1))

        self.W2 = np.random.randn(output_shape, self.hidden_num) * self.init_weights_coef
        self.b2 = np.zeros((output_shape, 1))

    def __propagate(self, X, y):
        mx = X.shape[0]

        # 前向传播
        Z1 = np.dot(self.W1, X.T) + self.b1  # h x m
        A1 = self.activate_fns[self.activate_fn](Z1)  # h x m
        Z2 = np.dot(self.W2, A1) + self.b2  # o x m
        A2 = self.__sigmoid(Z2)  # o x m

        cost = (-1.0 / mx) * np.sum(np.multiply(y, np.log(A2)) + np.multiply((1 - y), np.log(1 - A2)))

        # 反向传播
        dZ2 = A2 - y  # o x m
        dW2 = (1.0 / mx) * np.dot(dZ2, A1.T)  # o x h
        db2 = (1.0 / mx) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * self.derivative_activate_fns[self.activate_fn](Z1)  # h x m
        dW1 = (1.0 / mx) * np.dot(dZ1, X)  # h x n
        db1 = (1.0 / mx) * np.sum(dZ1, axis=1, keepdims=True)

        assert (self.W1.shape == dW1.shape)
        assert (self.W2.shape == dW2.shape)
        cost = np.squeeze(cost)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads, cost

    def fit(self, X, y):
        """
        Fit the data to train the model
        :param X: The features of training data, the shape of X should be m x n
        :param y: The targets of training data, the shape of y should be m x 1 or (m,)
        :return: 
        """
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        assert (X.ndim == 2)
        assert (X.shape[0] == y.shape[0])

        # initialize the params
        self.__initialize_weights(X.shape[1], y.shape[1])

        y = y.reshape(1, -1)

        # train the model
        for i in range(self.max_iter):

            grads, cost = self.__propagate(X, y)

            dW1 = grads.get("dW1")
            db1 = grads.get("db1")
            dW2 = grads.get("dW2")
            db2 = grads.get("db2")

            # update the params
            self.W1 = self.W1 - self.learning_rate * dW1
            self.b1 = self.b1 - self.learning_rate * db1
            self.W2 = self.W2 - self.learning_rate * dW2
            self.b2 = self.b2 - self.learning_rate * db2

            # save the cost
            if i % 10 == 0:
                self.costs.append(cost)

            # print the log
            if self.verbose and i % 100 == 0:
                print("Training times: {}, Training error: {}".format(i, cost))

    def __predict(self, X):
        assert (X.ndim == 2)

        Z1 = np.dot(self.W1, X.T) + self.b1  # h x m
        A1 = self.activate_fns[self.activate_fn](Z1)  # h x m
        Z2 = np.dot(self.W2, A1) + self.b2  # o x m
        preds = self.__sigmoid(Z2)  # o x m

        return preds

    def predict(self, X):
        pred = (self.__predict(X) >= 0.5).astype(np.int64)
        return pred

    def predict_proba(self, X):
        return self.__predict(X)

    @property
    def get_params(self):
        self.params = {
            "hidden_num": self.hidden_num,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "activate_fn": self.activate_fn,
            "init_weights_coef": self.init_weights_coef,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }
        return self.params

if __name__ == "__main__":
    # Load the data
    X, y = load_planar()

    # Train
    clf = SingleHiddenNeuralNetworkClassifier(max_iter=20000, activate_fn="relu", hidden_num=32, learning_rate=0.05, random_state=123)
    clf.fit(X, y)

    preds = clf.predict(X)
    print("Accuracy: {}%".format(100.0 * (preds.reshape(1, -1) == y.reshape(1, -1)).sum() / y.shape[0]))
