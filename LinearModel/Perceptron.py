# coding: utf-8
"""
The implementation of Basic Perceptron

"""
import numpy as np


class Perceptron(object):

    def __init__(self, iter_nums, learning_rate, init_mode="zeros", log_num_steps=10):
        """
        Constructor
        
        @iter_nums: The number of training steps
        @learning_rate: learning rate
        @init_mode: The initializer of parameters
        @log_num_steps: The number of steps to print log
        """
        self.model_name = "Basic Perceptron"
        self.iter_nums = iter_nums
        self.learning_rate = learning_rate
        self.init_mode = init_mode
        self.log_num_steps = log_num_steps

    def __initializer_parameters(self, shape):
        """
        The method of initializer (private method)
        
        :param shape: The shape of parameters
        :return: 
        """
        if self.init_mode == "zeros":
            self.w = np.zeros((shape, ))
            self.b = 0

    def __sign(self, x):
        return 1 if x >= 0 else -1

    def __calculate_loss(self, z, y):
        assert z.shape[0] == y.shape[0], "Inconsistent size"
        element_wise_result = z * y
        return np.abs(element_wise_result[element_wise_result <= 0].sum())

    def __forward(self, x):
        """
        forward calculation
        
        :param x: The input features vector, which is shaped as m times n
        :return: list: [logits, prediction result]
        """
        z = np.matmul(x, self.w) + self.b
        a = np.zeros_like(z)
        a[z >= 0] = 1
        a[z < 0] = -1
        return z, a

    def __backward(self, z, x, y):
        """
        backward calculation using SGD(stochastic gradient descent)
        :param z: The prediction logits
        :param x: The input features
        :param y: The true targets
        :return: None
        """

        assert z.shape[0] == y.shape[0], "Inconsistent size"

        for i in range(z.shape[0]):
            # Select the wrong case to update
            if self.__sign(z[i]) != y[i]:
                dw = - y[i] * x[i]
                db = - y[i]

                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db

    def fit(self, features, targets):
        """
        Train and fit the data set
        
        :param features: The input features of data
        :param targets: The targets of data
        :return: 
        """
        features = np.asarray(features)  # m x n
        targets = np.asarray(targets)  # m x 1

        assert features.ndim == 2, TypeError("Expected dims of input features: 2")
        assert features.shape[0] == targets.shape[0], "Inconsistent sample size between input features and targets"

        # Initialize the parameters
        self.__initializer_parameters(features.shape[1])

        # train
        for iter_num in range(self.iter_nums):
            # forward
            z, _ = self.__forward(features)

            # calculate the loss
            loss = self.__calculate_loss(z, targets)

            if (iter_num + 1) % self.log_num_steps == 0:
                print("iter: {}, cost: {}".format(iter_num + 1, loss))

            # backward
            self.__backward(z, features, targets)

    def predict(self, features):
        _, pred_results = self.__forward(features)
        return pred_results

    @property
    def get_parameters(self):
        print("-" * 10 + " Parameters " + "-" * 10)
        print("w: " + str(list(self.w)))
        print("b: " + str(self.b))
        print("-" * 14 + " End " + "-" * 14)

    def plot_boundary(self):
        pass


if __name__ == "__main__":
    x = np.asarray([[1, 0], [2, 0], [3, 0], [4, 0], [4, 3], [2, 1], [1, 2], [1, 3], [2, 9], [5, 7]])
    y = np.asarray([1, 1, 1, 1, 1, 1, -1, -1, -1, -1])

    model = Perceptron(iter_nums=20, learning_rate=1, log_num_steps=1)
    model.fit(x, y)
    model.get_parameters
    preds = model.predict(x)
    print("preds result: " + str(preds))
    print("accuracy: %.2f %%" % (100 * ((preds == y).sum() / y.shape[0])))
