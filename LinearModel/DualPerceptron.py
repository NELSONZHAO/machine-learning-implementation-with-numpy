# coding: utf-8
"""
The implementation of dual perceptron
"""

import numpy as np


class DualPerceptron(object):

    def __init__(self, iter_nums, learning_rate, init_mode="zeros", log_num_steps=5):
        """
        Constructor

        @iter_nums: The number of training steps
        @learning_rate: learning rate
        @init_mode: The initializer of parameters
        @log_num_steps: The number of steps to print log
        """
        self.model_name = "Dual Perceptron"
        self.iter_nums = iter_nums
        self.learning_rate = learning_rate
        self.init_mode = init_mode
        self.log_num_steps = log_num_steps

    def __initializer_parameters(self, shape):
        self.alpha = np.zeros((shape,))
        self.beta = 0

    def __parameter_transformation(self, x, y):
        self.w = np.sum(self.alpha * y * np.transpose(x), axis=1)
        self.b = self.beta

    def __gram_matrix(self, x, y):
        """
        Calculate the gram matrix of training data
        :param x: The shape of x is (m, n)
        :return: The m x m matrix
        """
        self.gram_matrix_with_y = np.transpose(y * np.dot(x, np.transpose(x)))

    def __sign(self, x):
        return 1 if x >= 0 else -1

    def __forward(self, x_id):
        """
        Forward calculate one sample
        :param x_id: the row index of sample
        :return: 
        """
        z = np.sum(self.alpha * self.gram_matrix_with_y[:, x_id]) + self.beta
        a = self.__sign(z)
        return z, a

    def __backward(self, x_id, y):
        self.alpha[x_id] = self.alpha[x_id] + self.learning_rate
        self.beta = self.beta + y

    def __calculate_loss(self, z, y):
        assert z.shape[0] == y.shape[0], "Inconsistent size"
        element_wise_result = z * y
        return np.abs(element_wise_result[element_wise_result <= 0].sum())

    def fit(self, features, targets):
        """
        Fit the model
        :param features: the features
        :param targets:  the targets
        :return: 
        """
        features = np.asarray(features)  # m x n
        targets = np.asarray(targets)  # m x 1

        # Initialize the parameters
        self.__gram_matrix(features, targets)
        self.__initializer_parameters(features.shape[0])

        assert features.ndim == 2, TypeError("Expected dims of input features: 2")
        assert features.shape[0] == targets.shape[0], "Inconsistent sample size between input features and targets"

        # train
        for iter_num in range(self.iter_nums):
            # forward
            for i in range(features.shape[0]):
                # forward
                z, _ = self.__forward(i)

                # backward if mis-classify
                if targets[i] * z <= 0:
                    self.__backward(i, targets[i])
                    self.__parameter_transformation(features, targets)

            # get loss
            loss = self.__calculate_loss(self.__predict_forward(features)[0], targets)
            if (iter_num) % self.log_num_steps == 0:
                print("iter: {}, cost: {}".format(iter_num + 1, loss))

    def __predict_forward(self, x):
        """
        Batch predict
        :return: 
        """
        z = np.dot(x, self.w) + self.b
        a = np.zeros_like(z)
        a[z >= 0] = 1
        a[z < 0] = -1
        return z, a

    def predict(self, features):
        _, a = self.__predict_forward(features)
        return a

    @property
    def __get_parameters(self):
        return self.w, self.b

    @property
    def get_parameters(self):
        print("-" * 10 + " Parameters " + "-" * 10)
        print("alpha: " + str(list(self.alpha)))
        print("beta: " + str(self.beta))
        print("-" * 10 + " Transform to basic parameter " + "-" * 10)
        print("w: " + str(list(self.__get_parameters[0])))
        print("b: " + str(self.__get_parameters[1]))
        print("-" * 14 + " End " + "-" * 14)

# test data
if __name__ == "__main__":
    x = [[3, 3], [4, 3], [1, 1]]
    y = [1, 1, -1]
    model = DualPerceptron(20, 1, log_num_steps=1)
    model.fit(x, y)
    model.get_parameters
    print(model.predict(x))
