# coding: utf-8

import numpy as np


class NaiveBayesian(object):
    def __init__(self, num_class=2, lambda_=1):
        self.num_class = num_class
        self.lambda_ = lambda_
        pass

    def __calculate_lookup_table(self, features, targets):
        """
        Calculate the prior prob table and conditional prob table
        :param features: 
        :param targets: 
        :return: 
        """
        # sample size
        m = len(features)

        # get freq of each class
        self.prior_table = dict()
        for y in targets:
            self.prior_table.setdefault(y, self.lambda_)  # default using Laplace Smoothing
            self.prior_table[y] += 1

        # get prior prob
        for k, v in self.prior_table.items():
            self.prior_table[k] = v / (m + self.num_class * self.lambda_)

        # get freq of joint appearing
        self.condition_table = dict()
        for i in range(m):
            for idx, x in enumerate(features[i]):  # idx represents the ith feature, x represents the value
                self.condition_table.setdefault(targets[i], {})
                self.condition_table[targets[i]].setdefault(x, self.lambda_)  # default using Laplace Smoothing
                self.condition_table[targets[i]][x] += 1

        # get condition prob
        for k, v in self.condition_table.items():
            denominator = self.condition_table[k].values()  # denominator of condition prob
            smoothing = len(self.condition_table[k])
            self.condition_table[k][v] = 0


    def fit(self, features, targets):
        """
        Fit the model and calculate the parameters
        :param features: 
        :param targets: 
        :return: 
        """

        # Calculate the prior prob table and conditional prob table
        self.__calculate_lookup_table(features, targets)


    def predict(self, features):
        pass

if __name__ == "__main__":
    x = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'],
         [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    navie_bayesian = NaiveBayesian(2, 1)
    navie_bayesian.fit(x, y)
