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
            self.prior_table[k] = v / (m + self.num_class * self.lambda_)  # using Laplace Smoothing

        # get freq of joint appearing
        self.condition_table = dict()
        self.feature_value_set = dict()
        for i in range(m):
            for idx, x in enumerate(features[i]):  # idx represents the ith feature, x represents the value
                y = targets[i]
                self.condition_table.setdefault(idx, {})
                self.condition_table[idx].setdefault(y, {})
                self.condition_table[idx][y].setdefault(x, self.lambda_)  # default using Laplace Smoothing
                self.condition_table[idx][y][x] += 1
                # feature value set, storing the distinct values of idx th feature
                self.feature_value_set.setdefault(idx, set())
                self.feature_value_set[idx].add(x)

        # get condition prob
        for idx, info in self.condition_table.items():
            for class_, v in info.items():
                denominator = sum(v.values())  # the number of samples belongs to class_
                # smoothing = len(self.feature_value_set[idx])  # the distinct value of certain feature
                for val, cnt in v.items():
                    v[val] = cnt / denominator

    def fit(self, features, targets):
        """
        Fit the model and calculate the parameters
        :param features: 
        :param targets: 
        :return: 
        """

        # Calculate the prior prob table and conditional prob table
        self.__calculate_lookup_table(features, targets)
        return

    def predict(self, features):
        """
        Predict class given features
        :param features: 
        :return: 
        """
        class_val = self.prior_table.keys()  # Get the potential class result

        # result
        pred_results = []

        for i, feature in enumerate(features):
            result = {}
            for class_ in class_val:
                pred_prob = self.prior_table[class_]
                for idx, val in enumerate(feature):
                    # idx: the id of feature
                    # val: the value of feature
                    pred_prob *= \
                    self.condition_table.get(idx).get(class_).get(val, 1.0 / len(self.feature_value_set[idx]) * self.lambda_)

                # store
                result[class_] = pred_prob

            pred_results.append(result)

        # Get the pred class
        pred_class = []
        for result in pred_results:
            pred_class.append(max(result.items(), key=lambda x: x[1]))

        return pred_class

    @property
    def get_parameters(self):
        return self.prior_table, self.condition_table

if __name__ == "__main__":
    x = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'],
         [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']]
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    navie_bayesian = NaiveBayesian(2, 1)
    navie_bayesian.fit(x, y)
    print(navie_bayesian.predict([[2, 'S'], [1, 'S'], [3, 'L']]))
