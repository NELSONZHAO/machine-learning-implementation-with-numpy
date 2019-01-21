# coding: utf-8
"""
The implementation of Naive KNN

pre-calculate and store the distance between each sample in training data set

Note: The kd-tree version is implemented in KNN.py
"""

import numpy as np


class NaiveKNN(object):
    def __init__(self, k, metric="l2", iter_nums=10, log_nums=10):
        self.k = k
        self.metric = metric
        self.iter_nums = iter_nums
        self.log_nums = log_nums

    def __get_distance(self, x1, x2):
        """
        Get the distance between x1 and x2
        :param x1: vector
        :param x2: vector
        :return: 
        """
        assert x1.shape[0] == x2.shape[0], TypeError("Expected dims of input features is not consistent")

        # The metric lp refers to Minkowski distance

        # Manhattan distance ( p = 1 )
        if self.metric == "l1":
            distance = sum(list(map(lambda x: np.abs(x), x1 - x2)))

        # Euclidean distance ( p = 2 )
        if self.metric == "l2":
            distance = np.sqrt(sum(list(map(lambda x: np.power(x, 2), x1 - x2))))

        # Chebyshev distance ( p = ∞ )
        if self.metric == "lp":
            distance = np.max(list(map(lambda x: np.abs(x), x1 - x2)))

        return distance

    def __calculate_distance(self, x, features, targets):
        """
        Calculate the distance matrix
        :param x: the input data
        :param features: The training data
        :param targets: 
        :return: 
        """
        # sample size
        m = features.shape[0]

        # calculate the distance
        distance_info = []

        for i in range(m):
            distance = self.__get_distance(x, features[i])
            distance_info.append((distance, targets[i]))  # dist and class

        # sort by distance
        distance_info = sorted(distance_info, key=lambda x: x[0], reverse=False)

        # vote for result
        vote_result = self.__vote_result(distance_info[:self.k])

        return vote_result

    def __vote_result(self, dist_class_pair):
        """
        return the vote classification result
        :param dist_class_pair: pair of (distance, class)
        :return: 
        """
        nk_class_list = list(map(lambda x: x[1], dist_class_pair))

        class_cnt_map = {}

        for c in nk_class_list:
            class_cnt_map.setdefault(c, 0)
            class_cnt_map[c] += 1

        return max(class_cnt_map.items(), key=lambda x: x[1])[0]

    def predict(self, samples, features, targets):
        """
        Fit the data, and calculate the distance between each sample
        :param samples: the input data set
        :param features: the features of training data
        :param targets: the targets of training data
        :return: 
        """
        features = np.asarray(features)  # m x n
        targets = np.asarray(targets)  # m x 1
        samples = np.asarray(samples)

        assert features.ndim == 2, TypeError("Expected dims of input features: 2")
        assert features.shape[0] == targets.shape[0], "Inconsistent sample size between input features and targets"

        # store the preds result
        preds = np.zeros((samples.shape[0]))

        for i, sample in enumerate(samples):
            pred = self.__calculate_distance(sample, features, targets)
            preds[i] = pred

        return preds

    @property
    def get_parameters(self):
        pass


if __name__ == "__main__":
    x = [[1, 1], [1, 2], [3, 4], [2, 5], [23, 12], [12, 14], [5, 18], [91, 1]]
    y = [1, 1, 1, 1, -1, -1, -1, -1]
    naive_knn = NaiveKNN(3)
    print(naive_knn.predict([[3, 6], [5, 4], [83, 82]], x, y))