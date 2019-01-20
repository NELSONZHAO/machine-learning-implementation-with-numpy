# coding: utf-8
"""
The implementation of k-NN
Three important elements:
1. The selection of "k"
2. The distance metrics
3. The voting method
"""


class TreeNode(object):
    def __init__(self):
        self.location
        self.cut_axis
        self.depth
        self.left_node
        self.right_node


class KNN(object):
    def __init__(self):
        pass

    def __generate_kd_tree(self):
        pass

    def __search_kd_tree(self):
        pass

    def fit(self, features, targets):
        self.__generate_kd_tree(features)

    def predict(self, x):
        pass

    @property
    def get_parameters(self):
        pass


if __name__ == "__main__":
    pass
