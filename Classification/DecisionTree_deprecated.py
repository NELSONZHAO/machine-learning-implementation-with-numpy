# coding: utf-8
import numpy as np
from collections import Counter
import pandas as pd


class Node(object):
    def __init__(self, attr=None, value=None, child_node=None, label=None):
        """
        A Tree Node
        :param attr: the split attribution
        :param value: the split value
        :param child_node: the child node
        :param label: the node label, if the node is a leaf node, else None
        """
        self.attr = attr
        self.value = value
        self.child_node = child_node
        self.label = label


class DecisionTree(object):
    def __init__(self, col_name, split_fn="entropy"):
        """
        Save some values such as column names of data and splitting method
        :param col_name: The features name in dataset
        :param split_fn: The split function of decision tree, such as entropy or gini index
        """
        self.col_name = col_name
        self.split_fn = split_fn

    def divide_set(self, data, attr, value):
        """
        Split data by given attribution and value
        :param data: the numpy format
        :param attr: the index of attr(or the column index of dataset)
        :param value: the value of attr(the value of split attr, type(value)=list)
        :return: The dict of splitted datasets
        """
        splitted_dataset = {}

        # Continuous value, using bi-partition
        if len(value) == 1:
            true_set = data[data[:, attr] >= value]
            false_set = data[data[:, attr] < value]
            # The "T" part and "F" part refer to data sets which suffice the conditions above
            splitted_dataset = {'T': true_set, 'F': false_set}

        # Discrete value or Categorical value
        else:
            for v in value:
                true_set = data[data[:, attr] == v]
                true_set = np.delete(true_set, attr, axis=1)  # Drop the column
                splitted_dataset[v] = true_set

        return splitted_dataset

    def unique_counts(self, data):
        """
        Counts the distribution of class in given dataset
        :param data: 
        :return: 
        """
        labels = data[:, -1]
        c = Counter(labels)
        label = c.most_common()  # Get the label in the current node

        return c, label

    def gini_impurity(self, data):
        """
        calculate the gini impurity of datasets
        :param data: 
        :return: 
        """
        # The number of data samples
        m = data.shape[0]
        counts, _ = self.unique_counts(data)
        purity = 0

        for k in counts:
            p = counts[k] * 1.0 / m
            purity += p * p

        impurity = 1 - purity
        return impurity

    def entropy(self, data):
        """
        Calculate the entropy of current dataset
        :param data: 
        :return: 
        """
        # The number of data samples
        m = data.shape[0]
        counts, _ = self.unique_counts(data)
        entropy = 0

        for k in counts:
            p = counts[k] * 1.0 / m
            entropy -= p * np.log2(p)

        return entropy

    def build_tree(self, data):
        m = data.shape[0]
        n = data.shape[1]
        # If there is no samples in current node
        if m == 0:
            return Node()

        # data = np.concatenate((x, y.reshape(m, 1)), axis=1)

        # Define some variables to split dataset
        if self.split_fn == "entropy":
            score_fn = self.entropy
        if self.split_fn == "gini":
            score_fn = self.gini_impurity

        current_score = score_fn(data)

        if current_score == 0:
            return Node(None, None, None, self.unique_counts(data))

        best_gain = 0.0
        best_attr = None
        best_value = None
        best_sets = None

        # Select the best attribution and value
        for col in range(n-1):
            # All of the values in current col
            col_values = np.unique(data[:, col])
            # Process categorical variable
            if isinstance(col_values[0], str):
                splitted_set = self.divide_set(data, col, col_values)

                total_ent = 0
                split_status = True
                for dataset in splitted_set.values():
                    p = dataset.shape[0] / m
                    if p == 0:
                        split_status = False  # Avoid no samples after splitting
                    ent = self.entropy(dataset)
                    total_ent += p * ent
                gain = current_score - total_ent

                if gain > best_gain and split_status:
                    best_gain = gain
                    best_attr = col
                    best_value = col_values
                    best_sets = splitted_set

            # Process continuous variable with bi-partition
            if isinstance(col_values[0], int) or isinstance(col_values[0], float):
                # Calculate the gain
                for value in col_values:
                    splitted_set = self.divide_set(data, col, [value])

                    split_status = True

                    p = (splitted_set["T"].shape[0]) / m
                    if p == 0 or p == 1:
                        split_status = False

                    gain = current_score - p * self.entropy(splitted_set["T"]) - (1 - p) * self.entropy(splitted_set["F"])

                    if gain > best_gain and split_status:
                        best_gain = gain
                        best_attr = col
                        best_value = value
                        best_sets = splitted_set

        # Create the child node
        child_node = {}
        if best_gain > 0:
            print(list(map(lambda x: {x[0]: len(x[1])}, best_sets.items())))
            for k, sets in best_sets.items():
                child_node[k] = self.build_tree(sets)
            print(child_node)
            return Node(best_attr, best_value, child_node, None)

        else:
            print(self.unique_counts(data)[1])
            return Node(None, None, None, label=self.unique_counts(data)[1])

if __name__ == "__main__":
    # Get data
    data_path = '/Users/Nelson/Desktop/ml_zhouzhihua_dt.xlsx'
    data = pd.read_excel(data_path)

    # Get the column name and data values
    col_name = data.columns.values
    data = data.values

    # Create a decision tree object
    dt = DecisionTree(col_name)

    # Build tree
    tree = dt.build_tree(data)
    # Show tree
    # dt.show_tree(tree)
