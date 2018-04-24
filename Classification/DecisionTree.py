# coding: utf-8
import numpy as np


class TreeNode(object):
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        Initialize a tree node
        :param col: The index of column to split
        :param value: The value of attribution to split
        :param results: The counter of results of current branch, if non-leaf node, the results are None
        :param tb: The true branch tree node
        :param fb: The false branch tree node
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def divide_set(data, column, value):
    """
    Split data by given column and value
    :param data: The dataset
    :param column: The attribution index
    :param value: The given value
    :return: The tuple of split datasets
    """
    split_fn = None
    # If the type of column is int or float
    if isinstance(value, int) or isinstance(value, float):
        split_fn = lambda row: row[column] >= value
    # If the type of column is str
    else:
        split_fn = lambda row: row[column] == value

    # Split data
    true_set = [row for row in data if split_fn(row)]
    false_set = [row for row in data if not split_fn(row)]

    return true_set, false_set


def unique_counts(data):
    """
    Counts the labels of data
    :param data: 
    :return: 
    """
    results = {}
    m = len(data[0])
    for row in data:
        # Get the label
        r = row[m-1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def entropy(data):
    """
    Calculate the entropy of given data
    :param data: The last column is the label of each sample
    :return: The entropy value of current data 
    """
    m = len(data)
    results = unique_counts(data)
    ent = 0.0

    for r in results.keys():
        p = results[r] * 1.0 / m
        ent -= p * np.log2(p)
    return ent


def gini_index(data):
    """
    Calculate the gini index of given data
    :param data: 
    :return: The gini index of current data
    """
    m = len(data)
    results = unique_counts(data)
    gini = 1.0
    for r in results.keys():
        p = results[r] * 1.0 / m
        gini -= p*p
    return gini


def build_tree(data, scoref=entropy, limit_gain=0.01):
    """
    Build the decision tree
    :param data: The given data
    :param scoref: The split method
    :param limit_gain: The minimum gain during each split
    :return: 
    """
    m = len(data)
    if m == 0:
        return TreeNode()

    current_score = scoref(data)

    # Initialize the value
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    # Iterate each feature
    n = len(data[0]) - 1
    for col in range(n):
        # Get the sets of current values
        values = set()
        for row in data:
            values.add(row[col])

        # Split data by each value to explore the best split
        for value in values:
            true_set, false_set = divide_set(data, col, value)

            # Calculate the score
            p = len(true_set) * 1.0 / m
            if p == 1 or p == 0:
                gain = 0
            else:
                gain = current_score - p * scoref(true_set) - (1 - p) * scoref(false_set)
            if gain > best_gain and len(true_set) > 0 and len(false_set) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (true_set, false_set)

    # Recursion
    if best_gain > limit_gain:
        true_branch = build_tree(best_sets[0])
        false_branch = build_tree(best_sets[1])
        return TreeNode(col=best_criteria[0], value=best_criteria[1], tb=true_branch, fb=false_branch)
    # leaf-node
    else:
        return TreeNode(results=unique_counts(data))


def print_tree(tree, indent=" "):
    if tree.results is not None:
        print(tree.results)
    else:
        print(str(tree.col) + ": " + str(tree.value) + "?")

        print(indent + "T->", end="")
        print_tree(tree.tb, indent + " ")
        print(indent + "F->", end="")
        print_tree(tree.fb, indent + " ")

if __name__ == "__main__":
    test_data = [['slashdot', 'USA', 'yes', 18, 'None'],
               ['google', 'France', 'yes', 23, 'Premium'],
               ['digg', 'USA', 'yes', 24, 'Basic'],
               ['kiwitobes', 'France', 'yes', 23, 'Basic'],
               ['google', 'UK', 'no', 21, 'Premium'],
               ['(direct)', 'New Zealand', 'no', 12, 'None'],
               ['(direct)', 'UK', 'no', 21, 'Basic'],
               ['google', 'USA', 'no', 24, 'Premium'],
               ['slashdot', 'France', 'yes', 19, 'None'],
               ['digg', 'USA', 'no', 18, 'None'],
               ['google', 'UK', 'no', 18, 'None'],
               ['kiwitobes', 'UK', 'no', 19, 'None'],
               ['digg', 'New Zealand', 'yes', 12, 'Basic'],
               ['slashdot', 'UK', 'no', 21, 'None'],
               ['google', 'UK', 'yes', 18, 'Basic'],
               ['kiwitobes', 'France', 'yes', 19, 'Basic']]

    watermelon_data = [["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697,	0.46,1],
                        ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774, 0.376,	1],
                        ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634, 0.264,	1],
                        ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608, 0.318,	1],
                        ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556, 0.215,	1],
                        ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403, 0.237,	1],
                        ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481, 0.149,	1],
                        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437, 0.211,	1],
                        ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666, 0.091,	0],
                        ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243, 0.267,	0],
                        ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245, 0.057,	0],
                        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343, 0.099,	0],
                        ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161,	0],
                        ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198,	0],
                        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.36, 0.37,	0],
                        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042,	0],
                        ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103,	0]]

    tree = build_tree(watermelon_data)
    print_tree(tree)