# coding: utf-8
"""
This part implements some initializing method in neural network
"""

import numpy as np


def initialize_params_zeros(layer_dims):
    """
    Initialize all the parameters with zeros 
    :param shape: variable parameters which defined the shape of weights
    :return: the initialized parameters
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def initialize_params_random(layer_dims, scaling_coef=0.1):
    """
    Initialize all the parameters randomly
    :param layer_dims: 
    :param scaling_coef: Scaling the range of weights
    :return: 
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scaling_coef
        parameters["b" + str(l)] = np.random.randn(layer_dims[l], 1) * scaling_coef

    return parameters


def initialize_params_xavier(layer_dims):
    """
    Initialize all the parameters with Xavier Initialization 
    :param layer_dims: 
    :return: 
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1.0 / layer_dims[l - 1])
        parameters["b" + str(l)] = np.random.randn(layer_dims[l], 1) * np.sqrt(2.0 / layer_dims[l - 1])

    return parameters


def initialize_params_he(layer_dims):
    """
    Initialize all the parameters using He's method
    :param layer_dims: 
    :return: 
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters["b" + str(l)] = np.random.randn(layer_dims[l], 1) * np.sqrt(2.0 / layer_dims[l-1])

    return parameters