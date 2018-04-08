# coding: utf-8

import numpy as np

class RNN(object):
    def __init__(self):
        pass

    def __softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def rnn_cell_forward(self, xt, a_prev, parameters):
        """
        Implements a single forward step of the RNN-cell, shape=(nx, m)
        :param xt: input data at timestep 't'
        :param a_prev: the hidden state at timestep 't-1'
        :param parameters: 
        :return: 
        """

        # Retrieve parameters
        Wax = parameters["Wax"]  # na, nx
        Waa = parameters["Waa"]  # na, na
        Wya = parameters["Way"]  # ny, na
        ba = parameters["ba"]  # na, 1
        by = parameters["by"]  # ny, 1

        # compute the next activation state
        a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)

        yt_pred = self.__softmax(np.dot(Wya, a_next) + by)

        # store the values
        cache = (a_next, a_prev, xt, parameters)

        return a_next, yt_pred, cache

    def rnn_forward(self, x, a0, parameters):
        """
        Implement the forward propagation of the RNN
        :param x: Input Data, of shape (nx, m, Tx)
        :param a0: the initial hidden state, of shape (na, m)
        :param parameters: 
        :return: 
        """
        # Initialize caches which will contain the list of all caches
        caches = []

        # Retrieve dimensions
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wya"].shape

        # Initialize "a" and "y" with zeros
        a = np.zeros((n_a, m, T_x))
        y_pred = np.zeros((n_y, m, T_x))

        # Initialize the a_next
        a_next = a0

        # Loop over all time-steps
        for t in range(T_x):
            # Update next hidden state
            a_next, yt_pred, cache = self.rnn_cell_forward(x[:, :, t], a_next, parameters)
            # Save the activation of time-step 't'
            a[:, :, t] = a_next
            # Save the value of the prediction in y
            y_pred[:, :, t] = yt_pred
            # Append cache
            caches.append(cache)

        # Store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y_pred, caches

    def rnn_cell_backward(self, da_next, cache):
        """
        Implements the backward pass for the RNN-cell of a single step
        :param da_next: Gradient loss of next hidden state
        :param parameters: 
        :return: 
        """
        # Retrieve the cache
        (a_next, a_prev, xt, parameters) = cache

        # Retrieve the parameters
        Waa = parameters["Waa"]
        Wax = parameters["Wax"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        # Compute the gradient
        dtanh = da_next * (1 - a_next ** 2)

        dxt = np.dot(Wax.T, dtanh)
        dWax = np.dot(dtanh, xt.T)

        da_prev = np.dot(Waa.T, dtanh)
        dWaa = np.dot(dtanh, a_prev.T)

        dba = np.sum(dtanh, axis=1, keepdims=True)

        # Store the gradients
        gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

        return gradients

    def rnn_backward(self, da, caches):
        """
        Implements the backward pass for the RNN-cell of all the timesteps
        :param da: 每个t下，来自y的gradient
        :param caches: caches
        :return: 
        """
        (caches, x) = caches
        (a1, a0, x1, parameters) = caches[0]

        # Retrieve the shape
        n_a, m, T_x = da.shape
        n_x, m = x1.shape

        # Initialize the gradients
        dx = np.zeros((n_x, m, T_x))
        dWax = np.zeros((n_a, n_x))
        dWaa = np.zeros((n_a, n_a))
        dba = np.zeros((n_a, 1))
        da0 = np.zeros((n_a, m))
        da_prev = np.zeros((n_a, m))

        # Loop through all timesteps
        for t in reversed(range(T_x)):
            # Compute gradients
            gradients = self.rnn_backward(da[:, :, t] + da_prev, caches[t])
            # Retrieve gradients
            dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
            # Increment global derivatives w.r.t parameters by adding their derivative at time-step t (≈4 lines)
            dx[:, :, t] = dxt
            dWax += dWaxt
            dWaa += dWaat
            dba += dbat

        # Set gradient at timestep 0
        da0 = da_prevt

        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

        return gradients