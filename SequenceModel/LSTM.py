# coding: utf-8
import numpy as np

class LSTM(object):
    def __init__(self):
        pass

    def __sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def __softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def lstm_cell_forward(self, xt, a_prev, c_prev, parameters):
        """
        Implement a single forward step of the LSTM-cell
        :param xt: Input data at time t
        :param a_prev: Hidden state at time t-1
        :param c_prev: Cell state or Memory state at time t-1
        :param parameters: 
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
        :return: 
        """
        # Retrieve the parameters
        Wf = parameters["Wf"]  # Forget gate
        bf = parameters["bf"]
        Wi = parameters["Wi"]  # Update gate
        bi = parameters["bi"]
        Wc = parameters["Wc"]  # tilde c
        bc = parameters["bc"]
        Wo = parameters["Wo"]  # Output gate
        bo = parameters["bo"]
        Wy = parameters["Wy"]  # y pred
        by = parameters["by"]

        # Retrieve dimensions of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape

        # Concat the a_prev and xt
        concat = np.concatenate((a_prev, xt), axis=0)  # na+nx, m

        # Compute the ft, it, cct, c_next, ot, a_next
        ft = self.__sigmoid(np.dot(Wf, concat) + bf)
        it = self.__sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = it * cct + ft * c_prev
        ot = self.__sigmoid(np.dot(Wo, concat) + bo)
        a_next = ot * np.tanh(c_next)

        # Compute preds
        yt_pred = self.__softmax(np.dot(Wy, a_next) + by)

        # Store the cache
        cache = [a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters]

        return a_next, c_next, yt_pred, cache

    def lstm_forward(self, x, a0, parameters):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell
        :param x: Input data of shape (n_x, m, T_x)
        :param a0: Initial hidden state
        :param parameters: 
        :return: 
        """
        # Initialize caches
        caches = []

        # Retrieve shapes
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wy"].shape

        # initialize "a", "c" and "y" with zeros
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))

        # Initialize zero state
        a_next = a0
        c_next = np.zeros_like(a0)

        # Loop over all the time-steps
        for t in range(T_x):
            a_next, c_next, yt_pred, cache = self.lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
            a[:, :, t] = a_next
            c[:, :, t] = c_next
            y[:, :, t] = yt_pred
            caches.append(cache)

        # Store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches
