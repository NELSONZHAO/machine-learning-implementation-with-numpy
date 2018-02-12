# coding: utf-8

import numpy as np
import h5py

class LogisticClassifier(object):

    def __init__(self, penalty="l2", tol=0.0001, max_iter=1000, reg_coef=1.0, learning_rate=0.1, fit_intercept=True,
                 class_weight=None, random_state=None, solver="liblinear", verbose=True, multi_class="ovr",
                 warm_start=False, init_mode='zeros'):
        """
        初始化函数
        :param penalty: 正则化方法，默认l2，可选l1或l2
        :param tol: 训练收敛误差。当t+1轮训练结果与t轮训练结果误差小于该值时，训练停止
        :param max_iter: 最大迭代次数
        :param reg_coef: 正则化系数
        :param learning_rate: 学习速率
        :param fit_intercept: 是否加入常数项。默认为True
        :param class_weight: 样本权重，默认所有权重均为1。支持dict格式
        :param random_state: 随机状态
        :param verbose: 是否打印日志
        :param solver: 
        :param multi_class: 多分类模式  
        :param warm_start: 是否加载上一次训练结果作为参数初始化状态
        :param init_mode: 参数初始化方法
        """
        self.penalty = penalty
        self.tol = tol
        self.max_iter = max_iter
        self.reg_coef = reg_coef
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose
        self.solver = solver
        self.multi_class = multi_class
        self.warm_start = warm_start
        self.init_mode = init_mode

        self.costs = []  # 记录训练过程中的损失

        # print(self.__init__)

    def __initialize_weights(self, shape, mode='zeros'):
        # 初始化为0
        if mode == "zeros":
            self.w = np.zeros((shape, 1))
            self.b = 0
        # 标准正态分布初始化
        elif mode == 'norm':
            self.w = np.random.randn(shape, 1)
            self.b = np.random.randn(1)[0]
        else:
            raise ValueError("参数初始化模式无法识别，请使用默认参数")

    def __sigmoid(self, z):
        s = 1.0 / (1 + np.exp(-z))
        return s

    def __derivative_sigmoid(self, z):
        return self.__sigmoid(z) * (1.0 - self.__sigmoid(z))

    def __propagate(self, X, y):
        m, n = X.shape
        y = y.reshape(1, m)

        # 前向传播
        z = np.dot(self.w.T, X.T) + self.b  # 1xn x nxm = 1xm
        a = self.__sigmoid(z)  # 1xm

        cost = (-1.0 / m) * np.sum(y * np.log(a) + (1-y) * np.log(1-a))

        # 反向传播
        dz = a - y  # 1xm
        dw = (1.0 / m) * np.dot(X.T, dz.T)  # nxm * mx1 = nx1
        db = (1.0 / m) * np.sum(dz)  # 1

        assert(dw.shape == self.w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def fit(self, X, y, samp_weights=None):
        """
        训练函数
        :param X: 训练特征数据, m x n
        :param y: 训练目标数据, 1 x m
        :param samp_weights: 样本权重
        :return: 
        """
        X = np.array(X)
        y = np.array(y)

        # 判断
        assert(X.ndim == 2)
        assert(X.shape[0] == y.shape[0])

        # 初始化参数
        self.__initialize_weights(X.shape[1], self.init_mode)

        # 优化训练
        for i in range(self.max_iter):

            grads, cost = self.__propagate(X, y)

            dw = grads.get("dw")
            db = grads.get("db")

            # 更新参数
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            # 记录cost
            if i % 10 == 0:
                self.costs.append(cost)

            # 打印日志
            if self.verbose and i % 100 == 0:
                print("训练第 {} 轮, 训练误差为: {}".format(i, cost))

    def __predict(self, X):
        assert(X.ndim == 2)

        pred_proba = np.dot(self.w.T, X.T) + self.b

        return pred_proba

    def predict(self, X):
        pred = (self.__predict(X) >= 0.5).astype(np.int64).squeeze()
        return pred

    def predict_proba(self, X):
        return self.__predict(X)

    @property
    def score(self):
        return NotImplemented

    @property
    def get_params(self):
        self.params = {
            "penalty": self.penalty,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "reg_coef": self.reg_coef,
            "learning_rate": self.learning_rate,
            "fit_intercept": self.fit_intercept,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "solver": self.solver,
            "multi_class": self.multi_class,
            "warm_start": self.warm_start,
            "init_mode": self.init_mode,
            "w": self.w,
            "b": self.b
        }
        return self.params


if __name__ == "__main__":
    # test case
    pass







