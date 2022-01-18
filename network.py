# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy as np
from functions import *
from mnist import load_mnist
import pickle


def get_data(normalize=True, flatten=True, one_hot_label=True):
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=normalize, flatten=flatten, one_hot_label=one_hot_label
    )
    return x_train, t_train, x_test, t_test


def init_network():
    network = dict()
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


def read_network(name):
    with open(name, "rb") as f:
        net = pickle.load(f)

    return net


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3

    y = identity(a3)

    return y


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def forward(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.forward(x)
        y = soft_max(z)
        loss = cross_entropy_error(y, t, one_hot_label=True, batch_size=1)
        return loss
