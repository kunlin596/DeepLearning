# -*- coding: utf-8 -*-
# !/usr/bin/env python
""" Activation Functions

This module includes all kinds of function that is useful
in deep learning, including all kinds of often used activation functions.

About activation function:

    must use non-linear functions,
    otherwise no matter how many middle layers you have,
    The output result will be,

        y = c1 * c2 * ... * cn * x

    and it is non sense in the context of neural network,
    since it simple acts like only one layer.

"""

import numpy as np


def sigmoid (x):
    """ Sigmoid Function

    This function takes x (either a plain number or a NumPy array),
    and perform the sigmoid function calculation.

    (Duck typing of parameters)
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def step (x):
    """ Step Function

    This function takes x (either a plain number or a NumPy array),
    and perform the step calculation.

    (Duck typing of parameters)

    If x is a NumPy array, for example,

        x = [1., -3., 2.]

    then the return value will be,

        y = [1, 0, 1]

    :param x:
    :return:
    """
    # y = x[x > 0]  # can be written as y = x > 0, but with warning
    y = x > 0
    return y.astype(np.int)


def relu (x):
    """ ReLU Function

    Rectified Linear Unit. This is recently broadly used activation function.

        f(x) = x, if x > 0
        f(x) = 0, if x <= 0

    :param x:
    :return:
    """
    return np.maximum(0, x)


def identity (x):
    """ Identity Function

    Return the input itself. Used by the output layer.
    Often used for classification problem

    :param x:
    :return:
    """
    return x


def soft_max (x):
    """ Soft Max Function

    Definition, used by the output layer.

        y_k = exp(a_k) / (sum(exp(a_i) from i to n))

    Often used in regression problem (make prediction)

    The below implementation will probably cause a overflow problem,

        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    for example,

        x = [1000, 1000 ,1000]

    The sum of the outputs will adds up to 1.

    :param x:
    :return:
    """
    c = np.max(x)
    exp_x = np.exp(x - c)  # overflow protection
    return exp_x / np.sum(exp_x)


def mean_squared_error (y, t):
    """ Mean Squared Error (MSE)

    Used as the loss function

    :param y
        The prediction
    :param t
        Label of training data
    :return:
        MSE between y and t
    """
    return np.sum((y - t) ** 2) * 0.5


def cross_entropy_error (y, t, one_hot_label = True, batch_size = None):
    """ Cross Entropy Error (CEE)

    Used as the loss function

    :param y
        The prediction
    :param t
        Label of training data
    :param one_hot_label
        TODO
    :param batch_size
        TODO
    :return:
        CEE between y and t
    """

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, t.size)

    delta = 1e-7  # protection of case of "log(0) -> inf"

    if one_hot_label:
        """ One hot label presentation,
        y[np.arange(batch_size), t] will produce the right answer

        Example:
            batch_size= = 5
            t = [0, 1, 0, 0, 0]
            np.arange(batch_size) = [0, 1, 2, 3, 4]

            y[np.arange(batch_size), t] = [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]]

            y[0] = [0.1, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01] # no.1 data in this batch
            y[0, 2] = 0.6 # The prediction of slot 0
        """
        return -(np.sum(np.log(y[np.arange(batch_size), t])) / batch_size)
    else:
        """ Non one hot label presentation
        """
        return -np.sum(t * np.log(y + delta)) / batch_size  # log = ln


def numerical_differentiation (func, x):
    """ Numerical Differentiation Computation

    :param func:
        Function to be evaluated
    :param x:
        point x
    :return:
        differentiation of function at point x
    """
    h = 0.0001
    return (func(x + h) - func(x - h)) / (2 * h)


def tangent_line (func, x):
    """ Tangent Line

    Compute k using numerical differentiation, and use y = kx + b to compute b.

    :param func:
    :param x:
    :return:
    """
    k = numerical_differentiation(func, x)
    b = func(x) - k * x
    return lambda t: k * t + b


def numerical_gradient (func, x):
    """ Numerical Gradient Implementation v1

    Compute the gradient at point x of function.

    x is a numpy array, by changing x[i] with +h/-h,
    we can get the differential of only x[i], since every elements will be 0 (f1 == f2)

    Example:

        numerical_gradient(func, np.array([2.0, 3.0]))
        numerical_gradient(func, np.array([4.0, 2.0]))
        numerical_gradient(func, np.array([2.0, 1.0]))

    :param func:
        partial derivative of some function
    :param x:
        point to be evaluated
    :return:

    """
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]

        x[i] = tmp_val + h
        f1 = func(x)

        x[i] = tmp_val - h
        f2 = func(x)

        grad[i] = (f1 - f2) / (2 * h)

    return


def gradient_descent (func, init_x, eta, max_step):
    """ Gradient Descent Method

    Gradient descent method for updating the x and find the local minimum with learning rate eta.

    :param func: the
    :param init_x: initial number of x
    :param eta: update rate (descent rate)
    :param max_step: maximum number of update
    :return:
    """
    x = init_x

    for i in range(max_step):
        grad = numerical_differentiation(func, x)  # the gradient vector at x
        x -= eta * grad  # move x along the gradient vector
    return x
