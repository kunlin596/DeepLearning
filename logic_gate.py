# -*- coding: utf-8 -*-
# !/usr/bin/env python

""" Logic Gates

This file consists of the basic logic gates,
which takes the linear combination of the input vector,
and use step function

    f(x) = 0, if x <= 0
    f(x) = 1, if x > 0

as the activation function of a perceptron.

Example:

    and_gate(1, 1) -> 1
    and_gate(1, 0) -> 0
    and_gate(0, 1) -> 0
    and_gate(0, 0) -> 0

TODO:

    1) The implementation of XNOR gate

"""

import numpy as np


def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def nand_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def or_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def xor_gate(x1, x2):
    s1 = nand_gate(x1, x2)
    s2 = or_gate(x1, x2)
    return and_gate(s1, s2)


def xnor_gate(x1, x2):
    pass
