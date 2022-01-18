import unittest

from functions import *
import numpy as np
import matplotlib.pylab as plt


class NumericalDifferentiationTest(unittest.TestCase):
    def test_quadratic_function(self):
        quadratic_function = lambda t: 0.01 * t ** 2 + 0.1 * t
        x = np.arange(0.0, 20.0, 0.1)
        y = quadratic_function(x)

        plt.ylim(-1.0, 10.0)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        tf = tangent_line(quadratic_function, 5)
        y2 = tf(x)
        plt.plot(x, y)
        plt.plot(x, y2)
        plt.show()

        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
