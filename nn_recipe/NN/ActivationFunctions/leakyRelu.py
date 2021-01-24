from .__activation import ActivationFunction
import numpy as np

class LeakyReLU(ActivationFunction):
    """
    Class represents the leaky relu activation function

    >>> x = np.array([1, 2, 3])           # input vector
    >>> f = LeakyReLU()                   # creating leaky relu object
    >>> print(f(x))                       # calculating relu of the input
    >>> print(f.local_grad)               # get local_grad of the relu at the input
    """

    def __init__(self, learning_rate=0.01):
        super(LeakyReLU, self).__init__()
        self._learning_rate = learning_rate

    def _forward(self,x):
        """
        - Forward pass of the leaky relu function
        - leaky_relu(x) = {
            x           for x >= 0
            lr * x      for x < 0
        }
        - visit https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu
        to get more info about leaky relu
        :param x: input that is wanted to calculate the relu at
        :return: leakyrelu value at input x
        :rtype: np.ndarray
        """
        X = np.copy(x)
        return np.where(X > 0, X, X * self._learning_rate)

    def _calc_local_grad(self, x):
        """
        - Backward pass of the leaky relu function
        - ∇ leaky_relu(x) = {
            1           for x >= 0
            lr          for x < 0
        }
        - visit https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu
        to get more info about leaky relu
        :param x: input that is wanted to calculate the leaky relu at
        :return: leaky relu gradient at input x
        :rtype: np.ndarray
        """
        dx = np.ones_like(x)
        dx[x < 0] = self._learning_rate
        return dx
