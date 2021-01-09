from src.NN.function import Function
import numpy as np


class Sigmoid(Function):
    """
    Class represents the sigmoid activation function

    >>> x = np.array([1, 2, 3])         # input vector
    >>> f = Sigmoid()                   # creating sigmoid object
    >>> print(f(x))                     # calculating sigmoid of the input
    >>> print(f.local_grad)             # get local_grad of the sigmoid at the input
    """
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def _forward(self, x):
        """
        - Forward path of the sigmoid function
        - sig(x) = 1 / (1 + exp(-x))
        - visit https://en.wikipedia.org/wiki/Sigmoid_function to get more info about sigmoid
        :param x: input that is wanted to calculate the sigmoid at
        :return: sigmoid value at input x
        :rtype: np.ndarray
        """
        return 1 / (1 + np.exp(-1*x))

    def _calc_local_grad(self, x):
        """
        - gradient path of the sigmoid function
        - ∇ sig(x) = a*(1-a),    a --> sig(x)
        - visit https://en.wikipedia.org/wiki/Sigmoid_function to get more info about sigmoid
        :param x: input that is wanted to calculate the sigmoid at
        :return: sigmoid gradient at input x
        :rtype: np.ndarray
        """
        return self._cache * (1 - self._cache)
