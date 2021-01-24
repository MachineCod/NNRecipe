from .__activation import ActivationFunction
import numpy as np


class ReLU(ActivationFunction):
    """
    Class represents the relu activation function

    >>> x = np.array([1, 2, 3])         # input vector
    >>> f = ReLU()                      # creating relu object
    >>> print(f(x))                     # calculating relu of the input
    >>> print(f.local_grad)             # get local_grad of the relu at the input
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def _forward(self, x):
        """
        - Forward pass of the relu function
        - relu(x) = x * max(0, x)
        - visit https://en.wikipedia.org/wiki/Sigmoid_functionhttps://en.wikipedia.org/wiki/Rectifier_(neural_networks)
         to get more info about relu
        :param x: input that is wanted to calculate the relu at
        :return: relu value at input x
        :rtype: np.ndarray
        """
        # x * (x > 0).astype(x.dtype)
        X = np.copy(x)
        return  np.maximum(0, X)

    def _calc_local_grad(self, x):
        """
        - Backward pass of the relu function
        - ∇ relu(x) = max(0, x)
        - visit https://en.wikipedia.org/wiki/Sigmoid_functionhttps://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        to get more info about sigmoid
        :param x: input that is wanted to calculate the relu at
        :return: relu gradient at input x
        :rtype: np.ndarray
        """
        X = np.copy(x)
        return (X > 0).astype(X.dtype)