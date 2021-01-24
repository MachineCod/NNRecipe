from .__activation import ActivationFunction
import numpy as np


class Identity(ActivationFunction):
    """
    Class represents the sigmoid activation function

    >>> x = np.array([1, 2, 3])         # input vector
    >>> f = Identity()                   # creating sigmoid object
    >>> print(f(x))                     # calculating sigmoid of the input
    >>> print(f.local_grad)             # get local_grad of the sigmoid at the input
    """

    def __init__(self):
        super(Identity, self).__init__()

    def _forward(self, x):
        """
        - Forward pass of the sigmoid function
        - Identity(x) = x
        :param x: input that is wanted to calculate the sigmoid at
        :return: Identity value at input x which is x
        :rtype: np.ndarray
        """
        return x

    def _calc_local_grad(self, x):
        """
        - Backward pass of the Identity function

        :param x: input that is wanted to calculate the identity at
        :return: identity gradient at input x
        :rtype: np.ndarray
        """
        return np.ones_like(x)
