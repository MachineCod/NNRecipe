from .__activation import ActivationFunction
import numpy as np


class Tanh(ActivationFunction):
    """
    Class represents the tanh activation function

    >>> x = np.array([1, 2, 3])           # input vector
    >>> f = Tanh()                        # creating tanhobject
    >>> print(f(x))                       # calculating tanh of the input
    >>> print(f.local_grad)               # get local_grad of the tanh at the input
    """

    def __init__(self):
        super(Tanh, self).__init__()


    def _forward(self, x):
        """
        - Forward pass of the Tanh function
        - tanh(x)
        - visit ////////// for more info on tanh func
        :param x: input that is wanted to calculate the tanh at
        :return: tanh value at input x
        :rtype: np.ndarray
        """

        return np.tanh(x)

    def _calc_local_grad(self, x):
        """
        - Backward pass of the tanh function
        - ∇ Tanh = 1-tanh**2
        - visit //////////////////////
        to get more info about Tanh
        :param x: input that is wanted to calculate the Tanh at
        :return: Tanh gradient at input x
        :rtype: np.ndarray
        """
        return 1 - (np.tanh(x) ** 2)
