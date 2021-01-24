from .__activation import ActivationFunction

import numpy as np



class HardTanh(ActivationFunction):
    """
    Class represents the hardtanh activation function

    >>> x = np.array([1, 2, 3])           # input vector
    >>> f = HardTanh()                    # creating HardTanh object
    >>> print(f(x))                       # calculating HardTanh of the input
    >>> print(f.local_grad)               # get local_grad of the HardTanh at the input
    """

    def __init__(self):
        super(HardTanh, self).__init__()

    def _forward(self, x):
        """
        - Forward pass of the HardTanh function
        - hardtanh(x) ={
                        1   x>1
                       -1   x<-1
        }
        - visit ////////// for more info on HardTanh func
        :param x: input that is wanted to calculate the HardTanh at
        :return: HardTanh value at input x
        :rtype: np.ndarray
        """
        X = np.copy(x)
        X[x > 1] = 1
        X[x < -1] = -1
        return X  # or np.maximum(-1, np.minimum(1, x))

    def _calc_local_grad(self, x):
        """
        - Backward pass of the tanh function
        - ∇ HardTanh = {
                            0        1 < x < -1
                            1       -1 <= x < 1
        }
        - visit //////////////////////
        to get more info about HardTanh
        :param x: input that is wanted to calculate the HardTanh at
        :return: HardTanh gradient at input x
        :rtype: np.ndarray
        """
        X = np.copy(x)
        X[x < -1 and x > 1] = 0
        X[x >= -1 and x < 1] = 1
        return X


