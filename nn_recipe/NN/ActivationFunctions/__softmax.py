from nn_recipe.NN.__function import Function
import numpy as np

# TODO needs to be edited soon


class Softmax(Function):
    """
    Class represents the softmax activation function

    >>> x = np.array([1, 2, 3])          # input vector
    >>> f = Softmax(x)                   # creating softmax object
    >>> print(f(x))                      # calculating softmax of the input

    for more info about softmax implementation visit:   https://deepnotes.io/softmax-crossentropy
https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def _forward(self, x, y, *args, **kwargs):
        """
        - Calculates the probabilities of the output nodes
        - softmax(x) = exp(x) / sum(exp(x[i])) i: 0-->N, N: number of classes
        - visit https://en.wikipedia.org/wiki/Softmax_function to get more info about softmax
        :param x: input that is wanted to calculate the softmax at
        :return: softmax value at input x
        :rtype: np.ndarray
        """
        expZ = np.exp(x - np.max(x))
        out = expZ / np.sum(expZ, axis=0)
        print(out.shape)
        return out

    def _calc_local_grad(self, x, y, *args, **kwargs):
        # print(self._cache[np.argmax(y, axis=0)])
        return np.multiply(self._cache[np.argmax(y, axis=0)][:,0].reshape((1,-1)), y-self._cache)

"""
# TODO Ask Eng/Rashad for this (we get dYtrue/dZ (vector) not dY/dZ (matrix)
x -> column matrix
YTrue = Y1
[ yTrue(0 - Y0) ]             [ HH - Yi ]             [ 0 ]
[ yTrue(1 - Y1) ] --> yTrue * [ HH - Yi ] --> Y[1] * ([ 1 ] - Y)
[ yTrue(0 - Y2) ]             [ HH - Yi ]             [ 0 ]

dYTrue/dZTrue = YTrue(1-YTrue)
dYTrue/dZi = -YTrue*Yi

"""