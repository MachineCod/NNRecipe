from .__loss_function import LossFunction
import numpy as np


class MClassLogisticLoss(LossFunction):
    """
    This class is responsible for calculating Cross Entropy Loss for multiclass

    - Input is the output of the last layer (not probabilities)
    - Softmax is applied to the input to get the probabilities of each class
    - Loss is -log(probability of the correct class)

     Examples
    ---------
    >>> from nn_recipe.utility import OneHotEncoder
    >>> import  numpy as np
    >>> X = np.array([[5, 6, 7], [5, 6, 7]])
    >>> Y = np.array([["Ahmed"], ["Ali"]])
    >>> encoder = OneHotEncoder(types=["Mohamed", "Ahmed", "Ali"])
    >>> loss(encoder.encode(Y), X)
    np.array([1.40760596, 0.40760596])
    >>> loss.local_grad
    np.array([[ 0.09003057, -0.75527153, 0.66524096], [0.09003057, 0.24472847, -0.33475904]]

    :ivar __softmax_value: Value of the softmax layer applied to the input
    :type __softmax_value: np.ndarray
    """
    ID = 3

    def _compute_loss(self, Y, Y_hat):
        """

        :param Y:
        :param Y_hat:
        :return:
        """
        expZ = np.exp(Y_hat - np.max(Y_hat))
        self.__softmax_value = expZ / np.sum(expZ, axis=1).reshape(-1, 1)
        return -1*np.log(self.__softmax_value[0, np.argmax(Y, axis=1)])

    def _compute_local_grad(self, Y, Y_Hat):
        self.__softmax_value[range(self.__softmax_value.shape[0]), np.argmax(Y, axis=1)] -= 1
        return self.__softmax_value
