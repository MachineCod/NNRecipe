from .__loss_function import LossFunction
import numpy as np


class HingeLoss(LossFunction):
    ID = 1

    def __init__(self):
        super(HingeLoss, self).__init__()

    def _compute_loss(self, Y, Y_Hat):
        """
        - computes the hinge_loss
        - hinge_loss (Y,Y_Hat) = max(0,1-Y*Y_Hat)
        :param Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        return np.maximum(0, 1 - (Y * Y_Hat))

    def _compute_local_grad(self, Y, Y_Hat):
        """
        - computes the grad of hinge_loss
        - ∇hinge_loss (Y,x) ={
                                0   Y*Y_Hat >= 1
                               -Yx  Y*Y_Hat <  1
        }
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
               Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
        :return:
        :rtype:
        """
        grad = np.zeros_like(Y)
        grad[self._cache > 0] = Y[self._cache > 0]
        return grad
