from .__loss_function import LossFunction
import numpy as np
import sys

#TODO replace function call with math formula
#TODO add normalization type
# regularization

EPSILON = sys.float_info.epsilon


class CrossEntropyLoss(LossFunction):
    ID = 0

    def _compute_loss(self, Y, Y_Hat):
        """
          - computes the cross entropy loss
          -cross_entropy_loss=−(Ylog(Y_Hat)+(1−Y)log(1−Y_Hat))
          :param Y   : numpy.ndarray Should contain class labels for each data point in x.
                Y_Hat: numpy.ndarray that contain the dot product of W(weights) and x(input)
          :return:
          :rtype:
        """
        return -(np.multiply(Y, np.log(Y_Hat + EPSILON)) + np.multiply((1-Y), (np.log(1-Y_Hat + EPSILON))))

    def _compute_local_grad(self, Y, Y_Hat):
        """
        - computes the grad of cross entropy loss
        - ∇ cross_entropy_loss_drv(Y,x) =
        :param x: input that is wanted to calculate the cross_entropy_loss_drv at
               Y:numpy.ndarray Should contain class labels for each data point in x.
        :return:
        :rtype:
        """
        return (-Y/(Y_Hat + EPSILON)) + (1-Y)/(1-Y_Hat + EPSILON)

    def _save(self):
        return self.ID

