from .__loss_function import LossFunction
import numpy as np


class MeanSquaredLoss(LossFunction):
    ID = 2

    def _compute_loss(self, Y, Y_hat):
        """
        Args:
            Y(array(N, 1)): vector of desired output
            Y_Hat(array(N, 1)): vector of actual output
        returns the loss value
        """
        return (1/(2*Y.shape[0]))*np.dot((Y_hat - Y).T, (Y_hat - Y))

    def _compute_local_grad(self, Y, Y_Hat):
        """returns the loss derivative wrt. the last output layer"""
        return Y_Hat - Y
