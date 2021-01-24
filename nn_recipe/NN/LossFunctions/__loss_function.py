from abc import ABC, abstractmethod
from nn_recipe.NN.__function import Function
import numpy as np


class LossFunction(Function):
    ID = -1

    def __init__(self, sum=False, axis=0):
        super(LossFunction, self).__init__()
        self.__sum = sum
        self.__axis = axis

    def _forward(self, Y, Y_hat):
        loss = self._compute_loss(Y, Y_hat)
        if self.__sum:
            loss = np.sum(loss, axis=self.__axis)
            if self.__axis == 0: loss = loss.reshape((1, -1))
            else: loss = loss.reshape((-1, 1))
        return loss

    @abstractmethod
    def _compute_loss(self, Y, Y_hat):
        pass

    def _calc_local_grad(self, Y, Y_hat):
        grad = self._compute_local_grad(Y, Y_hat)
        # if self.__sum:
        #     grad = np.sum(grad, axis=self.__axis)
        #     if self.__axis == 0: grad = grad.reshape((1, -1))
        #     else: grad = grad.reshape((-1, 1))
        return grad

    @abstractmethod
    def _compute_local_grad(self, Y, Y_hat):
        pass

    def save(self):
        return {
            "ID": self.ID,
            "sum": self.__sum,
            "axis": self.__axis
        }