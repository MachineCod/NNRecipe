from abc import ABC, abstractmethod
from nn_recipe.NN.__function import Function
import numpy as np


class LossFunction(Function):
    """
    This class is used as a base class for all loss function, and loss function must implement this class

    Examples
    --------

    >>> x = np,array([[4, 5, 6]])
    >>> act = LossFunction() # you can't create an instance from loss function as it's an abstract class
    >>> a = act(x)
    >>> grad = a.local_grad

    :cvar ID: unique id used by the loss function factory to build loss functions from the network descriptor
    :type ID: int
    :ivar __sum: flag used to sum the output of loss function (may be used in case of multiclass loss function)
    :type __sum: bool
    :ivar __axis: axis index at which the numpy sum will happen
    :type __axis: int
    """
    ID = -1

    def __init__(self, sum=False, axis=0):
        """
        :param sum: flag used to sum the output of loss function (may be used in case of multiclass loss function)
        :type sum: bool
        :ivar axis: axis index at which the numpy sum will happen
        :type axis: int
        """
        super(LossFunction, self).__init__()
        self.__sum = sum
        self.__axis = axis

    def _forward(self, Y, Y_hat):
        """
        This function is called when the object is called, the function calls it's subclass compute_loss then checks for
        the sum flag to sum the losses values in the specific axis
        """
        loss = self._compute_loss(Y, Y_hat)
        if self.__sum:
            loss = np.sum(loss, axis=self.__axis)
            if self.__axis == 0: loss = loss.reshape((1, -1))
            else: loss = loss.reshape((-1, 1))
        return loss

    @abstractmethod
    def _compute_loss(self, Y, Y_hat):
        """ Abstract method must be implemented by the user to compute the loss (forward path) """
        pass

    def _calc_local_grad(self, Y, Y_hat):
        grad = self._compute_local_grad(Y, Y_hat)
        return grad

    @abstractmethod
    def _compute_local_grad(self, Y, Y_hat):
        """ Abstract method must be implemented by the user to compute the loss gradient (forward path) """
        pass

    def save(self):
        """
       Methode used to get the data that will be saved in the save phase

       Expected Descriptor Structure:
          - ID: unique id for each layer (0 in case of Linear Layer)
          - sum: flag to indicate whether loss sum is needed or not
          - axis: axis at which the losses will be summed
          """
        return {
            "ID": self.ID,
            "sum": self.__sum,
            "axis": self.__axis
        }