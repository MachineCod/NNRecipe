from nn_recipe.NN.__function import Function
from abc import abstractmethod


class ActivationFunction(Function):
    """
    Base Class for all activation functions, All Activation functions must implement this class to add the save
    functionality

    :cvar ID: unique if for each activation function class
    :type ID: int
    """
    ID = -1

    def save(self):
        """
        Returns the activation function ID to be saved in the save phase
        """
        return self.ID
