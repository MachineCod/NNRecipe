from nn_recipe.NN.__function import Function
from abc import abstractmethod


class ActivationFunction(Function):
    """
    Base Class for all activation functions, All Activation functions must implement this class to add the save
    functionality

    :cvar ID: unique if for each activation function class
    """
    ID = -1

    def save(self):
        return self.ID
