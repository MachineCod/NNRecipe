from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    This class is responsible for updating the weights of the layer passed to is with different update methods,
    each method is given a unique positive ID to be called with while the abstract class (Optimizer) is given a negative
    ID as it will never be called
    """

    ID = -1

    @abstractmethod
    def optimize(self, layer, delta, number_of_examples, *args, **kwargs):
        pass

    class LearningRateValueError(Exception):
        """
        :note: This class is responsible for handling the negative values of the input learning rate, The correct value
        is a positive real number otherwise this class raise an error telling the user what is wrong with his/her input
        """
        def __init__(self, learning_rate_value):
            message = "Optimizer learning rate must be greater than zero, current value is " + str(learning_rate_value)
            super().__init__(message)

    class LearningRateTypeError(Exception):
        """
        This class is responsible for handling the string values of the input learning rate, The correct value
        is a positive real number otherwise this class raise an error telling the user what is wrong with his/her input
        """
        def __init__(self, learning_rate_type):
            message = "Optimizer learning rate must be a scalar real number current type is " + str(learning_rate_type)
            super().__init__(message)

    @abstractmethod
    def flush(self, layer):
        """
        :note: This function is used to remove the attributes the update weights method (optimize) creates to be used
        as accumulators after the training procedures is finished
        :param layer: a layer in the training network
        :type layer: layer
        """
        pass

    @abstractmethod
    def _save(self):
        pass

    def save(self):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        out = self._save()
        out["ID"] = self.ID
        return out

    @staticmethod
    @abstractmethod
    def load(data):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        pass
