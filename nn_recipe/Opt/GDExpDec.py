from .gd import GD
import numpy as np


class GDExpDec(GD):
    """
    :note: This class update the weights of the passed layer by using the exponential decay method.
    the needed hyper-parameters in this class are the learning rate (alpha), iteration number: number of epochs and
    k: which is a positive real number set by the user
    """
    ID = 4

    def __init__(self, k, *args, **kwargs):
        """
        :param learning_rate: learning rate
        :type learning_rate: positive real number with default value 0.01
        :param k: hyper-parameter
        :type k: positive real number
        """
        super(GDExpDec, self).__init__(*args, **kwargs)
        self._k = k

    def optimize(self, layer, delta: np.ndarray, number_of_examples, iteration, *args, **kwargs) -> None:
        """
          :note:  This function update the layer weights according to this algorithm
                modified learning rate = learning rate * exp(-k * iteration number)
                weights = weights - modified learning rate *  ∂L/∂W
        :param layer: a layer in the training network
        :type layer: layer
        :param delta: the chain rule of multiplying the partial derivative of the loss function by the desired
                      layer weights passing by all activation functions (backpropagation)
        :type delta: np.ndarray
        :param iteration: number of epochs
        :type iteration: positive integer
        :param number_of_examples: number of examples in the dataset
        :type number_of_examples: positive integer
        """
        delta_w, delta_b = self.update_delta(layer, delta, number_of_examples)
        learning_rate = self._learning_rate * np.exp(-self._k * iteration)
        layer.weights = layer.weights - learning_rate * delta_w
        layer.bias = layer.bias - learning_rate * delta_b

    def flush(self, layer):
        pass

    def _save(self):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        return {
            "lr": self._learning_rate,
            "k": self._k,
        }

    @staticmethod
    def load(data):
        """
        This function loads the hyper parameters of the optimizing technique
        """
        return GDExpDec(learning_rate=data["lr"], k=data["k"])





