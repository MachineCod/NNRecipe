from .gd import GD
import numpy as np


class GDAdaDelta(GD):
    """
    :note: This class update the weights of the passed layer by using AdaDelta method, the needed hyper-parameters in
    this class is (roh).
    Although GDAdaDelta inherits GD class, GDAdaDelta doesn't use it as it has dynamic learning rate
    """
    ID = 1

    def __init__(self, roh=0.9, *args, **kwargs):
        """
        :param roh: hyper-parameter set empirically by the user, value ]0 , 1[
        :type roh: positive real number
        """
        super(GDAdaDelta, self).__init__(*args, **kwargs)
        self.__roh = roh

    def optimize(self, layer, delta: np.ndarray, number_of_examples, *args, **kwargs) -> None:
        """
        :note:  1-This function update the layer weights according to this algorithm
                weights = weights - root( S / (A + epsilon) *  ∂L/∂W
                S = roh * S + (1 - roh) * S / (A + epsilon) * square(∂L/∂W)
                A = roh * A + (1 - roh) * square(∂L/∂W)
                A and S are accumulators initialized by zero matrix with dimensions like the layer's weights
                A and S have two parts the accumulator of the weights part and bias part
                A for weights ==> layer.a     for bias ==> layer.ao
                S for weights ==> layer.k     for bias ==> layer.ko
                2- Initialize accumulators as a layer attribute if they are not already there
        :param layer: a layer in the training network
        :type layer: layer
        :param delta: the chain rule of multiplying the partial derivative of the loss function by the desired layer
                      weights passing by all activation functions (backpropagation)
        :type delta: np.ndarray
        :param number_of_examples: number of examples in the dataset
        :type number_of_examples: positive integer
        """
        delta_w, delta_b = self.update_delta(layer, delta, number_of_examples)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        if not hasattr(layer, "k"):
            layer.k = np.zeros_like(layer.weights)
            layer.ko = np.zeros_like(layer.bias)

        layer.a = self.__roh * layer.a + (1 - self.__roh) * np.square(delta_w)
        layer.ao = self.__roh * layer.ao + (1 - self.__roh) * np.square(delta_b)

        layer.k = self.__roh * layer.k + (1 - self.__roh) * layer.k / (layer.a + np.finfo(float).eps) * np.square(delta_w)
        layer.ko = self.__roh * layer.ko + (1 - self.__roh) * layer.ko / (layer.ao + np.finfo(float).eps) * np.square(delta_b)

        layer.weights = layer.weights - np.power(layer.k / (layer.a + np.finfo(float).eps), 0.5) * delta_w
        layer.bias = layer.bias - np.power(layer.ko / (layer.ao + np.finfo(float).eps), 0.5) * delta_b

    def flush(self, layer):
        """
        :note: This function deletes the added attributes to objects (accumulators)
        :param layer: a layer in the training network
        :type layer: layer
        """
        del layer.ao
        del layer.a
        del layer.ko
        del layer.k

    def _save(self):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        return {
            "lr": self._learning_rate,
            "roh": self.__roh
        }

    @staticmethod
    def load(data):
        """
        This function loads the hyper parameters of the optimizing technique
        """
        return GDAdaDelta(learning_rate=data["lr"], roh=data["rho"])
