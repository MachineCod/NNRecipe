from .gd import GD
import numpy as np


class GDLeakyAdaGrad(GD):
    """
    :note: This class update the weights of the passed layer by using the leaky AdaGrad method.
    the needed hyper-parameters in this class are the learning rate (alpha), (roh).
    """
    ID = 7

    def __init__(self, roh, *args, **kwargs):
        """
        :param roh: hyper-parameter set empirically by the user, value ]0 , 1[
        :type roh: positive real number
        """
        super(GDLeakyAdaGrad, self).__init__(*args, **kwargs)
        self._roh=roh

    def optimize(self, layer, delta: np.ndarray, number_of_examples, *args, **kwargs) -> None:
        """
        :note:  1-This function update the layer weights according to this algorithm
                weights = weights - root( learning rate / (A + epsilon) *  ∂L/∂W
                A = roh * A + (1 - roh) * square(∂L/∂W)
                A is an accumulator initialized by zero matrix with dimensions like the layer's weights
                A has two parts the accumulator of the weights part and bias part
                A for weights ==> layer.a     for bias ==> layer.ao
                2- Initialize an accumulator as a layer attribute if it is not already there
        :param layer: a layer in the training network
        :type layer: layer
        :param delta: the chain rule of multiplying the partial derivative of the loss function by the desired
                      layer weights passing by all activation functions (backpropagation)
        :type delta: np.ndarray
        :param number_of_examples: number of examples in the dataset
        :type number_of_examples: positive integer
        """
        delta_w, delta_b = self.update_delta(layer, delta, number_of_examples)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(delta_w)
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(delta_b)

        layer.weights = layer.weights - np.multiply(self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5), delta_w)
        layer.bias = layer.bias - np.multiply(self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5), delta_b)

    def flush(self, layer):
        """
        :note: This function deletes the added attributes to objects (accumulators)
        :param layer: a layer in the training network
        :type layer: layer
        """
        del layer.ao
        del layer.a

    def _save(self):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        return {
            "lr": self._learning_rate,
            "roh": self._roh,
        }

    @staticmethod
    def load(data):
        """
        This function loads the hyper parameters of the optimizing technique
        """
        return GDLeakyAdaGrad(learning_rate=data["lr"], roh=data["roh"])


