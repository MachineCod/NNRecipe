from .gd import GD
import numpy as np


class GDMomentum(GD):
    """
    :note: This class update the weights of the passed layer by using the Momentum method.
    the needed hyper-parameters in this class are the learning rate (alpha), beta.
    """
    ID = 6

    def __init__(self, beta, *args, **kwargs):
        """
        :param learning_rate: learning rate
        :type learning_rate: positive real number with default value 0.01
        :param beta: hyper-parameter value ]0 , 1[
        :type beta: positive real number
        """
        super(GDMomentum, self).__init__(*args, **kwargs)
        self._beta = beta

    def optimize(self, layer, delta: np.ndarray, number_of_examples, *args, **kwargs) -> None:
        """
        :note:  1-This function update the layer weights according to this algorithm
                weights = weights + velocity
                velocity = beta * velocity - learning rate * ∂L/∂W
                velocity is an accumulator initialized by zero matrix with dimensions like the layer's weights
                velocity has two parts the accumulator of the weights part and bias part
                velocity for weights ==> layer.v     for bias ==> layer.vo
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
        if not hasattr(layer, "v"):
            layer.v = np.zeros_like(layer.weights)
            layer.vo = np.zeros_like(layer.bias)

        layer.v = self._beta * layer.v - self._learning_rate * delta_w
        layer.vo = self._beta * layer.vo - self._learning_rate * delta_b
        layer.weights = layer.weights + layer.v
        layer.bias = layer.bias + layer.vo

    def flush(self, layer):
        """
        :note: This function deletes the added attributes to objects (accumulators)
        :param layer: a layer in the training network
        :type layer: layer
        """
        del layer.vo
        del layer.v

    def _save(self):
        """
        This function saves the hyper parameters of the optimizing technique
        """
        return {
            "lr": self._learning_rate,
            "beta": self._beta,
        }

    @staticmethod
    def load(data):
        """
        This function loads the hyper parameters of the optimizing technique
        """
        return GDMomentum(learning_rate=data["lr"], beta=data["beta"])




