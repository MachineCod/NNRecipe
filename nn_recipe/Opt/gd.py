from.__optimizer import Optimizer
from ..NN.Layers.conv import Conv2D
import numpy as np


class GD(Optimizer):
    ID = 0

    def __init__(self, learning_rate: float = 0.01, *args, **kwargs):
        if learning_rate <= 0:
            raise Optimizer.LearningRateValueError(learning_rate)
        if type(learning_rate) is not float and type(learning_rate) is not int:
            raise Optimizer.LearningRateTypeError(type(learning_rate))
        self._learning_rate = learning_rate

    def update_delta(self, layer, delta: np.ndarray):
        """

        :param layer:
        :type layer: Layer
        :param delta:
        :type delta: np.ndarray
        """
        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w,delta_b

    def optimize(self, layer, delta: np.ndarray, *args, **kwargs) -> None:
        if isinstance(layer, Conv2D):
            layer.weights = layer.weights - self._learning_rate * delta
        else: 
            delta_w, delta_b = self.update_delta(layer, delta)
            layer.weights = layer.weights - self._learning_rate * delta_w
            layer.bias = layer.bias - self._learning_rate * delta_b

    def flush(self, layer):
        pass

    def _save(self):
        return {
            "lr": self._learning_rate
        }

    @staticmethod
    def load(data):
        return GD(learning_rate=data["lr"])

