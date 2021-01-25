from .gd import GD
import numpy as np


class GDLeakyAdaGrad(GD):
    ID = 7

    def __init__(self, roh, *args, **kwargs):
        super(GDLeakyAdaGrad, self).__init__(*args, **kwargs)
        self._roh=roh

    def update_delta(self, layer, delta: np.ndarray):

        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w,delta_b

    def optimize(self, layer, delta: np.ndarray, *args, **kwargs) -> None:
        delta_w, delta_b = self.update_delta(layer, delta)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(delta_w)
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(delta_b)

        layer.weights = layer.weights - np.multiply(self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5), delta_w)
        layer.bias = layer.bias - np.multiply(self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5), delta_b)

    def flush(self, layer):
        del layer.ao
        del layer.a

    def _save(self):
        return {
            "lr": self._learning_rate,
            "roh": self._roh,
        }

    @staticmethod
    def load(data):
        return GDLeakyAdaGrad(learning_rate=data["lr"], roh=data["roh"])


