from .gd import GD
import numpy as np


class GDAdaDelta(GD):
    ID = 1

    def __init__(self, roh, *args, **kwargs):
        super(GDAdaDelta, self).__init__(*args, **kwargs)
        self.__roh = roh

    def update_delta(self, layer, delta: np.ndarray):
        delta_w = np.dot(delta, layer.local_grad["dW"]) / 1
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w, delta_b

    def optimize(self, layer, delta: np.ndarray, *args, **kwargs) -> None:
        delta_w, delta_b = self.update_delta(layer, delta)
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
        del layer.ao
        del layer.a
        del layer.ko
        del layer.k

    def _save(self):
        return {
            "lr": self._learning_rate,
            "roh": self.__roh
        }

    @staticmethod
    def load(data):
        return GDAdaDelta(learning_rate=data["lr"], roh=data["rho"])
