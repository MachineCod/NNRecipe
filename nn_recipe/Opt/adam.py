from .leakyAdagrad import GDLeakyAdaGrad
import numpy as np


class GDAdam(GDLeakyAdaGrad):
    ID = 3

    def __init__(self, beta=0.999, *args, **kwargs):
        super(GDAdam, self).__init__(*args, **kwargs)
        self.__beta = beta

    def update_delta(self, layer, delta: np.ndarray):
        delta_w = np.dot(delta, layer.local_grad["dW"]) / 1
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w, delta_b

    def optimize(self, layer, delta: np.ndarray, iteration, *args, **kwargs) -> None:
        delta_w, delta_b = self.update_delta(layer, delta)
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        if not hasattr(layer, "f"):
            layer.f = np.zeros_like(layer.weights)
            layer.fo = np.zeros_like(layer.bias)

        layer.a = self._roh * layer.a + (1 - self._roh) * np.square(delta_w)
        layer.ao = self._roh * layer.ao + (1 - self._roh) * np.square(delta_b)

        layer.f = self.__beta * layer.f + (1 - self.__beta) * delta_w
        layer.fo = self.__beta * layer.fo + (1 - self.__beta) * delta_b

        learning_rate = self._learning_rate * np.power(1 - np.power(self._roh, iteration), 0.5) / \
                        (1 - np.power(self.__beta, iteration))

        layer.weights = layer.weights - learning_rate / np.power(layer.a, 0.5) * layer.f
        layer.bias = layer.bias - learning_rate / np.power(layer.ao, 0.5) * layer.fo

    def flush(self, layer):
        del layer.ao
        del layer.a
        del layer.fo
        del layer.f

    def _save(self):
        return {
            "lr": self._learning_rate,
            "beta": self.__beta,
            "roh": self._roh
        }

    @staticmethod
    def load(data):
        return GDAdam(learning_rate=data["lr"], beta=data["beta"], roh=data["roh"])
