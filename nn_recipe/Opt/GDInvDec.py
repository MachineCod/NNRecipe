from .GDExpDec import GDExpDec
import numpy as np


class GDInvDec(GDExpDec):
    ID = 5

    def __init__(self, *args, **kwargs):
        super(GDInvDec, self).__init__(*args, **kwargs)

    def update_delta(self, layer, delta: np.ndarray):

        delta_w = np.dot(delta, layer.local_grad["dW"]) / layer.weights.shape[1]
        delta_b = np.sum(delta, axis=1).reshape(-1, 1) / delta.shape[1]
        return delta_w,delta_b

    def optimize(self, layer, delta: np.ndarray, iteration, *args, **kwargs) -> None:
        delta_w, delta_b = self.update_delta(layer, delta)
        learning_rate = self._learning_rate / (1 + self._k * iteration)
        layer.weights = layer.weights - learning_rate * delta_w
        layer.bias = layer.bias - learning_rate * delta_b

    def flush(self, layer):
        pass

    def _save(self):
        return {
            "lr": self._learning_rate,
            "k": self._k,
        }

    @staticmethod
    def load(data):
        return GDInvDec(learning_rate=data["lr"], k=data["k"])




