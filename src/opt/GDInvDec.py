from src.opt.GDGDExpDec import GDExpDec
import numpy as np


class GDInvDec(GDExpDec):
    def __init__(self,*args, **kwargs):
        super(GDInvDec, self).__init__(*args,**kwargs)

    def optimize(self, layer, delta: np.ndarray) -> None:
        self._learning_rate = self._learning_rate / (1 + self._k * self._iteration_no)
        layer.weights = layer.weights - self._learning_rate * np.dot(delta, layer.local_grad["dW"])
        layer.bias = layer.bias - self._learning_rate * np.sum(delta) / delta.shape[1]




