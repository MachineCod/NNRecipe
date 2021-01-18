from src.opt.GD import GD
import numpy as np


class GDLeakyAdaGrad(GD):
    def __init__(self,roh,*args, **kwargs):
        super(GDLeakyAdaGrad, self).__init__(*args, **kwargs)
        self.__roh=roh

    def optimize(self, layer, delta : np.ndarray) -> None:
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = self.__roh * layer.a + (1 - self.__roh) * np.square(np.dot(delta, layer.local_grad["dW"]))
        layer.ao = self.__roh * layer.ao + (1 - self.__roh) * np.square(np.sum(delta)/delta.shape[1])
        "equivalent to taking square root of each alphat"
        #should add epsilon(avoid zero in denominator)
        layer.weights = layer.weights - np.multiply(self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5), np.dot(delta, layer.local_grad["dW"]))
        layer.bias = layer.bias - np.multiply(self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5), np.sum(delta)/delta.shape[1])