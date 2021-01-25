from src.opt.GD import GD
import numpy as np


class GDAdaGrad(GD):
    def __init__(self,*args, **kwargs):
        super(GDAdaGrad, self).__init__(*args, **kwargs)


    def optimize(self, layer, delta : np.ndarray) -> None:
        if not hasattr(layer, "a"):
            layer.a = np.zeros_like(layer.weights)
            layer.ao = np.zeros_like(layer.bias)

        layer.a = layer.a + np.square(np.dot(delta, layer.local_grad["dW"]))
        layer.ao = layer.ao + np.square(np.sum(delta)/delta.shape[1])
        "equivalent to taking square root of each alphat"
        #should add epsilon(avoid zero in denominator)
        layer.weights = layer.weights - np.multiply(self._learning_rate * np.power(layer.a + np.finfo(float).eps, -0.5), np.dot(delta, layer.local_grad["dW"]))
        layer.bias = layer.bias - np.multiply(self._learning_rate * np.power(layer.ao + np.finfo(float).eps, -0.5), np.sum(delta)/delta.shape[1])