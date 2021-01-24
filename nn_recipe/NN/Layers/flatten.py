class Flatten:
    """Flattening layer for multidimentional input"""

    def __call__(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def _forward(self, X):
        self._cache = X.shape
        return X.reshape(X.shape[0], 1, -1)
    
    def _calc_local_grad(self, dL):
        return dL.reshape(self._cache)
