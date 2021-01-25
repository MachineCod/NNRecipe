class Flatten:
    """Flattening layer for multidimentional input"""

    def __call__(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def _forward(self, X):
        self._cache = X.shape
        return X.reshape(X.shape[0], -1)
    
    def calc_local_grad(self, dL):
        print("Flatten", self._cache, dL.shape)
        return {
            'dY': dL.reshape(self._cache)
            }