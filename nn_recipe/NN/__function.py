from abc import ABC, abstractmethod


class Function(ABC):
    """
    - This class is the base for other classes such as Layer, Activation Functions, etc.
    - The class defines an interface of how the children objects be used

    >>> f = Function1()             # make an object from Function1 (child of Function)
    >>> out = f(X)                  # calculate the function output at input X
    >>> local_grad = f.local_grad   # get the function local grad at the input X

    - All subclasses must implement (_forward, _calc_local_grad) functions
    """
    def __init__(self):
        """ Initializing Cache variables to store the function output and local_grad"""
        self._cache = None                     # cache the function output 
        self.__grad = None                     # cache function local grad

    def __call__(self, x, *args, **kwargs):
        """Perform the function forward pass f(x), calculate the function gradient with respect to x"""
        self._cache = self._forward(x, *args, **kwargs)                 # forward pass
        self.__grad = self._calc_local_grad(x, *args, **kwargs)         # Gradient Calculation, caching
        return self._cache

    @abstractmethod
    def _forward(self, x, *args, **kwargs):
        """ This function resembles function forward pass (f(x)), must be implemented """
        pass

    @abstractmethod
    def _calc_local_grad(self, x, *args, **kwargs):
        """ This function calculate the gradient (∇f = ∂f/∂input) of the function with respect to input"""
        pass

    @property
    def local_grad(self):
        """Local grad getter"""
        return self.__grad

