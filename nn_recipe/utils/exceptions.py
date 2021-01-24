class ShapeError(Exception):
    def __init__(self, given_shape, required_shape):
        super(ShapeError, self).__init__(
            "Required Shape is {} but the given shape is {}".format(required_shape, given_shape)
        )


class OptimizerTypeError(Exception):
    def __init__(self):
        super(OptimizerTypeError, self).__init__("Optimizer object must be an instance of Optimizer class")


class LayerTypeError(Exception):
    def __init__(self):
        super(LayerTypeError, self).__init__("Layer object must be an instance of Layer class")


class LossFunctionTypeError(Exception):
    def __init__(self):
        super(LayerTypeError, self).__init__("Loss function object must be an instance of LossFunction class")


def check_integer(value_to_check, error_msg, condition=None):
    if type(value_to_check) is not int:
        if condition is not None and not condition(value_to_check):
            raise TypeError(error_msg)
    return value_to_check


def check_float(value_to_check, error_msg, condition=None):
    if type(value_to_check) is not float:
        if condition is not None and not condition(value_to_check):
            raise TypeError(error_msg)
    return value_to_check


__all__ = [
    "ShapeError", "OptimizerTypeError", "LayerTypeError", "LossFunctionTypeError"
]