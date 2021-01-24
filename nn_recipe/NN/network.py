from nn_recipe.NN.Layers.__factory import LayerFactory
from nn_recipe.NN.Layers.__layer import Layer
from nn_recipe.NN.LossFunctions.__factory import LossFunctionFactory
from nn_recipe.Opt.__factory import OptimizerFactory
from nn_recipe.Opt.__optimizer import Optimizer
from nn_recipe.NN.LossFunctions.__loss_function import LossFunction
from nn_recipe.Opt.gd import GD
from nn_recipe.NN.LossFunctions.meanSquared import MeanSquaredLoss
from nn_recipe.utils.exceptions import *
from nn_recipe.utils.exceptions import check_integer, check_float
import numpy as np
from typing import List, Tuple
import pickle


class Network:
    """
    This class is responsible for creating complete networks.

    Examples
    ---------
    >>> from nn_recipe.NN import *
    >>> net = Network(
    ...     Layers=[
    ...         Linear(in_dim=4, out_dim=3, activation=Sigmoid()),
    ...         Linear(in_dim=3, out_dim=3, activation=Sigmoid()),
    ...         Linear(in_dim=3, out_dim=2, activation=Sigmoid())
    ...     ],
    ...     optimizer=GD(learning_rate=0.5),
    ... )
    >>> net.train()
    >>> net.__feed_forward([1, 0.1, 0.5, 1.1])
    >>> net._save()

    :note: The class have the implementation for backprop so you can use the layer objects directly if u want to
                reimplement the backprop
    :note: Default training mode is Patch training with patch size equal to the input examples
    :ivar __layers: contains the Layers in the network
    :type __layer: list[Layer]
    :ivar __opt: Optimization object that will be used in weights update
    :type __opt: Optimizer
    :ivar __loss: Loss function that will be used to evaluate loss
    :type __loss: LossFunction
    :ivar __batch_size: batch size in the training
    :type __batch_size: ints
    """

    def __init__(self, layers, optimizer=GD(), loss_function=MeanSquaredLoss(), batch_size=None):
        """
        :param layers: List of Layers that will be contained in the network
        :type layers: List[Layer]
        :param optimizer: Optimizer object that will be used in the training process
        :type optimizer: Optimizer
        :param loss_function: Loss function that will be used to evaluate loss
        :type loss_function: LossFunction
        :param batch_size: batch size in the training
        :type batch_size: int
        :raise ShapeError: When the layer have an input size not equal to the output size of the previous layer
        :raise LayerTypeError: when the layer object is not a child of Layer
        :raise OptimizerTypeError: when the optimizer object is not a child of Optimizer
        """
        self.__layers: list[Layer] = []
        self.__opt = GD()
        self.__loss = MeanSquaredLoss()
        self.__batch_size = None
        # Set default values
        self.set_optimizer(optimizer)
        self.set_loss_function(loss_function)
        if batch_size is not None: self.set_batch_size(batch_size)
        self.add_layers(layers)

    def add_layers(self, layers):
        """
        Add Layers to the network

        :param layers: List of Layers that will be contained in the network
        :type layers: List[Layer]
        :raise ShapeError: When the layer have an input size not equal to the output size of the previous layer
        :raise LayerTypeError: when the layer object is not a child of Layer
        """
        # Size is None at the first layer as there is no previous layer in the network and the input dim is the network
        # input dim
        size = None
        for layer in layers:
            # check for the object parent class
            if not issubclass(layer.__class__, Layer):
                raise LayerTypeError()
            # check Layers dimensions
            if size is None:
                size = layer.size
            else:
                if layer.input_size != size:
                    raise ShapeError(required_shape=str(size), given_shape=layer.input_size)
                else:
                    size = layer.size
            # adding Layers to Layers list
            self.__layers.append(layer)

    def set_optimizer(self, optimizer:Optimizer):
        """
        :param optimizer: Optimizer object that will be used in the training process
        :type optimizer: Optimizer
        :raise OptimizerTypeError: when the optimizer object is not a child of Optimizer
        """
        if not issubclass(optimizer.__class__, Optimizer):
            raise OptimizerTypeError()
        self.__opt = optimizer

    def set_loss_function(self, loss_function):
        """
        :param loss_function: Loss function that will be used to evaluate loss
        :type loss_function: Function
        :raise LossFunctionTypeError: when the loss_function object is not a child of LossFunction  # TODO add Lossunction class
        """
        if not issubclass(loss_function.__class__, LossFunction):
            raise LossFunctionTypeError()
        self.__loss = loss_function

    def set_batch_size(self, batch_size):
        """
        :param batch_size: batch size in the training
        :type batch_size: int
        :raise TypeError: when batch_size value is less than zero or object type is not integer
        """
        check_integer(batch_size, "batch size value must be a positive real integer greater than zero")
        self.__batch_size = batch_size

    def train(self, X, Y, batch_size=None, epsilon=0.1, max_itr=100, notify_func=None, verify_func=None):
        """
        Train the network using the configurations added to the network

        :param X: input examples
        :type X: np.ndarray
        :param Y:  labels of the input examples
        :type Y: np.ndarray
        :param batch_size: batch size that will be used in the training process, if None the batch size will be equal
                to the number of the input examples
        :type batch_size: int
        :param epsilon: Value that will be compared to the loss to stop training
        :type epsilon: int
        :param max_itr: maximum number of iteration to be executed
        :type max_itr: int
        :notify_func: callback function used to report loss after training an epoch
        :type notify_func: Function[int]
        :return: loss value and number of iterations executed
        :rtype: Tuple[int, int]
        """
        # type checking for input configurations
        check_float(epsilon, "epsilon size value must be a positive real float greater than zero")
        check_integer(max_itr, "max_itr size value must be a positive real integer greater than zero")
        if batch_size is not None:
            check_integer(batch_size, "batch size value must be a positive real integer greater than zero")
            self.set_batch_size(batch_size)

        iteration = 0   # iteration number to break if greater than max_itr
        loss = None   # loss value to break if lower than epsilon
        opt_it = 0
        wrong_classified_examples = X.shape[0]
        if self.__batch_size is not None:
            # batch size equal to self.__batch_size
            Xbatches = np.array_split(X, X.shape[0]/self.__batch_size)
            Ybatches = np.array_split(Y, X.shape[0]/self.__batch_size)
            while True:
                for batch_index in range(len(Xbatches)):
                    x_batch = Xbatches[batch_index]
                    y_batch = Ybatches[batch_index]
                    out, loss = self.__propagate(x_batch, y_batch.reshape(-1, Y.shape[1]), opt_it, X.shape[0])
                    opt_it += 1
                    loss = np.sum(loss) / self.__batch_size
                    if notify_func is not  None: notify_func(loss)
                for layer in self.__layers:
                    self.__opt.flush(layer)
                if verify_func is not None:
                    error = verify_func()
                    print(error)
                    if error < wrong_classified_examples: wrong_classified_examples = error
                    else: break
                if loss < epsilon: break
                iteration += 1
                if iteration >= max_itr: break
        else:
            # batch size equal to number of input examples
            while True:
                out, loss = self.__propagate(X, Y, opt_it, X.shape[0])
                opt_it += 1
                loss = np.sum(loss) / loss.shape[0]
                if notify_func is not None: notify_func(loss)
                # if loss < epsilon: break
                iteration += 1
                for layer in self.__layers:
                    self.__opt.flush(layer)
                if verify_func is not None:
                    error = verify_func()
                    print(error)
                    if error < wrong_classified_examples: wrong_classified_examples = error
                    else: break
                if iteration >= max_itr: break

        return loss, iteration

    def __propagate(self, X, Y, opt_it, number_of_examples):
        """
        This function executes the forward path and the backward path for a one iteration

        :param X: input examples
        :type X: np.ndarray
        :param Y:  labels of the input examples
        :type Y: np.ndarray
        :return: value of forward path and loss value
        :rtype: Tuple[int, int]
        """
        out = self.__feed_forward(X)  # value of the forward path
        loss = self.__loss(Y, out)  # get loss value
        delta = self.__loss.local_grad  # get ∂loss/∂y
        # backpropagation path
        for layer in reversed(self.__layers):
            delta = np.multiply(delta.T, layer.local_grad["dZ"])  # delta * ∂y/∂z
            self.__opt.optimize(layer, delta, iteration=opt_it,
                                 number_of_examples=number_of_examples) # update weights and bias for a given layer
            delta = np.dot(delta.T, layer.local_grad["dX"]) # update the accumulated gradient ∂loss/∂x
        return out, loss

    def __feed_forward(self, X):
        """
        :param X: input examples
        :type X: np.ndarray
        :return: value of the forward path
        :rtype: np.ndarray
        """
        input_val = np.copy(X)
        for layer in self.__layers: input_val = layer(input_val)
        return input_val

    def evaluate(self, X):
        feed = self.__feed_forward(X)
        if feed.shape[1] == 1:
            return feed
        else:
            chosen_classes = np.argmax(feed, axis=1)
            out = np.zeros_like(feed)
            out[range(feed.shape[0]), chosen_classes] = 1
            return out

    def save(self, path:str):
        """
        Save the model into a pickle format
        :param path: path where the model will be saved
        """
        data = {"loss": self.__loss.save(), "layers": [], "batch_size": self.__batch_size, "opt": self.__opt.save()}
        for layer in self.__layers:
            data["layers"].append(layer.save())
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path:str):
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        layers = LayerFactory(data["layers"])
        loss_func = LossFunctionFactory(data["loss"])
        batch_size = data["batch_size"]
        opt = OptimizerFactory(data["opt"])
        return Network(layers=layers, optimizer=opt, loss_function=loss_func, batch_size=batch_size)