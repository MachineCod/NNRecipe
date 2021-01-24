from __future__ import annotations

from enum import Enum, auto
from itertools import product
from math import sqrt

import numpy as np

# from .layer import*


class PaddingType(Enum):
    SAME = "SAME"
    VALID = "VALID"
    FULL = "FULL"


class ActivationFunction(Enum):
    LINEAR = auto()
    SIGMOID = auto()
    TANH = auto()
    RELU = auto()
    LEAKY_RELU = auto()


class Initializer(Enum):
    RANDOM_NORMAL = auto()
    RANDOM_UNIFORM = auto()
    ZEROS = auto()
    ONES = auto()

class ConvolutionType(Enum):
    NORMAL = auto()
    FULL = auto()

class Conv2D():
    """Convlution layer for 2D inputs"""
    def __init__(self, inChannels, filters, 
            kernelSize=(3, 3),
            strides=(1, 1),
            padding=PaddingType.VALID, **kwargs):

        self.inChannels = inChannels
        self.filters = filters
        self.kernelSize = kernelSize if isinstance(kernelSize, tuple) else (kernelSize, kernelSize)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding

        if "activation" not in kwargs:
            kwargs["activation"] = ActivationFunction.LINEAR
        self.activation = kwargs["activation"]
        if "filters_values" not in kwargs:
            self._init_params()
        else:
            self._bias = np.zeros((self.filters, 1))
            try:
                self._weights = kwargs["filters_values"].reshape((filters, *kernelSize, inChannels))
                print("\nweights", self._weights.shape)
            except ValueError:
                self._weights = np.empty((filters, *kernelSize, inChannels))
                print("\nweights2", self._weights.shape)
                for i in range(filters):
                    self._weights[i] = kwargs['filters_values']


    def __call__(self, x, *args, **kwargs):
        """Perform the function forward pass f(x), calculate the function gradient with respect to x"""
        self._cache = self._forward(x, *args, **kwargs)                     # forward pass
        # self._grad = self._calc_local_grad(x, dY,*args, **kwargs)         # Gradient Calculation, caching
        return self._cache

    def _init_params(self):
        """ Initializing the parameters that will be used by the layer object (bias, weights) """
        factor = np.sqrt(1/(self.inChannels * self.filters))
        self._bias = np.zeros((self.filters, 1))
        # self._weights = np.random.randn(self.filters, self.kernelSize[0], self.kernelSize[1], self.inChannels)
        self._weights = np.random.normal(0, factor, (self.filters, self.kernelSize[0], self.kernelSize[1], self.inChannels))
        # print(self._weights)

    def generate_initial_input_and_output(self, x, outChannels, kernelSize, strides, padding):
        if len(x.shape) == 3:   # one image
            height, width, inChannels = x.shape
            batch_size = 1
        if len(x.shape) == 4:   # multiples images
            batch_size, height, width, inChannels = x.shape
        x_copy = np.copy(x).reshape((batch_size, height, width, inChannels))
        # self._inputShape = x_copy.shape

        if padding == "VALID":      
            outputHeight = (height - kernelSize[0]) // strides[0] + 1
            outputWidth = (width - kernelSize[1]) // strides[1] + 1
            outputSize = (batch_size, outputHeight, outputWidth, outChannels)
            newInput = x_copy
        elif padding== "SAME":
            outputSize = (batch_size, height, width, outChannels)
            paddingHeight = ((strides[0] - 1) * height - strides[0] + kernelSize[0]) // 2
            paddingWidth = ((strides[1] - 1) * width - strides[1] + kernelSize[1]) // 2
            npad = ((0, 0), (paddingHeight, paddingHeight), (paddingWidth, paddingWidth), (0, 0))
            newInput = np.pad(x_copy, pad_width=npad, mode='constant', constant_values=0) 
            # newInput = np.zeros((batch_size, height + paddingHeight, width + paddingWidth, inChannels))
            # newInput[:, paddingHeight:paddingWidth + height, paddingWidth:paddingWidth + width, :] = x_copy
        else:
            # padding == "FULL"
            (batch_size, height, width, outChannels) = outputSize = self._inputShape
            b, h, w, n_C = x_copy.shape
            paddingHeight = ((height - 1) * strides[0] - h + kernelSize[0]) // 2
            paddingWidth = ((width - 1) * strides[1] - w + kernelSize[1]) // 2
            npad = ((0, 0), (paddingHeight, paddingHeight), (paddingWidth, paddingWidth), (0, 0))
            newInput = np.pad(x_copy, pad_width=npad, mode='constant', constant_values=0) 
        output = np.zeros(outputSize)
        return newInput, output, x_copy.shape

    def _forward(self, x):
        self._newInput, output, self._inputShape = self.generate_initial_input_and_output(x, self.filters, self.kernelSize, self.strides, self.padding)
        batch_size, outputHeight, outputWidth, outChannels = output.shape
        for b in range(batch_size):
            for ch in range(outChannels):
                for i in range(outputHeight):
                    heightStart = i * self.strides[0]
                    heightEnd = heightStart + self.kernelSize[0]
                    if heightEnd > self._newInput.shape[1]:
                        break
                    for j in range(outputWidth):
                        widthStart = j * self.strides[1]    
                        widthEnd = widthStart + self.kernelSize[1]
                        if widthEnd > self._newInput.shape[2]:
                            break
                        window = self._newInput[b, heightStart:heightEnd, widthStart:widthEnd, ch]
                        # print(b, ch, i, j, self._weights[ch].shape, window.shape, self._bias[ch].shape)
                        output[b, i, j, ch] = np.sum(self._weights[b, :, :, ch]*window) #+ self._bias[ch]
                
        return np.maximum(output, 0)
        
    def _convolute(self, a, b, convType="NORMAL"):
        """
        a: to be padded variable 
        b: the sliding window
            in forwardprop: kernel
            in backprop: dL/dOutput
        """
        kernelSize = b.shape[1:3]
        if convType=="FULL":
            paddedInput, output, _ = self.generate_initial_input_and_output(a, self.inChannels, kernelSize, self.strides, "FULL")
            # print("paddedInput", paddedInput.shape, paddedInput)
            # print("output", output.shape, output)
        if convType=="NORMAL":
            paddedInput, output, _ = self.generate_initial_input_and_output(a, self.inChannels, kernelSize, self.strides, self.padding)
        
        batch_size, outputHeight, outputWidth, outChannels = output.shape
        for n in range(batch_size):
            for ch in range(self.inChannels):
                for i in range(outputHeight):
                    heightStart = i * self.strides[0]
                    heightEnd = heightStart + kernelSize[0]
                    if heightEnd > paddedInput.shape[1]:
                        break
                    for j in range(outputWidth):
                        widthStart = j * self.strides[1]    # 30    30:31
                        widthEnd = widthStart + kernelSize[1]  # 30 + 5
                        if widthEnd > paddedInput.shape[2]:
                            break
                        window = paddedInput[n, heightStart:heightEnd, widthStart:widthEnd, :]
                        # print(b, ch, i, j, self._weights[ch].shape, window.shape, self._bias[ch].shape)
                        output[n, i, j, ch] = np.sum(b[ch]*window)
        return output

    def _calc_local_grad(self, dY):
        """
        Backpropagation in convolutional layer
            1. dW: ∂L/∂W = convolution between (padded input x, ∂L/∂Y)
            2. dX: ∂L/∂X = full convolution between (filter rotatated 180°, ∂L/∂Y)
            where Y is the output of the convolution layer in the forward pass
        """
        return {
            'dW': self._convolute(self._newInput, dY, "NORMAL"),
            'dX': np.flip(np.flip(self._convolute(np.flip(np.flip(self._weights, 2), 1),  dY, "FULL"), 2), 1)
            # self._convolute(x, self._weights)
        }
    
    @property
    def weights(self):
        """ Layer's weights getter"""
        return self._weights

    @weights.setter
    def weights(self, value):
        """ Layer's weights setter"""
        # TODO add type checking for weights setter
        assert self._weights.shape == value.shape
        self._weights = value
    
    @property
    def local_grad(self):
        """Local grad getter"""
        return self._grad