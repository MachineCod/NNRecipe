from __future__ import annotations

import numpy as np
from nn_recipe.NN.__function import Function
from enum import Enum, auto

class PaddingType(Enum):
    SAME = "SAME"
    VALID = "VALID"
    FULL = "FULL"


class MaxPool2D:
    def __init__(self, kernelSize, strides, padding=PaddingType.SAME, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernelSize = kernelSize if isinstance(kernelSize, tuple) else (kernelSize, kernelSize)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding: PaddingType = padding
        self._local_grad = None
        self._cache = None

    def __call__(self, x, *args, **kwargs):
        return self._forward(x, *args, **kwargs)                 # forward pass
        

    def _generate_initial_input_and_output(self, x, kernelSize, strides, padding: PaddingType):
        # if len(x.shape) == 2:   # Gray scale image
        #     height, width = x.shape
        #     n_channels = 1
        if len(x.shape) == 3:   # colored image
            x_copy = np.expand_dims(x, axis=0)
            batch_size = 1
        else:
            x_copy = np.copy(x)
        batch_size, height, width, n_channels = x_copy.shape
        if padding == "VALID":    # "VALID"
            self._pad = (0, 0)
            outputHeight = (height - kernelSize[0]) // strides[0] + 1
            outputWidth = (width - kernelSize[1]) // strides[1] + 1
            outputSize = (batch_size, outputHeight, outputWidth, n_channels)
            newInput = x_copy
        else:   # padding is SAME
            outputSize = (batch_size, height, width, n_channels)
            paddingHeight = ((strides[0] - 1) * height - strides[0] + kernelSize[0]) // 2
            paddingWidth = ((strides[1] - 1) * width - strides[1] + kernelSize[1]) // 2
            self._pad = (paddingHeight, paddingWidth)
            npad = ((0, 0), (paddingHeight, paddingHeight), (paddingWidth, paddingWidth), (0, 0))
            newInput = np.pad(x_copy, pad_width=npad, mode='constant', constant_values=0)
            # newInput = np.zeros((height + 2*paddingHeight, width + 2*paddingWidth, n_channels))
            # newInput[paddingHeight:paddingHeight + height, paddingWidth:paddingWidth + width, :] = x_copy
        self._local_grad = np.zeros(x_copy.shape)   # previously x.shape
        output = np.zeros(outputSize)
        
        return newInput, output

    def _forward(self, x, *args, **kwargs):
        newInput, output = self._generate_initial_input_and_output(x, self.kernelSize, self.strides, self.padding)
        batch_size, outputHeight, outputWidth, outChannels = output.shape
        KH, KW = self.kernelSize
        s1, s2 = self.strides
        H, W = x.shape[0], x.shape[1]
        i_accumulator = 0
        # for batch in range(batch_size):
        for i in range(outputHeight):
            j_accumulator = 0 
            heightStart = i * s1
            heightEnd = heightStart + s1
            for j in range(outputWidth):
                widthStart = j * s2
                widthEnd = widthStart + s2
                window = newInput[:, heightStart:heightEnd, widthStart:widthEnd, :]
                output[:, i, j, :] = np.max(window, axis=(1,2))    
            
                for n in range(x.shape[-1]):    # in_channels
                    index = tuple(np.add(np.unravel_index(np.argmax(window[:, :, :, n]), (batch_size, KH, KW, 1)), (0, i_accumulator, j_accumulator, 0)))
                    index = tuple(np.subtract(index, (0, self._pad[0], self._pad[1],0)))
                    # print("index", index)
                    try:
                        self._local_grad[index] = 1
                    except IndexError: 
                        pass
                j_accumulator += self.strides[1]
            i_accumulator += self.strides[0]
        # ------------------------------------------------------------------
        # for h in range(0, H//KH): 
        #     for w in range(0, W//KW):
        #         h_offset, w_offset = h*KH, w*KW
        #         window = x[h_offset:h_offset+KH, w_offset:w_offset+KW, :]
        #         output[h, w, :] = np.max(window, axis=(0, 1))
        #         for kh in range(KH):
        #             for kw in range(KW):
        #                 self._local_grad[h_offset+kh, w_offset+kw, :] = (x[h_offset+kh, w_offset+kw, :] >= output[h, w, :])
        # print(self._local_grad)
        return output

    def calc_local_grad(self, dY):
        """
        example:
        input:
        1 2 3 
        4 5 6
        7 8 9
        padded Input: pad=1 0 0 0 1
        0 0 0 0 0
        0 1 2 3 0
        0 4 5 6 0
        0 7 8 9 0
        0 0 0 0 0

        output when stride=2 kernel(2,2)
        1 3 0
        7 9 0
        0 0 0


        dY: 1 2     
            3 4
        self._local_grad:
        0 1 1 0
        0 0 0 0
        0 1 0 0
        0 0 0 1
        
        dY after repeat:
        1 1 2 2
        1 1 2 2
        3 3 4 4
        3 3 4 4
        
        new global grad (dY)
        0 1 2 0
        0 0 0 0
        0 3 0 0
        0 0 0 4
        """
        dY = np.repeat(np.repeat(dY, repeats=self.kernelSize[0], axis=0),
                repeats=self.kernelSize[1], axis=1)
        return {
            'dY': self._local_grad * dY
        }

class AvgPool2D:
    def __init__(self, kernelSize, strides, padding=PaddingType.SAME, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(kernelSize, int):
            kernelSize = (kernelSize, kernelSize)
        if isinstance(strides, int):
            strides = (strides, strides)
        # General format (x,y)
        self.kernelSize: tuple = kernelSize
        # General format (x,y)
        self.strides: tuple = strides
        self.padding: PaddingType = padding

    def __call__(self, x, *args, **kwargs):
        return self._forward(x)

    def _forward(self, x):
        # numOfBatches, numOfChannels, height, width = x.shape
        height, width = x.shape
        # Calculate the size of the output
        if self.padding == "VALID":
            outputHeight = (height - self.kernelSize[0]) // self.strides[0] + 1
            outputWidth = (width - self.kernelSize[1]) // self.strides[1] + 1
            outputSize = (outputHeight, outputWidth)
            newInput = x
        else:
            outputSize = (height, width)
            paddingHeight = ((self.strides[0] - 1) * height - self.strides[0] + self.kernelSize[0]) // 2
            paddingWidth = ((self.strides[1] - 1) * width - self.strides[1] + self.kernelSize[1]) // 2
            newInput = np.zeros((height + 2*paddingHeight, width + 2*paddingWidth))
            newInput[paddingHeight:paddingWidth + height, paddingWidth:paddingWidth + width] = x
        output = np.zeros(outputSize)
        outputHeight = outputSize[0]
        outputWidth = outputSize[1]
        for i in range(outputHeight):
            for j in range(outputWidth):
                heightStart = i * self.strides[0]
                heightEnd = heightStart + self.strides[0]
                widthStart = j * self.strides[1]
                widthEnd = widthStart + self.strides[1]
                output[i][j] = np.average(newInput[heightStart:heightEnd, widthStart:widthEnd])

        return {'output': output}

    def calc_local_grad(self, dY):
        dY = np.repeat(np.repeat(dY, repeats=self.kernel_size[0], axis=0),
        repeats=self.kernel_size[1], axis=1)
        n = x.shape[0]
        return {
            'dX': self._local_grad * dY * (1/(n*n))
        }