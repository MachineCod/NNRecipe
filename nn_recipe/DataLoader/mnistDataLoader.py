import os
import tarfile
import urllib

from .dataLoader import DataLoader
import gzip
import numpy as np


class MNISTDataLoader(DataLoader):
    RESOURCES = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
         "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
         "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
         "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
         "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    CLASSES = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    def __init__(self, rootPath="../../mnist", valRatio=0.177, download=False):
        super(MNISTDataLoader, self).__init__(rootPath, MNISTDataLoader.RESOURCES, valRatio, download)
        self.labelNames = MNISTDataLoader.CLASSES

    def read4bytes(self, bytestream):
        # Big Endian ('>')
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_images(self, file):
        with gzip.GzipFile(fileobj=file) as bytestream:
            fileID = self.read4bytes(bytestream)
            if fileID != 2051:
                raise ValueError()
            numOfImages = self.read4bytes(bytestream)
            rows = self.read4bytes(bytestream)
            cols = self.read4bytes(bytestream)
            buffer = bytestream.read(rows * cols * numOfImages)
            data = np.frombuffer(buffer, dtype=np.uint8)
            # (28*28*60000,)
            data = data.reshape(numOfImages, rows, cols, 1)
            # (60000, 28, 28, 1)
            return data

    def extract_labels(self, file):
        with gzip.GzipFile(fileobj=file) as bytestream:
            fileID = self.read4bytes(bytestream)
            if fileID != 2049:
                raise ValueError()
            numOfItems = self.read4bytes(bytestream)
            buffer = bytestream.read(numOfItems)
            labels = np.frombuffer(buffer, dtype=np.uint8)
            return labels

    def load(self):
        # Load Train Data & Labels
        trainDataFilename = os.path.join(self.rootPath, MNISTDataLoader.TRAIN_IMAGES)
        with open(trainDataFilename, 'rb') as file:
            self.trainData = self.extract_images(file)

        trainLabelsFilename = os.path.join(self.rootPath, MNISTDataLoader.TRAIN_LABELS)
        with open(trainLabelsFilename, 'rb') as file:
            self.trainLabels = self.extract_labels(file)

        # Load Test Data & Labels
        testDataFilename = os.path.join(self.rootPath, MNISTDataLoader.TEST_IMAGES)
        with open(testDataFilename, 'rb') as file:
            self.testData = self.extract_images(file)

        testLabelsFilename = os.path.join(self.rootPath, MNISTDataLoader.TEST_LABELS)
        with open(testLabelsFilename, 'rb') as file:
            self.testLabels = self.extract_labels(file)

        # Split Train Data To Train & Validate
        numOfTrainImage = int(self.trainData.shape[0] * (1 - self.valRatio))
        self.trainData, self.validationData = self.trainData[:numOfTrainImage, :, :, :], \
                                              self.trainData[numOfTrainImage:, :, :, :]
        self.trainLabels, self.validationLabels = self.trainLabels[:numOfTrainImage], \
                                                  self.trainLabels[numOfTrainImage:]


if __name__ == "__main__":
    mnistDataLoader = MNISTDataLoader(download=False)
    mnistDataLoader.load()
    print("Labels: " + str(mnistDataLoader.get_label_names()))
    print(mnistDataLoader.get_train_data().shape)
    print(mnistDataLoader.get_train_labels().shape)
    print(mnistDataLoader.get_validation_data().shape)
    print(mnistDataLoader.get_validation_labels().shape)
    print(mnistDataLoader.get_test_data().shape)
    print(mnistDataLoader.get_test_labels().shape)
