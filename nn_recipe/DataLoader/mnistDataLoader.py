import os
import gzip
import numpy as np
from .__dataLoader import DataLoader


class MNISTDataLoader(DataLoader):
    """
    The class is responsible for loading the MNIST dataset. It is responsible for loading all the images in the
    datasets and all the corresponding labels. It is also responsible for dividing the train dataset from the files
    to two parts: train data and validation data.
    """
    # The resources that should be downloaded to be loaded
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
    # The label names
    CLASSES = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']
    # The filenames of the files to be loaded
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    def __init__(self, rootPath="../../mnist", valRatio=0.177, download=False):
        """
        :param rootPath: The path to the directory that contains the MNIST dataset files
        :type rootPath: str
        :param valRatio: The ratio of the train dataset to be considered as validation dataset
        :type valRatio: float
        :param download: Indicates whether the Data Loader should download the needed resources or not
        :type download: bool
        """
        super(MNISTDataLoader, self).__init__(rootPath, MNISTDataLoader.RESOURCES, valRatio, download)
        self.labelNames = MNISTDataLoader.CLASSES

    def read4bytes(self, bytestream):
        """
        :param bytestream: The bytestream to be read from
        :type bytestream: gzip.GzipFile
        :return: The next 4 bytes from the bytestream in a np.uint32 format
        """
        # The data type is a np.uint32 loaded with big-endian
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def extract_images(self, file):
        """
        :param file: The file object to be read from
        :type file: BinaryIO
        :return: The extracted images from the file in a numpy.ndarray format with 4 Dimensions
        """
        with gzip.GzipFile(fileobj=file) as bytestream:
            magicNum = self.read4bytes(bytestream)
            # Check if the magic number is as expected
            if magicNum != 2051:
                raise ValueError()
            numOfImages = self.read4bytes(bytestream)
            rows = self.read4bytes(bytestream)
            cols = self.read4bytes(bytestream)
            # Read the bytes of all the images
            buffer = bytestream.read(rows * cols * numOfImages)
            # Put the bytes in a numpy.ndarray
            data = np.frombuffer(buffer, dtype=np.uint8)
            # Reshape the data to be consistent with the required shape (num_of_images, rows, cols, num_of_channels)
            data = data.reshape(numOfImages, rows, cols, 1)
            return data

    def extract_labels(self, file):
        """
        :param file: The file object to be read from
        :type file: BinaryIO
        :return: The extracted labels from the file in a numpy.ndarray format as a vector
        """
        with gzip.GzipFile(fileobj=file) as bytestream:
            magicNum = self.read4bytes(bytestream)
            # Check if the magic number is as expected
            if magicNum != 2049:
                raise ValueError()
            numOfItems = self.read4bytes(bytestream)
            buffer = bytestream.read(numOfItems)
            labels = np.frombuffer(buffer, dtype=np.uint8)
            return labels

    def load(self):
        """
        This method is responsible for loading the data and the labels from the MNIST Dataset.
        It is also responsible for dividing the train dataset to train data and validation data and the same with
        the labels.
        """
        # Load Train Data
        if not os.path.isdir(self.rootPath):
            os.mkdir(self.rootPath)
        trainDataFilename = os.path.join(self.rootPath, MNISTDataLoader.TRAIN_IMAGES)
        with open(trainDataFilename, 'rb') as file:
            self.trainData = self.extract_images(file)
        # Load Train Labels
        trainLabelsFilename = os.path.join(self.rootPath, MNISTDataLoader.TRAIN_LABELS)
        with open(trainLabelsFilename, 'rb') as file:
            self.trainLabels = self.extract_labels(file)
        # Load Test Data
        testDataFilename = os.path.join(self.rootPath, MNISTDataLoader.TEST_IMAGES)
        with open(testDataFilename, 'rb') as file:
            self.testData = self.extract_images(file)
        # Load Test Labels
        testLabelsFilename = os.path.join(self.rootPath, MNISTDataLoader.TEST_LABELS)
        with open(testLabelsFilename, 'rb') as file:
            self.testLabels = self.extract_labels(file)
        # Split Train Data & Labels To Train & Validation
        numOfTrainImage = int(self.trainData.shape[0] * (1 - self.valRatio))
        self.trainData, self.validationData = self.trainData[:numOfTrainImage, :, :, :], \
                                              self.trainData[numOfTrainImage:, :, :, :]
        self.trainLabels, self.validationLabels = self.trainLabels[:numOfTrainImage], \
                                                  self.trainLabels[numOfTrainImage:]


if __name__ == "__main__":
    mnistDataLoader = MNISTDataLoader(download=True)
    mnistDataLoader.load()
    print("Labels: " + str(mnistDataLoader.get_label_names()))
    print(mnistDataLoader.get_train_data().shape)
    print(mnistDataLoader.get_train_labels().shape)
    print(mnistDataLoader.get_validation_data().shape)
    print(mnistDataLoader.get_validation_labels().shape)
    print(mnistDataLoader.get_test_data().shape)
    print(mnistDataLoader.get_test_labels().shape)
