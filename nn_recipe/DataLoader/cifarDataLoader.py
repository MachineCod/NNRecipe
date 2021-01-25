import pickle
import tarfile
import numpy as np
from .__dataLoader import *


class CifarDataLoader(DataLoader):
    """
    The class is responsible for loading the Cifar10 dataset. It is responsible for loading all the images in the
    datasets and all the corresponding labels. It is also responsible for dividing the train dataset from the files
    to two parts: train data and validation data.
    """
    # The resources that should be downloaded to be loaded
    RESOURCES = [
        ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
         'c58f30108f718f92721af3b95e74349a')
        # ('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'eb9058c3a382ffc7106e4002c42a8d85'),
    ]

    def __init__(self, rootPath="../../cifar-10-batches-py", valRatio=0.177, download=False):
        """
        :param rootPath: The path to the directory that contains the MNIST dataset files
        :type rootPath: str
        :param valRatio: The ratio of the train dataset to be considered as validation dataset
        :type valRatio: float
        :param download: Indicates whether the Data Loader should download the needed resources or not
        :type download: bool
        """
        super(CifarDataLoader, self).__init__(rootPath, CifarDataLoader.RESOURCES, valRatio, download)
        self.labelNames = None

    def unpickle(self, file: str):
        with open("{}/{}".format(self.rootPath, file), 'rb') as fo:
            dataDict = pickle.load(fo, encoding='bytes')
        return dataDict

    def load(self):
        """
        This method is responsible for loading the data and the labels from the Cifar10 Dataset.
        It is also responsible for dividing the train dataset to train data and validation data and the same with
        the labels.
        """
        # Unzip the tar file
        tarfilePath = "{}/{}".format(self.rootPath, "cifar-10-python.tar.gz")
        tar = tarfile.open(tarfilePath)
        tarfileContentDest = os.path.join(self.rootPath, "..")
        tar.extractall(tarfileContentDest)
        tar.close()
        # Load Label Names
        batchesMeta = self.unpickle("batches.meta")
        self.labelNames = np.array(batchesMeta[b"label_names"])
        # Load Train Data & Labels
        trainDataFilenames = ["data_batch_{}".format(i) for i in range(1, 6)]
        trainDataDicts = [self.unpickle(trainDataFilename) for trainDataFilename in trainDataFilenames]
        for i in range(5):
            if i == 0:
                self.trainData = trainDataDicts[i][b"data"]
                self.trainLabels = trainDataDicts[i][b"labels"]
            else:
                self.trainData = np.vstack((self.trainData, trainDataDicts[i][b"data"]))
                self.trainLabels = self.trainLabels + trainDataDicts[i][b"labels"]
        self.trainData = self.trainData.reshape((len(self.trainData), 3, 32, 32))
        self.trainData = self.trainData.transpose(0, 2, 3, 1).astype(np.float32)
        self.trainLabels = np.array(self.trainLabels)
        # Split Train Data & Labels To Train & Validation
        numOfTrainImage = int(self.trainData.shape[0] * (1 - self.valRatio))
        self.trainData, self.validationData = self.trainData[:numOfTrainImage, :, :, :], \
                                              self.trainData[numOfTrainImage:, :, :, :]
        self.trainLabels, self.validationLabels = self.trainLabels[:numOfTrainImage], \
                                              self.trainLabels[numOfTrainImage:]
        # Load Test Data & Labels
        testDataFilename = "test_batch"
        testDataDict = self.unpickle(testDataFilename)
        self.testData = testDataDict[b"data"]
        self.testLabels = testDataDict[b"labels"]
        self.testData = self.testData.reshape((len(self.testData), 3, 32, 32))
        self.testData = self.testData.transpose(0, 2, 3, 1).astype(np.float32)
        self.testLabels = np.array(self.testLabels)


if __name__ == "__main__":
    cifarDataLoader = CifarDataLoader(download=False)
    cifarDataLoader.load()
    print("Labels: " + str(cifarDataLoader.get_label_names()))
    print(cifarDataLoader.get_train_data().shape)
    print(cifarDataLoader.get_train_labels().shape)
    print(cifarDataLoader.get_validation_data().shape)
    print(cifarDataLoader.get_validation_labels().shape)
    print(cifarDataLoader.get_test_data().shape)
    print(cifarDataLoader.get_test_labels().shape)
