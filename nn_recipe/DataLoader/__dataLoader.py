import os
import requests


class DataLoader:
    """
    This class is the base class of the Data Loaders. The data loaders are responsible for loading the datasets
    such as MNIST and Cifar10. It is responsible for loading all the images in the datasets and all the corresponding
    labels. It is also responsible for dividing the train dataset from the files to two parts: train data
    and validation data.
    """
    def __init__(self, rootPath, resources, valRatio=0.177, download=False):
        """
        :param rootPath: The path to the directory that contains the dataset files
        :type rootPath: str
        :param resources: A list which contains all the resources to be downloaded
        :type resources: list
        :param valRatio: The ratio of the train dataset to be considered as validation dataset
        :type valRatio: float
        :param download: Indicates whether the Data Loader should download the needed resources or not
        :type download: bool
        """
        self.rootPath = rootPath
        self.valRatio = valRatio
        self.resources = resources
        self.trainData = None
        self.trainLabels = None
        self.validationData = None
        self.validationLabels = None
        self.testData = None
        self.testLabels = None
        self.labelNames = None
        if download:
            self.download()

    def download(self):
        """
        This method downloads the required datasets to be loaded such as MNIST and Cifar10. This method is
        activated only if it is specified in the constructor that the class should download the files.
        """
        # Loop over the resources that should be downloaded
        for (url, md5) in self.resources:
            # get the filename only from the url
            filename = url.rpartition('/')[2]
            # if the file was in the directory
            if filename not in os.listdir(os.getcwd()):
                # Download the resource
                r = requests.get(url, allow_redirects=True)
                # Save the resource in the specified directory
                open(os.path.join(os.getcwd(), self.rootPath, filename), 'wb').write(r.content)

    def load(self):
        """
        This method is an empty method that should be implemented in the child classes. This method is responsible
        for loading the data and the labels from the resources. It is also responsible for dividing the train
        dataset to train data and validation data and the same with the labels.
        """
        pass

    def get_label_names(self):
        """
        :return: The array that contains the label names
        """
        return self.labelNames

    def get_train_data(self):
        """
        :return: A 4-D np.ndarray that contains the train data (num_of_train_images, rows, cols, num_of_channels)
        """
        return self.trainData

    def get_train_labels(self):
        """
        :return: A vector that contains the train labels (num_of_train_images,)
        """
        return self.trainLabels

    def get_validation_data(self):
        """
        :return: A 4-D np.ndarray that contains the validation data (num_of_validation_images, rows, cols, num_of_channels)
        """
        return self.validationData

    def get_validation_labels(self):
        """
        :return: A vector that contains the validation labels (num_of_validation_images,)
        """
        return self.validationLabels

    def get_test_data(self):
        """
        :return: A 4-D np.ndarray that contains the test data (num_of_test_images, rows, cols, num_of_channels)
        """
        return self.testData

    def get_test_labels(self):
        """
        :return: A vector that contains the test labels (num_of_test_images,)
        """
        return self.testLabels
