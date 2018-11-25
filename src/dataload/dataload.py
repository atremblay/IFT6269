import os
import importlib
from torch.utils.data import Dataset
from skimage import io
from PIL import ImageFile



class DatasetLib:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.datasets = self._find_dataset()

    def _find_dataset(self):
        # Data set dictionary
        datasets = {}
        for p in os.listdir(self.data_dir):
            dataset_class = getattr(importlib.import_module('.'+p, 'src.dataload'), p.upper())
            datasets[p] = dataset_class(p, os.path.join(self.data_dir, p))

        return datasets

    def __getitem__(self, dataset_name):
        return self.datasets[dataset_name]


class DataSet(Dataset):
    def __init__(self, name, dataset_dir):
        self.name = name
        self.dir = dataset_dir
        self.data = {'Test': None, 'Train': None}
        self.mode = 'Train'
        self.task = None

    def load(self):
        """
        Function to load train, or test dataset
        :return:
        """
        self.data[self.mode] = self.load_specific(self.mode.lower())

    def unload(self):
        """
        Function to unload train, or test dataset
        :return:
        """
        self.data[self.mode] = None

    @staticmethod
    def _load_image(path_image):
        try:
            img = io.imread(path_image)
        except ValueError as e:
            ImageFile.LOAD_TRUNCATED_IMAGES = True  # Some Images have are truncated and raise an error at load. This disables the error
            img = io.imread(path_image)
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            print('Warning - '+path_image + ' possibly invalid: ' + str(e).split('\n')[1])

        return img

    def __len__(self):
        return len(self.data[self.mode][0])

    def __getitem__(self, idx):

        sample = self.data[self.mode][idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


