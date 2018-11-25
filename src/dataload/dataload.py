import os
import importlib
from torch.utils.data import Dataset
from skimage import io


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
        return io.imread(path_image)

    def __len__(self):
        return len(self.data[self.mode][0])

    def __getitem__(self, idx):

        sample = self.data[self.mode][idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


