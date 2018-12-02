from .dataload import DataSet
import os
from scipy import io
from torchvision import transforms
import torch
import numpy as np
from PIL import Image


class NYUV2(DataSet):
    """
    Dataset found at http://dl.caffe.berkeleyvision.org/nyud.tar.gz
    It is not the original dataset, but it is the dataset with slightly smaller images with the 40 classes preprocessed.
    """
    def __init__(self, name, dataset_dir):
        super().__init__(name, dataset_dir)
        self.transform =  {False: transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),]),
                           True: transforms.Compose([transforms.ToTensor(),])
                           }

        if self.task == 'classification':
            self.transform_target = {False: transforms.Compose([transforms.CenterCrop(224)]),
                                     True: transforms.Compose([])
                                    }
        else:
            #Todo not good for regression
            self.transform_target = {False: transforms.Compose([transforms.CenterCrop(224)]),
                                     True: transforms.Compose([])
                                    }
        self.number_of_classes = 40 + 1

    def load_specific(self, d):
        """ Private function to load train, or tests dataset.

        First, creates the mapping between labels files, and inputs files.

        :param d:
        :return:
        """

        files = self._load_mat(os.path.join(self.dir, d + '.mat'), 'split_ids')

        if self.task == 'classification':
            return [(self._load_image(os.path.join(self.dir, 'data', 'images', f[0] + '.png')),
                     self._load_mat(os.path.join(self.dir, 'segmentation', f[0] + '.mat'), 'segmentation'))
                    for f in files]

        elif self.task == 'regression':
            return [(self._load_image(os.path.join(self.dir, 'data', 'images', f[0] + '.png')),
                     self._load_mat(os.path.join(self.dir, 'benchmarkData', 'groundTruth', f[0] + '.mat'), 'groundTruth'))
                    for f in files]
        else:
            raise ValueError('Unknown task: ' + self.task)

    @staticmethod
    def _load_mat(file_path, field):
        if field == 'groundTruth':
            tmp = io.loadmat(file_path)[field][0][0][0][0]
            return tmp # todo: not in good format for image 3 channels

        else:
            return io.loadmat(file_path)[field].squeeze()

    def __getitem__(self, idx):

        inp, labels = self.data[self.mode][idx]

        inp = self.transform[self.fine_tune](inp)
        labels = Image.fromarray(labels)
        labels = self.transform_target[self.fine_tune](labels)
        labels = torch.from_numpy(np.asarray(labels, dtype=np.long))

        return inp, labels


