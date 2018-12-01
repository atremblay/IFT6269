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

    def init_transforms(self, task=None):

        self.task = task

        self.transform =  {False: transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),]),
                           True: transforms.Compose([transforms.ToTensor(),])
                           }

        assert self.task is not None

        if self.task == 'classification':
            self.transform_target = {False: transforms.Compose([Image.fromarray, transforms.CenterCrop(224), self.from_image_to_longtensor]),
                                     True: transforms.Compose([Image.fromarray, self.from_image_to_longtensor])
                                    }
            self.number_of_classes = 40 + 1
        else:
            #Todo not good for regression
            self.transform_target = {False: transforms.Compose([transforms.CenterCrop(224), self.from_image_to_floattensor,]),
                                     True: transforms.Compose([self.from_image_to_floattensor, ])

                                    }
            self.number_of_classes = 1


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
                     self._load_image(os.path.join(self.dir, 'data', 'depth', f[0] + '.png')))
                    for f in files]
        else:
            raise ValueError('Unknown task: ' + self.task)

    @staticmethod
    def _load_mat(file_path, field):
        if field == 'groundTruth':
            tmp = io.loadmat(file_path)[field][0][0][0][0]
            print([tmp[i].max() for i in range(3)])
            return tmp # todo: not in good format for image 3 channels

        else:
            return io.loadmat(file_path)[field].squeeze()

    @staticmethod
    def from_image_to_longtensor(x):
        return torch.from_numpy(np.asarray(x, dtype=np.long))

    @staticmethod
    def from_image_to_floattensor(x):
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    def __getitem__(self, idx):

        inp, labels = self.data[self.mode][idx]

        inp = self.transform[self.fine_tune](inp)
        labels = self.transform_target[self.fine_tune](labels)

        return inp, labels


