from .dataload import DataSet
import os
from torchvision import transforms
import numpy as np
import torch


class CAMVID(DataSet):
    """ CamVid dataset loading taking from https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

    """
    def init_transforms(self, task=None):

        self.task = 'classification'

        self.transform =  {False: transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),]),
                           True: transforms.Compose([transforms.ToTensor(),])
                           }
        self.transform_target = {False: transforms.Compose([transforms.CenterCrop(224)]),
                                 True: transforms.Compose([])
                                }
        self.number_of_classes = 11 + 1


    def load_specific(self, d):
        """ Private function to load train, or tests dataset.

        First, creates the mapping between labels files, and inputs files.

        :param d:
        :return:
        """
        input_dir = os.path.join(self.dir, d)
        label_dir = os.path.join(self.dir, d + 'annot')

        mapping = self.map_files(input_dir, label_dir)
        return [(self._load_image(f[0]), self._load_image(f[1])) for f in mapping]

    @staticmethod
    def map_files(input_dir, label_dir):
        """ Creates mapping between label files, and inputs files, and asserts data consistency.

        """
        input_files = os.listdir(input_dir)
        label_files = os.listdir(label_dir)

        # Cross compare both folder contents
        input_file_reminder = [f for f in input_files if f not in label_files]
        label_files_reminder = [f for f in label_files if f not in input_files]

        assert len(input_file_reminder) == 0 and len(label_files_reminder) == 0

        # Complete file path mapping
        return [(os.path.join(input_dir, f), os.path.join(label_dir, f)) for f in input_files]

    def __getitem__(self, idx):

        inp, labels = self.data[self.mode][idx]

        inp = self.transform[self.fine_tune](inp)
        labels = self.transform_target[self.fine_tune](labels)
        labels = torch.from_numpy(np.asarray(labels, dtype=np.long))

        return inp, labels

