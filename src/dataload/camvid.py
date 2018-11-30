from .dataload import DataSet
import os
from torchvision import transforms

class CAMVID(DataSet):
    """ CamVid dataset loading taking from https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

    """
    def __init__(self, name, dataset_dir):
        super().__init__(name, dataset_dir)
        self.transform =  transforms.Compose([
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),])

    def load_specific(self, d):
        """ Private function to load train, or test dataset.

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



