from .dataload import DataSet
import os
from scipy import io


class MAKE3D(DataSet):
    """
    Dataset found at ...

    """

    def load_specific(self, d):
        """ Private function to load train, or test dataset.

        First, creates the mapping between labels files, and inputs files.

        :param d:
        :return:
        """

        input_dir = os.path.join(self.dir, d)
        label_dir = os.path.join(self.dir, d + 'Depth')

        mapping = self.map_files(input_dir, label_dir)

        return [(self._load_image(f[0]), self._load_mat(f[1], 'Position3DGrid')) for f in mapping]

    def map_files(self, input_dir, label_dir):
        """ Creates mapping between label files, and inputs files, and asserts data consistency.

        """
        input_files = os.listdir(input_dir)
        label_files = os.listdir(label_dir)

        input_file_wo_ext = [self._base_img_file_name(f) for f in input_files]
        label_files_wo_ext = [self._base_depth_file_name(f) for f in label_files]

        # Cross compare both folder contents
        input_file_reminder = [f for f in input_file_wo_ext if f not in label_files_wo_ext]
        label_files_reminder = [f for f in label_files_wo_ext if f not in input_file_wo_ext]

        assert len(input_file_reminder) == 0 and len(label_files_reminder) == 0

        # Complete file path mapping
        return [(os.path.join(input_dir, 'img-'+f+'.jpg'),
                 os.path.join(label_dir, 'depth_sph_corr-'+f+'.mat'))
                for f in input_file_wo_ext]

    @staticmethod
    def _base_img_file_name(f):
        return os.path.splitext(f)[0][len('img-'):]

    @staticmethod
    def _base_depth_file_name(f):
        return os.path.splitext(f)[0][len('depth_sph_corr-'):]

    @staticmethod
    def _load_mat(file_path, field):

        tmp = io.loadmat(file_path)[field].squeeze()
        return tmp

