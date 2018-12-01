import logging
import csv
import numpy as np

class Job:

    def __init__(self, save_file_path, data_loader, net):
        self.save_file_path = save_file_path
        self.data_loader = data_loader
        self.net = net
        self.logger = logging.getLogger(str(type(self)))
        self.name = str(type(self)).split('.')[-1]

    def append_save_data(self, content):
        with open(self.save_file_path, 'a') as f:
            content_format = ','.join(['{}' for _ in content]) + '\n'
            f.write(content_format.format(*content))

    def load_save_data(self):
        with open(self.save_file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            data = np.asarray(list(reader), dtype=np.float)

        return data

    @staticmethod
    def get_predictions(output_batch):
        bs, c, h, w = output_batch.size()
        tensor = output_batch.data
        values, indices = tensor.cpu().max(1)
        indices = indices.view(bs, h, w)
        return indices

    @staticmethod
    def error(preds, targets):
        assert preds.size() == targets.size()
        bs, h, w = preds.size()
        n_pixels = bs * h * w
        incorrect = preds.ne(targets).cpu().sum().item()
        err = incorrect / n_pixels
        return round(err, 5)

