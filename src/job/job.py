import logging
import csv
import numpy as np

class Job:

    def __init__(self, save_file_path, data_loader, net, loss):
        self.save_file_path = save_file_path
        self.data_loader = data_loader
        self.net = net
        self.logger = logging.getLogger(str(type(self)))
        self.name = str(type(self)).split('.')[-1]
        self.loss = loss

    def init_transforms(self, task=None):
        pass

    def append_save_data(self, content):
        with open(self.save_file_path, 'a') as f:
            content_format = ','.join(['{}' for _ in content]) + '\n'
            f.write(content_format.format(*content))

    def load_save_data(self):
        with open(self.save_file_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='|')
            data = np.asarray(list(reader), dtype=np.float)

        return data

    def get_predictions(self, output_batch):

        def classification(output_batch):

            bs, c, h, w = output_batch.size()
            tensor = output_batch.data
            values, indices = tensor.cpu().max(1)
            indices = indices.view(bs, h, w)
            return indices

        def regression(output_batch):
            tensor = output_batch.data
            return tensor

        mapping = {'regression': regression,
                   'classification':classification}
        return mapping[self.data_loader.dataset.task](output_batch)

    def error(self, preds, targets):

        def classification(preds, targets):
            assert preds.size() == targets.size()
            bs, h, w = preds.size()
            n_pixels = bs * h * w
            incorrect = preds.ne(targets).cpu().sum().item()
            err = incorrect / n_pixels
            return round(err, 5)

        def regression(preds, targets):
            return ((preds.cpu().squeeze()-targets)**2).mean()

        mapping = {'regression': regression,
                   'classification':classification}

        return mapping[self.data_loader.dataset.task](preds, targets)

