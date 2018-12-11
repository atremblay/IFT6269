import logging
import csv
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import gc


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

    def clean(self):
        self.data_loader = None
        self.net = None
        self.loss = None
        gc.collect

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
            tensor = output_batch.data.cpu()
            return tensor

        mapping = {'regression': regression,
                   'classification':classification}
        return mapping[self.data_loader.dataset.task](output_batch)

    def error(self, preds, targets):

        def classification(preds, targets):
            assert preds.size() == targets.size()
            bs, h, w = preds.size()
            n_pixels = bs * h * w
            incorrect = preds.ne(targets).sum().item()
            err = incorrect / n_pixels
            return round(err, 5)

        def regression(preds, targets):
            return sqrt(mean_squared_error(preds.squeeze().flatten(), targets.flatten()))

        mapping = {'regression': regression,
                   'classification':classification}

        return mapping[self.data_loader.dataset.task](preds, targets)

    def specific_metrics(self, preds, targets):

        def IoU(preds, targets):

            return [['IoU', np.mean([self.IoU(preds[i], targets.data.cpu()[i]) for i in range(preds.shape[0])])]]

        def reg(preds, targets):

            def rel(preds, targets):
                return np.mean(np.abs(preds-targets)/targets)

            def rms(preds, targets):
                return np.sqrt(np.mean(np.linalg.norm(preds - targets)))

            def log_10(preds, targets):
                return np.sqrt(np.mean(np.linalg.norm(np.log10(preds) - np.log10(targets))))

            targets = targets.cpu().data.numpy()
            preds = preds.cpu().data.numpy()

            return [['rel', rel(preds, targets)],
                    ['rms', rms(preds, targets)],
                    ['log_10', log_10(preds, targets)],
                    ]

        mapping = {'regression':  reg,
                   'classification':  IoU}

        return mapping[self.data_loader.dataset.task](preds, targets)

    def IoU(self, pred, target):
        n_classes = self.data_loader.dataset.number_of_classes
        target = target.data.numpy()
        pred = pred.data.numpy()
        # Macro IoU is over all classes at once.
        # It sums up the number of pixels that intersect accross all classes
        # Same thing for union. This counter balance rare classes in
        # an image where the prediction could be perfect, biasing
        # the average of all the IoU of all the classes
        macro_intersection = 0
        macro_union = 0
        for i in range(n_classes):
            target_mask = (target == i).astype(bool)
            pred_mask = (pred == i).astype(bool)
            intersection = target_mask & pred_mask
            macro_intersection += intersection.sum()
            union = target_mask | pred_mask
            macro_union += union.sum()
        return macro_intersection / macro_union