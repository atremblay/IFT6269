import logging

class Job:

    def __init__(self, save_file_path, loader, net):
        self.save_file_path = save_file_path
        self.loader = loader
        self.net = net
        self.logger = logging.getLogger(str(type(self)))

    def append_save_data(self, content):
        with open(self.save_file_path, 'a') as f:
            content_format = ', '.join(['{}' for _ in content]) + '\n'
            f.write(content_format.format(*content))

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