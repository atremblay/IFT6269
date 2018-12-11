from .job import Job
from utils.device import device


class Val(Job):

    def mode(self):
        self.data_loader.dataset.mode = 'Val'

    def __call__(self, epoch):
        self.mode()

        self.net.eval()
        test_loss = 0
        incorrect = 0
        metric_value = None
        for data, target in self.data_loader:

            data, target = device(data), device(target)

            output = self.net(data)
            test_loss += self.loss(*output, target).item()
            pred = self.get_predictions(output[0])
            incorrect += self.error(pred, target.data.cpu())

            # Calculate Task specific metric
            metric = self.specific_metrics(pred, target)

            if metric_value is None:
                metric_value = []
                for m in metric:
                    metric_value.append(m)
            else:
                for i, m in enumerate(metric):
                    metric_value[i][1] += m[1]

        test_loss /= len(self.data_loader)  # loss function already averages over batch size

        # Average metric
        for i in range(len(metric_value)):
            metric_value[i][1] /= len(self.data_loader)

        nTotal = len(self.data_loader.dataset)
        err = 100. * incorrect / nTotal

        self.logger.info(
            'Epoch: {}, Loss: {:.4f}, Error: {}/{} ({:.0f}%), {}, Device: {}'.format(epoch,
            test_loss, incorrect, nTotal, err, metric_value, device))

        self.append_save_data([epoch, test_loss, err])


