from .job import Job
from utils.device import device


class Train(Job):

    def __call__(self, epoch, optimizer):
        self.data_loader.dataset.mode = 'Train'

        self.net.train()
        nProcessed = 0
        nTrain = len(self.data_loader.dataset)
        for batch_idx, (data, target) in enumerate(self.data_loader):

            target = device(target)

            optimizer.zero_grad()
            output = self.net(device(data))
            loss = self.loss(*output, target)

            loss.backward()
            optimizer.step()

            nProcessed += len(data)
            pred = self.get_predictions(output[0])
            incorrect = self.error(pred, target.data.cpu())
            err = 100. * incorrect / len(data)
            partialEpoch = epoch + batch_idx / len(self.data_loader)
            metric = self.specific_metrics(pred, target)

            self.logger.info(
                'Epoch: {:.2f} [{}/{} ({:.0f}%)], Loss: {:.6f}, Error: {:.6f}, , {}, Device: {}'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(self.data_loader),
                loss.item(), err, metric, device)
                )

            self.append_save_data([partialEpoch, loss.item(), err])

