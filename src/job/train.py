from torch.nn import functional as F
from .job import Job
from utils.device import device


class Train(Job):

    def __call__(self, epoch, optimizer):
        self.net.train()
        nProcessed = 0
        nTrain = len(self.loader.dataset)
        for batch_idx, (data, target) in enumerate(self.loader):

            data, target = device(data), device(target)

            optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
            loss.backward()
            optimizer.step()
            nProcessed += len(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect = pred.ne(target.data).cpu().sum()
            err = 100. * incorrect.item() / len(data)
            partialEpoch = epoch + batch_idx / len(self.loader) - 1

            self.logger.info(
                'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f} (Device: {})'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(self.loader),
                loss.item(), err, device)
                )

            self.append_save_data([partialEpoch, loss.item(), err])
