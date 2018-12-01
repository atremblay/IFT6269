from torch.nn import functional as F
from .job import Job
from utils.device import device

class Test(Job):

    def __call__(self, epoch):
        self.net.eval()
        test_loss = 0
        incorrect = 0
        for data, target in self.loader:

            data, target = device(data), device(target)

            output = self.net(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(self.loader)  # loss function already averages over batch size
        nTotal = len(self.loader.dataset)
        err = 100. * incorrect.item() / nTotal

        self.logger.info(
            'Test set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%) (Device: {})\n'.format(
            test_loss, incorrect, nTotal, err, device))

        self.append_save_data([epoch, test_loss, err])
