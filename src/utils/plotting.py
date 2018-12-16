import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn.functional as F
import torch
from matplotlib import cm
from matplotlib.colors import Normalize
from utils.device import device


class PlotHelper:

    def __init__(self, execution_data_directory):
        self.execution_data_directory = execution_data_directory
        self.show=True

    def plot_loss_vs_epochs(self, train_data, test_data):

        # Get test Loss
        test_loss = test_data[:, 1]

        # Get train Loss for last batch of epoch
        epochs = [int(pe[0]) for pe in train_data]

        train_loss = [np.average([l for x, l in zip(epochs, train_data[:, 1]) if x == i]) for i in range(1,len(test_loss)+1)]

        assert len(train_loss) == len(test_loss)

        plt.plot(range(len(train_loss)), train_loss)
        plt.plot(range(len(train_loss)), test_loss)

        plt.savefig(os.path.join(self.execution_data_directory, 'loss_vs_epochs.svg'))

        if self.show:
            plt.show()

    @staticmethod
    def data_target_prediction(row, fs, data, target, task):

        n_plots = 3

        n, c, h, w = fs.size()
        assert n == 1

        fs = fs.squeeze(dim=0)

        fig, ax = plt.subplots(row, n_plots)
        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        if row == 1:
            ax[0].set_axis_off()
            ax[0].imshow(np.moveaxis(data.cpu().squeeze().numpy(), 0, -1))

            ax[1].set_axis_off()
            ax[1].imshow(target.detach().cpu().squeeze().numpy(), norm=Normalize(target.min(), target.max()))

            ax[2].set_axis_off()
            pred = fs.cpu().numpy()

            if task == 'classification':
                pred = pred.argmax(axis=0)
            else:
                pred = pred.squeeze()

            ax[2].imshow(pred, norm=Normalize(pred.min(), pred.max()))

        else:
            ax[0, 0].set_axis_off()
            ax[0, 0].imshow(np.moveaxis(data.cpu().squeeze().numpy(), 0, -1))

            ax[0, 1].set_axis_off()
            ax[0, 1].imshow(target.detach().cpu().squeeze().numpy(), norm=Normalize(target.min(), target.max()))

            ax[0, 2].set_axis_off()
            pred = fs.cpu().numpy()

            if task == 'classification':
                pred = pred.argmax(axis=0)
            else:
                pred = pred.squeeze()

            ax[0, 2].imshow(pred, norm=Normalize(pred.min(), pred.max()))

        return fig, ax

    @staticmethod
    def heat_map(ax, fs, sigmas, predicted_variance, T=50):

        def get_epsilon(size):
            mean = torch.zeros(size)
            var = torch.ones(size)
            eps_normal = torch.distributions.Normal(mean, var)
            return eps_normal.sample()

        n, c, h, w = fs.size()

        sigmas = sigmas.squeeze(dim=0)
        fs = fs.squeeze(dim=0)

        softmax = torch.zeros((c, h, w))
        for t in range(T):
            x = (fs + sigmas * get_epsilon(size=(c, h, w))).squeeze(dim=2)
            softmax += F.softmax(x, dim=0)

        assert (((softmax/T).sum(dim=0) - 1).abs() < 0.000001).all()

        mean_softmax = softmax/T
        entropy = -(mean_softmax * np.log(mean_softmax)).sum(dim=0)

        ax[1, 0].set_axis_off()
        ax[1, 0].imshow(entropy.numpy(), norm=Normalize(entropy.min(), entropy.max()), cmap=cm.jet)

        ax[1, 1].set_axis_off()
        ax[1, 1].imshow(predicted_variance.numpy(), norm=Normalize(predicted_variance.min(), predicted_variance.max()), cmap=cm.jet)

        combined = entropy + predicted_variance
        ax[1, 2].set_axis_off()
        ax[1, 2].imshow(combined.numpy(), norm=Normalize(combined.min(), combined.max()), cmap=cm.jet)

    def plot_helper(self, fig, i=0):

        fig.savefig(os.path.join(self.execution_data_directory, 'heat_map_'+str(i)+'.svg'))

        if self.show:
            plt.show()







