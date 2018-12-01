import matplotlib.pyplot as plt
import os
import numpy as np


class PlotHelper:

    def __init__(self, execution_data_directory):
        self.execution_data_directory = execution_data_directory
        self.show=True

    def plot_loss_vs_epochs(self, train_data, test_data):

        # Get test Loss
        test_loss = test_data[:, 1]

        # Get train Loss for last batch of epoch
        epochs = [int(pe[0]) for pe in train_data]

        train_loss = [np.average([l for  x, l in zip(epochs, train_data[:, 1]) if x == i]) for i in range(1,len(test_loss)+1)]

        assert len(train_loss) == len(test_loss)

        plt.plot(range(len(train_loss)), train_loss)
        plt.plot(range(len(train_loss)), test_loss)

        plt.savefig(os.path.join(self.execution_data_directory, 'loss_vs_epochs.svg'))

        if self.show:
            plt.show()
