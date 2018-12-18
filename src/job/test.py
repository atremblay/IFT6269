from .job import Job
from utils.device import device
import numpy as np
import torch.nn.functional as F
import torch

class Test(Job):

    def mode(self):
        self.data_loader.dataset.mode = 'Test'

    def __call__(self, plot_helper):

        if self.data_loader.dataset.has_plot_variance:
            self.net.train()
        else:
            self.net.eval()

        metric_value = None
        test_loss = 0
        for i, (data, target) in enumerate(self.data_loader):

            data = device(data)

            if self.data_loader.dataset.has_plot_variance:
                fs, sigma, pred_var = montecarlo_prediction(self.net, data, T=50)
                if i == 0 or i == 50:
                    fig, ax = plot_helper.data_target_prediction(2, fs, data, target, self.data_loader.dataset.task)
                    plot_helper.heat_map(ax, fs, sigma, pred_var, T=50)

                output = device(fs), device(sigma)
            else:

                output = self.net(data)
                output = [o.detach() for o in output]

                if i == 0 or i == 50:
                    fig, ax = plot_helper.data_target_prediction(2, output[0], data, target, self.data_loader.dataset.task)
                    plot_helper.heat_map_softmax(ax, output[0])

            if i == 0 or i == 50:
                plot_helper.plot_helper(fig, i)

            target = device(target)
            test_loss += self.loss(*output, target).item()
            pred = self.get_predictions(output[0].cpu())
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

        self.logger.info(
            'Loss: {:.4f}, {}'.format(test_loss, metric_value))

        self.append_save_data([0, test_loss, "N/A"])


# model - the trained classifier(C classes)
#					where the last layer applies softmax
# X_data - a list of input data(size N)
# T - the number of monte carlo simulations to run
def montecarlo_prediction(model, X_data, T=50):
    # shape: (T, C, H, W)
    predictions = []
    sigma = []
    for i in range(T):
        predictions.append(F.softmax(model(X_data)[0], dim=1).detach().cpu().squeeze(dim=0).numpy())
        sigma.append(model(X_data)[1].detach().cpu().squeeze(dim=0).numpy())

    # shape: (C, H, W)
    prediction_probabilities = np.mean(np.array(predictions), axis=0)
    sigma_mean = np.mean(np.array(sigma), axis=0)

    # shape: (H, W)
    prediction_variances = np.apply_along_axis(predictive_entropy, axis=0, arr=prediction_probabilities)

    return torch.from_numpy(prediction_probabilities).unsqueeze(dim=0), torch.from_numpy(sigma_mean).unsqueeze(dim=0),\
           torch.from_numpy(prediction_variances.astype(np.float32))


# prob - prediction probability for each class(C). Shape: (N, C)
# returns - Shape: (N)
def predictive_entropy(prob):

    return -1 * np.sum(np.log(prob) * prob, axis=0)
