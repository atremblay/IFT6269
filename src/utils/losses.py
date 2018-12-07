import torch
from .device import device
import torch.nn.functional as F


def hc_loss(fs, sigmas, target, T=50):
    """
    loss function of heteroscedastic classifciation for semantic segmentation
    :param fs: output before activation of output layer
    :param sigmas: sigma of variance corresponding with fs given by different head beside fs in output layer
    :param target: labels of dataset
    :param T: number of dropout samples
    :return: loss for semantic segmentation
    """

    def get_epsilon(size):
        mean = torch.zeros(size)
        var = torch.ones(size)
        eps_normal = torch.distributions.Normal(mean, var)
        return eps_normal.sample()

    n, c, h, w = fs.size()
    nt, ht, wt = target.size()

    assert nt == n and ht == h and wt == w

    out_size = (n,) + fs.size()[2:]
    if target.size()[1:] != fs.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(out_size, target.size()))
    fs = fs.contiguous().view(n, c, 1, -1)
    sigmas = sigmas.contiguous().view(n, c, 1, -1) #add soft plus to sigma
    target = target.contiguous().view(n, 1, -1)

    # generalize epsilon(T, c, 1, h*w) from gaussian distribution
    eps = device(get_epsilon(size=(T, c, 1, h*w)))

    loss = device(torch.zeros(n, h * w))
    # https://github.com/kyle-dorman/bayesian-neural-network-blogpost
    for t in range(T):
        x = F.log_softmax((fs + sigmas * eps[t]).squeeze(), dim=1)
        loss += F.cross_entropy(x, target.squeeze())

    return (loss.sum(dim=1)/T).mean()


def aleatoric_loss(true, pred, var):
    """
    Taken from https://arxiv.org/pdf/1703.04977.pdf

    Theory says we should implement equation (5),
    but practice says equation (8).

    This paper is for computer vision, but the theory behind it applies to
    any neural network model. Here we are using it for NLP.

    Params
    ======
    true: torch tensor
        The true targets
    pred: torch tensor
        The predictions
    var: torch tensor
        The uncertainty of every prediction (actually log(var)).
    """
    loss = torch.exp(-var) * (true - pred)**2 / 2
    loss += 0.5 * var
    return torch.mean(loss)
