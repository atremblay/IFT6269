import torch
from .device import device
import torch.nn.functional as F
import gc


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
    sigmas = sigmas.contiguous().view(n, c, 1, -1)
    target = target.contiguous().view(n, 1, -1)

    log_softmax = device(torch.zeros(n, c, h * w))
    # https://github.com/kyle-dorman/bayesian-neural-network-blogpost
    for t in range(T):
        x = (fs + sigmas * device(get_epsilon(size=(c, 1, h * w)))).squeeze(dim=2)
        b = x.max(dim=1, keepdim=True)[0].repeat(1, 12, 1)
        log_softmax += x-b-torch.logsumexp(x-b, dim=1, keepdim=True).repeat(1, 12, 1)

    return F.nll_loss(log_softmax/T, target.squeeze(dim=1))


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
