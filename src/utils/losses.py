import numpy as np
import torch
import torch.nn.functional as F


def tile(input, dim, n_tile):
    """
    tile the dimention of input tensor by T times
    :param input: input tensor
    :param dim: dimention of input
    :param n_tile: number of tiling
    :return: resulted tensor
    """
    init_dim = input.size(dim)
    repeat_idx = [1] * input.dim()
    repeat_idx[dim] = n_tile
    input = input.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(input, dim, order_index)


def heteroscedastic_classification_loss(fs, sigmas, target, T=50):
    """
    loss function of heteroscedastic classifciation for semantic segmentation
    :param fs: output before activation of output layer
    :param sigmas: sigma of variance corresponding with fs given by different head beside fs in output layer
    :param target: labels of dataset
    :param T: number of dropout samples
    :return: loss for semantic segmentation
    """

    n, c, h, w = fs.size()

    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        fs = F.interpolate(fs, size=(ht, wt), mode="bilinear", align_corners=True)
        sigmas = F.interpolate(sigmas, size=(ht, wt), mode="bilinear", align_corners=True)

    # generalize epsilon(n*h*w, T, c) from gaussian distribution
    c = [n * h * w, T, c]
    mean = torch.zeros(c)
    var = torch.ones(c)
    eps = torch.distributions.Normal(mean, var)
    # reshape to (n*h*w, 1, c)
    fs = fs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 1, c)
    sigmas = sigmas.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 1, c)
    # tile each pixel by T times and get (n*h*w, T, c)
    fs = tile(fs, 1, T)
    sigmas = tile(sigmas, 1,T)

    target = target.view(-1)  # reshape target to (n*h*w)
    target_onehot = torch.FloatTensor(n * h * w, c)  # define onehot coding tensor (n*h*w, c)
    target_onehot.zero_()  # reset to zero
    target_onehot.scatter_(1, target, 1)  # get one-hot coding(n*h*w, c)
    target_onehot = tile(target_onehot, 1, T) # tile each pixel by T times and get (n*h*w, T, c)

    x = fs + sigmas * eps
    x_c = torch.sum(x*target_onehot, dim=2)
    loss = torch.sum(-torch.log(torch.sum(torch.exp(x_c - torch.logsumexp(x, dim=2)),dim=1)))

    return loss


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
