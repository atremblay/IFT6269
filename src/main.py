from dataload import dataload
from torch.utils.data import DataLoader
from model.densenet import DenseNet, summary
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument(
        '--dataset',
        required=True,
        choices=["nnyuv2", "camvid", "make3d"]
    )
    parser.add_argument(
        "--task", choices=['regression', 'classification']
    )
    args = parser.parse_args()
    if args.cuda and not torch.nn.is_available():
        print("CUDA not available on your machine. Setting it back to False")
        args.cuda = False

    if args.dataset == 'nnyuv2':
        if 'task' not in args:
            msg = 'Need to provide --task when you select --dataset nnyuv2'
            raise Exception(msg)
    return args


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


if __name__ == '__main__':
    args = parse_args()

    # Data Path
    data_sets = dataload.DatasetLib('/home/data/')
    print('Available Datasets:' + str(list(data_sets.datasets.keys())))

    d_train = data_sets[args.dataset]  # nnyuv2, camvid, or make3d
    d_test = data_sets[args.dataset]  # nnyuv2, camvid, or make3d
    if args.dataset == 'nnyuv2':
        d_train.task = args.task
        d_test.task = args.task
    d_train.mode = 'Train'  # Train or Test
    d_test.mode = 'Test'  # Train or Test
    d_train.load()
    d_test.load()
    train_loader = DataLoader(d_train, batch_size=args.batch_size)
    test_loader = DataLoader(d_test, batch_size=args.batch_size)

    # Took these values directly from the other implementation
    net = DenseNet(
        growthRate=12,
        depth=100,
        reduction=0.5,
        bottleneck=True,
        nClasses=10  # Need to set this to the appropriate number of classes
    )

    summary(net)
