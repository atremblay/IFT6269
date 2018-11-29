from dataload import dataload
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
    args = parser.parse_args()
    if args.cuda and not torch.nn.is_available():
        print("CUDA not available on your machine. Setting it back to False")
        args.cuda = False
    return args


if __name__ == '__main__':
    args = parse_args()

    # Data Path
    data_sets = dataload.DatasetLib('/home/data/')
    print('Available Datasets:' + str(list(data_sets.datasets.keys())))

    d = data_sets['make3d']  #nnyuv2, camvid, or make3d
    d.mode = 'Test'  # Train or Test
    # d.task = 'regression' # regression or classification (for nyuv2 only)
    d.load()
    d.unload()

    # Took these values directly from the other implementation
    net = DenseNet(
        growthRate=12,
        depth=100,
        reduction=0.5,
        bottleneck=True,
        nClasses=10  # Need to set this to the appropriate number of classes
    )

    summary(net)
