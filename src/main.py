from dataload import dataload
from torch.utils.data import DataLoader
from model.densenet import FCDenseNet, summary
import argparse
import torch
import torch.optim as optim
import os


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


def prepare_dataset(args):
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
    train_loader = DataLoader(
        d_train,
        batch_size=args.batch_size,
        # transform=d_train.transform
    )
    test_loader = DataLoader(
        d_test,
        batch_size=args.batch_size,
        # transform=d_test.transform
    )

    return train_loader, test_loader


def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100. * incorrect / len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err)
        )

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()


if __name__ == '__main__':
    args = parse_args()
    train_loader, test_loader = prepare_dataset(args)

    # Took these values directly from the other implementation
    net = FCDenseNet(
        growthRate=12,
        depth=100,
        reduction=0.5,
        bottleneck=True,
        nClasses=10  # Need to set this to the appropriate number of classes
    )

    # Todo: Move to appropriate train function
    for samples in train_loader:
        print(samples)

    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(
            net.parameters(),
            lr=1e-1,
            momentum=0.9,
            weight_decay=1e-4
        )
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    summary(net)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        # adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, train_loader, optimizer, trainF)
        test(args, epoch, net, test_loader, optimizer, testF)
        # torch.save(net, os.path.join(args.save, 'latest.pth'))
        # os.system('./plot.py {} &'.format(args.save))
