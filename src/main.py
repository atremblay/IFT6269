import logging as logging
from logger_toolbox import setup_logging

from model.tiramisu import FCDenseNet103
import os
from job.train import Train
from job.test import Test
from arguments import Args
import getpass


if __name__ == '__main__':

    # Prepare execution as per arguments
    args = Args()
    setup_logging(execution_data_folder_path=os.path.join(args.args.save, getpass.getuser()))
    # Find root logging
    logger = logging.getLogger(__name__)

    train_loader, test_loader = args.resolve_dataset()

    # Took these values directly from the other implementation
    net = FCDenseNet103(n_classes=40)
    net = args.resolve_cuda(net)
    optimizer = args.resolve_optimizer(net)

    #summary(net)
    # Initiate jobs
    train = Train(save_file_path=os.path.join(args.args.save, 'train.csv'), loader=train_loader, net=net)
    test = Test(save_file_path=os.path.join(args.args.save, 'test.csv'), loader=test_loader, net=net)

    for epoch in range(1, args.args.epochs + 1):
        # adjust_opt(args.opt, optimizer, epoch)
        train(epoch, optimizer)
        test(epoch)
        # torch.save(net, os.path.join(args.save, 'latest.pth'))
        # os.system('./plot.py {} &'.format(args.save))
