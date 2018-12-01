import logging as logging
from logger_toolbox import setup_logging

from torch.utils.data import DataLoader
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

    d = args.resolve_dataset()

    data_loader = DataLoader(d, batch_size=args.args.batch_size)

    # Took these values directly from the other implementation
    net = FCDenseNet103(n_classes=40)
    net = args.resolve_cuda(net)
    optimizer = args.resolve_optimizer(net)

    #summary(net)
    # Initiate jobs
    train = Train(save_file_path=os.path.join(args.args.save, 'train.csv'), loader=data_loader, net=net)
    test = Test(save_file_path=os.path.join(args.args.save, 'test.csv'), loader=data_loader, net=net)

    # Raw Training
    for epoch in range(1, args.args.epochs + 1):
        train(epoch, optimizer)
        test(epoch)

    # Fine-tune
    data_loader = DataLoader(d, batch_size=1)
    train.loader = data_loader
    test.loader = data_loader
    for epoch in range(1, args.args.finetuneepochs + 1):
        train(epoch, optimizer)
        test(epoch)

        ### Adjust Lr ###
        # import utils.training
        # LR_DECAY = 0.995
        # DECAY_EVERY_N_EPOCHS = 1
        # utils.training.adjust_learning_rate(args.args.lr, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)


        # torch.save(net, os.path.join(args.save, 'latest.pth'))
        # os.system('./plot.py {} &'.format(args.save))
