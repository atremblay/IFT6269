import logging as logging
from logger_toolbox import setup_logging

from torch.utils.data import DataLoader
from model.tiramisu import FCDenseNet103
import os
from utils.plotting import PlotHelper
from job.train import Train
from job.test import Test
from arguments import Args
import getpass
import time


def training_loop(start_epoch, end_epoch, optimizer, train, test, logger):

    for epoch in range(start_epoch + 1, start_epoch + end_epoch + 1):
        t = time.time()
        train(epoch, optimizer)
        test(epoch)
        logger.info('Time elapsed {:.3f}, Epoch {}\n'.format(time.time()-t, epoch))

        ### Adjust Lr ###
        # import utils.training
        # LR_DECAY = 0.995
        # DECAY_EVERY_N_EPOCHS = 1
        # utils.training.adjust_learning_rate(args.args.lr, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)


        # torch.save(net, os.path.join(args.save, 'latest.pth'))
        # os.system('./plot.py {} &'.format(args.save))


if __name__ == '__main__':

    # Prepare execution as per arguments
    args = Args()
    execution_data_folder = os.path.join(args.args.save, getpass.getuser())
    setup_logging(execution_data_folder)
    # Find root logging
    logger = logging.getLogger(__name__)

    d = args.resolve_dataset()

    data_loader = DataLoader(d, batch_size=args.args.batch_size)

    # Took these values directly from the other implementation
    net = FCDenseNet103(n_classes=d.number_of_classes)
    net = args.resolve_cuda(net)
    optimizer = args.resolve_optimizer(net)

    # Raw Training
    # Initiate jobs
    execution_data_folder += os.path.sep + 'pid' + str(os.getpid())

    train = Train(save_file_path=os.path.join(execution_data_folder,  'train.csv'), data_loader=data_loader, net=net)
    test = Test(save_file_path=os.path.join(execution_data_folder, 'test.csv'), data_loader=data_loader, net=net)
    training_loop(0, args.args.epochs, optimizer, train, test, logger)


    # Fine-tune
    data_loader = DataLoader(d, batch_size=1)
    # Update jobs
    train.data_loader = data_loader
    test.data_loader = data_loader
    training_loop(args.args.epochs, args.args.finetuneepochs, optimizer, train, test, logger)


    # Plotting time
    plot_helper = PlotHelper(execution_data_folder)
    train_data = train.load_save_data()
    test_data = test.load_save_data()
    plot_helper.plot_loss_vs_epochs(train_data, test_data)





