import logging as logging
from logger_toolbox import setup_logging
from torch.utils.data import DataLoader
from model.tiramisu import FCDenseNet103
import os
from utils.plotting import PlotHelper
from utils.device import device
from job.train import Train
from job.val import Val
from job.test import Test
from arguments import Args
import getpass
import time
import torch


def training_loop(start_epoch, end_epoch, net, optimizer, train, val, logger, execution_data_folder, offset):

    for epoch in range(offset+start_epoch + 1, offset+start_epoch + end_epoch + 1):
        t = time.time()
        train(epoch, optimizer)
        val(epoch)
        torch.save(net.state_dict(), os.path.join(execution_data_folder,  'Epoch_'+str(epoch)+'_model.pth'))
        logger.info('Time elapsed {:.3f}, Epoch {}\n'.format(time.time()-t, epoch))


def prepare_training(weights_file_path=None):

    # Prepare execution as per arguments
    args = Args()
    user_folder = os.path.join(args.args.save, getpass.getuser())
    setup_logging(user_folder)
    # Find root logging
    logger = logging.getLogger(__name__)

    dataset = args.resolve_dataset()

    # Took these values directly from the other implementation
    bnn = True if args.args.loss in ['hc_loss', 'aleatoric_loss'] else False
    net = FCDenseNet103(n_classes=dataset.number_of_classes, bnn=bnn)
    net = args.resolve_cuda(net)

    if weights_file_path is not None:
        net.load_state_dict(torch.load(weights_file_path, map_location=lambda storage, loc: storage))

    loss = args.resolve_loss()

    return net, logger, dataset, loss, user_folder, args


def raw_training(net, logger, dataset, loss, execution_data_folder, args, offset=0):

    data_loader = DataLoader(dataset, batch_size=args.args.batch_size, shuffle=True)
    optimizer = args.resolve_optimizer(net)

    # Raw Training
    # Initiate jobs

    args.save(execution_data_folder)

    train = Train(save_file_path=os.path.join(execution_data_folder, 'train.csv'), data_loader=data_loader, net=net,
                  loss=loss)
    val = Val(save_file_path=os.path.join(execution_data_folder, 'val.csv'), data_loader=data_loader, net=net,
              loss=loss)

    training_loop(0, args.args.epochs, net, optimizer, train, val, logger, execution_data_folder, offset)

    train.clean(), val.clean()

    return train, val


def fine_tune_training(net, logger, dataset, loss, execution_data_folder, args, offset=0):

    # Fine-tune
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # GPU is not big enough for fine-tune images
    device.isCuda = False
    net = net.cpu()
    dataset.fine_tune = True
    optimizer = args.resolve_optimizer(net)

    train = Train(save_file_path=os.path.join(execution_data_folder, 'train.csv'), data_loader=data_loader, net=net,
                  loss=loss)
    val = Val(save_file_path=os.path.join(execution_data_folder, 'val.csv'), data_loader=data_loader, net=net,
              loss=loss)

    # Update jobs
    train.data_loader = data_loader
    val.data_loader = data_loader
    training_loop(args.args.epochs, args.args.finetuneepochs, net, optimizer, train, val, logger, execution_data_folder, offset)

    train.clean(), val.clean()

    return train, val


def evaluate(dataset, train, val, execution_data_folder, net, loss):
    # Plotting time
    plot_helper = PlotHelper(execution_data_folder)
    train_data = train.load_save_data()
    test_data = val.load_save_data()
    plot_helper.plot_loss_vs_epochs(train_data, test_data)

    # Unload memory
    dataset.mode = 'Train'
    dataset.unload()
    dataset.mode = 'Val'
    dataset.unload()

    dataset.mode = 'Test'
    dataset.load()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test = Test(save_file_path=os.path.join(execution_data_folder, 'test.csv'), data_loader=data_loader, net=net, loss=loss)
    test(plot_helper)


if __name__ == '__main__':

    net, logger, dataset, loss, user_folder, args = prepare_training()

    execution_data_folder = user_folder + os.path.sep + 'pid' + str(os.getpid())

    train, val = raw_training(net, logger, dataset, loss, execution_data_folder, args)
    #train, val = fine_tune_training(net, logger, dataset, loss, execution_data_folder, args)

    evaluate(dataset, train, val, execution_data_folder, net, loss)









