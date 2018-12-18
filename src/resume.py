import main
import sys
import os
from job.train import Train
from job.val import Val


def load(path):
    with open(os.path.join(path, 'commandline_args.txt'), 'r') as f:
        return [l.strip() for l in f.readlines()]


def resolve_epochs(args, init_epoch):

    args.args.finetuneepochs = max(0, args.args.finetuneepochs+args.args.epochs-init_epoch)
    args.args.epochs = max(0, args.args.epochs-init_epoch)


if __name__ == '__main__':

    execution_data_folder = sys.argv[1]

    sys.argv = sys.argv[:1] + load(execution_data_folder)

    init_epoch = max([int(f.split('_')[1]) for f in os.listdir(execution_data_folder) if f.endswith('.pth')])

    weights_file_path = os.path.join(execution_data_folder, 'Epoch_'+str(init_epoch)+'_model.pth')

    net, logger, dataset, loss, user_folder, args = main.prepare_training(weights_file_path)

    resolve_epochs(args, init_epoch)

    train, val = main.raw_training(net, logger, dataset, loss, execution_data_folder, args, offset=init_epoch)
    #train, val = main.fine_tune_training(net, logger, dataset, loss, execution_data_folder, args, offset=init_epoch)

    train = Train(save_file_path=os.path.join(execution_data_folder, 'train.csv'), data_loader=None, net=None, loss=None)
    val = Val(save_file_path=os.path.join(execution_data_folder, 'val.csv'), data_loader=None, net=None, loss=None)

    main.evaluate(dataset, train, val, execution_data_folder, net, loss)