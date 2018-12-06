import argparse
import torch
from dataset import dataload
import torch.optim as optim
import logging
from utils.device import device
from torch.nn import functional as F
from utils.losses import aleatoric_loss, hc_loss

class Args:

    def __init__(self):
        # Register instance class logger
        self.logger = logging.getLogger(__name__)

        self.parser = argparse.ArgumentParser()
        # Add arguments
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--epochs', type=int, default=20)
        self.parser.add_argument('--finetuneepochs', type=int, default=10)
        self.parser.add_argument('--cuda', default=False, action='store_true')
        self.parser.add_argument('--save', type=str, default='/home/execution')
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
        self.parser.add_argument('--dataset', required=True, choices=["nyuv2", "camvid", "make3d"])
        self.parser.add_argument("--task", choices=['regression', 'classification'])
        self.parser.add_argument("--data_folder", default='/home/data/', type=str)
        self.parser.add_argument("--lr", default=1e-3, type=float)
        self.parser.add_argument('--loss', required=True, choices=['nll_loss', 'mse_loss', 'hc_loss', 'aleatoric_loss'])
        self.args = self.parser.parse_args()

    def resolve_dataset(self):

        # Resolve task
        if self.args.dataset == 'nyuv2':
            if self.args.task is None:
                msg = 'Need to provide --task when you select --dataset nyuv2'
                self.logger.error(msg)
                raise Exception(msg)

        # Data Path
        data_sets = dataload.DatasetLib(self.args.data_folder)

        d = data_sets[self.args.dataset]

        d.init_transforms(self.args.task)
        d.mode = 'Train'  # Train or Test
        d.load()
        d.mode = 'Test'  # Train or Test
        d.load()

        return d

    def resolve_cuda(self, net):

        device.isCuda = self.args.cuda

        if device.isCuda and not torch.cuda.is_available():
            print("CUDA not available on your machine. Setting it back to False")
            self.args.cuda = False
            device.isCuda=False

        if device.isCuda:
            net = net.cuda()

        return net

    def resolve_optimizer(self, net):

        if self.args.opt == 'sgd':
            optimizer = optim.SGD(
                net.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.args.opt == 'adam':
            optimizer = optim.Adam(net.parameters(), weight_decay=1e-4, lr=self.args.lr)
        elif self.args.opt == 'rmsprop':
            optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4, lr=self.args.lr)
        else:
            self.parser.print_help()
            raise ValueError( 'Invalid optimizer value fro argument --opt:' + self.args.opt)

        return optimizer

    def resolve_loss(self):
        if self.args.dataset == 'camvid' and self.args.loss not in ['nll_loss', 'hc_loss']:
            raise ValueError('Loss ' + self.args.loss + 'is not available for dataset '+ self.args.dataset)
        elif self.args.dataset == 'make3d' and self.args.loss not in ['mse_loss', 'aleatoric_loss']:
            raise ValueError('Loss ' + self.args.loss + 'is not available for dataset ' + self.args.dataset)
        elif self.args.dataset == 'nyuv2' and (
            (self.args.task == 'regression' and self.args.loss not in ['mse_loss', 'aleatoric_loss']) or
             self.args.task == 'classification' and self.args.loss not in ['nll_loss', 'hc_loss']):
            raise ValueError('Loss ' + self.args.loss + ' is not available for task ' + self.args.task)


        loss_fct = {'nll_loss' : F.nll_loss,
                    'mse_loss' : F.mse_loss,
                    'hc_loss': hc_loss,
                    'aleatoric_loss': aleatoric_loss,
                   }

        return loss_fct[self.args.loss]

