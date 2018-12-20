from model.layers import *


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12, bnn=False):
        """

        :param in_channels:
        :param down_blocks:
        :param up_blocks:
        :param bottleneck_layers:
        :param growth_rate:
        :param out_chans_first_conv:
        :param n_classes: If n_classes==0, model is set in regression task
        """
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        if n_classes == 0:
            self.task = 'regression'
        else:
            self.task = 'classification'

        self.bnn = bnn

        #  First Convolution ##

        self.add_module(
            'firstconv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_chans_first_conv, kernel_size=3,
                stride=1, padding=1, bias=True)
        )
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module(
            'bottleneck',
            Bottleneck(cur_channels_count,
                       growth_rate, bottleneck_layers)
        )
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(
                prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + \
                skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        # Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + \
            skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        # Softmax ##
        if self.task == 'regression':
            n_classes = 1

        self.finalConv = nn.Conv2d(
            in_channels=cur_channels_count,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # If Bayesian Network for classification task
        if self.bnn and self.task == 'classification':

            self.finalConv_sigma = nn.Conv2d( #second header of last layer to get sigma
                in_channels=cur_channels_count,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
            self.format = nn.Softplus()
        elif self.bnn and self.task == 'regression':
            raise ValueError('Not Implemented')

        elif self.task == 'classification':
            self.format = nn.LogSoftmax(dim=1)
        elif self.task == 'regression':
            self.format = nn.ReLU()
        else:
            raise ValueError('Set up not defined')

        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.zero_()

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        f = self.finalConv(out)

        if self.bnn:
            sigma = self.finalConv_sigma(out)
            return f, self.format(sigma)
        else:
            return self.format(f),


def FCDenseNet57(n_classes, bnn):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes, bnn=bnn)


def FCDenseNet67(n_classes, bnn):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, bnn=bnn)


def FCDenseNet103(n_classes, bnn):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, bnn=bnn)
