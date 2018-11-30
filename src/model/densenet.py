import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce


def summary(model):
    print(model)
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        num_params = reduce(lambda x, y: x * y, p.shape)
        if p.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params
    print("{} trainable parameters".format(trainable_params))
    print("{} non trainable parameters".format(non_trainable_params))


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(
            nChannels,
            interChannels,
            kernel_size=1,
            bias=False
        )
        self.dropout1 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(
            interChannels,
            growthRate,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout2(out)
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(
            nChannels,
            growthRate,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout1(out)
        out = torch.cat((x, out), 1)
        return out


class TransitionDown(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(TransitionDown, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(
            nChannels,
            nOutChannels,
            kernel_size=1,
            bias=False
        )
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout1(out)
        out = F.avg_pool2d(out, 2)
        return out


class TransitionUp(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(TransitionUp, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(
            nChannels,
            nOutChannels,
            kernel_size=1,
            bias=False
        )
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(
            3,
            nChannels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = self._make_dense(
            nChannels,
            growthRate,
            nDenseBlocks,
            bottleneck
        )
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = TransitionDown(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(
            nChannels,
            growthRate,
            nDenseBlocks,
            bottleneck
        )
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = TransitionDown(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(
            nChannels,
            growthRate,
            nDenseBlocks,
            bottleneck
        )
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out
