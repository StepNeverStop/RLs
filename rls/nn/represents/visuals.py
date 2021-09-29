import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

Vis_REGISTER = {}


class SimpleConvNetwork(nn.Module):

    def __init__(self, visual_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(visual_dim[-1], 16, kernel_size=8, stride=4),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(inplace=True),
            nn.Flatten()
        )
        with th.no_grad():
            self.output_dim = np.prod(
                self.net(th.zeros(1, visual_dim[-1], visual_dim[0], visual_dim[1])).shape[1:])

    def forward(self, x):
        return self.net(x)


class NatureConvNetwork(nn.Module):

    def __init__(self, visual_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(visual_dim[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        with th.no_grad():
            self.output_dim = np.prod(
                self.net(th.zeros(1, visual_dim[-1], visual_dim[0], visual_dim[1])).shape[1:])

    def forward(self, x):
        return self.net(x)


class Match3ConvNetwork(nn.Module):

    def __init__(self, visual_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(visual_dim[-1], 35, kernel_size=3, stride=3),
            nn.ELU(inplace=True),
            nn.Conv2d(35, 144, kernel_size=1, stride=1),
            nn.ELU(inplace=True),
            nn.Flatten()
        )
        with th.no_grad():
            self.output_dim = np.prod(
                self.net(th.zeros(1, visual_dim[-1], visual_dim[0], visual_dim[1])).shape[1:])

    def forward(self, x):
        return self.net(x)


class DeepConvNetwork(nn.Sequential):

    def __init__(self,
                 visual_dim,
                 out_channels=[16, 32],
                 kernel_sizes=[[8, 8], [4, 4]],
                 stride=[[4, 4], [2, 2]],

                 use_bn=False,

                 max_pooling=False,
                 avg_pooling=False,
                 pool_sizes=[[2, 2], [2, 2]],
                 pool_strides=[[1, 1], [1, 1]],
                 ):
        super().__init__()
        conv_layers = len(out_channels)
        in_channels = [visual_dim[-1]] + out_channels[:-1]
        for i in range(conv_layers):
            self.add_module(f'conv2d_{i}', nn.Conv2d(in_channels=in_channels[i],
                                                     out_channels=out_channels[i],
                                                     kernel_size=kernel_sizes[i],
                                                     stride=stride[i]))
            self.add_module(f'relu_{i}', nn.ReLU())
            if use_bn:
                self.add_module(f'bachnorm2d_{i}', nn.BatchNorm2d(out_channels[i]))

            if max_pooling:
                self.add_module(f'maxpool2d_{i}', nn.MaxPool2d(kernel_size=pool_sizes[i],
                                                               stride=pool_strides[i]))
            elif avg_pooling:
                self.add_module(f'avgpool2d_{i}', nn.AvgPool2d(kernel_size=pool_sizes[i],
                                                               stride=pool_strides[i]))
        self.add_module('flatten', nn.Flatten())

        with th.no_grad():
            self.output_dim = np.prod(
                self(th.zeros(1, visual_dim[-1], visual_dim[0], visual_dim[1])).shape[1:])


class ResnetNetwork(nn.Module):

    def __init__(self, visual_dim):
        super().__init__()
        self.out_channels = [16, 32, 32]
        in_channels = [visual_dim[-1]] + self.out_channels[:-1]
        self.res_blocks = 2
        for i in range(len(self.out_channels)):
            setattr(self, 'conv' + str(i), nn.Conv2d(
                in_channels=in_channels[i], out_channels=self.out_channels[i], kernel_size=[3, 3], stride=(1, 1)))
            setattr(self, 'pool' + str(i),
                    nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding='same'))
            for j in range(self.res_blocks):
                setattr(self, 'resblock' + str(i) + 'conv' + str(j), nn.Conv2d(
                    in_channels=self.out_channels[i], out_channels=self.out_channels[i], kernel_size=[3, 3],
                    stride=(1, 1), padding='same'))
        self.flatten = nn.Flatten()

        with th.no_grad():
            self.output_dim = np.prod(
                self.net(th.zeros(1, visual_dim[-1], visual_dim[0], visual_dim[1])).shape[1:])

    def forward(self, x):
        """
           -----------------------------------multi conv layer---------------------------------
           ↓                                             ----multi residual block-------      ↑
           ↓                                             ↓                             ↑      ↑
        x - > conv -> x -> max_pooling -> x(block_x) -> relu -> x -> resnet_conv -> x => x ↘ ↑
                                               ↓                                         +    x -> relu -> x -> flatten -> x
                                               --------------residual add----------------↑ ↗
        """
        for i in range(len(self.out_channels)):
            x = getattr(self, 'conv' + str(i))(x)
            block_x = x = getattr(self, 'pool' + str(i))(x)
            for j in range(self.res_blocks):
                x = F.relu(x)
                x = getattr(self, 'resblock' + str(i) + 'conv' + str(j))(x)
            x += block_x
        x = F.relu(x)
        x = self.flatten(x)
        return x


Vis_REGISTER['simple'] = SimpleConvNetwork
Vis_REGISTER['nature'] = NatureConvNetwork
Vis_REGISTER['match3'] = Match3ConvNetwork
Vis_REGISTER['deepconv'] = DeepConvNetwork
Vis_REGISTER['resnet'] = ResnetNetwork
