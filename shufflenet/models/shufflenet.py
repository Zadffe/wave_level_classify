# import torch
# import torch.nn as nn

# class ShuffleBlock(nn.Module):
#     def __init__(self, groups=2):
#         super(ShuffleBlock, self).__init__()
#         self.groups = groups

#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         channels_per_group = channels // self.groups
#         x = x.view(batch_size, self.groups, channels_per_group, height, width)
#         x = torch.transpose(x, 1, 2).contiguous()
#         x = x.view(batch_size, -1, height, width)
#         return x

# class ShuffleUnitV2(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ShuffleUnitV2, self).__init__()
#         self.stride = stride
#         self.out_channels = out_channels
        
#         if stride == 1:
#             # 分支1: 通道分离
#             self.branch1 = nn.Sequential()
#             # 分支2: 主要计算分支
#             self.branch2 = nn.Sequential(
#                 nn.Conv2d(in_channels//2, in_channels//2, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(in_channels//2),
#                 nn.ReLU(True),
#                 nn.Conv2d(in_channels//2, in_channels//2, 3, stride, 1, groups=in_channels//2, bias=False),
#                 nn.BatchNorm2d(in_channels//2),
#                 nn.Conv2d(in_channels//2, out_channels//2, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_channels//2),
#                 nn.ReLU(True)
#             )
#         else:
#             self.branch1 = nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
#                 nn.BatchNorm2d(in_channels),
#                 nn.Conv2d(in_channels, out_channels//2, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_channels//2),
#                 nn.ReLU(True)
#             )
#             self.branch2 = nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(in_channels),
#                 nn.ReLU(True),
#                 nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
#                 nn.BatchNorm2d(in_channels),
#                 nn.Conv2d(in_channels, out_channels//2, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_channels//2),
#                 nn.ReLU(True)
#             )

#     def forward(self, x):
#         if self.stride == 1:
#             x1, x2 = x.chunk(2, dim=1)
#             out = torch.cat((x1, self.branch2(x2)), dim=1)
#         else:
#             out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        
#         return self.channel_shuffle(out, 2)
    
#     def channel_shuffle(self, x, groups):
#         batch_size, channels, height, width = x.size()
#         channels_per_group = channels // groups
#         x = x.view(batch_size, groups, channels_per_group, height, width)
#         x = torch.transpose(x, 1, 2).contiguous()
#         x = x.view(batch_size, -1, height, width)
#         return x

# class ShuffleNetV2(nn.Module):
#     def __init__(self, num_classes=3):
#         super(ShuffleNetV2, self).__init__()
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 24, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(24),
#             nn.ReLU(True)
#         )
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.stage2 = self._make_stage(24, 116, 4)
#         self.stage3 = self._make_stage(116, 232, 8)
#         self.stage4 = self._make_stage(232, 464, 4)
        
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(464, 1024, 1, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(True)
#         )
        
#         self.fc = nn.Linear(1024, num_classes)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#     def _make_stage(self, in_channels, out_channels, num_blocks):
#         layers = []
#         layers.append(ShuffleUnitV2(in_channels, out_channels, 2))
#         for _ in range(num_blocks-1):
#             layers.append(ShuffleUnitV2(out_channels, out_channels, 1))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool(x)
        
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         x = self.conv5(x)
        
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# def shufflenet(num_classes=3):
#     return ShuffleNetV2(num_classes=num_classes)






# """shufflenet in pytorch

#     此为github上的shufflenet神经网络模型

# [1] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun.

#     ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
#     https://arxiv.org/abs/1707.01083v2
# """

# from functools import partial

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

    # def forward(self, x):
    #     batch_size, channels, height, width = x.shape
    #     channels_per_group = channels // self.groups
        
    #     # 使用 reshape 代替 view+transpose
    #     x = x.reshape(batch_size, self.groups, channels_per_group, height, width)
    #     x = x.permute(0, 2, 1, 3, 4)
    #     return x.reshape(batch_size, -1, height, width)

class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups):
        super().__init__()

        #"""Similar to [9], we set the number of bottleneck channels to 1/4
        #of the output channels for each ShuffleNet unit."""
        self.bottlneck = nn.Sequential(
            PointwiseConv2d(
                input_channels,
                int(output_channels / 4),
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        #"""Note that for Stage 2, we do not apply group convolution on the first pointwise
        #layer because the number of input channels is relatively small."""
        if stage == 2:
            self.bottlneck = nn.Sequential(
                PointwiseConv2d(
                    input_channels,
                    int(output_channels / 4),
                    groups=groups
                ),
                nn.ReLU(inplace=True)
            )

        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            groups=int(output_channels / 4),
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv2d(
            int(output_channels / 4),
            output_channels,
            groups=groups
        )

        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        #"""As for the case where ShuffleNet is applied with stride,
        #we simply make two modifications (see Fig 2 (c)):
        #(i) add a 3 × 3 average pooling on the shortcut path;
        #(ii) replace the element-wise addition with channel concatenation,
        #which makes it easy to enlarge channel dimension with little extra
        #computation cost.
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            self.expand = PointwiseConv2d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups
            )

            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output

class ShuffleNet(nn.Module):

    def __init__(self, num_blocks, num_classes=3, groups=3):
        super().__init__()

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv2d(1, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[0],
            out_channels[1],
            stride=2,
            stage=2,
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[1],
            out_channels[2],
            stride=2,
            stage=3,
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups):
        """make shufflenet stage

        Args:
            block: block type, shuffle unit
            out_channels: output depth channel number of this stage
            num_blocks: how many blocks per stage
            stride: the stride of the first block of this stage
            stage: stage index
            groups: group number of group convolution
        Return:
            return a shuffle net stage
        """
        strides = [stride] + [1] * (num_blocks - 1)

        stage = []

        for stride in strides:
            stage.append(
                block(
                    self.input_channels,
                    output_channels,
                    stride=stride,
                    stage=stage,
                    groups=groups
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)

def shufflenet(num_classes=3):
    return ShuffleNet([4, 8, 4], num_classes=num_classes, groups=2)




