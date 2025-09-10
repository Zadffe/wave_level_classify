"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from torchvision import models

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=3):
        super().__init__()
        self.features = features

        # 添加自适应平均池化层，确保输出大小为 7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平特征图，保持批次维度

        # # output = output.view(output.size()[ 0], -1)
        x = self.classifier(x)

        # output = self.features(x)
        # output = output.view(output.size()[0], -1)
        # output = self.classifier(output)

        return x

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 1
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn(num_classes=3):
    model = VGG(make_layers(cfg['D'], batch_norm=True), num_class=num_classes)
    # 修改第一层卷积的输入通道
    first_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    model.features[0] = first_conv
    return model

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))



def vgg16_pretrained(num_classes=3, pretrained=True):
    """
    获取预训练的VGG16模型
    Args:
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
    Returns:
        model: 修改后的VGG16模型
    """
    # 加载预训练的VGG16模型
    model = models.vgg16_bn(pretrained=pretrained)
    
    # 修改第一个卷积层以适应单通道输入

    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    original_conv = model.features[0]
    # 将三通道预训练权重平均到单通道
    with torch.no_grad():
        new_weight = original_conv.weight.sum(dim=1, keepdim=True) / 3.0
        model.features[0].weight.copy_(new_weight)

    
    # 修改分类器
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(inplace=True),
        # nn.Dropout(),
        # nn.Linear(4096, 4096),
        # nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    
    return model


