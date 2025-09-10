import torch
import torch.nn as nn
import torchvision.models as models

def resnet18_pretrained(num_classes=3, fine_tune=True):
    """
    获取预训练的 ResNet 模型并适配灰度图输入
    Args:
        num_classes: 分类数量
        fine_tune: 是否进行微调训练
    Returns:
        model: 修改后的 ResNet 模型
    """
    # 加载预训练的 ResNet18
    model = models.resnet18(pretrained=True)
    
    # 修改第一个卷积层以接受单通道输入
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # 将预训练权重的第一个卷积层权重调整为单通道
    # 方法1：取RGB通道的平均值
    model.conv1.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)
    
    # 是否进行微调
    if fine_tune:
        # 选择性地冻结部分层
        for name, param in model.named_parameters():
            if "layer1" in name or "conv1" in name:
                param.requires_grad = False
    else:
        # 冻结所有预训练参数
        for param in model.parameters():
            param.requires_grad = False
    
    # 修改最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# """resnet in pytorch



# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

#     Deep Residual Learning for Image Recognition
#     https://arxiv.org/abs/1512.03385v1
# """

# # import torch
# # import torch.nn as nn

# class BasicBlock(nn.Module):
#     """ResNet的基本块结构"""
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         # 主要路径
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                               stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                               stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         # shortcut连接
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * self.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * self.expansion,
#                          kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion)
#             )

#     def forward(self, x):
#         identity = self.shortcut(x)

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity
#         out = self.relu(out)

#         return out

# class ResNet18(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
        
#         # 初始层 - 适用于灰度图输入
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
        
#         # 主干网络
#         self.layer1 = self._make_layer(64, 64, 2, stride=1)
#         self.layer2 = self._make_layer(64, 128, 2, stride=2)
#         self.layer3 = self._make_layer(128, 256, 2, stride=2)
#         self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
#         # 分类头
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )
        
#         # 权重初始化
#         self._initialize_weights()

#     def _make_layer(self, in_channels, out_channels, num_blocks, stride):
#         layers = []
#         layers.append(BasicBlock(in_channels, out_channels, stride))
#         for _ in range(1, num_blocks):
#             layers.append(BasicBlock(out_channels, out_channels))
#         return nn.Sequential(*layers)

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv1(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
        
#         return x

# def resnet18(num_classes=3):
#     """
#     获取ResNet18模型实例
#     Args:
#         num_classes: 分类数量
#     Returns:
#         model: ResNet18模型
#     """
#     return ResNet18(num_classes=num_classes)




# 此为github上开源resnet模型

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=3):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18(num_classes=3):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=3):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=3):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=3):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=3):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)





