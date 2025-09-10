import torch
from torchsummary import summary
from models.resnet import resnet18, resnet18_pretrained  # 确保导入你的模型定义

# 构建模型
model = resnet18_pretrained(num_classes=3)  # 根据你的模型定义
model.eval()

# 打印模型结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
summary(model, input_size=(1, 224, 224))  # 假设输入是单通道 224x224 图像