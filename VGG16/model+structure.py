import torch
from torchviz import make_dot
from torchsummary import summary
from models.vgg import vgg16_bn  # 确保导入你的模型定义

# 加载模型权重
state_dict = torch.load("./checkpoints/vgg_unpre.pth")  # 替换为你的 .pth 文件路径

# 检查权重文件的内容
print("模型权重中的键：", state_dict.keys())

# 构建与权重匹配的模型

model = vgg16_bn(num_classes=3)  # 根据你的模型定义
model.load_state_dict(state_dict, strict=False)  # 加载权重
model.eval()

# 打印模型结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
summary(model, input_size=(1, 224, 224))  # 假设输入是单通道 224x224 图像

# 绘制模型计算图
dummy_input = torch.randn(1, 1, 224, 224).to(device)  # 创建一个虚拟输入
output = model(dummy_input)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("model_structure")  # 保存为 model_structure.png