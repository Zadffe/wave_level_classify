import torch
from PIL import Image
from torchvision import transforms
from models.resnet import resnet18_pretrained
import argparse
import logging

def setup_transforms():
    """设置图像预处理流程"""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )
    ])

def predict_image(image_path, model_path, class_names):
    """
    预测单张图片的海况等级
    Args:
        image_path: 图片路径
        model_path: 模型路径
        class_names: 类别名称列表
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = resnet18_pretrained(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 加载并预处理图片
    try:
        image = Image.open(image_path).convert('L')
    except Exception as e:
        logging.error(f"无法打开图片: {e}")
        return None
    
    # 图像转换
    transform = setup_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
    
    # 获取结果
    predicted_label = class_names[predicted_class.item()]
    confidence = predicted_prob.item()
    
    return predicted_label, confidence

def main():
    parser = argparse.ArgumentParser(description="海况等级预测")
    parser.add_argument("--image_path", type=str, default=r"E:\Research__dir\Wave_level_Classfied\image_00339.jpg", help="输入图片路径")
    parser.add_argument("--model_path", type=str, default="./resnet_pre.pth", help="模型路径")
    args = parser.parse_args()
    
    # 定义类别名称（需要与训练时保持一致）
    class_names = ["Level_2", "Level_3", "Level_4"]  # 根据实际情况修改
    
    # 预测
    result, confidence = predict_image(args.image_path, args.model_path, class_names)
    
    # 输出结果
    print(f"\n预测结果:")
    print(f"图片路径: {args.image_path}")
    print(f"海况等级: {result}")
    print(f"置信度: {confidence:.2%}")

if __name__ == "__main__":
    main()