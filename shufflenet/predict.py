import torch
from PIL import Image
import torchvision.transforms as transforms
from models.vgg import vgg16_bn
import argparse

class WavePredictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # 初始化模型
        self.model = vgg16_bn(num_classes=3).to(device)
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])
        
        # 类别映射
        self.class_names = ['level2', 'level3', 'level4']  # 根据实际类别修改
        
    def predict(self, image_path):
        # 加载并预处理图像
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return {
            'class_name': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence * 100,  # 转换为百分比
            'probabilities': {
                class_name: prob.item() * 100
                for class_name, prob in zip(self.class_names, probabilities[0])
            }
        }

def main():
    parser = argparse.ArgumentParser(description='预测海况等级')
    parser.add_argument('--model_path', type=str, default="./checkpoints/vgg.pth", help='模型权重路径')
    parser.add_argument('--image_path', type=str, default="E:\Research__dir\Wave_level_Classfied\image_00039.jpg", help='要预测的图片路径')
    args = parser.parse_args()
    
    # 创建预测器实例
    predictor = WavePredictor(args.model_path)
    
    # 进行预测
    result = predictor.predict(args.image_path)
    
    # 打印预测结果
    print(f"\n预测结果：{result['class_name']}")
    print(f"置信度：{result['confidence']:.2f}%")
    print("\n各类别概率：")
    for class_name, prob in result['probabilities'].items():
        print(f"{class_name}: {prob:.2f}%")

if __name__ == '__main__':
    main()