from PIL import Image
import os

# 目标裁剪尺寸
crop_width = 1856
crop_height = 1458

# 仅处理这个分辨率的图像
target_original_size = (2456, 2058)

# 文件夹路径（修改为你的路径）
image_folder = r"E:\Research__dir\Wave_level_Classfied\ResNet\data\test\level2"  # 例如：r"D:\your\images"

# 支持的图像扩展名
extensions = [".jpg", ".jpeg", ".png"]

# 批量处理
for filename in os.listdir(image_folder):
    if any(filename.lower().endswith(ext) for ext in extensions):
        img_path = os.path.join(image_folder, filename)
        with Image.open(img_path) as img:
            if img.size == target_original_size:
                # 计算中心裁剪区域坐标
                left = (img.width - crop_width) // 2
                top = (img.height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                # 裁剪并保存
                cropped = img.crop((left, top, right, bottom))
                cropped.save(img_path)
                print(f"已裁剪并保存：{filename}")
            else:
                print(f"跳过：{filename}（尺寸为 {img.size}）")
