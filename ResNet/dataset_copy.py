import os
import shutil
import random

def copy_random_images(source_folder, dest_folder, num_images=750):
    # 支持的图片格式列表（可根据需要扩展）
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    # 获取源文件夹中所有图片文件
    all_files = [f for f in os.listdir(source_folder) 
                if f.lower().endswith(image_extensions)]
    
    if not all_files:
        print("源文件夹中没有找到图片文件")
        return
    
    # 确定实际要复制的数量
    num_to_copy = min(num_images, len(all_files))
    
    # 随机选择文件
    selected_files = random.sample(all_files, num_to_copy)
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(dest_folder, exist_ok=True)
    
    # 复制文件
    for i, filename in enumerate(selected_files, 1):
        src_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        shutil.move(src_path, dest_path)
        print(f"\r已移动 {i}/{num_to_copy} 文件", end='', flush=True)
    
    print(f"\n成功移动 {num_to_copy} 张图片到 {dest_folder}")

if __name__ == "__main__":
    # 用户需要修改以下路径
    source_directory = r"E:\Research__dir\Wave_level_Classfied\data\test\level4"  # 替换为你的源文件夹路径
    destination_directory = r"E:\Research__dir\Wave_level_Classfied\data_instance\test\level4"  # 替换为目标文件夹路径
    
    copy_random_images(source_directory, destination_directory)