import os
import random
import uuid

def shuffle_and_rename_images(folder_path):
    """
    对指定文件夹中的图片进行随机排序并按顺序重命名为00001、00002...格式
    保留原始文件扩展名，确保无文件名冲突
    """
    # 支持的图片格式（可根据需要扩展）
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', 
                       '.bmp', '.tiff', '.webp', '.heic')
    
    # 获取所有图片文件
    original_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(image_extensions)]
    
    if not original_files:
        print("文件夹中没有找到图片文件")
        return

    # 第一阶段：重命名所有文件为唯一临时文件名
    temp_files = []
    for filename in original_files:
        # 生成唯一临时文件名
        ext = os.path.splitext(filename)[1]
        temp_name = f"temp_{uuid.uuid4().hex}{ext}"
        
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, temp_name)
        
        os.rename(src, dst)
        temp_files.append(temp_name)

    # 打乱文件顺序
    random.shuffle(temp_files)

    # 第二阶段：重命名为数字序号
    for index, temp_name in enumerate(temp_files, 1):
        # 生成新文件名（五位数字 + 原扩展名）
        ext = os.path.splitext(temp_name)[1]
        new_name = f"image_{index:05d}{ext}"
        
        src = os.path.join(folder_path, temp_name)
        dst = os.path.join(folder_path, new_name)
        
        os.rename(src, dst)
        print(f"已重命名：{new_name}")

    print(f"\n操作完成！共处理 {len(original_files)} 张图片")

if __name__ == "__main__":
    # 需要修改的目标文件夹路径
    target_folder = r"E:\Research__dir\Wave_level_Classfied\ResNet\data\test\level4"
    
    # 执行操作
    shuffle_and_rename_images(target_folder)