import cv2
import numpy as np

def check_image_type(image_path):
    """
    判断图像是灰度图还是RGB图
    
    参数:
        image_path (str): 图像文件路径
        
    返回:
        str: "GRAY" (灰度图), "RGB" (RGB彩色图) 或 "UNKNOWN" (其他类型)
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 获取图像形状
    shape = img.shape
    dim = len(shape)
    
    # 判断图像类型
    if dim == 2:
        return "GRAY"  # 二维数组 = 灰度图
    
    elif dim == 3:
        channels = shape[2]
        
        if channels == 1:
            return "GRAY"  # 三维数组但只有1个通道 = 灰度图
            
        elif channels == 3:
            # 检查是否是真正的RGB图（三个通道值不同）
            # 灰度图存储为RGB时三个通道值相同
            b, g, r = cv2.split(img)
            
            # 比较三个通道的差异
            diff_rg = np.sum(cv2.absdiff(r, g))
            diff_rb = np.sum(cv2.absdiff(r, b))
            
            # 如果通道间存在明显差异，则是真正的RGB图
            if diff_rg > 1000 or diff_rb > 1000:
                return "RGB"
            else:
                return "GRAY (stored as RGB)"
        
        elif channels == 4:
            return "RGBA (with alpha channel)"
    
    return "UNKNOWN"

# 示例用法
if __name__ == "__main__":
    image_path = r"E:\Research__dir\Wave_level_Classfied\data\train\level2\image_00002.jpg"  # 替换为你的图像路径
    result = check_image_type(image_path)
    
    print(f"图像类型: {result}")
    print(f"详细解释:")
    
    if result == "GRAY":
        print(" - 单通道灰度图像")
    elif result == "RGB":
        print(" - 三通道RGB彩色图像")
    elif result == "GRAY (stored as RGB)":
        print(" - 实际是灰度图像，但以RGB格式存储（三个通道值相同）")
    elif result == "RGBA (with alpha channel)":
        print(" - 四通道图像（RGB + 透明度通道）")
    else:
        print(" - 未知图像类型")