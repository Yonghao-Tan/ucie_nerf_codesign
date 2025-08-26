import os
import numpy as np
from PIL import Image, ImageDraw
import cv2

def handle_patches():
    """
    处理图片patches：提取、分割和降采样
    """
    # 输入图片路径
    input_image_path = "/home/ytanaz/access/IBRNet/data/nerf_synthetic/lego/val/r_2.png"
    input_image_path = "/home/ytanaz/access/IBRNet/data/nerf_synthetic/ship/val/r_70.png"
    
    # 创建patches文件夹
    patches_dir = "./patches"
    os.makedirs(patches_dir, exist_ok=True)
    
    # 读取图片
    try:
        image = Image.open(input_image_path)
        print(f"成功读取图片: {input_image_path}")
        print(f"图片尺寸: {image.size}")
    except Exception as e:
        print(f"读取图片失败: {e}")
        return
    
    # 提取240*40的区域，从坐标(100, 300)开始
    x_start, y_start = 100, 400
    patch_size = 100
    width, height = patch_size*6, patch_size
    
    # 检查坐标是否在图片范围内
    if x_start + width > image.width or y_start + height > image.height:
        print(f"警告: 提取区域超出图片边界")
        print(f"图片尺寸: {image.size}, 提取区域: ({x_start}, {y_start}) 到 ({x_start + width}, {y_start + height})")
    
    # 提取区域
    extracted_region = image.crop((x_start, y_start, x_start + width, y_start + height))
    print(f"提取区域尺寸: {extracted_region.size}")
    
    # 保存原始完整图片
    original_filename = os.path.join(patches_dir, "original_image.png")
    image.save(original_filename)
    print(f"保存原始图片: {original_filename}")
    
    # 创建带红色框的图片副本
    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)
    
    # 画红色矩形框，线宽为3像素
    box_coords = [x_start, y_start, x_start + width, y_start + height]
    draw.rectangle(box_coords, outline="red", width=3)
    
    # 保存带红色框的图片
    boxed_filename = os.path.join(patches_dir, "original_with_selection.png")
    image_with_box.save(boxed_filename)
    print(f"保存带红色框的图片: {boxed_filename}")
    print(f"红色框区域: ({x_start}, {y_start}) 到 ({x_start + width}, {y_start + height})")
    
    # 分割为6个40*40的patch
    patches = []
    
    for i in range(6):
        x_patch_start = i * patch_size
        patch = extracted_region.crop((x_patch_start, 0, x_patch_start + patch_size, patch_size))
        patches.append(patch)
        
        # 保存原始patch
        patch_filename = os.path.join(patches_dir, f"patch_{i+1}.png")
        patch.save(patch_filename)
        print(f"保存patch {i+1}: {patch_filename}")
    
    # 对每个patch进行降采样到20*20
    for i, patch in enumerate(patches):
        # 使用PIL的NEAREST方法进行降采样（round to nearest）
        downsampled_patch = patch.resize((patch_size//4, patch_size//4), Image.NEAREST)
        
        # 保存降采样后的patch
        low_reso_filename = os.path.join(patches_dir, f"patch_{i+1}_low_reso.png")
        downsampled_patch.save(low_reso_filename)
        print(f"保存降采样patch {i+1}: {low_reso_filename}")
    
    print(f"\n处理完成！所有文件保存在: {patches_dir}")
    print(f"生成了:")
    print(f"- 1个原始完整图片")
    print(f"- 1个带红色选择框的图片")
    print(f"- 6个40x40的原始patches")
    print(f"- 6个20x20的降采样patches")

if __name__ == "__main__":
    handle_patches()
