from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

# 检查一个具体例子
scene = 'horns'
idx = 0

print('=== 检查数据质量 ===')
# 加载图像
org_path = f'../../eval/llff_test/eval_llff/{scene}/{idx}_pred_fine.png'
sr_path = f'../../eval/llff_test/eval_llff_sr/{scene}/{idx}_pred_sr.png' 
fine_path = f'../../eval/llff_test/eval_llff_sr/{scene}/{idx}_pred_fine.png'
gt_path = f'../../eval/llff_test/eval_llff_sr/{scene}/{idx}_gt_rgb.png'

org_img = np.array(Image.open(org_path))
sr_img = np.array(Image.open(sr_path))
fine_img = np.array(Image.open(fine_path))
gt_img = np.array(Image.open(gt_path))

print(f'ORG (pred_fine from eval_llff): {org_img.shape}')
print(f'SR (pred_sr from eval_llff_sr): {sr_img.shape}')
print(f'Fine (pred_fine from eval_llff_sr): {fine_img.shape}')
print(f'GT (gt_rgb from eval_llff_sr): {gt_img.shape}')

# 调整GT尺寸匹配SR
import cv2
if gt_img.shape != sr_img.shape:
    gt_resized = cv2.resize(gt_img, (sr_img.shape[1], sr_img.shape[0]))
else:
    gt_resized = gt_img

print(f'GT resized: {gt_resized.shape}')

# 计算PSNR
sr_psnr = psnr(gt_resized, sr_img)
org_psnr = psnr(gt_resized, org_img)

print(f'SR vs GT: {sr_psnr:.3f} dB')
print(f'ORG vs GT: {org_psnr:.3f} dB')

print(f'ORG是否比SR更好: {org_psnr > sr_psnr}')
print(f'差异: {org_psnr - sr_psnr:.3f} dB')