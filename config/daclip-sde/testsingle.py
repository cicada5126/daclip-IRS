import os
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import lpips
import utils as util
# 假设您已经有了两张图片的路径
dir1 = r"C:\Users\86136\Desktop\LQ_test\old photo\GT\176_GT.jpg"
dir2 = r"C:\Users\86136\Desktop\daclip-uir-main\results\daclip-sde\universal-ir\Test\176_GT.png"

img1 = np.array(Image.open(dir1).convert('RGB'))
img2 = np.array(Image.open(dir2).convert('RGB'))

# 确保两张图片都是 uint8 格式
# img1 = img1 / 255.0 * 255
# img2 = img2 / 255.0 * 255
# img1 = img1.astype(np.uint8)
# img2 = img2.astype(np.uint8)

psnr = calculate_psnr(img1, img2, data_range=255)
print(f"PSNR: {psnr} dB")

# 确保图像尺寸至少为7x7
if min(img1.shape[:2]) < 7:
    raise ValueError("Images must be at least 7x7 pixels in size.")

# 计算 SSIM
# 对于RGB图像，通道位于第三个轴（轴索引为2）
ssim = calculate_ssim(img1, img2, win_size=11, channel_axis=2, multichannel=True)
print(f"SSIM: {ssim:.4f}")

# 计算 LPIPS
# 假设您已经有了一个预先加载的 LPIPS 模型
lpips_model = lpips.LPIPS(net='alex')  # 这里使用默认的 'alex' 网络
# 将图片转换为 torch 张量并标准化到 [-1, 1] 范围
img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).float()
img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).float()
img1_tensor = (img1_tensor - 0.5) * 2
img2_tensor = (img2_tensor - 0.5) * 2
# 计算 LPIPS 距离
lpips_score = lpips_model(img1_tensor.unsqueeze(0), img2_tensor.unsqueeze(0)).squeeze().item()
print(f"LPIPS: {lpips_score}")