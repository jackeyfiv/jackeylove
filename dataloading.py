import numpy as np
from skimage.measure import label, regionprops
import cv2

def keep_largest_region(mask):
    """
    保留二值分割结果中的最大连通区域

    参数:
    mask: numpy数组，二值分割结果（0为背景，1为前景）

    返回:
    largest_region: numpy数组，只包含最大连通区域的二值图像
    """
    # 检查输入是否为二值图像
    if mask.max() == 0:
        return mask  # 全背景图像直接返回

    # 标记连通区域
    labeled_mask = label(mask, connectivity=2)

    # 获取所有区域属性
    regions = regionprops(labeled_mask)

    # 如果没有找到任何区域，直接返回原图
    if not regions:
        return mask

    # 找到面积最大的区域
    largest_region = max(regions, key=lambda r: r.area)

    # 创建只包含最大区域的二值图像
    result_mask = np.zeros_like(mask)
    result_mask[labeled_mask == largest_region.label] = 1

    return result_mask
import matplotlib.pyplot as plt
from skimage.io import imread

# 读取二值分割图像
binary_mask = imread(r"E:\1\segcode\2024-08-27-10-14-52-350046_monu_10_1e-05_2_100_0.0_0\40000_val_ema_major/211008-24+4w_1-1_0_gt.png", as_gray=True)
 # 确保是二值图像
binary_mask = cv2.bitwise_not(binary_mask)
# 保留最大区域
filtered_mask = keep_largest_region(binary_mask)

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(binary_mask, cmap='gray')
axes[0].set_title('Original Mask')
axes[0].axis('off')

axes[1].imshow(filtered_mask, cmap='gray')
axes[1].set_title('Largest Region Only')
axes[1].axis('off')

plt.tight_layout()
plt.show()