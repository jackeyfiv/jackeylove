import numpy as np
import os
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from matplotlib import pylab as plt
from scipy import ndimage as ndi


def draw_fig(img_mat, title='img_mat'):
    plt.close()
    plt.title(title)
    plt.imshow(img_mat, cmap='gray')
    plt.show()
    plt.close()


def image_fill(image):
    """
    选取最大连通分量，根据图像特征，漫水算法的起始点是图像的中心。
    :param image: [h,w]
    :return: 选取最大连通分量之后的label。[h,w]
    """
    ori = image.copy()
    src = image.copy()

    # 确保是二值图像
    if len(np.unique(ori)) > 2:
        _, ori = cv2.threshold(ori, 127, 255, cv2.THRESH_BINARY)
        _, src = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)

    mask = np.zeros([src.shape[0] + 2, src.shape[1] + 2], np.uint8)

    # 找到中心点
    center_x = src.shape[1] // 2
    center_y = src.shape[0] // 2

    # 执行漫水填充
    cv2.floodFill(
        src,
        mask,
        seedPoint=(center_x, center_y),
        newVal=255,
        loDiff=0,
        upDiff=0,
        flags=cv2.FLOODFILL_FIXED_RANGE
    )

    output = ori - src
    blank_pic = np.zeros_like(src)
    mask_circle = cv2.circle(
        blank_pic,
        (center_x, center_y),
        min(center_x, center_y) - 5,
        1,
        -1
    )
    output = output * mask_circle

    return output


def fill_hole(image):
    """填充图像中的空洞"""
    # 确保是二值图像
    if len(np.unique(image)) > 2:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    src = image.copy()
    mask = np.zeros([src.shape[0] + 2, src.shape[1] + 2], np.uint8)

    # 找到背景点
    isbreak = False
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i, j] == 0:
                seedpoint = (j, i)  # 注意坐标顺序 (x,y)
                isbreak = True
                break
        if isbreak:
            break

    cv2.floodFill(src, mask, seedpoint, 255)
    img_floofill_inv = cv2.bitwise_not(src)
    im_out = image | img_floofill_inv

    return im_out


def post_process(origin_mask):
    """后处理函数，保留最大连通区域"""
    # 确保是二值图像
    if len(np.unique(origin_mask)) > 2:
        _, origin_mask = cv2.threshold(origin_mask, 127, 255, cv2.THRESH_BINARY)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iterations = 2
    processed = cv2.morphologyEx(origin_mask, cv2.MORPH_OPEN, kernel, iterations)

    # 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)

    # 如果没有连通区域，直接返回
    if num_labels <= 1:
        return processed

    # 找到最大的连通区域（跳过背景）
    max_size = 0
    max_label = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_size:
            max_size = area
            max_label = label

    # 创建只包含最大连通区域的图像
    output = np.zeros_like(processed)
    output[labels == max_label] = 255

    return output


if __name__ == "__main__":
    # 1. 读取图像 (自动转换为灰度图)
    image_path = r"E:\1\segcode\2024-08-27-10-14-52-350046_monu_10_1e-05_2_100_0.0_0\40000_val_ema_major/211008-24+4w_1-1_0_gt.png"
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # 2. 互换前景和背景值
    # 使用位运算取反，将黑色变白色，白色变黑色
    inverted = cv2.bitwise_not(original)

    # 3. 处理图像 (使用反转后的图像)
    processed = image_fill(inverted.copy())

    # 4. 后处理 (可选)
    final_result = post_process(processed)

    # 5. 显示结果
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(inverted, cmap='gray')
    plt.title("Inverted Image (前景背景互换)")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(final_result, cmap='gray')
    plt.title("Processed Result")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=300)
    plt.show()

    # 保存结果
    cv2.imwrite("inverted_image.png", inverted)
    cv2.imwrite("processed_result.png", final_result)
    print("结果已保存: inverted_image.png, processed_result.png")